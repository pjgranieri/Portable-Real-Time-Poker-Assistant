#include <WiFi.h>
#include <WiFiClient.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <base64.h>
#include "esp_camera.h"
#include "driver/ledc.h"

// WiFi credentials
const char* ssid     = "designlab";
const char* password = "designlab1";

// Server details
const char* serverURL = "http://20.246.97.176:3000/api/upload-image";
const char* apiKey = "ewfgjiohewiuhwe8934yt83gigiuewhui83h8ge84849g4h489g";

// Camera pin configuration - ESP32-S3 + OV5640
#define PWDN_GPIO_NUM   10
#define RESET_GPIO_NUM  11
#define XCLK_GPIO_NUM   15
#define SIOD_GPIO_NUM   8
#define SIOC_GPIO_NUM   9

#define Y9_GPIO_NUM     47
#define Y8_GPIO_NUM     48
#define Y7_GPIO_NUM     45
#define Y6_GPIO_NUM     42
#define Y5_GPIO_NUM     41
#define Y4_GPIO_NUM     40
#define Y3_GPIO_NUM     39
#define Y2_GPIO_NUM     38
#define VSYNC_GPIO_NUM  12
#define HREF_GPIO_NUM   13
#define PCLK_GPIO_NUM   14

#define LED_GPIO_NUM    -1  // No LED on this module

// Timing and memory constants
#define WIFI_TIMEOUT 20000     // 20 seconds
#define HTTP_TIMEOUT 30000     // 30 seconds
#define CAPTURE_INTERVAL 10000 // 10 seconds between full capture cycles
#define MIN_FREE_HEAP 50000    // Minimum heap before upload

// ROI (Region of Interest) definitions for 640x480 image
// Left section: 0-212 (213 pixels wide)
// Middle section: 213-426 (214 pixels wide)
// Right section: 427-639 (213 pixels wide)
#define IMAGE_WIDTH 640
#define IMAGE_HEIGHT 480
#define LEFT_WIDTH 213
#define MIDDLE_WIDTH 214
#define RIGHT_WIDTH 213

// Counter for debugging
int captureCount = 0;
int failureCount = 0;
int currentROI = 0; // 0 = left, 1 = middle, 2 = right

// =============================
// Initialize 24 MHz XCLK for OV5640
// =============================
void initXCLK() {
  Serial.println("🔧 Configuring 24 MHz XCLK for OV5640...");
  
  ledc_timer_config_t ledc_timer = {
    .speed_mode       = LEDC_LOW_SPEED_MODE,
    .duty_resolution  = LEDC_TIMER_1_BIT,
    .timer_num        = LEDC_TIMER_0,
    .freq_hz          = 24000000,
    .clk_cfg          = LEDC_AUTO_CLK
  };
  ledc_timer_config(&ledc_timer);
  
  ledc_channel_config_t ledc_channel = {
    .gpio_num       = XCLK_GPIO_NUM,
    .speed_mode     = LEDC_LOW_SPEED_MODE,
    .channel        = LEDC_CHANNEL_0,
    .intr_type      = LEDC_INTR_DISABLE,
    .timer_sel      = LEDC_TIMER_0,
    .duty           = 1,
    .hpoint         = 0
  };
  ledc_channel_config(&ledc_channel);
  
  Serial.println("✅ 24 MHz clock started on GPIO15");
}

// =============================
// Camera configuration for OV5640
// =============================
void initCamera() {
  // Wake up the camera first
  pinMode(PWDN_GPIO_NUM, OUTPUT);
  pinMode(RESET_GPIO_NUM, OUTPUT);
  digitalWrite(PWDN_GPIO_NUM, LOW);   // Power ON (active HIGH to power down)
  digitalWrite(RESET_GPIO_NUM, HIGH); // Release reset (active LOW)
  delay(50);
  
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;

  config.xclk_freq_hz = 24000000;       // OV5640 works best at 24MHz
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size   = FRAMESIZE_VGA;  // 640x480
  config.jpeg_quality = 10;             // Lower quality = smaller size
  config.fb_location  = CAMERA_FB_IN_PSRAM; // ESP32-S3 has PSRAM
  config.fb_count     = 1;
  config.grab_mode    = CAMERA_GRAB_LATEST;

  // Initialize camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("❌ Camera init failed: 0x%x\n", err);
    while(1) {
      delay(1000);
    }
  }

  Serial.println("✅ Camera initialized successfully!");

  // Get sensor and apply OV5640-specific settings
  sensor_t *s = esp_camera_sensor_get();
  if (s) {
    Serial.printf("📷 Detected sensor PID: 0x%04X\n", s->id.PID);
    
    // OV5640-specific sensor settings
    s->set_brightness(s, 0);      // -2 to 2
    s->set_contrast(s, 0);        // -2 to 2
    s->set_saturation(s, 0);      // -2 to 2
    s->set_whitebal(s, 1);        // Enable auto white balance
    s->set_awb_gain(s, 1);        // Enable auto white balance gain
    s->set_exposure_ctrl(s, 1);   // Enable auto exposure
    s->set_aec2(s, 1);            // Enable AEC sensor
    s->set_gain_ctrl(s, 1);       // Enable auto gain
    s->set_agc_gain(s, 0);        // Auto gain value (0-30)
    s->set_bpc(s, 1);             // Enable black pixel correction
    s->set_wpc(s, 1);             // Enable white pixel correction
    s->set_lenc(s, 1);            // Enable lens correction
    s->set_dcw(s, 1);             // Enable downsize cropping window
    s->set_colorbar(s, 0);        // Disable color bar test pattern
    
    // OV5640 supports autofocus - trigger it
    if (s->id.PID == 0x5640) {  // OV5640 PID
      Serial.println("🎯 OV5640 detected - autofocus available");
      // Note: Autofocus can be triggered per capture or continuously
      // For continuous AF, you might need additional register writes
    }
  }
  
  // Warm up camera
  Serial.println("🔥 Warming up camera...");
  for (int i = 0; i < 3; i++) {  // Take a few frames to stabilize
    camera_fb_t* fb = esp_camera_fb_get();
    if (fb) {
      esp_camera_fb_return(fb);
    }
    delay(100);
  }
  Serial.println("✅ Camera warmup successful!");
}

// =============================
// WiFi connection
// =============================
bool connectWiFi() {
  Serial.println("🔗 Connecting to WiFi...");
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  
  unsigned long startTime = millis();
  while (WiFi.status() != WL_CONNECTED) {
    if (millis() - startTime > WIFI_TIMEOUT) {
      Serial.println("❌ WiFi connection timeout");
      return false;
    }
    delay(500);
    Serial.print(".");
  }
  
  Serial.println();
  Serial.println("✅ WiFi connected!");
  Serial.print("📡 IP address: ");
  Serial.println(WiFi.localIP());
  return true;
}

// =============================
// Capture full image and upload ROI
// =============================
bool captureAndUploadROI(int roiIndex) {
  captureCount++;
  
  const char* roiNames[] = {"LEFT", "MIDDLE", "RIGHT"};
  int roiWidths[] = {LEFT_WIDTH, MIDDLE_WIDTH, RIGHT_WIDTH};
  int roiStartX[] = {0, LEFT_WIDTH, LEFT_WIDTH + MIDDLE_WIDTH};
  
  Serial.printf("\n📸 Capturing %s ROI (Attempt #%d)...\n", roiNames[roiIndex], captureCount);
  Serial.printf("📊 Statistics: %d captures, %d failures\n", captureCount - 1, failureCount);
  Serial.printf("🔲 ROI: x=%d, width=%d, height=%d\n", roiStartX[roiIndex], roiWidths[roiIndex], IMAGE_HEIGHT);
  
  // Check WiFi connection
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("❌ WiFi disconnected, reconnecting...");
    if (!connectWiFi()) {
      return false;
    }
  }

  // Check available memory
  size_t freeHeap = ESP.getFreeHeap();
  Serial.printf("📊 Free heap: %u bytes\n", freeHeap);
  
  if (freeHeap < MIN_FREE_HEAP) {
    Serial.println("❌ Insufficient memory");
    return false;
  }

  // Capture full image
  Serial.println("📷 Requesting full frame from camera...");
  unsigned long captureStart = millis();
  camera_fb_t* fb = esp_camera_fb_get();
  unsigned long captureTime = millis() - captureStart;
  
  if (!fb) {
    Serial.printf("❌ Camera capture failed (took %lu ms)\n", captureTime);
    Serial.println("🔄 Retrying camera capture...");
    delay(100);
    
    captureStart = millis();
    fb = esp_camera_fb_get();  // Retry once
    captureTime = millis() - captureStart;
    
    if (!fb) {
      Serial.printf("❌ Camera capture retry failed (took %lu ms)\n", captureTime);
      Serial.println("⚠️ Camera may be stuck. Consider power cycling the device.");
      failureCount++;
      return false;
    }
  }
  
  Serial.printf("✅ Captured full image: %d bytes, %dx%d (took %lu ms)\n", fb->len, fb->width, fb->height, captureTime);
  
  // Validate frame buffer
  if (fb->len == 0 || fb->buf == NULL) {
    Serial.println("❌ Invalid frame buffer (empty or null)");
    esp_camera_fb_return(fb);
    failureCount++;
    return false;
  }

  // For JPEG images, we'll send metadata about the ROI with the full image
  // The server can handle the cropping, or we send the full image with ROI metadata
  Serial.printf("🔄 Preparing %s ROI for upload...\n", roiNames[roiIndex]);
  
  // Encode full image to base64
  Serial.println("🔄 Encoding full image to base64...");
  String imageBase64 = base64::encode(fb->buf, fb->len);
  
  if (imageBase64.length() == 0) {
    Serial.println("❌ Base64 encoding failed");
    esp_camera_fb_return(fb);
    failureCount++;
    return false;
  }

  Serial.printf("✅ Base64 encoded: %d bytes\n", imageBase64.length());

  // Create JSON payload with ROI information
  Serial.println("🔄 Creating JSON payload with ROI metadata...");
  DynamicJsonDocument doc(imageBase64.length() + 2048);
  
  doc["device_id"] = "ESP32S3_OV5640_001";
  doc["timestamp"] = millis();
  doc["image_format"] = "jpg";
  doc["image_width"] = fb->width;
  doc["image_height"] = fb->height;
  doc["image_size"] = fb->len;
  doc["image_data"] = imageBase64;
  
  // Add ROI metadata
  JsonObject roi = doc.createNestedObject("roi");
  roi["section"] = roiNames[roiIndex];
  roi["x"] = roiStartX[roiIndex];
  roi["y"] = 0;
  roi["width"] = roiWidths[roiIndex];
  roi["height"] = IMAGE_HEIGHT;
  roi["index"] = roiIndex;

  String jsonString;
  serializeJson(doc, jsonString);
  
  Serial.printf("📦 JSON payload size: %d bytes\n", jsonString.length());

  // Free camera buffer
  esp_camera_fb_return(fb);

  // Send HTTP POST request
  Serial.println("🌐 Sending HTTP request...");
  
  HTTPClient http;
  http.begin(serverURL);
  http.setTimeout(HTTP_TIMEOUT);
  http.addHeader("Content-Type", "application/json");
  http.addHeader("X-API-Key", apiKey);

  int httpResponseCode = http.POST(jsonString);
  
  bool success = false;
  if (httpResponseCode > 0) {
    String response = http.getString();
    Serial.printf("✅ Upload successful! Response code: %d\n", httpResponseCode);
    Serial.println("📄 Response: " + response);
    success = true;
  } else {
    Serial.printf("❌ Upload failed! Error code: %d\n", httpResponseCode);
    String error = http.errorToString(httpResponseCode);
    Serial.println("🔍 Error: " + error);
    failureCount++;
  }

  http.end();
  
  Serial.printf("📊 Free heap after upload: %u bytes\n", ESP.getFreeHeap());
  
  return success;
}

// =============================
// Setup
// =============================
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n=== ESP32-S3 OV5640 Cropped Camera Upload Client ===");
  Serial.printf("PSRAM: %s (%u bytes)\n", psramFound() ? "YES" : "NO", ESP.getPsramSize());
  Serial.printf("Initial free heap: %u bytes\n", ESP.getFreeHeap());
  Serial.println("📐 ROI Configuration:");
  Serial.printf("   Left:   %d x %d pixels (x=0)\n", LEFT_WIDTH, IMAGE_HEIGHT);
  Serial.printf("   Middle: %d x %d pixels (x=%d)\n", MIDDLE_WIDTH, IMAGE_HEIGHT, LEFT_WIDTH);
  Serial.printf("   Right:  %d x %d pixels (x=%d)\n", RIGHT_WIDTH, IMAGE_HEIGHT, LEFT_WIDTH + MIDDLE_WIDTH);

  // Check PSRAM
  if (!psramFound()) {
    Serial.println("⚠️ PSRAM not found! Camera may not work properly.");
  }

  // Initialize XCLK first (required for OV5640)
  initXCLK();

  // Initialize camera
  Serial.println("\n📷 Initializing OV5640 camera...");
  initCamera();

  // Connect to WiFi
  if (!connectWiFi()) {
    Serial.println("💀 WiFi connection failed - halting");
    while(1) delay(1000);
  }

  Serial.println("🚀 Setup complete! Starting ROI capture loop...\n");
  Serial.println("📋 Sequence: LEFT → MIDDLE → RIGHT → (repeat)\n");
}

// =============================
// Main loop
// =============================
void loop() {
  // Capture and upload current ROI
  if (captureAndUploadROI(currentROI)) {
    Serial.printf("✅ %s ROI upload successful!\n\n", 
                  (currentROI == 0) ? "LEFT" : (currentROI == 1) ? "MIDDLE" : "RIGHT");
  } else {
    Serial.printf("❌ %s ROI upload failed!\n\n",
                  (currentROI == 0) ? "LEFT" : (currentROI == 1) ? "MIDDLE" : "RIGHT");
  }
  
  // Move to next ROI (cycle through 0, 1, 2)
  currentROI = (currentROI + 1) % 3;
  
  // Wait before next capture
  // Since we're sending 3 ROIs per cycle, we divide the interval by 3
  int waitTime = CAPTURE_INTERVAL / 3;
  Serial.printf("⏰ Waiting %d seconds before next ROI capture...\n", waitTime / 1000);
  Serial.printf("📍 Next: %s section\n\n", 
                (currentROI == 0) ? "LEFT" : (currentROI == 1) ? "MIDDLE" : "RIGHT");
  
  // Yield to allow other tasks to run
  delay(100);
  yield();
  delay(waitTime - 100);
}