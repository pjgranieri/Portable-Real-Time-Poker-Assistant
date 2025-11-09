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

#define LED_GPIO_NUM    -1

// Timing and memory constants
#define WIFI_TIMEOUT 20000
#define HTTP_TIMEOUT 30000
#define CAPTURE_INTERVAL 10000
#define MIN_FREE_HEAP 50000

// Counters
int captureCount = 0;
int failureCount = 0;

// =============================
// Initialize 24 MHz XCLK for OV5640
// =============================
void initXCLK() {
  Serial.println("🔧 Configuring 24 MHz XCLK for OV5640...");
  
  ledc_timer_config_t timer_cfg = {
    .speed_mode = LEDC_LOW_SPEED_MODE,
    .duty_resolution = LEDC_TIMER_1_BIT,
    .timer_num = LEDC_TIMER_0,
    .freq_hz = 24000000,
    .clk_cfg = LEDC_AUTO_CLK
  };
  ledc_timer_config(&timer_cfg);

  ledc_channel_config_t ch_cfg = {
    .gpio_num = XCLK_GPIO_NUM,
    .speed_mode = LEDC_LOW_SPEED_MODE,
    .channel = LEDC_CHANNEL_0,
    .intr_type = LEDC_INTR_DISABLE,
    .timer_sel = LEDC_TIMER_0,
    .duty = 1,
    .hpoint = 0
  };
  ledc_channel_config(&ch_cfg);
  
  Serial.println("✅ 24 MHz clock started on GPIO15");
}

// =============================
// Camera initialization
// =============================
void initCamera() {
  camera_config_t cam_cfg;
  cam_cfg.ledc_channel = LEDC_CHANNEL_0;
  cam_cfg.ledc_timer = LEDC_TIMER_0;
  cam_cfg.pin_d0 = Y2_GPIO_NUM;
  cam_cfg.pin_d1 = Y3_GPIO_NUM;
  cam_cfg.pin_d2 = Y4_GPIO_NUM;
  cam_cfg.pin_d3 = Y5_GPIO_NUM;
  cam_cfg.pin_d4 = Y6_GPIO_NUM;
  cam_cfg.pin_d5 = Y7_GPIO_NUM;
  cam_cfg.pin_d6 = Y8_GPIO_NUM;
void initCamera() {
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

  // Get sensor and apply settings
  sensor_t *s = esp_camera_sensor_get();
  if (s) {
    Serial.printf("📷 Detected sensor PID: 0x%04X\n", s->id.PID);
    
    // Apply same basic settings as ESP32-Cropped
    s->set_brightness(s, 0);
    s->set_contrast(s, 0);
    s->set_saturation(s, 0);
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
    s->set_exposure_ctrl(s, 1);
    s->set_gain_ctrl(s, 1);
    s->set_lenc(s, 1);
    
    if (s->id.PID == 0x5640) {
      Serial.println("🎯 OV5640 detected - autofocus available");
    }
  }
  
  // Match ESP32-Cropped warmup (single capture)
  Serial.println("🔥 Warming up camera...");
  camera_fb_t* fb = esp_camera_fb_get();
  if (fb) {
    esp_camera_fb_return(fb);
    Serial.println("✅ Camera warmup successful!");
  } else {
    Serial.println("⚠️ Camera warmup failed, but continuing...");
  }
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
  
  Serial.printf("\n📸 Capturing %s ROI (#%d)...\n", roiNames[roiIndex], captureCount);
  
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("❌ WiFi disconnected");
    if (!connectWiFi()) return false;
  }

  size_t freeHeap = ESP.getFreeHeap();
  Serial.printf("📊 Free heap: %u bytes\n", freeHeap);
  
  if (freeHeap < MIN_FREE_HEAP) {
    Serial.println("❌ Insufficient memory");
    return false;
  }

  // Single capture attempt (like ESP32-Cropped)
  camera_fb_t* fb = esp_camera_fb_get();
  
  if (!fb) {
    Serial.println("❌ Camera capture failed");
    failureCount++;
    return false;
  }
  
  Serial.printf("✅ Captured: %d bytes, %dx%d\n", fb->len, fb->width, fb->height);
  
  // Encode to base64
  String imageBase64 = base64::encode(fb->buf, fb->len);
  
  if (imageBase64.length() == 0) {
    Serial.println("❌ Base64 encoding failed");
    esp_camera_fb_return(fb);
    failureCount++;
    return false;
  }

  // Create JSON with ROI metadata
  DynamicJsonDocument doc(imageBase64.length() + 2048);
  
  doc["device_id"] = "ESP32S3_OV5640";
  doc["timestamp"] = millis();
  doc["image_format"] = "jpg";
  doc["image_width"] = fb->width;
  doc["image_height"] = fb->height;
  doc["image_size"] = fb->len;
  doc["image_data"] = imageBase64;
  
  JsonObject roi = doc.createNestedObject("roi");
  roi["section"] = roiNames[roiIndex];
  roi["x"] = roiStartX[roiIndex];
  roi["y"] = 0;
  roi["width"] = roiWidths[roiIndex];
  roi["height"] = IMAGE_HEIGHT;

  String jsonString;
  serializeJson(doc, jsonString);
  
  esp_camera_fb_return(fb);

  // HTTP POST
  HTTPClient http;
  http.begin(serverURL);
  http.setTimeout(HTTP_TIMEOUT);
  http.addHeader("Content-Type", "application/json");
  http.addHeader("X-API-Key", apiKey);

  int httpResponseCode = http.POST(jsonString);
  
  bool success = false;
  if (httpResponseCode > 0) {
    Serial.printf("✅ Upload OK: %d\n", httpResponseCode);
    success = true;
  } else {
    Serial.printf("❌ Upload failed: %d\n", httpResponseCode);
    failureCount++;
  }

  http.end();
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