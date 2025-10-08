#include <WiFi.h>
#include <WiFiClient.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <base64.h>
#include "esp_camera.h"
#include "sensor.h"  // For OV2640_PID constant

// WiFi credentials
const char* ssid = "iphone (566)";
const char* password = "hotdogpizza";

// Server details
const char* serverURL = "http://20.246.97.176:3000/api/upload-image";
const char* apiKey = "ewfgjiohewiuhwe8934yt83gigiuewhui83h8ge84849g4h489g";

// Camera pin configuration - AI-Thinker ESP32-CAM
#define PWDN_GPIO_NUM   32
#define RESET_GPIO_NUM  -1
#define XCLK_GPIO_NUM   0
#define SIOD_GPIO_NUM   26
#define SIOC_GPIO_NUM   27

#define Y9_GPIO_NUM     35
#define Y8_GPIO_NUM     34
#define Y7_GPIO_NUM     39
#define Y6_GPIO_NUM     36
#define Y5_GPIO_NUM     21
#define Y4_GPIO_NUM     19
#define Y3_GPIO_NUM     18
#define Y2_GPIO_NUM     5
#define VSYNC_GPIO_NUM  25
#define HREF_GPIO_NUM   23
#define PCLK_GPIO_NUM   22

// 4 for flash led or 33 for normal led
#define LED_GPIO_NUM    4

// Timing and memory constants
#define WIFI_TIMEOUT 20000     // 20 seconds
#define HTTP_TIMEOUT 30000     // 30 seconds
#define CAPTURE_INTERVAL 10000 // 10 seconds between captures
#define MIN_FREE_HEAP 50000    // Minimum heap before upload

// Counter for debugging
int captureCount = 0;
int failureCount = 0;

// =============================
// Camera configuration
// =============================
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

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size   = FRAMESIZE_VGA;  // 640x480
  config.jpeg_quality = 12;             // 0-63, lower means higher quality
  config.fb_location  = CAMERA_FB_IN_PSRAM;
  config.fb_count     = 2;  // Changed from 1 to 2 for better stability
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
    
    // Apply sensor settings for better image quality
    s->set_brightness(s, 0);     // -2 to 2
    s->set_contrast(s, 0);       // -2 to 2
    s->set_saturation(s, 0);     // -2 to 2
    s->set_whitebal(s, 1);       // Enable white balance
    s->set_awb_gain(s, 1);       // Enable auto white balance gain
    s->set_exposure_ctrl(s, 1);  // Enable auto exposure
    s->set_gain_ctrl(s, 1);      // Enable auto gain
    s->set_lenc(s, 1);           // Enable lens correction
    s->set_wb_mode(s, 0);        // Auto white balance mode
    
    // For OV2640, set special mode bit
    if (s->id.PID == OV2640_PID) {
      s->set_special_effect(s, 0);  // No special effect
    }
  }
  
  // Warm up the camera - take and discard first frame
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
// Capture and upload image
// =============================
bool captureAndUpload() {
  captureCount++;
  Serial.printf("\n📸 Capturing image (Attempt #%d)...\n", captureCount);
  Serial.printf("📊 Statistics: %d captures, %d failures\n", captureCount - 1, failureCount);
  
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

  // Capture image with timeout
  Serial.println("📷 Requesting frame from camera...");
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
  
  Serial.printf("✅ Captured: %d bytes, %dx%d (took %lu ms)\n", fb->len, fb->width, fb->height, captureTime);
  
  // Validate frame buffer
  if (fb->len == 0 || fb->buf == NULL) {
    Serial.println("❌ Invalid frame buffer (empty or null)");
    esp_camera_fb_return(fb);
    failureCount++;
    return false;
  }

  // Encode to base64
  Serial.println("🔄 Encoding to base64...");
  String imageBase64 = base64::encode(fb->buf, fb->len);
  
  if (imageBase64.length() == 0) {
    Serial.println("❌ Base64 encoding failed");
    esp_camera_fb_return(fb);
    return false;
  }

  Serial.printf("✅ Base64 encoded: %d bytes\n", imageBase64.length());

  // Create JSON payload
  Serial.println("🔄 Creating JSON payload...");
  DynamicJsonDocument doc(imageBase64.length() + 1024);
  
  doc["device_id"] = "ESP32_CAM_001";
  doc["timestamp"] = millis();
  doc["image_format"] = "jpg";
  doc["image_width"] = fb->width;
  doc["image_height"] = fb->height;
  doc["image_size"] = fb->len;
  doc["image_data"] = imageBase64;

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

  Serial.println("\n=== ESP32 Camera Upload Client ===");
  Serial.printf("PSRAM: %s (%u bytes)\n", psramFound() ? "YES" : "NO", ESP.getPsramSize());
  Serial.printf("Initial free heap: %u bytes\n", ESP.getFreeHeap());

  // Check PSRAM
  if (!psramFound()) {
    Serial.println("⚠️ PSRAM not found! Camera may not work properly.");
  }

  // Initialize camera
  Serial.println("\n📷 Initializing camera...");
  initCamera();

  // Connect to WiFi
  if (!connectWiFi()) {
    Serial.println("💀 WiFi connection failed - halting");
    while(1) delay(1000);
  }

  Serial.println("🚀 Setup complete! Starting capture loop...\n");
}

// =============================
// Main loop
// =============================
void loop() {
  // Capture and upload
  if (captureAndUpload()) {
    Serial.println("✅ Capture and upload successful!\n");
  } else {
    Serial.println("❌ Capture or upload failed!\n");
  }
  
  // Wait before next capture
  Serial.printf("⏰ Waiting %d seconds before next capture...\n\n", CAPTURE_INTERVAL / 1000);
  
  // Yield to allow other tasks to run
  delay(100);
  yield();
  delay(CAPTURE_INTERVAL - 100);
}
