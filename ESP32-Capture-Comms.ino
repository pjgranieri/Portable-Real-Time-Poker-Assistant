#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <base64.h>
#include "esp_camera.h"

// WiFi credentials (update with your actual WiFi)


// === ESP32-S3-CAM pin map (from Image_Capture.ino) ===
#define XCLK_GPIO     15
#define SIOD_GPIO      4
#define SIOC_GPIO      5
#define VSYNC_GPIO     6
#define HREF_GPIO      7
#define PCLK_GPIO     13
#define Y2_GPIO       11
#define Y3_GPIO        9
#define Y4_GPIO        8
#define Y5_GPIO       10
#define Y6_GPIO       12
#define Y7_GPIO       18
#define Y8_GPIO       17
#define Y9_GPIO       16

#define CAM_RESET_GPIO  -1
#define CAM_PWDN_GPIO   -1

// Memory and timing constants
#define MAX_IMAGE_SIZE 100000  // 100KB max
#define WIFI_TIMEOUT 20000     // 20 seconds
#define HTTP_TIMEOUT 30000     // 30 seconds
#define MIN_FREE_HEAP 50000    // Minimum heap before upload

// =============================
// Camera config helper (from Image_Capture.ino)
// =============================
static void fill_cfg(camera_config_t &cfg, framesize_t fs, int jpeg_quality, int xclk_hz = 20000000) {
  cfg.ledc_channel = LEDC_CHANNEL_0;
  cfg.ledc_timer   = LEDC_TIMER_0;
  cfg.pin_d0 = Y2_GPIO; cfg.pin_d1 = Y3_GPIO; cfg.pin_d2 = Y4_GPIO; cfg.pin_d3 = Y5_GPIO;
  cfg.pin_d4 = Y6_GPIO; cfg.pin_d5 = Y7_GPIO; cfg.pin_d6 = Y8_GPIO; cfg.pin_d7 = Y9_GPIO;
  cfg.pin_xclk     = XCLK_GPIO;
  cfg.pin_pclk     = PCLK_GPIO;
  cfg.pin_vsync    = VSYNC_GPIO;
  cfg.pin_href     = HREF_GPIO;
  cfg.pin_sscb_sda = SIOD_GPIO;
  cfg.pin_sscb_scl = SIOC_GPIO;
  cfg.pin_pwdn     = CAM_PWDN_GPIO;
  cfg.pin_reset    = CAM_RESET_GPIO;

  cfg.xclk_freq_hz = xclk_hz;
  cfg.pixel_format = PIXFORMAT_JPEG;
  cfg.frame_size   = fs;
  cfg.jpeg_quality = jpeg_quality;
  cfg.fb_location  = CAMERA_FB_IN_PSRAM;
  cfg.fb_count     = 1;
  cfg.grab_mode    = CAMERA_GRAB_LATEST;
}

// =============================
// Apply sensor tuning (from Image_Capture.ino)
// =============================
void tune_sensor(sensor_t *s, int preset) {
  if (!s) return;

  if (s->id.PID == OV2640_PID) {
    Serial.println("Tuning OV2640...");
    s->set_brightness(s, 0);
    s->set_contrast(s, 0);
    s->set_saturation(s, 0);
    s->set_whitebal(s, 1);
    s->set_exposure_ctrl(s, 1);
    s->set_gain_ctrl(s, 1);
    s->set_framesize(s, FRAMESIZE_VGA);
    s->set_quality(s, 12); // Slightly higher compression for upload
  }

  if (s->id.PID == OV5640_PID) {
    Serial.println("Tuning OV5640...");
    s->set_brightness(s, 0);
    s->set_contrast(s, 1);
    s->set_saturation(s, 2);
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
    s->set_exposure_ctrl(s, 1);
    s->set_gain_ctrl(s, 1);
    s->set_lenc(s, 1);
    
    // Auto mode for uploads
    s->set_wb_mode(s, 0);
    s->set_ae_level(s, 0);
    s->set_framesize(s, FRAMESIZE_VGA); // Start with VGA for reliable uploads
    s->set_quality(s, 12); // Higher compression for upload
  }
}

// =============================
// WiFi connection with timeout
// =============================
bool connectWiFi() {
  Serial.println("🔗 Connecting to WiFi...");
  WiFi.begin(ssid, password);
  
  unsigned long startTime = millis();
  while (WiFi.status() != WL_CONNECTED) {
    if (millis() - startTime > WIFI_TIMEOUT) {
      Serial.println("❌ WiFi connection timeout");
      return false;
    }
    delay(1000);
    Serial.print(".");
  }
  
  Serial.println();
  Serial.println("✅ WiFi connected!");
  Serial.print("📡 IP address: ");
  Serial.println(WiFi.localIP());
  return true;
}

// =============================
// Memory check function
// =============================
bool checkMemory(size_t requiredBytes) {
  size_t freeHeap = ESP.getFreeHeap();
  Serial.printf("📊 Free heap: %u bytes, Required: %u bytes\n", freeHeap, requiredBytes);
  
  if (freeHeap < requiredBytes + MIN_FREE_HEAP) {
    Serial.println("❌ Insufficient memory for operation");
    return false;
  }
  return true;
}

// =============================
// Main capture and upload function
// =============================
void captureAndUpload() {
  Serial.println("\n📸 Starting capture and upload...");
  
  // Check WiFi connection
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("❌ WiFi disconnected, reconnecting...");
    if (!connectWiFi()) {
      return;
    }
  }

  // Capture image
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("❌ Camera capture failed");
    return;
  }

  Serial.printf("📷 Captured: %d bytes, %dx%d\n", fb->len, fb->width, fb->height);

  // Check if image is too large
  if (fb->len > MAX_IMAGE_SIZE) {
    Serial.printf("❌ Image too large: %d bytes (max: %d)\n", fb->len, MAX_IMAGE_SIZE);
    esp_camera_fb_return(fb);
    return;
  }

  // Calculate memory requirements for base64 encoding
  size_t base64Size = ((fb->len + 2) / 3) * 4; // Base64 size calculation
  size_t jsonSize = base64Size + 1000; // JSON overhead
  
  // Check memory before proceeding
  if (!checkMemory(jsonSize)) {
    esp_camera_fb_return(fb);
    return;
  }

  // Encode to base64
  Serial.println("🔄 Encoding to base64...");
  String imageBase64 = base64::encode(fb->buf, fb->len);
  
  if (imageBase64.length() == 0) {
    Serial.println("❌ Base64 encoding failed");
    esp_camera_fb_return(fb);
    return;
  }

  // Create JSON payload with proper size
  Serial.println("🔄 Creating JSON payload...");
  DynamicJsonDocument doc(jsonSize);
  
  doc["device_id"] = "ESP32_001";
  doc["timestamp"] = millis();
  doc["image_format"] = "jpg";
  doc["image_width"] = fb->width;
  doc["image_height"] = fb->height;
  doc["image_size"] = fb->len;
  doc["image_data"] = imageBase64;

  // Check for JSON serialization errors
  if (doc.overflowed()) {
    Serial.println("❌ JSON document overflow");
    esp_camera_fb_return(fb);
    return;
  }

  String jsonString;
  serializeJson(doc, jsonString);
  
  Serial.printf("📦 JSON payload size: %d bytes\n", jsonString.length());

  // Free camera buffer early
  esp_camera_fb_return(fb);

  // Send HTTPS request
  Serial.println("🌐 Sending HTTPS request...");
  
  WiFiClient client;
  client.setTimeout(HTTP_TIMEOUT / 1000); // Convert to seconds
  
  HTTPClient http;
  http.begin(client, serverURL);
  http.setTimeout(HTTP_TIMEOUT);
  http.addHeader("Content-Type", "application/json");
  http.addHeader("X-API-Key", apiKey);

  int httpResponseCode = http.POST(jsonString);
  
  if (httpResponseCode > 0) {
    String response = http.getString();
    Serial.printf("✅ Upload successful! Response code: %d\n", httpResponseCode);
    Serial.println("📄 Response: " + response);
  } else {
    Serial.printf("❌ Upload failed! Error code: %d\n", httpResponseCode);
    String error = http.errorToString(httpResponseCode);
    Serial.println("🔍 Error: " + error);
  }

  http.end();
  
  // Print memory status after upload
  Serial.printf("📊 Free heap after upload: %u bytes\n", ESP.getFreeHeap());
}

// =============================
// Setup function
// =============================
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n=== ESP32-S3-CAM HTTPS Upload Client ===");
  Serial.printf("PSRAM: %s (%u bytes)\n", psramFound() ? "YES" : "NO", ESP.getPsramSize());
  Serial.printf("Initial free heap: %u bytes\n", ESP.getFreeHeap());

  // Check PSRAM availability
  if (!psramFound()) {
    Serial.println("❌ PSRAM not found! Camera may not work properly.");
  }

  // Initialize camera with error handling
  Serial.println("📷 Initializing camera...");
  camera_config_t cfg;
  fill_cfg(cfg, FRAMESIZE_VGA, 12); // VGA resolution, quality 12 for uploads
  
  esp_err_t err = esp_camera_init(&cfg);
  if (err != ESP_OK) {
    Serial.printf("❌ Camera init failed: 0x%x\n", err);
    while(1) {
      delay(1000);
      Serial.println("💀 Camera initialization failed - halting");
    }
  }

  // Get sensor and apply tuning
  sensor_t *s = esp_camera_sensor_get();
  if (s) {
    Serial.printf("✅ Detected sensor PID: 0x%04X\n", s->id.PID);
    tune_sensor(s, 0); // Auto mode
  } else {
    Serial.println("❌ Failed to get camera sensor");
  }

  // Connect to WiFi
  if (!connectWiFi()) {
    Serial.println("💀 WiFi connection failed - continuing without network");
  }

  Serial.println("🚀 Setup complete! Starting capture loop...");
}

// =============================
// Main loop
// =============================
void loop() {
  captureAndUpload();
  
  // Wait 30 seconds between uploads
  Serial.println("⏰ Waiting 30 seconds before next capture...\n");
  delay(30000);
}