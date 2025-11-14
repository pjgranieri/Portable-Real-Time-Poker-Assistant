#include <WiFi.h>
#include <WiFiClient.h>
#include <HTTPClient.h>
#include "esp_camera.h"
#include "driver/ledc.h"
#include "SPIFFS.h"
#include "FS.h"

const char* ssid     = "designlab";
const char* password = "designlab1";
const char* serverURL = "http://20.246.97.176:3000/api/upload-image-raw";
const char* apiKey = "ewfgjiohewiuhwe8934yt83gigiuewhui83h8ge84849g4h489g";

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

#define WIFI_TIMEOUT 20000
#define HTTP_TIMEOUT 30000
#define CAPTURE_INTERVAL 10000
#define MIN_FREE_HEAP 50000
#define MAX_CONSECUTIVE_FAILURES 5
#define BACKOFF_MULTIPLIER 2
#define MAX_BACKOFF_DELAY 60000

int captureCount = 0;
int failureCount = 0;
int consecutiveFailures = 0;
unsigned long currentBackoffDelay = CAPTURE_INTERVAL;

void setupCamera() {
  camera_config_t cam;
  cam.ledc_channel = LEDC_CHANNEL_0;
  cam.ledc_timer = LEDC_TIMER_0;
  cam.pin_d0 = Y2_GPIO_NUM;
  cam.pin_d1 = Y3_GPIO_NUM;
  cam.pin_d2 = Y4_GPIO_NUM;
  cam.pin_d3 = Y5_GPIO_NUM;
  cam.pin_d4 = Y6_GPIO_NUM;
  cam.pin_d5 = Y7_GPIO_NUM;
  cam.pin_d6 = Y8_GPIO_NUM;
  cam.pin_d7 = Y9_GPIO_NUM;
  cam.pin_xclk = XCLK_GPIO_NUM;
  cam.pin_pclk = PCLK_GPIO_NUM;
  cam.pin_vsync = VSYNC_GPIO_NUM;
  cam.pin_href = HREF_GPIO_NUM;
  cam.pin_sscb_sda = SIOD_GPIO_NUM;
  cam.pin_sscb_scl = SIOC_GPIO_NUM;
  cam.pin_pwdn = PWDN_GPIO_NUM;
  cam.pin_reset = RESET_GPIO_NUM;
  cam.xclk_freq_hz = 20000000;  // Reduced from 24MHz for stability
  cam.pixel_format = PIXFORMAT_JPEG;
  cam.frame_size = FRAMESIZE_VGA;
  cam.jpeg_quality = 12;  // 10-63, lower = higher quality
  cam.fb_count = 2;  // Use 2 frame buffers
  cam.grab_mode = CAMERA_GRAB_LATEST;
  
  // Try PSRAM first, fall back to DRAM if not available
  if (psramFound()) {
    Serial.println("✅ PSRAM found");
    cam.fb_location = CAMERA_FB_IN_PSRAM;
  } else {
    Serial.println("⚠️  No PSRAM, using DRAM");
    cam.fb_location = CAMERA_FB_IN_DRAM;
  }

  esp_err_t err = esp_camera_init(&cam);
  if (err != ESP_OK) {
    Serial.printf("❌ Camera init failed: 0x%x\n", err);
    while(1) delay(1000);
  }

  sensor_t *sensor = esp_camera_sensor_get();
  if (sensor) {
    // Reset to defaults first
    sensor->set_brightness(sensor, 0);
    sensor->set_contrast(sensor, 0);
    sensor->set_saturation(sensor, 0);
    sensor->set_sharpness(sensor, 0);
    sensor->set_denoise(sensor, 0);
    sensor->set_gainceiling(sensor, GAINCEILING_2X);
    sensor->set_quality(sensor, 12);
    sensor->set_colorbar(sensor, 0);
    sensor->set_whitebal(sensor, 1);
    sensor->set_gain_ctrl(sensor, 1);
    sensor->set_exposure_ctrl(sensor, 1);
    sensor->set_hmirror(sensor, 0);
    sensor->set_vflip(sensor, 0);
    sensor->set_awb_gain(sensor, 1);
    sensor->set_agc_gain(sensor, 0);
    sensor->set_aec_value(sensor, 300);
    sensor->set_aec2(sensor, 0);
    sensor->set_dcw(sensor, 1);
    sensor->set_bpc(sensor, 0);
    sensor->set_wpc(sensor, 1);
    sensor->set_raw_gma(sensor, 1);
    sensor->set_lenc(sensor, 1);
    sensor->set_special_effect(sensor, 0);
    sensor->set_wb_mode(sensor, 0);
    sensor->set_ae_level(sensor, 0);
  }
  
  // Discard first few frames (they're often corrupted)
  Serial.println("🔄 Warming up camera...");
  for (int i = 0; i < 3; i++) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (fb) {
      esp_camera_fb_return(fb);
      Serial.printf("  Discarded frame %d\n", i + 1);
    }
    delay(100);
  }
  
  Serial.println("✅ Camera ready");
}

bool setupWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED) {
    if (millis() - start > WIFI_TIMEOUT) {
      Serial.println("❌ WiFi timeout");
      return false;
    }
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("\n✅ WiFi connected");
  Serial.println(WiFi.localIP());
  return true;
}

// Comprehensive JPEG validation
bool isValidJPEG(const uint8_t* data, size_t len) {
  if (len < 10) {
    Serial.println("❌ Image too small");
    return false;
  }
  
  // Check SOI (Start of Image) marker
  if (data[0] != 0xFF || data[1] != 0xD8) {
    Serial.printf("❌ Invalid JPEG SOI: 0x%02X 0x%02X (expected 0xFF 0xD8)\n", data[0], data[1]);
    return false;
  }
  
  // Check EOI (End of Image) marker
  if (data[len-2] != 0xFF || data[len-1] != 0xD9) {
    Serial.printf("❌ Invalid JPEG EOI: 0x%02X 0x%02X (expected 0xFF 0xD9)\n", data[len-2], data[len-1]);
    return false;
  }
  
  // Check for JFIF or Exif marker
  bool hasValidHeader = false;
  for (size_t i = 2; i < len - 10; i++) {
    if (data[i] == 0xFF && data[i+1] == 0xE0) {  // JFIF
      hasValidHeader = true;
      break;
    }
    if (data[i] == 0xFF && data[i+1] == 0xE1) {  // Exif
      hasValidHeader = true;
      break;
    }
    if (i > 100) break;  // Don't search too far
  }
  
  if (!hasValidHeader) {
    Serial.println("⚠️  No JFIF/Exif header found");
  }
  
  return true;
}

// Save image locally for debugging
void saveToSPIFFS(const uint8_t* data, size_t len) {
  if (!SPIFFS.begin(true)) {
    Serial.println("⚠️  SPIFFS mount failed");
    return;
  }
  
  String filename = "/debug_" + String(millis()) + ".jpg";
  File file = SPIFFS.open(filename, FILE_WRITE);
  if (file) {
    file.write(data, len);
    file.close();
    Serial.printf("💾 Saved debug copy to SPIFFS: %s\n", filename.c_str());
  }
}

bool uploadImage() {
  captureCount++;
  
  // Check WiFi
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("⚠️ WiFi disconnected, reconnecting...");
    if (!setupWiFi()) {
      Serial.println("❌ WiFi reconnection failed");
      return false;
    }
  }
  
  // Check heap
  uint32_t freeHeap = ESP.getFreeHeap();
  if (freeHeap < MIN_FREE_HEAP) {
    Serial.printf("❌ Low memory: %d bytes free (need %d)\n", freeHeap, MIN_FREE_HEAP);
    return false;
  }

  // Capture image
  camera_fb_t* img = esp_camera_fb_get();
  if (!img) {
    Serial.println("❌ Camera capture failed");
    failureCount++;
    return false;
  }

  Serial.printf("📸 Captured: %dx%d, %d bytes, format: %d\n", 
                img->width, img->height, img->len, img->format);
  
  // Print first 16 bytes for debugging
  Serial.print("   First bytes: ");
  for (int i = 0; i < 16 && i < img->len; i++) {
    Serial.printf("%02X ", img->buf[i]);
  }
  Serial.println();
  
  // Validate JPEG
  if (!isValidJPEG(img->buf, img->len)) {
    Serial.println("❌ Invalid JPEG - discarding");
    
    // Save for debugging on first failure
    if (failureCount == 0) {
      saveToSPIFFS(img->buf, img->len);
    }
    
    esp_camera_fb_return(img);
    failureCount++;
    return false;
  }

  // Send HTTP request
  HTTPClient http;
  http.begin(serverURL);
  http.setTimeout(HTTP_TIMEOUT);
  http.addHeader("Content-Type", "image/jpeg");
  http.addHeader("X-API-Key", apiKey);
  http.addHeader("X-Device-ID", "ESP32S3_OV5640");
  http.addHeader("X-Image-Width", String(img->width));
  http.addHeader("X-Image-Height", String(img->height));
  http.addHeader("X-Timestamp", String(millis()));

  Serial.println("📤 Uploading raw binary...");
  int code = http.POST(img->buf, img->len);
  String response = http.getString();
  http.end();

  esp_camera_fb_return(img);

  // Handle response
  if (code == 200 || code == 201) {
    Serial.printf("✅ Upload #%d successful (HTTP %d)\n", captureCount, code);
    consecutiveFailures = 0;
    currentBackoffDelay = CAPTURE_INTERVAL;
    return true;
  } else if (code > 0) {
    Serial.printf("⚠️ Upload failed with HTTP %d: %s\n", code, response.c_str());
  } else {
    Serial.printf("❌ Upload failed: %s\n", http.errorToString(code).c_str());
  }
  
  failureCount++;
  consecutiveFailures++;
  
  if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
    Serial.printf("🚨 ALERT: %d consecutive failures!\n", consecutiveFailures);
    currentBackoffDelay = min(currentBackoffDelay * BACKOFF_MULTIPLIER, (unsigned long)MAX_BACKOFF_DELAY);
    Serial.printf("⏱️ New delay: %d ms\n", currentBackoffDelay);
  }
  
  return false;
}

void printStats() {
  Serial.println("\n📊 Statistics:");
  Serial.printf("  Total captures: %d\n", captureCount);
  Serial.printf("  Total failures: %d\n", failureCount);
  Serial.printf("  Consecutive failures: %d\n", consecutiveFailures);
  Serial.printf("  Success rate: %.1f%%\n", captureCount > 0 ? (100.0 * (captureCount - failureCount) / captureCount) : 0);
  Serial.printf("  Free heap: %d bytes\n", ESP.getFreeHeap());
  Serial.printf("  PSRAM free: %d bytes\n", ESP.getFreePsram());
  Serial.printf("  Current delay: %d ms\n\n", currentBackoffDelay);
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n=== ESP32-S3 OV5640 Camera Debug ===");
  Serial.printf("ESP-IDF Version: %s\n", esp_get_idf_version());
  Serial.printf("Chip Model: %s\n", ESP.getChipModel());
  Serial.printf("PSRAM: %s\n", psramFound() ? "Yes" : "No");
  Serial.printf("Free Heap: %d bytes\n", ESP.getFreeHeap());
  if (psramFound()) {
    Serial.printf("Free PSRAM: %d bytes\n", ESP.getFreePsram());
  }
  Serial.println();
  
  setupCamera();
  
  if (!setupWiFi()) {
    Serial.println("❌ WiFi initialization failed");
    while(1) delay(1000);
  }
  
  Serial.println("🚀 Ready\n");
}

void loop() {
  uploadImage();
  
  if (captureCount % 10 == 0) {
    printStats();
  }
  
  delay(currentBackoffDelay);
}