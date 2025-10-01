#include <Arduino.h>
#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include <ArduinoJson.h>
#include <base64.h>

// === WiFi Configuration ===
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// === Azure Configuration ===
const char* azureHost = "your-app-name.azurewebsites.net";
const char* azureEndpoint = "/api/upload-image";
const char* apiKey = "YOUR_API_KEY"; // Optional, for authentication

// === FORIOT / ESP32-S3-EYE style pin map ===
#define XCLK_GPIO     15
#define SIOD_GPIO     4
#define SIOC_GPIO     5
#define VSYNC_GPIO    6
#define HREF_GPIO     7
#define PCLK_GPIO     13
#define Y2_GPIO       11
#define Y3_GPIO       9
#define Y4_GPIO       8
#define Y5_GPIO       10
#define Y6_GPIO       12
#define Y7_GPIO       18
#define Y8_GPIO       17
#define Y9_GPIO       16

#define CAM_RESET_GPIO  -1
#define CAM_PWDN_GPIO   -1

// Global camera configuration
camera_config_t camera_config;
bool camera_initialized = false;

void initWiFi() {
  Serial.print("Connecting to WiFi");
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.println("✅ WiFi Connected!");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println();
    Serial.println("❌ WiFi Connection Failed!");
  }
}

void initCamera() {
  camera_config.ledc_channel = LEDC_CHANNEL_0;
  camera_config.ledc_timer   = LEDC_TIMER_0;
  camera_config.pin_d0 = Y2_GPIO;
  camera_config.pin_d1 = Y3_GPIO;
  camera_config.pin_d2 = Y4_GPIO;
  camera_config.pin_d3 = Y5_GPIO;
  camera_config.pin_d4 = Y6_GPIO;
  camera_config.pin_d5 = Y7_GPIO;
  camera_config.pin_d6 = Y8_GPIO;
  camera_config.pin_d7 = Y9_GPIO;
  camera_config.pin_xclk     = XCLK_GPIO;
  camera_config.pin_pclk     = PCLK_GPIO;
  camera_config.pin_vsync    = VSYNC_GPIO;
  camera_config.pin_href     = HREF_GPIO;
  camera_config.pin_sscb_sda = SIOD_GPIO;
  camera_config.pin_sscb_scl = SIOC_GPIO;
  camera_config.pin_pwdn     = CAM_PWDN_GPIO;
  camera_config.pin_reset    = CAM_RESET_GPIO;

  camera_config.xclk_freq_hz = 20000000;
  camera_config.pixel_format = PIXFORMAT_JPEG;
  camera_config.frame_size   = FRAMESIZE_VGA;  // Start with VGA
  camera_config.jpeg_quality = 10;
  camera_config.fb_count     = 2;
  camera_config.fb_location  = CAMERA_FB_IN_PSRAM;
  camera_config.grab_mode    = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&camera_config);
  if (err != ESP_OK) {
    Serial.printf("❌ Camera init failed: 0x%x\n", err);
    camera_initialized = false;
  } else {
    Serial.println("✅ Camera initialized successfully");
    camera_initialized = true;
  }
}

bool uploadImageToAzure(camera_fb_t* fb) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("❌ WiFi not connected");
    return false;
  }

  WiFiClientSecure client;
  client.setInsecure(); // For testing - in production, use proper certificates
  
  HTTPClient https;
  
  String url = "https://" + String(azureHost) + String(azureEndpoint);
  
  if (!https.begin(client, url)) {
    Serial.println("❌ HTTPS connection failed");
    return false;
  }

  // Set headers
  https.addHeader("Content-Type", "application/json");
  if (strlen(apiKey) > 0) {
    https.addHeader("X-API-Key", apiKey);
  }
  https.addHeader("User-Agent", "ESP32-S3-Camera/1.0");

  // Create JSON payload
  DynamicJsonDocument doc(fb->len * 2 + 1000); // Extra space for metadata
  
  // Encode image to base64
  String encodedImage = base64::encode(fb->buf, fb->len);
  
  doc["device_id"] = WiFi.macAddress();
  doc["timestamp"] = millis();
  doc["image_format"] = "jpeg";
  doc["image_width"] = fb->width;
  doc["image_height"] = fb->height;
  doc["image_size"] = fb->len;
  doc["image_data"] = encodedImage;

  String jsonString;
  serializeJson(doc, jsonString);

  Serial.printf("📤 Uploading image: %dx%d (%d bytes)\n", 
                fb->width, fb->height, fb->len);
  Serial.printf("JSON payload size: %d bytes\n", jsonString.length());

  int httpResponseCode = https.POST(jsonString);
  
  if (httpResponseCode > 0) {
    String response = https.getString();
    Serial.printf("✅ HTTP Response: %d\n", httpResponseCode);
    Serial.printf("Response: %s\n", response.c_str());
    
    https.end();
    return (httpResponseCode >= 200 && httpResponseCode < 300);
  } else {
    Serial.printf("❌ HTTP Error: %d\n", httpResponseCode);
    https.end();
    return false;
  }
}

bool captureAndUpload() {
  if (!camera_initialized) {
    Serial.println("❌ Camera not initialized");
    return false;
  }

  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("❌ Camera capture failed");
    return false;
  }

  bool success = uploadImageToAzure(fb);
  
  esp_camera_fb_return(fb);
  return success;
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n=== ESP32-S3-CAM Azure HTTPS Upload ===");
  Serial.printf("PSRAM: %s (%u bytes)\n",
                psramFound() ? "YES" : "NO", ESP.getPsramSize());

  // Initialize WiFi
  initWiFi();
  
  if (WiFi.status() == WL_CONNECTED) {
    // Initialize Camera
    initCamera();
    
    if (camera_initialized) {
      Serial.println("🚀 System ready! Starting image capture and upload...");
      
      // Take initial test shot
      delay(2000);
      captureAndUpload();
    }
  } else {
    Serial.println("❌ System initialization failed - no WiFi");
  }
}

void loop() {
  if (WiFi.status() == WL_CONNECTED && camera_initialized) {
    delay(10000); // Wait 10 seconds between captures
    
    Serial.println("\n--- Capturing new frame ---");
    if (captureAndUpload()) {
      Serial.println("✅ Upload successful!");
    } else {
      Serial.println("❌ Upload failed!");
    }
  } else {
    Serial.println("⚠️ System not ready - checking connections...");
    
    // Try to reconnect WiFi if needed
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("Attempting WiFi reconnection...");
      initWiFi();
    }
    
    delay(5000);
  }
}