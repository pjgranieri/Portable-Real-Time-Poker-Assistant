#include <WiFi.h>
#include <HTTPClient.h>
#include "esp_camera.h"
#include "driver/ledc.h"

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

#define CAPTURE_INTERVAL 10000
#define MAX_RETRIES 5

int captureCount = 0;
int successCount = 0;
int failureCount = 0;

void setupCamera() {
  // Power on camera
  pinMode(PWDN_GPIO_NUM, OUTPUT);
  pinMode(RESET_GPIO_NUM, OUTPUT);
  digitalWrite(PWDN_GPIO_NUM, LOW);
  digitalWrite(RESET_GPIO_NUM, HIGH);
  delay(50);
  
  // Setup 10MHz clock
  ledc_timer_config_t timer_conf = {
    .speed_mode = LEDC_LOW_SPEED_MODE,
    .duty_resolution = LEDC_TIMER_1_BIT,
    .timer_num = LEDC_TIMER_0,
    .freq_hz = 10000000,
    .clk_cfg = LEDC_AUTO_CLK
  };
  ledc_timer_config(&timer_conf);

  ledc_channel_config_t channel_conf = {
    .gpio_num = XCLK_GPIO_NUM,
    .speed_mode = LEDC_LOW_SPEED_MODE,
    .channel = LEDC_CHANNEL_0,
    .intr_type = LEDC_INTR_DISABLE,
    .timer_sel = LEDC_TIMER_0,
    .duty = 1,
    .hpoint = 0
  };
  ledc_channel_config(&channel_conf);
  
  // Configure camera
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 10000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_VGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;

  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("Camera init failed");
    while(1) delay(1000);
  }

  // Flip image vertically only (no mirror)
  sensor_t *s = esp_camera_sensor_get();
  if (s) {
    s->set_hmirror(s, 0);  // No horizontal mirror
    s->set_vflip(s, 1);    // Vertical flip only
  }

  // Warm up
  for (int i = 0; i < 3; i++) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (fb) esp_camera_fb_return(fb);
    delay(100);
  }
  
  Serial.println("Camera ready");
}

bool isValidJPEG(const uint8_t* data, size_t len) {
  if (len < 10) return false;
  if (data[0] != 0xFF || data[1] != 0xD8) return false;
  if (data[len-2] != 0xFF || data[len-1] != 0xD9) return false;
  return true;
}

bool captureAndUpload() {
  captureCount++;
  
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected");
    return false;
  }
  
  // Try multiple times to get a valid frame
  camera_fb_t* fb = nullptr;
  for (int attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    if (attempt > 1) {
      Serial.printf("Retry %d/%d...\n", attempt, MAX_RETRIES);
      delay(200);
    }
    
    fb = esp_camera_fb_get();
    if (!fb) continue;
    
    if (isValidJPEG(fb->buf, fb->len)) {
      Serial.printf("Valid frame on attempt %d: %d bytes\n", attempt, fb->len);
      break;
    }
    
    esp_camera_fb_return(fb);
    fb = nullptr;
  }
  
  if (!fb) {
    Serial.println("Failed to get valid frame");
    failureCount++;
    return false;
  }
  
  // Upload
  HTTPClient http;
  http.begin(serverURL);
  http.setTimeout(30000);
  http.addHeader("Content-Type", "image/jpeg");
  http.addHeader("X-API-Key", apiKey);
  http.addHeader("X-Device-ID", "ESP32S3_OV5640");
  http.addHeader("X-Image-Width", String(fb->width));
  http.addHeader("X-Image-Height", String(fb->height));
  http.addHeader("X-Timestamp", String(millis()));

  int code = http.POST(fb->buf, fb->len);
  http.end();
  esp_camera_fb_return(fb);

  if (code == 200 || code == 201) {
    successCount++;
    Serial.printf("Upload #%d OK (success rate: %.1f%%)\n\n", 
                  captureCount, 100.0 * successCount / captureCount);
    return true;
  }
  
  Serial.printf("Upload failed: HTTP %d\n", code);
  failureCount++;
  return false;
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n=== ESP32-S3 OV5640 ===");
  Serial.println("Hardware JPEG with validation & retry\n");
  
  setupCamera();
  
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.printf("\nConnected: %s\n\n", WiFi.localIP().toString().c_str());
}

void loop() {
  captureAndUpload();
  delay(CAPTURE_INTERVAL);
}