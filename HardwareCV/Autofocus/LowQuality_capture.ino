#include "Arduino.h"
#include "driver/ledc.h"
#include "esp_camera.h"

// ==== Pin map for your ESP32-S3 + OV5640 ====
#define PWDN_GPIO_NUM   10    // Power down (active HIGH)
#define RESET_GPIO_NUM  11    // Reset (active LOW)
#define XCLK_GPIO_NUM   15    // External clock output
#define SIOD_GPIO_NUM   8     // SDA
#define SIOC_GPIO_NUM   9     // SCL

#define Y2_GPIO_NUM     38    // D2
#define Y3_GPIO_NUM     39    // D3
#define Y4_GPIO_NUM     40    // D4
#define Y5_GPIO_NUM     41    // D5
#define Y6_GPIO_NUM     42    // D6
#define Y7_GPIO_NUM     45    // D7
#define Y8_GPIO_NUM     48    // D8
#define Y9_GPIO_NUM     47    // D9

#define VSYNC_GPIO_NUM  12    // VS
#define HREF_GPIO_NUM   13    // HS
#define PCLK_GPIO_NUM   14    // PC

void setup() {
  Serial.begin(115200);
  delay(800);
  Serial.println("\n-- OV5640 Init + Frame Capture Test (Final) --");

  // ==== Generate 24 MHz XCLK on GPIO15 ====
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

  // ==== Prepare camera config ====
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;

  config.xclk_freq_hz = 24000000;
  config.pixel_format = PIXFORMAT_JPEG;   // JPEG = fastest sanity check
  config.frame_size   = FRAMESIZE_QQVGA;
  config.jpeg_quality = 12;
  config.fb_count     = 1;
  config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location  = CAMERA_FB_IN_DRAM;

  // ==== Wake camera ====
  pinMode(PWDN_GPIO_NUM, OUTPUT);
  pinMode(RESET_GPIO_NUM, OUTPUT);
  digitalWrite(PWDN_GPIO_NUM, LOW);   // awake
  digitalWrite(RESET_GPIO_NUM, HIGH); // release reset
  delay(50);

  // ==== Initialize camera ====
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("❌ Camera init failed: 0x%x\n", err);
    return;
  }
  Serial.println("✅ Camera initialized!");

  // ==== Capture one frame ====
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("❌ Frame capture failed!");
    return;
  }

  Serial.printf("✅ Captured frame: %u bytes, %dx%d\n",
                fb->len, fb->width, fb->height);

  esp_camera_fb_return(fb);
  Serial.println("🎉 Camera test complete!");
}

void loop() {}
