#include <Arduino.h>
#include "esp_camera.h"

// === FORIOT / ESP32-S3-EYE style pin map (confirmed from silkscreen) ===
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

// Optional — set if you discover RESET/PWDN pads; else leave -1
#define CAM_RESET_GPIO  -1
#define CAM_PWDN_GPIO   -1

void setupCamera(int xclk_hz, framesize_t fs) {
  camera_config_t cfg;
  cfg.ledc_channel = LEDC_CHANNEL_0;
  cfg.ledc_timer   = LEDC_TIMER_0;
  cfg.pin_d0 = Y2_GPIO;
  cfg.pin_d1 = Y3_GPIO;
  cfg.pin_d2 = Y4_GPIO;
  cfg.pin_d3 = Y5_GPIO;
  cfg.pin_d4 = Y6_GPIO;
  cfg.pin_d5 = Y7_GPIO;
  cfg.pin_d6 = Y8_GPIO;
  cfg.pin_d7 = Y9_GPIO;
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
  cfg.frame_size   = fs;           // FRAMESIZE_QVGA = 320x240
  cfg.jpeg_quality = 12;           // lower = better quality
  cfg.fb_count     = 1;
  cfg.fb_location  = CAMERA_FB_IN_PSRAM;
  cfg.grab_mode    = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&cfg);
  if (err != ESP_OK) {
    Serial.printf("❌ Camera init failed (xclk=%d): 0x%x\n", xclk_hz, err);
    return;
  }

  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.printf("❌ Capture failed (xclk=%d)\n", xclk_hz);
  } else {
    Serial.printf("✅ Frame captured: %d bytes, %dx%d\n", fb->len, fb->width, fb->height);

    // --- Send JPEG over Serial with markers ---
    Serial.println("===FRAME_START===");
    Serial.write(fb->buf, fb->len);
    Serial.println("===FRAME_END===");

    esp_camera_fb_return(fb);
  }

  esp_camera_deinit();
}

void setup() {
  Serial.begin(115200);
  delay(800);
  Serial.println("\n=== ESP32-S3-CAM OV2640 Serial Capture ===");
  Serial.printf("PSRAM: %s (%u bytes)\n", psramFound() ? "YES" : "NO", ESP.getPsramSize());

  // Try 320x240 first (Qvga). Change to FRAMESIZE_VGA, SVGA, UXGA if stable
  setupCamera(20000000, FRAMESIZE_QVGA);
}

void loop() {
  delay(5000);   // capture every 5s
  setupCamera(20000000, FRAMESIZE_QVGA);
}
