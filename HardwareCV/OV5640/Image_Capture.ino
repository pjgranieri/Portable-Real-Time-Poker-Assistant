#include <Arduino.h>
#include "esp_camera.h"

// === FORIOT / ESP32-S3-CAM pin map ===
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

// =============================
// Camera config helper
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
  cfg.fb_count     = 1;                  // use 1 for stability at high res
  cfg.grab_mode    = CAMERA_GRAB_LATEST;
}

// =============================
// Apply sensor tuning
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
    s->set_quality(s, 10);
  }

  if (s->id.PID == OV5640_PID) {
    Serial.println("Tuning OV5640...");

    // Base fixes
    s->set_brightness(s, 0);
    s->set_contrast(s, 1);
    s->set_saturation(s, 2);   // stronger color
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
    s->set_exposure_ctrl(s, 1);
    s->set_gain_ctrl(s, 1);
    s->set_lenc(s, 1);

    // Apply presets
    switch (preset) {
      case 0: // Auto
        s->set_wb_mode(s, 0);
        s->set_ae_level(s, 0);
        break;
      case 1: // Indoor (warm)
        s->set_wb_mode(s, 4);  // home/indoor
        s->set_ae_level(s, -1);
        break;
      case 2: // Outdoor (sunny)
        s->set_wb_mode(s, 1);  // sunny
        s->set_ae_level(s, 0);
        break;
      case 3: // Low light
        s->set_wb_mode(s, 0);
        s->set_ae_level(s, 2);
        break;
    }

    // Safe resolution start
    s->set_framesize(s, FRAMESIZE_SVGA);  // 800x600
    s->set_quality(s, 8);
  }
}

// =============================
// Capture + send over Serial
// =============================
static void capture_and_send() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("❌ Capture failed");
    return;
  }

  Serial.printf("✅ Frame: %d bytes, %dx%d\n", fb->len, fb->width, fb->height);
  Serial.println("===FRAME_START===");
  Serial.write(fb->buf, fb->len);
  Serial.println("===FRAME_END===");
  esp_camera_fb_return(fb);
}

// =============================
// Setup + Loop
// =============================
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n=== ESP32-S3-CAM Auto (OV2640 / OV5640) ===");
  Serial.printf("PSRAM: %s (%u bytes)\n",
                psramFound() ? "YES" : "NO", ESP.getPsramSize());

  camera_config_t cfg;
  fill_cfg(cfg, FRAMESIZE_VGA, 10); // safe default
  esp_err_t err = esp_camera_init(&cfg);
  if (err != ESP_OK) {
    Serial.printf("❌ Camera init failed: 0x%x\n", err);
    return;
  }

  // Get sensor + tune
  sensor_t *s = esp_camera_sensor_get();
  if (s) {
    Serial.printf("Detected sensor PID: 0x%04X\n", s->id.PID);
    // preset: 0=auto, 1=indoor, 2=outdoor, 3=lowlight
    tune_sensor(s, 0);  
  }

  // First capture
  capture_and_send();
}

void loop() {
  delay(5000);
  capture_and_send();
}
