#include <WiFi.h>
#include <HTTPClient.h>
#include "esp_camera.h"
#include "driver/ledc.h"
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// ========== LCD Setup ==========
#define SDA_PIN 2
#define SCL_PIN 3
LiquidCrystal_I2C lcd(0x27, 16, 2);  // Change 0x27 to 0x3F if your LCD uses that address

// Helper function to print to Serial + LCD
void lcdPrintln(const String &msg) {
  Serial.println(msg);
  lcd.clear();
  lcd.setCursor(0, 0);
  if (msg.length() <= 16) {
    lcd.print(msg);
  } else {
    // Split long message over 2 lines
    lcd.print(msg.substring(0, 16));
    lcd.setCursor(0, 1);
    lcd.print(msg.substring(16, min((int)msg.length(), 32)));
  }
}

// ========== Wi-Fi and Server Config ==========
const char* ssid     = "designlab";
const char* password = "designlab1";
const char* serverURL = "http://20.246.97.176:3000/api/upload-image-raw";
const char* apiKey = "ewfgjiohewiuhwe8934yt83gigiuewhui83h8ge84849g4h489g";

// ========== Camera Pins ==========
#define PWDN_GPIO_NUM   -1
#define RESET_GPIO_NUM  -1
#define XCLK_GPIO_NUM   4
#define SIOD_GPIO_NUM   17
#define SIOC_GPIO_NUM   16
#define Y9_GPIO_NUM     9
#define Y8_GPIO_NUM     18
#define Y7_GPIO_NUM     15
#define Y6_GPIO_NUM     14
#define Y5_GPIO_NUM     13
#define Y4_GPIO_NUM     12
#define Y3_GPIO_NUM     11
#define Y2_GPIO_NUM     10
#define VSYNC_GPIO_NUM  6
#define HREF_GPIO_NUM   7
#define PCLK_GPIO_NUM   5

#define CAPTURE_INTERVAL 3000
#define MAX_RETRIES 5

int captureCount = 0;
int successCount = 0;
int failureCount = 0;

void setupCamera() {
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
    lcdPrintln("Cam init failed");
    while(1) delay(1000);
  }

  sensor_t *s = esp_camera_sensor_get();
  if (s) {
    s->set_hmirror(s, 0);
    s->set_vflip(s, 1);
  }

  lcdPrintln("Camera ready");
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
    lcdPrintln("WiFi disconnected");
    return false;
  }
  
  camera_fb_t* fb = nullptr;
  for (int attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    if (attempt > 1) {
      lcdPrintln("Retry " + String(attempt));
      delay(200);
    }
    
    fb = esp_camera_fb_get();
    if (!fb) continue;
    
    if (isValidJPEG(fb->buf, fb->len)) {
      lcdPrintln("Frame OK " + String(fb->len) + "B");
      break;
    }
    
    esp_camera_fb_return(fb);
    fb = nullptr;
  }
  
  if (!fb) {
    lcdPrintln("Bad frame");
    failureCount++;
    return false;
  }
  
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
  //
  //Return string for LCD
  String responseString;
  if (code > 0) {
    responseString = http.getString();  // Get the server response as a string
  } else {
    responseString = "HTTP error: " + String(code);  // Prepare error string if HTTP request failed
  }
  //
  http.end();
  esp_camera_fb_return(fb);

  if (code == 200 || code == 201) {
    successCount++;

    lcd.clear();
    lcd.setCursor(0, 0);
    lcdPrintln("Upload OK #" + String(captureCount));
    //
    //Display server response on the LCD
    lcd.setCursor(0, 1);
    String displayMessage = responseString.substring(0, 16);
    lcdPrintln(responseString);  // Display the response
    //
    return true;
  }
  
  lcdPrintln("Upload fail " + String(code));
  failureCount++;
  return false;
}

void setup() {
  Serial.begin(115200);
  Wire.begin(SDA_PIN, SCL_PIN);
  lcd.init();
  lcd.backlight();
  lcdPrintln("Init...");

  delay(1000);
  
  lcdPrintln("Cam setup...");
  setupCamera();
  
  WiFi.begin(ssid, password);
  lcdPrintln("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  lcdPrintln("WiFi OK");
  lcd.setCursor(0, 1);
  lcd.print(WiFi.localIP());
  Serial.printf("\nConnected: %s\n\n", WiFi.localIP().toString().c_str());
}

void loop() {
  captureAndUpload();
  delay(CAPTURE_INTERVAL);
}
