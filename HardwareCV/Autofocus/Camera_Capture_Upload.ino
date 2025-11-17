#include <WiFi.h>
#include <HTTPClient.h>
#include "esp_camera.h"
#include "driver/ledc.h"
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// ========== LCD Configuration ==========
#define LCD_SDA_PIN 2
#define LCD_SCL_PIN 3
#define LCD_ADDRESS 0x27  // Change to 0x3F if needed
#define LCD_COLS 16
#define LCD_ROWS 2
LiquidCrystal_I2C display(LCD_ADDRESS, LCD_COLS, LCD_ROWS);

// ========== Network Configuration ==========
const char* ssid = "designlab";
const char* password = "designlab1";
const char* serverURL = "http://20.246.97.176:3000/api/upload-image-raw";
const char* coachActionURL = "http://20.246.97.176:3000/api/coach-action";
const char* apiKey = "ewfgjiohewiuhwe8934yt83gigiuewhui83h8ge84849g4h489g";

// ========== Camera Pin Definitions ==========
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

// ========== Timing Constants ==========
#define CAPTURE_INTERVAL_MS 8000
#define MAX_CAPTURE_RETRIES 5
#define ACTION_POLL_INTERVAL_MS 3000
#define ACTION_DISPLAY_DURATION_MS 5000

// ========== Statistics ==========
int totalCaptures = 0;
int successfulUploads = 0;
int failedUploads = 0;

// ========== Coach Action State ==========
unsigned long lastActionPollTime = 0;
unsigned long actionDisplayEndTime = 0;
String activeAction = "";
String activeValue = "";
bool showingAction = false;

// ========== LCD Display Functions ==========
void updateDisplay(const String &topLine, const String &bottomLine) {
  Serial.println(topLine);
  Serial.println(bottomLine);
  
  display.clear();
  display.setCursor(0, 0);
  display.print(topLine.substring(0, LCD_COLS));
  display.setCursor(0, 1);
  display.print(bottomLine.substring(0, LCD_COLS));
}

void showSingleLineDisplay(const String &message) {
  Serial.println(message);
  
  display.clear();
  display.setCursor(0, 0);
  
  if (message.length() <= LCD_COLS) {
    display.print(message);
  } else {
    display.print(message.substring(0, LCD_COLS));
    display.setCursor(0, 1);
    display.print(message.substring(LCD_COLS, min((int)message.length(), LCD_COLS * 2)));
  }
}

// ========== Camera Setup ==========
void setupCamera() {
  pinMode(PWDN_GPIO_NUM, OUTPUT);
  pinMode(RESET_GPIO_NUM, OUTPUT);
  digitalWrite(PWDN_GPIO_NUM, LOW);
  digitalWrite(RESET_GPIO_NUM, HIGH);
  delay(50);
  
  // Configure PWM for camera clock
  ledc_timer_config_t timerConfig = {
    .speed_mode = LEDC_LOW_SPEED_MODE,
    .duty_resolution = LEDC_TIMER_1_BIT,
    .timer_num = LEDC_TIMER_0,
    .freq_hz = 10000000,
    .clk_cfg = LEDC_AUTO_CLK
  };
  ledc_timer_config(&timerConfig);

  ledc_channel_config_t channelConfig = {
    .gpio_num = XCLK_GPIO_NUM,
    .speed_mode = LEDC_LOW_SPEED_MODE,
    .channel = LEDC_CHANNEL_0,
    .intr_type = LEDC_INTR_DISABLE,
    .timer_sel = LEDC_TIMER_0,
    .duty = 1,
    .hpoint = 0
  };
  ledc_channel_config(&channelConfig);
  
  // Camera configuration
  camera_config_t camConfig;
  camConfig.ledc_channel = LEDC_CHANNEL_0;
  camConfig.ledc_timer = LEDC_TIMER_0;
  camConfig.pin_d0 = Y2_GPIO_NUM;
  camConfig.pin_d1 = Y3_GPIO_NUM;
  camConfig.pin_d2 = Y4_GPIO_NUM;
  camConfig.pin_d3 = Y5_GPIO_NUM;
  camConfig.pin_d4 = Y6_GPIO_NUM;
  camConfig.pin_d5 = Y7_GPIO_NUM;
  camConfig.pin_d6 = Y8_GPIO_NUM;
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

#define ACTION_CHECK_INTERVAL 3000  // Check every 3 seconds
unsigned long lastActionCheck = 0;
String currentAction = "";
String currentValue = "";

void checkCoachAction() {
  if (WiFi.status() != WL_CONNECTED) {
    return;
  }
  
  HTTPClient http;
  String actionURL = "http://20.246.97.176:3000/api/coach-action";
  
  http.begin(actionURL);
  http.setTimeout(5000);
  http.addHeader("X-API-Key", apiKey);
  
  int code = http.GET();
  
  if (code == 200) {
    String response = http.getString();
    
    // Parse JSON response (simple parsing)
    // Response format: {"action":"raise","value":10,"timestamp":...}
    int actionStart = response.indexOf("\"action\":\"") + 10;
    int actionEnd = response.indexOf("\"", actionStart);
    String newAction = response.substring(actionStart, actionEnd);
    
    int valueStart = response.indexOf("\"value\":") + 8;
    int valueEnd = response.indexOf(",", valueStart);
    String newValue = response.substring(valueStart, valueEnd);
    
    // Only display if action changed
    if (newAction != currentAction || newAction == "null") {
      currentAction = newAction;
      currentValue = newValue;
      
      if (currentAction != "null" && currentAction.length() > 0) {
        Serial.println("\n🎯 ===== COACH ACTION =====");
        Serial.print("   Action: ");
        Serial.println(currentAction);
        
        if (currentValue != "0" && currentValue.length() > 0) {
          Serial.print("   Value: $");
          Serial.println(currentValue);
        }
        
        Serial.println("=========================\n");
      }
    }
  }
  
  http.end();
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
  
  // Add action checking
  unsigned long now = millis();
  if (now - lastActionCheck >= ACTION_CHECK_INTERVAL) {
    lastActionCheck = now;
    checkCoachAction();
  }
}