// filepath: /home/azureuser/Computer-Vision-Powered-AI-Poker-Coach/HardwareCV/Autofocus/Camera_Capture_Upload.ino
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

// ========== Camera Pins ==========
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
#define CAPTURE_INTERVAL_MS 3000
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

// ADD WINNER STATE
String lastWinner = "";
unsigned long winnerDisplayEndTime = 0;
bool showingWinner = false;

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
  camConfig.pin_d7 = Y9_GPIO_NUM;
  camConfig.pin_xclk = XCLK_GPIO_NUM;
  camConfig.pin_pclk = PCLK_GPIO_NUM;
  camConfig.pin_vsync = VSYNC_GPIO_NUM;
  camConfig.pin_href = HREF_GPIO_NUM;
  camConfig.pin_sccb_sda = SIOD_GPIO_NUM;
  camConfig.pin_sccb_scl = SIOC_GPIO_NUM;
  camConfig.pin_pwdn = PWDN_GPIO_NUM;
  camConfig.pin_reset = RESET_GPIO_NUM;
  camConfig.xclk_freq_hz = 10000000;
  camConfig.pixel_format = PIXFORMAT_JPEG;
  camConfig.frame_size = FRAMESIZE_VGA;
  camConfig.jpeg_quality = 12;
  camConfig.fb_count = 1;
  camConfig.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  camConfig.fb_location = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;

  if (esp_camera_init(&camConfig) != ESP_OK) {
    showSingleLineDisplay("Camera init fail");
    while(1) delay(1000);
  }

  // Set image orientation
  sensor_t *sensor = esp_camera_sensor_get();
  if (sensor) {
    sensor->set_hmirror(sensor, 0);
    sensor->set_vflip(sensor, 1);
  }

  // Warm up camera
  for (int i = 0; i < 3; i++) {
    camera_fb_t* warmupFrame = esp_camera_fb_get();
    if (warmupFrame) esp_camera_fb_return(warmupFrame);
    delay(100);
  }
  
  showSingleLineDisplay("Camera ready");
}

// ========== JPEG Validation ==========
bool validateJPEGData(const uint8_t* buffer, size_t length) {
  if (length < 10) return false;
  if (buffer[0] != 0xFF || buffer[1] != 0xD8) return false;
  if (buffer[length-2] != 0xFF || buffer[length-1] != 0xD9) return false;
  return true;
}

// ========== Image Capture & Upload ==========
bool performCaptureAndUpload() {
  totalCaptures++;
  
  if (WiFi.status() != WL_CONNECTED) {
    updateDisplay("WiFi error", "Disconnected");
    return false;
  }
  
  // Attempt to capture valid frame
  camera_fb_t* frameBuffer = nullptr;
  for (int attemptNum = 1; attemptNum <= MAX_CAPTURE_RETRIES; attemptNum++) {
    if (attemptNum > 1) {
      // REMOVED LCD display for retry attempts
      delay(200);
    }
    
    frameBuffer = esp_camera_fb_get();
    if (!frameBuffer) continue;
    
    if (validateJPEGData(frameBuffer->buf, frameBuffer->len)) {
      Serial.printf("Valid frame: %d bytes (attempt %d)\n", frameBuffer->len, attemptNum);
      break;
    }
    
    esp_camera_fb_return(frameBuffer);
    frameBuffer = nullptr;
  }
  
  if (!frameBuffer) {
    // REMOVED LCD display for bad frame
    failedUploads++;
    return false;
  }
  
  // Upload to server
  HTTPClient httpClient;
  httpClient.begin(serverURL);
  httpClient.setTimeout(30000);
  httpClient.addHeader("Content-Type", "image/jpeg");
  httpClient.addHeader("X-API-Key", apiKey);
  httpClient.addHeader("X-Device-ID", "ESP32S3_OV5640");
  httpClient.addHeader("X-Image-Width", String(frameBuffer->width));
  httpClient.addHeader("X-Image-Height", String(frameBuffer->height));
  httpClient.addHeader("X-Timestamp", String(millis()));

  int httpCode = httpClient.POST(frameBuffer->buf, frameBuffer->len);
  httpClient.end();
  esp_camera_fb_return(frameBuffer);

  if (httpCode == 200 || httpCode == 201) {
    successfulUploads++;
    // REMOVED upload success LCD display - keep it in Serial only
    Serial.printf("Upload #%d OK (rate: %.1f%%)\n", totalCaptures, 100.0 * successfulUploads / totalCaptures);
    return true;
  }
  
  // REMOVED upload failed LCD display
  Serial.printf("Upload failed: HTTP %d\n", httpCode);
  failedUploads++;
  return false;
}

// ========== Coach Action Polling ==========
void pollCoachAction() {
  if (WiFi.status() != WL_CONNECTED) return;
  
  HTTPClient httpClient;
  httpClient.begin(coachActionURL);
  httpClient.setTimeout(5000);
  httpClient.addHeader("X-API-Key", apiKey);
  
  int httpCode = httpClient.GET();
  
  if (httpCode == 200) {
    String jsonResponse = httpClient.getString();
    
    // Parse action from JSON
    int actionKeyPos = jsonResponse.indexOf("\"action\":\"") + 10;
    int actionEndPos = jsonResponse.indexOf("\"", actionKeyPos);
    String newAction = jsonResponse.substring(actionKeyPos, actionEndPos);
    
    int valueKeyPos = jsonResponse.indexOf("\"value\":") + 8;
    int valueEndPos = jsonResponse.indexOf(",", valueKeyPos);
    if (valueEndPos < 0) valueEndPos = jsonResponse.indexOf("}", valueKeyPos);
    String newValue = jsonResponse.substring(valueKeyPos, valueEndPos);
    newValue.trim();
    
    // Display if action changed
    if (newAction != activeAction && newAction != "null" && newAction.length() > 0) {
      activeAction = newAction;
      activeValue = newValue;
      showingAction = true;
      actionDisplayEndTime = millis() + ACTION_DISPLAY_DURATION_MS;
      
      Serial.println("\n=== COACH ACTION ===");
      Serial.print("Action: ");
      Serial.println(activeAction);
      
      // Display on LCD
      String line1 = "Coach: " + activeAction;
      String line2 = "";
      if (activeValue != "0" && activeValue.length() > 0) {
        line2 = "$" + activeValue;
        Serial.print("Value: $");
        Serial.println(activeValue);
      }
      Serial.println("====================\n");
      
      updateDisplay(line1, line2);
    }
  }
  
  httpClient.end();
}

// Poll for winner
void pollWinner() {
  if (WiFi.status() != WL_CONNECTED) return;
  
  HTTPClient httpClient;
  httpClient.begin("http://20.246.97.176:3000/api/winner");
  httpClient.setTimeout(5000);
  httpClient.addHeader("X-API-Key", apiKey);
  
  int httpCode = httpClient.GET();
  
  if (httpCode == 200) {
    String jsonResponse = httpClient.getString();
    
    // CHECK IF needsDisplay FLAG IS TRUE
    int needsDisplayPos = jsonResponse.indexOf("\"needsDisplay\":");
    if (needsDisplayPos > 0) {
      int boolStartPos = needsDisplayPos + 15;
      String needsDisplayValue = jsonResponse.substring(boolStartPos, boolStartPos + 4);
      
      // ONLY PROCESS IF needsDisplay IS TRUE
      if (needsDisplayValue == "true") {
        // Parse winner from JSON
        int winnerKeyPos = jsonResponse.indexOf("\"winner\":\"") + 10;
        int winnerEndPos = jsonResponse.indexOf("\"", winnerKeyPos);
        String newWinner = jsonResponse.substring(winnerKeyPos, winnerEndPos);
        
        int amountKeyPos = jsonResponse.indexOf("\"amount\":") + 9;
        int amountEndPos = jsonResponse.indexOf(",", amountKeyPos);
        if (amountEndPos < 0) amountEndPos = jsonResponse.indexOf("}", amountKeyPos);
        String winAmount = jsonResponse.substring(amountKeyPos, amountEndPos);
        winAmount.trim();
        
        // Display winner (needsDisplay ensures this is new)
        if (newWinner.length() > 0 && newWinner != "null") {
          lastWinner = newWinner;
          showingWinner = true;
          winnerDisplayEndTime = millis() + 8000; // Show for 8 seconds
          
          Serial.println("\n=== WINNER ===");
          Serial.print("Winner: ");
          Serial.println(newWinner);
          Serial.print("Amount: $");
          Serial.println(winAmount);
          Serial.println("==============\n");
          
          // Display on LCD
          String line1 = newWinner;
          String line2 = "wins $" + winAmount;
          
          updateDisplay(line1, line2);
        }
      }
    }
  }
  
  httpClient.end();
}

// ========== Setup ==========
void setup() {
  Serial.begin(115200);
  
  // Initialize LCD
  Wire.begin(LCD_SDA_PIN, LCD_SCL_PIN);
  display.init();
  display.backlight();
  showSingleLineDisplay("Starting...");
  
  delay(1000);
  
  Serial.println("\n=== ESP32-S3 OV5640 ===");
  Serial.println("Camera + Coach Display\n");
  
  setupCamera();
  
  // Connect to WiFi
  WiFi.begin(ssid, password);
  showSingleLineDisplay("WiFi connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  updateDisplay("WiFi connected", WiFi.localIP().toString());
  Serial.printf("\nConnected: %s\n\n", WiFi.localIP().toString().c_str());
  delay(2000);
}

// ========== Main Loop ==========
void loop() {
  unsigned long currentTime = millis();
  
  // Check if winner display time expired
  if (showingWinner && currentTime >= winnerDisplayEndTime) {
    showingWinner = false;
    lastWinner = ""; // Reset so same winner can be shown again next hand
  }
  
  // Check if action display time expired
  if (showingAction && currentTime >= actionDisplayEndTime) {
    showingAction = false;
  }
  
  // Display "Please Standby" if nothing is being shown
  if (!showingAction && !showingWinner) {
    updateDisplay("Please Standby", "");
  }
  
  // Perform image capture/upload (don't show upload status on LCD)
  performCaptureAndUpload();
  
  delay(CAPTURE_INTERVAL_MS);
  
  // Poll for coach actions (every 3 seconds)
  if (currentTime - lastActionPollTime >= ACTION_POLL_INTERVAL_MS) {
    lastActionPollTime = currentTime;
    pollCoachAction();
    pollWinner(); // Winner polling happens at same interval, but only displays when needsDisplay=true
  }
}