#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <ArduinoJson.h>

// WiFi credentials
const char* ssid = "YOUR_WIFI_NAME";
const char* password = "YOUR_WIFI_PASSWORD";

// Server details
const char* server = "20.246.97.176";
const int port = 3000;
const char* apiEndpoint = "/api/signal";
const char* serverURL = "http://20.246.97.176:3000/api/signal";
const char* apiKey = "ewfgjiohewiuhwe8934yt83gigiuewhui83h8ge84849g4h489g";

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);

  // Wait for WiFi connection
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi!");
}

void loop() {
  WiFiClient client;

  if (!client.connect(server, port)) {
    Serial.println("Connection to server failed!");
    return;
  }

  // Create JSON payload
  DynamicJsonDocument doc(1024);
  doc["device_id"] = "ESP32_001";
  doc["signal"] = "ping";

  String jsonString;
  serializeJson(doc, jsonString);

  // Send HTTP POST request
  client.println("POST " + String(serverURL) + " HTTP/1.1");
  client.println("Host: " + String(server));
  client.println("Content-Type: application/json");
  client.println("X-API-Key: " + String(apiKey)); // Add API key header
  client.println("Content-Length: " + String(jsonString.length()));
  client.println();
  client.print(jsonString);

  // Read response
  while (client.connected()) {
    String line = client.readStringUntil('\n');
    if (line == "\r") break; // Headers end
  }
  String response = client.readString();
  Serial.println("Response: " + response);

  delay(10000); // Wait 10 seconds before sending the next signal
}
