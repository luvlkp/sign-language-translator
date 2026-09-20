/*
  ESP32-S3-DevKitC-1 (N8R8) + ArduCAM Mini 2MP / 2MP Plus (OV2640, SPI)
  Serves one JPEG photo per request at:   http://<esp-ip>/capture
  The laptop's app.py fetches these photos in a loop.

  Arduino IDE settings (Tools menu):
    Board:            ESP32S3 Dev Module
    Flash Size:       8MB (64Mb)
    PSRAM:            OPI PSRAM
    USB CDC On Boot:  Disabled  if you plug into the port labelled "UART" (recommended)
                      Enabled   if you plug into the port labelled "USB"

  Library: ArduCAM (github.com/ArduCAM/Arduino). In
  Documents/Arduino/libraries/ArduCAM/memorysaver.h, uncomment
  OV2640_MINI_2MP_PLUS (or OV2640_MINI_2MP) and comment out the others.
*/

#include <WiFi.h>
#include <WebServer.h>
#include <Wire.h>
#include <SPI.h>
#include <ArduCAM.h>
#include <esp_system.h>
#include "memorysaver.h"

#if !(defined(OV2640_MINI_2MP_PLUS) || defined(OV2640_MINI_2MP))
#error "Open Documents/Arduino/libraries/ArduCAM/memorysaver.h and uncomment #define OV2640_MINI_2MP_PLUS"
#endif

// ======================= SETTINGS =======================
// 0 = join an existing 2.4 GHz Wi-Fi network (recommended)
// 1 = the ESP creates its own Wi-Fi network (laptop AND phone must join it)
#define USE_ACCESS_POINT 0

const char* WIFI_SSID     = "Weston's Laptop";      // used when USE_ACCESS_POINT = 0
const char* WIFI_PASSWORD = "jeffreyjoe";

// Fixed camera address (only used when USE_ACCESS_POINT = 0), so the
// camera URL never changes. It must match the network the ESP joins:
//   Windows laptop hotspot:  192.168.137.50,  gateway 192.168.137.1,  subnet 255.255.255.0
//   iPhone hotspot:          172.20.10.10,    gateway 172.20.10.1,    subnet 255.255.255.240
#define USE_STATIC_IP 1
IPAddress STATIC_IP(192, 168, 137, 50);
IPAddress GATEWAY(192, 168, 137, 1);
IPAddress SUBNET(255, 255, 255, 0);

const char* AP_SSID       = "SignTranslator-Cam";   // used when USE_ACCESS_POINT = 1
const char* AP_PASSWORD   = "signlanguage";         // must be at least 8 characters

// Image size: OV2640_320x240 is fastest. Try OV2640_640x480 if hands
// are detected poorly (slower, but more detail).
#define IMAGE_SIZE OV2640_320x240

// Set to 1 if the Serial Monitor reports a BROWNOUT reset. It lowers the
// Wi-Fi transmit power, which is fine when the laptop is within a few metres.
#define LOW_WIFI_POWER 0

// Wiring (ESP32-S3 GPIO numbers)
const int PIN_CS   = 10;
const int PIN_MOSI = 11;
const int PIN_SCK  = 12;
const int PIN_MISO = 13;
const int PIN_SDA  = 8;
const int PIN_SCL  = 9;
// ========================================================

ArduCAM myCAM(OV2640, PIN_CS);
WebServer server(80);

const size_t FRAME_BUF_SIZE = 0x60000;   // 384 KB, same as the camera's FIFO
uint8_t* frameBuf = nullptr;

bool cameraSetup() {
  // 1. Check the SPI connection (CS, MOSI, MISO, SCK)
  myCAM.write_reg(ARDUCHIP_TEST1, 0x55);
  if (myCAM.read_reg(ARDUCHIP_TEST1) != 0x55) {
    Serial.println("SPI error: check CS, MOSI, MISO, SCK wiring.");
    return false;
  }

  // 2. Check the sensor over I2C (SDA, SCL)
  uint8_t vid = 0, pid = 0;
  myCAM.wrSensorReg8_8(0xff, 0x01);
  myCAM.rdSensorReg8_8(OV2640_CHIPID_HIGH, &vid);
  myCAM.rdSensorReg8_8(OV2640_CHIPID_LOW, &pid);
  if (vid != 0x26 || (pid != 0x41 && pid != 0x42)) {
    Serial.printf("OV2640 not found (vid=0x%02X pid=0x%02X): check SDA, SCL wiring.\n", vid, pid);
    return false;
  }

  // 3. Configure JPEG output
  myCAM.set_format(JPEG);
  myCAM.InitCAM();
  myCAM.OV2640_set_JPEG_size(IMAGE_SIZE);
  delay(500);
  myCAM.clear_fifo_flag();
  Serial.println("Camera OK.");
  return true;
}

// Finds the actual JPEG inside the FIFO data (skips padding before/after it)
size_t findJpeg(uint8_t* buf, size_t len, uint8_t** start) {
  size_t s = SIZE_MAX;
  for (size_t i = 0; i + 1 < len; i++) {
    if (buf[i] == 0xFF && buf[i + 1] == 0xD8) { s = i; break; }
  }
  if (s == SIZE_MAX) return 0;
  for (size_t i = s + 2; i + 1 < len; i++) {
    if (buf[i] == 0xFF && buf[i + 1] == 0xD9) {
      *start = buf + s;
      return i + 2 - s;
    }
  }
  return 0;
}

size_t captureJpeg(uint8_t** jpg) {
  myCAM.flush_fifo();
  myCAM.clear_fifo_flag();
  myCAM.start_capture();

  unsigned long t0 = millis();
  while (!myCAM.get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) {
    if (millis() - t0 > 3000) {
      Serial.println("Capture timed out.");
      return 0;
    }
    delay(1);
  }

  uint32_t len = myCAM.read_fifo_length();
  if (len == 0 || len >= MAX_FIFO_SIZE || len > FRAME_BUF_SIZE) {
    Serial.printf("Bad frame length: %u\n", (unsigned)len);
    myCAM.clear_fifo_flag();
    return 0;
  }

  memset(frameBuf, 0, len);
  myCAM.CS_LOW();
  myCAM.set_fifo_burst();
  SPI.transfer(frameBuf, len);   // read the whole frame in one go (much faster than byte by byte)
  myCAM.CS_HIGH();
  myCAM.clear_fifo_flag();

  return findJpeg(frameBuf, len, jpg);
}

void handleCapture() {
  unsigned long t0 = millis();
  uint8_t* jpg = nullptr;
  size_t n = captureJpeg(&jpg);
  if (n == 0) {
    server.send(503, "text/plain", "Capture failed");
    return;
  }
  WiFiClient client = server.client();
  client.printf("HTTP/1.1 200 OK\r\n"
                "Content-Type: image/jpeg\r\n"
                "Content-Length: %u\r\n"
                "Cache-Control: no-store\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Connection: close\r\n\r\n", (unsigned)n);
  client.write(jpg, n);
  Serial.printf("Sent frame: %u bytes in %lu ms\n", (unsigned)n, millis() - t0);
}

void handleRoot() {
  server.send(200, "text/html",
              "<h2>ESP32 camera is running</h2>"
              "<p><a href='/capture'>/capture</a> returns one photo. Refresh to take another.</p>");
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("\nStarting ESP32-S3 + ArduCAM...");

  esp_reset_reason_t reason = esp_reset_reason();
  if (reason == ESP_RST_BROWNOUT) {
    Serial.println("!! Last restart was a BROWNOUT (power dip). Try another USB cable/port, or set LOW_WIFI_POWER to 1.");
  } else if (reason == ESP_RST_PANIC) {
    Serial.println("!! Last restart was a CRASH. Copy the error text above this line and share it.");
  }

  // Frame buffer in PSRAM
  frameBuf = (uint8_t*)ps_malloc(FRAME_BUF_SIZE);
  if (!frameBuf) {
    Serial.println("PSRAM allocation failed. Set Tools > PSRAM to 'OPI PSRAM' and re-upload.");
    while (true) delay(1000);
  }

  // Buses
  Wire.begin(PIN_SDA, PIN_SCL);
  pinMode(PIN_CS, OUTPUT);
  digitalWrite(PIN_CS, HIGH);
  SPI.begin(PIN_SCK, PIN_MISO, PIN_MOSI, PIN_CS);
  SPI.setFrequency(8000000);   // lower to 4000000 if frames come out corrupted

  // Reset the camera's controller chip, then initialise (retry until wiring is right)
  myCAM.write_reg(0x07, 0x80);
  delay(100);
  myCAM.write_reg(0x07, 0x00);
  delay(100);
  while (!cameraSetup()) {
    delay(2000);
  }

  // Wi-Fi
#if USE_ACCESS_POINT
  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASSWORD);
  Serial.printf("Created Wi-Fi network \"%s\"\n", AP_SSID);
  Serial.print("Camera URL: http://");
  Serial.print(WiFi.softAPIP());
  Serial.println("/capture");
#else
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);   // lower latency
#if USE_STATIC_IP
  if (!WiFi.config(STATIC_IP, GATEWAY, SUBNET)) {
    Serial.println("!! Could not set the fixed IP address.");
  }
#endif
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
#if LOW_WIFI_POWER
  WiFi.setTxPower(WIFI_POWER_8_5dBm);
#endif
  Serial.printf("Connecting to \"%s\"", WIFI_SSID);
  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    if (millis() - t0 > 20000) {
      Serial.println("\nCould not connect. The ESP32 only supports 2.4 GHz Wi-Fi. Check the name/password. Restarting...");
      delay(1000);
      ESP.restart();
    }
  }
  Serial.println(" connected!");
  Serial.print("Camera URL: http://");
  Serial.print(WiFi.localIP());
  Serial.println("/capture");
#endif

  server.on("/", handleRoot);
  server.on("/capture", handleCapture);
  server.begin();
}

void loop() {
  server.handleClient();

#if !USE_ACCESS_POINT
  // Report Wi-Fi drops so they show up in the Serial Monitor
  static bool wasConnected = true;
  bool connected = (WiFi.status() == WL_CONNECTED);
  if (connected != wasConnected) {
    Serial.println(connected ? "Wi-Fi reconnected." : "!! Wi-Fi connection lost.");
    wasConnected = connected;
  }
#endif
}
