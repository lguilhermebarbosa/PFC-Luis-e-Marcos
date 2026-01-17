/*
 * 1 (HSPI): CE=16, CSN=15, SCK=14, MOSI=13, MISO=12
 * 2 (VSPI): CE=22, CSN=21, SCK=18, MOSI=23, MISO=19
 * OLED (HW I2C):  SDA=4, SCL=5
 * BOOT:           GPIO 0
 * LED:            GPIO 2
*/

#include <Arduino.h>
#include <SPI.h>
#include <RF24.h>
//#include <U8g2lib.h> # for ESP32 oled
#include <Wire.h> 
#include <esp_wifi.h>
#include <esp_bt.h>

// 1 (HSPI)
#define R1_CE   16
#define R1_CSN  15

// 2 (VSPI)
#define R2_CE   22
#define R2_CSN  21

// Display OLED
//#define OLED_SDA 4
//#define OLED_SCL 5

// SPI 16MHz
RF24 radio1(R1_CE, R1_CSN, 16000000);
RF24 radio2(R2_CE, R2_CSN, 16000000);

SPIClass *hspi = NULL;
SPIClass *vspi = NULL;

//U8G2_SSD1306_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, /* reset=*/ U8X8_PIN_NONE);

int currentMode = 0; 
const int MAX_MODES = 4;

void updateDisplayUI();

void initRadio(RF24* radio, SPIClass* spiBus, const char* name) {
  if(radio->begin(spiBus)) {
    Serial.print(name); Serial.println(" START OK");
    radio->setAutoAck(false);
    radio->stopListening();
    radio->setRetries(0, 0); 
    radio->setPALevel(RF24_PA_MAX, true); 
    radio->setDataRate(RF24_2MBPS);
    radio->setCRCLength(RF24_CRC_DISABLED); 
  } else {
    Serial.print(name); Serial.println(" FALHOU");
  }
}

void setup() {
  esp_wifi_stop();
  esp_bt_controller_deinit();
  
  Serial.begin(115200);
  pinMode(0, INPUT_PULLUP); // Boot
  pinMode(2, OUTPUT);       // LED

  /*
  Wire.begin(OLED_SDA, OLED_SCL); 
  u8g2.begin();
  
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_7x14B_tr);
  u8g2.drawStr(10, 20, "HYPER JAMMER");
  u8g2.drawStr(10, 40, "SYSTEM READY");
  u8g2.sendBuffer();
  delay(1000);
  */

  hspi = new SPIClass(HSPI);
  hspi->begin(14, 12, 13, 15); 
  
  vspi = new SPIClass(VSPI);
  vspi->begin(18, 19, 23, 21); 

  initRadio(&radio1, hspi, "Radio 1");
  initRadio(&radio2, vspi, "Radio 2");

  radio1.powerDown();
  radio2.powerDown();

  updateDisplayUI();
}

void hopBluetooth() {
  int ch1 = random(0, 80);
  int ch2 = random(0, 80);
  radio1.setChannel(ch1);
  radio2.setChannel(ch2);
  delayMicroseconds(random(20, 50)); 
}

void hopBLE() {
  int ch1 = random(0, 40) * 2;
  int ch2 = random(0, 40) * 2;
  radio1.setChannel(ch1);
  radio2.setChannel(ch2);
  delayMicroseconds(random(20, 50));
}

void hopWiFi() {
  int ch1 = random(0, 85);
  int ch2 = random(0, 85);
  radio1.setChannel(ch1);
  radio2.setChannel(ch2);
  delayMicroseconds(random(100, 200)); 
}

void hopRC() {
  int ch1 = random(0, 126);
  int ch2 = random(0, 126);
  radio1.setChannel(ch1);
  radio2.setChannel(ch2);
  delayMicroseconds(random(30, 80));
}

void updateDisplayUI() {

  /*
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_ncenB08_tr);
  
  if (currentMode == 0) {
    u8g2.drawStr(0, 10, "STATUS: STANDBY");
  } else {
    u8g2.drawStr(0, 10, "STATUS: JAMMING");
  }
  
  u8g2.drawLine(0, 12, 128, 12);
  
  u8g2.setFont(u8g2_font_profont12_tr);
  u8g2.setCursor(0, 30);
  */

  switch(currentMode) {
    case 0:
      //u8g2.print("MODE: IDLE / OFF"); 
      //u8g2.setCursor(0, 45); u8g2.print("Radios Paused");
      Serial.println("MODE: 0 (STANDBY)");
      break;
    case 1: 
      //u8g2.print("MODE: BT Classic"); 
      //u8g2.setCursor(0, 45); u8g2.print("Range: 0-80 CH");
      Serial.println("MODE: 1 (BLUETOOTH CLASSIC)");
      break;
    case 2: 
      //u8g2.print("MODE: BLE (Smart)");
      //u8g2.setCursor(0, 45); u8g2.print("Range: Even CHs");
      Serial.println("MODE: 2 (BLE SMART)");
      break;
    case 3: 
      //u8g2.print("MODE: Wi-Fi Full");
      //u8g2.setCursor(0, 45); u8g2.print("Range: 0-85 CH");
      Serial.println("MODE: 3 (WI-FI FULL)");
      break;
    case 4: 
      //u8g2.print("MODE: RC / Drone");
      //u8g2.setCursor(0, 45); u8g2.print("Range: 0-125 CH");
      Serial.println("MODE: 4 (RC / DRONE)");
      break;
  }
  //u8g2.sendBuffer();
}

void checkButton() {
  if(digitalRead(0) == LOW) {
    delay(20); 
    if(digitalRead(0) == LOW) {
      
      currentMode++;
      if(currentMode > MAX_MODES) currentMode = 0; // Volta para 0
      if (currentMode == 0) {
        radio1.powerDown();
        radio2.powerDown();
      } else {
        radio1.powerUp();
        radio2.powerUp();
        delay(2); 
        radio1.startConstCarrier(RF24_PA_MAX, 0); 
        radio2.startConstCarrier(RF24_PA_MAX, 0); 
      }

      for(int i = 0; i < 3; i++) {
        digitalWrite(2, HIGH); delay(100);
        digitalWrite(2, LOW);  delay(100);
      }

      updateDisplayUI();
      while(digitalRead(0) == LOW) { delay(10); } 
    }
  }
}

void loop() {
  checkButton();

  if (currentMode == 0) {
    digitalWrite(2, LOW);
  } else {
    if(millis() % 1000 < 500) digitalWrite(2, HIGH); else digitalWrite(2, LOW);
  }

  switch(currentMode) {
    case 0: 
      delay(10); 
      break;
    case 1: hopBluetooth(); break;
    case 2: hopBLE(); break;
    case 3: hopWiFi(); break;
    case 4: hopRC(); break;
  }
}
