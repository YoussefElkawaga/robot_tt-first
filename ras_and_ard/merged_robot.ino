#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <LedControl.h>

// Servo controller setup
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define SERVOMIN  150   // نبضة زاوية 0°
#define SERVOMAX  600   // نبضة زاوية 180°

// LED Matrix setup
LedControl mx = LedControl(9, 11, 10, 4);

/* قنوات السيرفو */
#define HEAD_CH         0
#define RIGHT_HAND_CH   1
#define RIGHT_SH_CH     2
#define LEFT_HAND_CH    3
#define LEFT_SH_CH      4
#define NECK_CH         5
/* -------------- */

// العين اليسرى (الوحدة 3)
const byte eyeLeft[8] = {
  B00000000,
  B00111100,
  B01111110,
  B01111110,
  B01111110,
  B01111110,
  B00111100,
  B00000000
};

// العين اليمنى (الوحدة 1)
const byte eyeRight[8] = {
  B00000000,
  B00111100,
  B01111110,
  B01111110,
  B01111110,
  B01111110,
  B00111100,
  B00000000
};

// رمشة للعين (صفوف مطفية كأن العين مغمضة)
const byte eyeBlink[8] = {
  B00000000,
  B00000000,
  B00000000,
  B01111110,
  B01111110,
  B00000000,
  B00000000,
  B00000000
};

// الابتسامة - الجزء الأيسر (الوحدة 0)
const byte smileLeft[8] = {
  B00000000,
  B00000000,
  B00011000,
  B00011000,
  B00011000,
  B00011000,
  B00011000,
  B00011000
};

// الابتسامة - الجزء الأيمن (الوحدة 2)
const byte smileRight[8] = {
  B00011000,
  B00011000,
  B00011000,
  B00011000,
  B00011000,
  B00011000,
  B00000000,
  B00000000
};

// توقيتات الرمش
unsigned long lastBlinkTime = 0;
bool isBlinking = false;
const unsigned long blinkInterval = 5000;  // كل 5 ثوانٍ
const unsigned long blinkDuration = 150;  // مدة غمضة العين

int angleToPulse(int angle) {
  return map(angle, 0, 180, SERVOMIN, SERVOMAX);
}

void setup() {
  Serial.begin(9600);
  
  // Initialize servo controller
  pwm.begin();
  pwm.setPWMFreq(50);      // 50 Hz
  delay(10);
  idlePose();              // ابدأ بالوضع الطبيعي
  
  // Initialize LED matrices
  for (int i = 0; i < 4; i++) {
    mx.shutdown(i, false);
    mx.setScanLimit(i, 7);
    mx.setIntensity(i, 8);
    mx.clearDisplay(i);
  }

  drawEyes();
  drawSmile();
}

void loop() {
  // Check for serial commands
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if      (cmd == "idle")        idlePose();
    else if (cmd == "talk")        talkPose();
    else if (cmd == "happy")       happyDance();
    else if (cmd == "shake_hand")  shakeHand();
  }
  
  // Handle eye blinking
  handleBlinking();
}

/* --------- الحركات ---------- */

void idlePose() {
  pwm.setPWM(HEAD_CH,        0, angleToPulse(65));
  pwm.setPWM(RIGHT_HAND_CH,  0, angleToPulse(0));
  pwm.setPWM(RIGHT_SH_CH,    0, angleToPulse(100));
  pwm.setPWM(LEFT_HAND_CH,   0, angleToPulse(0));
  pwm.setPWM(LEFT_SH_CH,     0, angleToPulse(120));
  pwm.setPWM(NECK_CH,        0, angleToPulse(0));
}

void talkPose() {
  pwm.setPWM(RIGHT_HAND_CH, 0, angleToPulse(50));
  delay(300);
  pwm.setPWM(RIGHT_HAND_CH, 0, angleToPulse(0));
}

void happyDance() {
  for (int i = 0; i < 3; i++) {
    pwm.setPWM(LEFT_HAND_CH, 0, angleToPulse(150));
    delay(300);
    pwm.setPWM(LEFT_HAND_CH, 0, angleToPulse(120));
    delay(300);
  }
}

void shakeHand() {
  // وضعية بداية المصافحة
  pwm.setPWM(LEFT_HAND_CH, 0, angleToPulse(100));   // اليد
  pwm.setPWM(LEFT_SH_CH,   0, angleToPulse(120));   // الكتف
  delay(500);

  // اهتزاز المصافحة (3 مرات)
  for (int i = 0; i < 3; i++) {
    pwm.setPWM(LEFT_HAND_CH, 0, angleToPulse(90));
    delay(250);
    pwm.setPWM(LEFT_HAND_CH, 0, angleToPulse(100));
    delay(250);
  }

  // إعادة الذراع اليسرى إلى وضع idle
  pwm.setPWM(LEFT_HAND_CH, 0, angleToPulse(0));
  pwm.setPWM(LEFT_SH_CH,   0, angleToPulse(120));
}

/* --------- وظائف LED Matrix ---------- */

// ترسم العينين الطبيعية
void drawEyes() {
  for (int row = 0; row < 8; row++) {
    mx.setRow(3, row, eyeLeft[row]);
    mx.setRow(1, row, eyeRight[row]);
  }
}

// ترسم الابتسامة
void drawSmile() {
  for (int row = 0; row < 8; row++) {
    mx.setRow(0, row, smileLeft[row]);
    mx.setRow(2, row, smileRight[row]);
  }
}

// ترمش تلقائيًا كل فترة
void handleBlinking() {
  unsigned long now = millis();

  if (!isBlinking && now - lastBlinkTime > blinkInterval) {
    // ابدأ الرمش
    isBlinking = true;
    for (int row = 0; row < 8; row++) {
      mx.setRow(3, row, eyeBlink[row]);
      mx.setRow(1, row, eyeBlink[row]);
    }
    lastBlinkTime = now;
  }

  if (isBlinking && now - lastBlinkTime > blinkDuration) {
    // انتهت الرمشة، رجع العينين
    drawEyes();
    isBlinking = false;
    lastBlinkTime = now;
  }
} 