#include <Wire.h>
#include <VL53L1X.h>

// Define two sensors
// The sensor on the left end of the track should be senser 0, whose xshut pin must be connected to Pin 4 in Arduino board.
// Similarly, the sensor on the right end should be sensor 1, connecting to Pin 5.
const uint8_t xshutPins[2] = { 4, 5 };
VL53L1X sensors[2];

// Define Reward Pin
const uint8_t pin_RewardE = 6;

uint8_t mode = 0; // 0 for no-task, 2 for left, 3 for right
uint16_t distance[2] = {0xFFFF, 0xFFFF};
uint32_t start_time;
uint32_t last_reward_time;
uint32_t task_duration;
bool task_finished = true;

// Define Reward Regions in mm(milli-meters)
const uint16_t LRegionLeftEnd = 100;
const uint16_t LRegionRightEnd = 900;
const uint16_t RRegionLeftEnd = 1100;
const uint16_t RRegionRightEnd = 1900;

// Offset for sensors in mm
// offsetLsensor = -100 means sensor0 (on the left end) is set to be 100mm apart from the left end of the linear track.
const uint16_t offsetLsensor = -100;
const uint16_t offsetRsensor = 2100;
#define DistanceOffsetCorrection(dis, end) ((1 - end) * offsetLsensor + end * offsetRsensor + (1 - 2 * end) * dis)

// Define Reward Time Interval (msec)
const uint32_t rewardTimeInterval = 5000;
const uint32_t initialNoRewardTime = 5000;


void reward(uint32_t* time){
  digitalWrite(pin_RewardE, HIGH); 
  time = millis();
  delay(1);
  digitalWrite(pin_RewardE, LOW); 
}

void setup()
{
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(400000);

  pinMode(pin_RewardE, OUTPUT); // For MFB stimulation

  for (uint8_t i = 0; i < 2; i++)
  {
    pinMode(xshutPins[i], OUTPUT);
    digitalWrite(xshutPins[i], LOW);
  }

  for (uint8_t i = 0; i < 2; i++)
  {
    pinMode(xshutPins[i], INPUT);
    delay(10);

    sensors[i].setTimeout(500);
    if (!sensors[i].init())
    {
      Serial.print("Failed to detect and initialize sensor ");
      Serial.println(i);
      while (1);
    }

    sensors[i].setAddress(0x2A + i);
    sensors[i].setDistanceMode(VL53L1X::Medium);
    sensors[i].setMeasurementTimingBudget(50000);
    sensors[i].startContinuous(50); // measuring at 20 Hz 
  }
}

void loop()
{
  uint32_t time = last_reward_time; // If the value of sensors is not updated, the program will not enter the reward section

  // Receive session start notification
  if (Serial.available() >= 3){ 
    auto byte = Serial.read();
    mode = (uint8_t) byte; // mode change (2: Left, 3: Right)
    task_duration = 0;
    for (int i = 0; i < 2; i++){
      byte = Serial.read();
      task_duration += byte << (i * 8);
    }
    task_finished = false;
    start_time = millis();
    last_reward_time = start_time - rewardTimeInterval + initialNoRewardTime;
  }
  if(mode > 3) {
    Serial.println("Serial error. Please reset");
    return;
  }

  // Sensor
  if(sensors[0].dataReady() && sensors[1].dataReady()){
    char payload[40];
    for (uint8_t i = 0; i < sensorCount; i++)
    {
      distance[i] = DistanceOffsetCorrection(sensors[i].read(false), i);
      if (sensors[i].timeoutOccurred()) {
        distance[i] = 0xFFFF;   // when timed out
        break;
      }
    }
    time = millis();
    sprintf(payload, "Dist: %d %d (%d ms)", distance[0], distance[1], time - start_time);
    Serial.println(payload);    
  }

  // Reward
  if (time - rewardTimeInterval <= task_duration) {
    if (time - last_reward_time >= rewardTimeInterval){
      bool stim = false;

      switch (mode) {
        case 2:   // left
          for (int i = 0; i < 2; i++) if(distance[i] < LRegionLeftEnd || LRegionRightEnd < distance[i]) break;
          stim = true;
          break;

        case 3:   // right
          for (int i = 0; i < 2; i++) if(distance[i] < RRegionLeftEnd || RRegionRightEnd < distance[i]) break;
          stim = true;
          break;

        default:  // no task
          break;
      }

      if (stim) {
        reward(&time);
        char payload[20];
        sprintf(payload, "Reward: %d ms", time - start_time);
        Serial.println(payload);
      }
    }
  } else if (!task_finished) {
    time = millis();
    char payload[40];
    sprintf(payload, "Session finished: %d ms", time - start_time);
    Serial.println(payload);
    task_finished = true;
  }
}