
#include <Wire.h>
#include <VL53L1X.h>

// Define one sensor. This is for the right sensor. 
VL53L1X sensor;

// Define Reward Pin
const uint8_t pin_RewardE = 6;
bool stim = false;

uint8_t mode = 0; // 0 for no-task, 2 for left, 3 for right
int16_t distance = 0x7FFF;
uint32_t start_time = 0;
uint32_t last_reward_time;
uint32_t task_duration = 0;
bool task_finished = true;

// Offset for sensors in mm
// offsetLsensor = -100 means sensor0 (on the left end) is set to be 100mm apart from the left end of the linear track.
const uint16_t offsetLsensor = -100;
const uint16_t offsetRsensor = 2100;
#define DistanceOffsetCorrection(dis, end) ((1 - end) * offsetLsensor + end * offsetRsensor + (1 - 2 * end) * dis)

// Define Reward Time Interval (msec)
const uint32_t rewardTimeInterval = 5000;
const uint32_t initialNoRewardTime = 5000;

// serial communication
uint8_t buff[4] = {0};
int counter = 0;

uint32_t reward(){
  digitalWrite(pin_RewardE, HIGH); 
  auto time = millis();
  delay(1);
  digitalWrite(pin_RewardE, LOW);
  return time; 
}

void setup()
{
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(400000);

  pinMode(pin_RewardE, OUTPUT); // For MFB stimulation

  sensor.setTimeout(500);
  if (!sensor.init())
  {
    Serial.println("Failed to detect and initialize sensor!");
    while (1);
  }

  sensor.setDistanceMode(VL53L1X::Long);
  sensor.setMeasurementTimingBudget(80000);
  sensor.startContinuous(100); // measuring at 10 Hz 
  
}

void loop()
{
  uint32_t time = millis();

  // Receive session start notification

  if (Serial.available() > 0){ 
    auto data = Serial.read();
    if(data != -1){
      buff[counter] = data;
      counter++;
    }
    if(counter == 4){
      counter = 0;
      if(buff[3] == 0){
        if ((uint8_t)buff[0] != 64){ // 64 is for reward stim 
          mode = (uint8_t) buff[0]; // mode change (2: Left, 3: Right)
          task_duration = buff[1] + (buff[2] * 255);
          task_duration *= 1000;
          task_finished = false;
          start_time = millis();
          last_reward_time = start_time - rewardTimeInterval + initialNoRewardTime;
          stim = false;
        } else stim = true;
      } else mode = 0x7F;
    }
  }
  if(mode > 3) {
    Serial.println("Serial error. Please reset");
    return;
  }

  // Sensor
  if(sensor.dataReady()){
    char payload[40];

    distance = DistanceOffsetCorrection(sensor.read(false), 1); // 0 for the left sensor, 1 for the right sensor
    if (sensor.timeoutOccurred()) {
      distance = 0x7FFF;   // when timed out
    }
    time = millis();
    sprintf(payload, "Dist(Left): %d (%ld ms)", distance, time - start_time);
    Serial.println(payload);Serial.println(task_duration);
  }

  // Reward
  if (time - start_time <= task_duration) {
    if (time - last_reward_time >= rewardTimeInterval){

      if (stim) {
        time = reward();
        char payload[20];
        sprintf(payload, "Reward: %ld ms", time - start_time);
        Serial.println(payload);
        last_reward_time = time;
      }
      stim = false;
    }
  } else if (!task_finished) {
    time = millis();
    char payload[40];
    sprintf(payload, "Session finished: %ld ms", time - start_time);
    Serial.println(payload);
    task_finished = true;
  }
  delay(1);
}
