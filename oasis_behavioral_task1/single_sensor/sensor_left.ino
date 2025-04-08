#include <Wire.h>
#include <VL53L1X.h>

// Define one sensor. This is for the left sensor. 
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


void reward(uint32_t* time){
  digitalWrite(pin_RewardE, HIGH); 
  *time = millis();
  delay(1);
  digitalWrite(pin_RewardE, LOW); 
}

void setup()
{
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(400000);

  pinMode(pin_RewardE, OUTPUT); // For MFB stimulation

  delay(100);

  sensor.setTimeout(500);
  if (!sensor.init())
  {
    Serial.println("Failed to detect and initialize sensor!");
    while (1);
  }

  sensor.setDistanceMode(VL53L1X::Medium);
  sensor.setMeasurementTimingBudget(80000);
  sensor.startContinuous(100); // measuring at 10 Hz 
  
}

void loop()
{
  uint32_t time = last_reward_time; // If the value of sensors is not updated, the program will not enter the reward section

  // Receive session start notification
  if (Serial.available() > 0){ 
    int counter = 0;
    uint8_t buff[5];
    uint8_t data = 0x7F;
    while(data != '\0'){
      data = Serial.read();
      buff[counter] = data;
      counter++;
      if (counter > 4) break;
    }
    if(counter == 4){
      if ((uint8_t)buff[0] != 64){ // 64 is for reward stim 
        mode = (uint8_t) buff[0]; // mode change (2: Left, 3: Right)
        task_duration = buff[1] + buff[2] << 8;
        task_duration *= 1000;
        task_finished = false;
        start_time = millis();
        last_reward_time = start_time - rewardTimeInterval + initialNoRewardTime;
      } else stim = true;
    } else mode = 0x7F;
  }
  if(mode > 3) {
    Serial.println("Serial error. Please reset");
    return;
  }

  // Sensor
  if(sensor.dataReady()){
    char payload[40];

    distance = DistanceOffsetCorrection(sensor.read(false), 0); // 0 for the left sensor, 1 for the right sensor
    if (sensor.timeoutOccurred()) {
      distance = 0x7FFF;   // when timed out
      break;
    }
    time = millis();
    sprintf(payload, "Dist(Left): %d (%d ms)", distance, time - start_time);
    Serial.println(payload);    
  }

  // Reward
  if (time - start_time <= task_duration) {
    if (time - last_reward_time >= rewardTimeInterval){

      if (stim) {
        reward(&time);
        char payload[20];
        sprintf(payload, "Reward: %d ms", time - start_time);
        Serial.println(payload);
        last_reward_time = time;
      }
      stim = false;
    }
  } else if (!task_finished) {
    time = millis();
    char payload[40];
    sprintf(payload, "Session finished: %d ms", time - start_time);
    Serial.println(payload);
    task_finished = true;
  }
}
