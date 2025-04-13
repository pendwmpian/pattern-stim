
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

int split(String data, char delimiter, String *dst){
  int idx = 0; 
  int len = data.length();
  for (int i = 0; i < len; i++) {
    char tmp = data.charAt(i);
    if ( tmp == delimiter ) {
        idx++;
    }
    else dst[idx] += tmp;
  }
  return (idx + 1);
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
    auto data = Serial.readString();
    data.trim();
    String dat[3] = {"\0"};
    int index = split(data, ',', dat);
    if(dat[0] == "Ses"){  // "Session 2 10": mode 2 (left), duration 10sec
        mode = dat[1].toInt();
        task_duration = dat[2].toInt() * 1000;
        task_finished = false;
        start_time = millis();
        last_reward_time = start_time - rewardTimeInterval + initialNoRewardTime;
        stim = false;
        char payload[80];
        sprintf(payload, "Session started: mode %d, duration %ld msec", mode, task_duration);
        Serial.println(payload);
    } else if (dat[0] == "Reward"){
      stim = true;
    }
  }
  if(mode > 3) {
    Serial.println("Serial error. Please reset");
    return;
  }

  // Sensor
  if(true){
    char payload[40];

    distance = DistanceOffsetCorrection(sensor.read(), 1); // 0 for the left sensor, 1 for the right sensor
    if (sensor.timeoutOccurred()) {
      distance = 0x7FFF;   // when timed out
    }
    time = millis();
    sprintf(payload, "Dist(Right): %d (%ld ms)", distance, time - start_time);
    Serial.println(payload);
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
