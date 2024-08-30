#include <Servo.h>

Servo servo;

void setup() {
  Serial.begin(9600); 
  servo.attach(9);  
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n'); 
    
    if (command == "apple") {
      // "apple" means move the servo left
      servo.write(180); 
    } else if (command == "banana") {
      // "banana" means move the servo right
      servo.write(-180); 
    }
  }
}
