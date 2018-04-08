from Tkinter import *
import RPi.GPIO as GPIO
import time

motorFreq = 150
servoFreq = 100
motorPin = 21
servoPin = 18
motorStartDuty = 40
servoStartDuty = 95


GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPin, GPIO.OUT)
GPIO.setup(motorPin, GPIO.OUT)
pwmServo = GPIO.PWM(servoPin, servoFreq)
pwmMotor = GPIO.PWM(motorPin, motorFreq)
pwmServo.start(servoStartDuty)
pwmMotor.start(motorStartDuty)

class App:

    def __init__(self, master):
        frame = Frame(master)
        frame.pack()
        scale = Scale(frame, from_=95, to=180,
              orient=HORIZONTAL, command=self.updateServo)
	scale2 = Scale(frame, from_=0, to=100,
	      orient=VERTICAL, command=self.updateMotor)
	scale.set(120)
	scale2.set(40)
       	scale.grid(row=0)
       	scale2.grid(row=100)
		

    def updateServo(self, angle):
        dutyServo = float(angle) / 10.0 + 2.5
        pwmServo.ChangeDutyCycle(dutyServo)

    def updateMotor(self, speed):
	dutyMotor = float(speed) / 10.0 + 15
	pwmMotor.ChangeDutyCycle(dutyMotor)

root = Tk()
root.wm_title('Servo Control')
app = App(root)
root.geometry("200x200+0+0")
root.mainloop()
