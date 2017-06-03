#!/usr/bin/env python
# coding=utf-8

import sys
import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setup(25, GPIO.OUT)

def blink():
    GPIO.output(25, GPIO.HIGH)
    time.sleep(0.5)
    GPIO.output(25, GPIO.LOW)


while True:
    string = sys.stdin.readline().strip()
    if not string:
        break
    if string == 'Spotting!':
        blink()

GPIO.cleanup()
