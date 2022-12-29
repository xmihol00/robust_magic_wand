import serial
import re
import numpy as np
import matplotlib.pyplot as plt
import math

PATTERN = r"^(.*?),(.*?)\r$"
PORT = '/dev/ttyACM0'
BAUD_RATE = 9600
SAMPLES_PER_MEASUREMENT = 119

#output_file = open(write_to_file_path, "w+");
arduino = serial.Serial(PORT, BAUD_RATE)
line_count = 0
accelometr_measurement = np.zeros((SAMPLES_PER_MEASUREMENT, 3))
gyroskop_measurement = np.zeros((SAMPLES_PER_MEASUREMENT, 3))
magnetometr_measurement = np.zeros((SAMPLES_PER_MEASUREMENT, 3))
colors = np.zeros((SAMPLES_PER_MEASUREMENT, 3)) + 1
color = np.linspace(255 - 2 * SAMPLES_PER_MEASUREMENT + 2, 255, SAMPLES_PER_MEASUREMENT) / 255
colors[:, 0] = color
colors[:, 1] = color
colors[:, 2] = color

while True:
    line = arduino.readline()
    line = line.decode("utf-8")

    if line_count < SAMPLES_PER_MEASUREMENT:
        measurements = np.array(list(map(lambda x: float(x), line.split()))).reshape(3, 3)
        accelometr_measurement[line_count] = measurements[0]
        gyroskop_measurement[line_count] = measurements[1]
        magnetometr_measurement[line_count] = measurements[2]

        line_count += 1
    else:
        if line_count == SAMPLES_PER_MEASUREMENT:
            # TODO

            arduino.write([1])

        line_count = 0