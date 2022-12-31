import serial
import numpy as np
import matplotlib.pyplot as plt

PORT = "/dev/ttyACM0"
BAUD_RATE = 9600
IMAGE_HEIGHT = 40
IMAGE_WIDTH = 40

arduino = serial.Serial(PORT, BAUD_RATE)
line_count = 0
image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))

while True:
    line = arduino.readline()
    line = line.decode("utf-8")

    if line_count < IMAGE_HEIGHT:
        row = np.array(list(map(lambda x: float(x), line.split())))
        image[line_count] = row

        line_count += 1
    else:
        if line_count == IMAGE_HEIGHT:
            plt.imshow(image / 255, cmap="gray")
            plt.show()

            arduino.write([1])

        line_count = 0
