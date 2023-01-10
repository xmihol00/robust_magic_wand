import serial
import numpy as np
import matplotlib.pyplot as plt

PORT = "/dev/ttyACM0"
BAUD_RATE = 9600
SAMPLES_PER_MEASUREMENT = 119
IMAGE_WIDTH_HEIGHT = 30
IMAGE_WIDTH_HEIGHT_IDX = IMAGE_WIDTH_HEIGHT - 1

arduino = serial.Serial(PORT, BAUD_RATE)

colors = np.linspace(255 - 2 * SAMPLES_PER_MEASUREMENT + 2, 255, SAMPLES_PER_MEASUREMENT) / 255
while True:
    image = np.zeros((IMAGE_WIDTH_HEIGHT, IMAGE_WIDTH_HEIGHT))
    for i in range(SAMPLES_PER_MEASUREMENT):
        x, y = arduino.readline().decode("utf-8").split()
        x, y = int(float(x) * IMAGE_WIDTH_HEIGHT_IDX), int(float(y) * IMAGE_WIDTH_HEIGHT_IDX)
        image[y, x] = colors[i]
    
    for _ in range(5):
        print(arduino.readline().decode("utf-8"), end='')

    label = arduino.readline().decode("utf-8").strip()
    
    figure, axis = plt.subplots(1, 1)
    axis.imshow(image, cmap="gray")
    figure.suptitle(label, fontsize=100)
    axis.set_xticks([], [])
    axis.set_yticks([], [])
    axis.set_frame_on(False)

    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.84, wspace=0.25, hspace=0.5)
    plt.get_current_fig_manager().full_screen_toggle()
    plt.show(block=False)
    plt.pause(3)
    plt.close()

    print("")
    
    arduino.write([1]) # deadlock may appear, in such a case restart the arduino
    