import serial
import numpy as np
import matplotlib.pyplot as plt
import sys
import tty
import termios

PATTERN = r"^(.*?),(.*?)\r$"
PORT = '/dev/ttyACM0'
BAUD_RATE = 9600
SAMPLES_PER_MEASUREMENT = 119
IMAGE_WIDTH_HEIGHT = 80
IMAGE_WIDTH_HEIGHT_INDEX = IMAGE_WIDTH_HEIGHT - 1

arduino = serial.Serial(PORT, BAUD_RATE)
line_count = 0
accelometer_samples = np.zeros((SAMPLES_PER_MEASUREMENT, 3))
gyroscope_samples = np.zeros((SAMPLES_PER_MEASUREMENT, 3))
magnetometer_samples = np.zeros((SAMPLES_PER_MEASUREMENT, 3))
orientation_samples = np.zeros((SAMPLES_PER_MEASUREMENT, 3))
stroke_samples = np.zeros((SAMPLES_PER_MEASUREMENT, 2))

colors = np.linspace(255 - SAMPLES_PER_MEASUREMENT + 1, 255, SAMPLES_PER_MEASUREMENT) / 255

file = open(sys.argv[1], 'a')
samples = ""

stdin_fd = sys.stdin.fileno()
term_settings = termios.tcgetattr(stdin_fd)
saved = 0

while True:
    line = arduino.readline()
    line = line.decode("utf-8")
    samples += line.replace(' ', ',')

    if line_count < SAMPLES_PER_MEASUREMENT:
        measurements = np.array(list(map(lambda x: float(x), line.split()))).reshape(3, 3)
        accelometer_samples[line_count] = measurements[0]
        gyroscope_samples[line_count] = measurements[1]
        magnetometer_samples[line_count] = measurements[2]
        line_count += 1

    elif line_count == SAMPLES_PER_MEASUREMENT:
        # average acceleration
        acceleration_average = np.average(accelometer_samples, axis=0)

        # calcualte orientation
        previous_orientation = np.zeros(3)
        for i, gyro_sample in enumerate(gyroscope_samples):
            orientation_samples[i] = previous_orientation + gyro_sample / SAMPLES_PER_MEASUREMENT
            previous_orientation = orientation_samples[i]     
        orientation_avg = np.average(orientation_samples, axis=0) # average orientation

        # calculate stroke
        acceleration_magnitude = np.sqrt(acceleration_average.dot(acceleration_average.T)) # dot product insted of squaring
        acceleration_magnitude += (acceleration_magnitude < 0.0001) * 0.0001 # prevent division by 0
        normalzied_acceleration = acceleration_average / acceleration_magnitude
        normalized_orientation = orientation_samples - orientation_avg
        stroke_samples[:, 0] = -normalzied_acceleration[1] * normalized_orientation[:, 1] - normalzied_acceleration[2] * normalized_orientation[:, 2]
        stroke_samples[:, 1] =  normalzied_acceleration[1] * normalized_orientation[:, 2] - normalzied_acceleration[2] * normalized_orientation[:, 1]

        # rasterize stroke
        stroke_samples -= np.min(stroke_samples, axis=0) # make samples in range from 0 to x
        pixels = np.round(stroke_samples * IMAGE_WIDTH_HEIGHT_INDEX / np.max(stroke_samples, axis=0), 0).astype(np.uint8) # normalize samples to the whole image
        image = np.zeros((IMAGE_WIDTH_HEIGHT, IMAGE_WIDTH_HEIGHT))
        image[pixels[:, 1], pixels[:, 0]] = colors
        plt.imshow(image, cmap="gray")
        plt.show()

        tty.setraw(stdin_fd)
        character = sys.stdin.read(1)
        termios.tcsetattr(stdin_fd, termios.TCSADRAIN, term_settings)

        if character == 'w':
            file.write(samples)
            saved += 1
            print(f"saved {saved}")
        else:
            print("not saving")

        samples = ""
        arduino.write([1]) # allow arduino to collect another sample
        line_count = 0