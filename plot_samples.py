import matplotlib.pyplot as plt
import numpy as np
import sys

SAMPLES_PER_MEASUREMENT = 119
IMAGE_WIDTH_HEIGHT = 40
IMAGE_WIDTH_HEIGHT_INDEX = IMAGE_WIDTH_HEIGHT - 1

if len(sys.argv) < 2:
    print("Specify file name relative to the script.", file=sys.stderr)
    exit(1)

orientation_samples = np.zeros((SAMPLES_PER_MEASUREMENT, 3))
stroke_samples = np.zeros((SAMPLES_PER_MEASUREMENT, 2))
colors = np.linspace(255 - SAMPLES_PER_MEASUREMENT + 1, 255, SAMPLES_PER_MEASUREMENT) / 255

file = open(sys.argv[1], "r")
file.readline() # skip header
data = file.read()
file.close()

rows_of_samples = [list(map(lambda x: float(x), line.split(','))) for line in data.split('\n') if line] # remove empty lines and convert CSV to float lists

for i, idx in enumerate(range(0, len(rows_of_samples), SAMPLES_PER_MEASUREMENT)): 
    measurment = np.array(rows_of_samples[idx: idx+SAMPLES_PER_MEASUREMENT])
    acceleration_average = np.average(measurment[:, 0:3], axis=0)

    # calcualte orientation
    previous_orientation = np.zeros(3)
    for j, gyro_sample in enumerate(measurment[:, 3:6]):
        orientation_samples[j] = previous_orientation + gyro_sample / SAMPLES_PER_MEASUREMENT
        previous_orientation = orientation_samples[j]     
    orientation_avg = np.average(orientation_samples, axis=0) # average orientation

    # calculate stroke
    acceleration_magnitude = np.sqrt(acceleration_average.dot(acceleration_average.T)) # dot product insted of squaringw
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
    plt.show()
    plt.title(f"Sample {i + 1}")
    plt.imshow(image, cmap="gray")
