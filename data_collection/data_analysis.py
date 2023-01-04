import numpy as np
import matplotlib.pyplot as plt
import glob
import random

IMAGE_HEIGHT = 40
IMAGE_WIDTH = 40
SAMPLES_PER_MEASUREMENT = 119
IMAGE_WIDTH_HEIGHT_INDEX = IMAGE_WIDTH - 1

def calculate_stroke(samples):
    acceleration_average = np.average(samples[:, 0:2], axis=0)
    # calcualte angle
    previous_angle = np.zeros(2)
    stroke_samples = np.zeros((119, 2))
    for j, gyro_sample in enumerate(samples[:, 2:4]):
        stroke_samples[j] = previous_angle + gyro_sample / 119
        previous_angle = stroke_samples[j]     
    angle_avg = np.average(stroke_samples, axis=0) # average angle

    # calculate stroke
    acceleration_magnitude = np.sqrt(acceleration_average.dot(acceleration_average.T)) # dot product insted of squaring
    acceleration_magnitude += (acceleration_magnitude < 0.0001) * 0.0001 # prevent division by 0
    normalzied_acceleration = acceleration_average / acceleration_magnitude
    normalized_angle = stroke_samples - angle_avg
    stroke_samples[:, 0] = -normalzied_acceleration[0] * normalized_angle[:, 0] - normalzied_acceleration[1] * normalized_angle[:, 1]
    stroke_samples[:, 1] =  normalzied_acceleration[0] * normalized_angle[:, 1] - normalzied_acceleration[1] * normalized_angle[:, 0]
    stroke_samples -= np.min(stroke_samples, axis=0) # make samples in range from 0 to x
    stroke_samples /= np.max(stroke_samples, axis=0) # normalize values from 0 to 1

    pixels = np.round(stroke_samples * IMAGE_WIDTH_HEIGHT_INDEX, 0).astype(np.uint8) # normalize samples to the whole image
    image = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT))
    image[pixels[:, 1], pixels[:, 0]] = np.linspace(255 - 2 * SAMPLES_PER_MEASUREMENT + 2, 255, SAMPLES_PER_MEASUREMENT) / 255

    return stroke_samples, image

time = np.linspace(0, 1, SAMPLES_PER_MEASUREMENT)

for i, file_name in enumerate(glob.glob("data/*.csv")):
    file = open(file_name, "r")
    file.readline() # skip header
    samples = np.array([list(map(lambda x: float(x), line.split(','))) for line in file.read().split("\n") if line])[:, [1, 2, 4, 5]]
    file.close()
    samples = samples.reshape(-1, 119, 4)

    for _ in range(10):
        sample_A = random.randint(0, samples.shape[0] - 1)
        sample_B = random.randint(0, samples.shape[0] - 1)

        sample_A = samples[sample_A]
        sample_B = samples[sample_B]

        figure, axis = plt.subplots(3, 4)

        axis[0, 0].plot(time, sample_A[:, 0], color="green")
        axis[0, 0].plot(time, sample_B[:, 0], color="blue")
        axis[0, 0].legend(["sample A", "sample B"])
        axis[0, 0].set_title("Acceleration X coordinate")
        axis[0, 0].set_xlabel("time [$s$]")
        axis[0, 0].set_ylabel("acceleration [$m/s^{2}$]")

        axis[0, 1].plot(time, sample_A[:, 1], color="green")
        axis[0, 1].plot(time, sample_B[:, 1], color="blue")
        axis[0, 1].legend(["sample A", "sample B"])
        axis[0, 1].set_title("Acceleration Y coordinate")
        axis[0, 1].set_xlabel("time [$s$]")
        axis[0, 1].set_ylabel("acceleration [$m/s^{2}$]")

        axis[1, 0].plot(time, sample_A[:, 2], color="green")
        axis[1, 0].plot(time, sample_B[:, 2], color="blue")
        axis[1, 0].legend(["sample A", "sample B"])
        axis[1, 0].set_title("Angular velocity X coordinate")
        axis[1, 0].set_xlabel("time [$s$]")
        axis[1, 0].set_ylabel("angular velocity [$s^{-1}$]")

        axis[1, 1].plot(time, sample_A[:, 3], color="green")
        axis[1, 1].plot(time, sample_B[:, 3], color="blue")
        axis[1, 1].legend(["sample A", "sample B"])
        axis[1, 1].set_title("Angular velocity Y coordinate")
        axis[1, 1].set_xlabel("time [$s$]")
        axis[1, 1].set_ylabel("angular velocity [$s^{-1}$]")

        stroke_A, image_A = calculate_stroke(sample_A)
        stroke_B, image_B = calculate_stroke(sample_B)

        axis[2, 0].plot(time, stroke_A[:, 0], color="green")
        axis[2, 0].plot(time, stroke_B[:, 0], color="blue")
        axis[2, 0].legend(["sample A", "sample B"])
        axis[2, 0].set_title("Stroke X coordinate")
        axis[2, 0].set_xlabel("time [$s$]")
        axis[2, 0].set_ylabel("relative position")

        axis[2, 1].plot(time, stroke_A[:, 1], color="green")
        axis[2, 1].plot(time, stroke_B[:, 1], color="blue")
        axis[2, 1].legend(["sample A", "sample B"])
        axis[2, 1].set_title("Stroke Y coordinate")
        axis[2, 1].set_xlabel("time [$s$]")
        axis[2, 1].set_ylabel("relative position")

        axis = plt.subplot(1, 4, 3)
        axis.imshow(image_A, cmap="gray")
        axis.set_title("Sample A rasterized stroke")
        axis.set_xticks([], [])
        axis.set_yticks([], [])
        axis.set_frame_on(False)

        axis = plt.subplot(1, 4, 4)
        axis.set_title("Sample B rasterized stroke")
        axis.imshow(image_B, cmap="gray")
        axis.set_xticks([], [])
        axis.set_yticks([], [])
        axis.set_frame_on(False)
        
        figure.set_size_inches(17, 13)
        plt.subplots_adjust(left=0.05, bottom=0.08, right=0.97, top=0.95, wspace=0.25, hspace=0.5)
        plt.show()

