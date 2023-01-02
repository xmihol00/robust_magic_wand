import tensorflow as tf
import tensorflow.keras.utils as tfu
import glob
import sklearn.model_selection as skms
import numpy as np

IMAGE_HEIGHT = 40
IMAGE_WIDTH = 40
SAMPLES_PER_MEASUREMENT = 119
LINES_PER_MEASUREMENT = SAMPLES_PER_MEASUREMENT + 1
IMAGE_WIDTH_HEIGHT_INDEX = IMAGE_WIDTH - 1

def get_stroke_samples(data):
    orientation_samples = np.zeros((SAMPLES_PER_MEASUREMENT, 3))
    stroke_samples = np.zeros((SAMPLES_PER_MEASUREMENT, 2))
    rows_of_samples = [list(map(lambda x: float(x), line.split(','))) for line in data.split('\n') if line]

    for i in range(0, len(rows_of_samples), SAMPLES_PER_MEASUREMENT): 
        measurment = np.array(rows_of_samples[i: i+SAMPLES_PER_MEASUREMENT])
        acceleration_average = np.average(measurment[:, 0:3], axis=0)

        # calcualte orientation
        previous_orientation = np.zeros(3)
        for j, gyro_sample in enumerate(measurment[:, 3:6]):
            orientation_samples[j] = previous_orientation + gyro_sample / SAMPLES_PER_MEASUREMENT
            previous_orientation = orientation_samples[j]     
        orientation_avg = np.average(orientation_samples, axis=0) # average orientation

        # calculate stroke
        acceleration_magnitude = np.sqrt(acceleration_average.dot(acceleration_average.T)) # dot product insted of squaring
        acceleration_magnitude += (acceleration_magnitude < 0.0001) * 0.0001 # prevent division by 0
        normalzied_acceleration = acceleration_average / acceleration_magnitude
        normalized_orientation = orientation_samples - orientation_avg
        stroke_samples[:, 0] = -normalzied_acceleration[1] * normalized_orientation[:, 1] - normalzied_acceleration[2] * normalized_orientation[:, 2]
        stroke_samples[:, 1] =  normalzied_acceleration[1] * normalized_orientation[:, 2] - normalzied_acceleration[2] * normalized_orientation[:, 1]
        yield stroke_samples

def load_as_images(one_hot=True, seed=42):
    data = ""
    labels = []
    for i, file_name in enumerate(glob.glob("data/*.csv")):
        file = open(file_name, "r")
        file.readline() # skip header
        read_lines = file.read()
        labels += [i] * (read_lines.count("\n") // LINES_PER_MEASUREMENT)
        data += read_lines
        file.close()

    colors = np.linspace(255 - 2 * SAMPLES_PER_MEASUREMENT + 2, 255, SAMPLES_PER_MEASUREMENT) / 255
    images = np.zeros((len(labels), IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.float32)

    for i, stroke_samples in enumerate(get_stroke_samples(data)): 
        # rasterize stroke
        stroke_samples -= np.min(stroke_samples, axis=0) # make samples in range from 0 to x
        pixels = np.round(stroke_samples * IMAGE_WIDTH_HEIGHT_INDEX / np.max(stroke_samples, axis=0), 0).astype(np.uint8) # normalize samples to the whole image
        image = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT))
        image[pixels[:, 1], pixels[:, 0]] = colors
        images[i] = image.reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 1).astype(np.float32)

    X_train, X_test, y_train, y_test = skms.train_test_split(images, labels, test_size=0.2, random_state=seed)
    if one_hot:
        # one-hot encoding of labels
        y_train = tfu.to_categorical(y_train, num_classes=5)

    return X_train, X_test, y_train, np.array(y_test)

def load_as_array(one_hot=True, seed=42):
    data = ""
    labels = []
    for i, file_name in enumerate(glob.glob("data/*.csv")):
        file = open(file_name, "r")
        file.readline() # skip header
        read_lines = file.read()
        labels += [i] * (read_lines.count("\n") // LINES_PER_MEASUREMENT)
        data += read_lines
        file.close()

    arrays = np.zeros((len(labels), 2 * SAMPLES_PER_MEASUREMENT), dtype=np.float32)

    for i, stroke_samples in enumerate(get_stroke_samples(data)): 
        stroke_samples -= np.min(stroke_samples, axis=0) # make samples in range from 0 to x
        stroke_samples /= np.max(stroke_samples, axis=0) # normalize values from 0 to 1
        arrays[i] = stroke_samples.reshape(-1)

    X_train, X_test, y_train, y_test = skms.train_test_split(arrays, labels, test_size=0.2, random_state=seed)
    if one_hot:
        # one-hot encoding of labels
        y_train = tfu.to_categorical(y_train, num_classes=5)

    return X_train, X_test, y_train, np.array(y_test)
    