import matplotlib.pyplot as plt
import data_loaders

IMAGE_HEIGHT = 40
IMAGE_WIDTH = 40

X_train, X_test, y_train, y_test = data_loaders.load_as_images(one_hot=False)

spells = ["Avada Kedavra", "Locomotor", "Arresto Momentum", "Revelio", "Alohomora"]

for image, label in zip(X_train[:100], y_train[:100]):
    plt.imshow(image.reshape(IMAGE_WIDTH, IMAGE_HEIGHT), cmap="gray")
    plt.title(spells[label])
    plt.show()

for image, label in zip(X_test, y_test):
    plt.imshow(image.reshape(IMAGE_WIDTH, IMAGE_HEIGHT), cmap="gray")
    plt.title(spells[label])
    plt.show()
