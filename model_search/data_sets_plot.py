import matplotlib.pyplot as plt
import data_loaders

X_train, X_test, y_train, y_test = data_loaders.load_as_images("data", one_hot=False)

spells = ["Alohomora", "Arresto Momentum", "Avada Kedavra", "Locomotor", "Revelio"]

for image, label in zip(X_train[:100], y_train[:100]):
    plt.imshow(image, cmap="gray")
    plt.title(spells[label])
    plt.show()

for image, label in zip(X_test, y_test):
    plt.imshow(image, cmap="gray")
    plt.title(spells[label])
    plt.show()
