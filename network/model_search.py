import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import tensorflow as tf
import tensorflow.keras.utils as tfu
import tensorflow.keras.models as tfm
import tensorflow.keras.layers as tfl
import tensorflow.keras.callbacks as tfc
import tensorflow.keras.regularizers as tfr
import tensorflow.keras.initializers as tfi
import sklearn.model_selection as skm
import numpy as np
import matplotlib.pyplot as plt
import random
import re
import keras_flops as kf
import time
import functools

import data_loaders

SEED = 42
IMAGE_HEIGHT = 40
IMAGE_WIDTH = 40
SAMPLES_PER_MEASUREMENT = 119
LINES_PER_MEASUREMENT = SAMPLES_PER_MEASUREMENT + 1
IMAGE_WIDTH_HEIGHT_INDEX = IMAGE_WIDTH - 1

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

def representative_dataset(data_set):
    for sample in data_set:
        yield [np.expand_dims(sample, 0)]

def collect_model_summary(summary_line, model_dict):
    match = re.match(r"(.*?): ([\d,]+)", summary_line)
    if match:
        match = match.groups()
        model_dict[match[0].replace("params", "parameters")] = int(match[1].replace(',', ''))

hidden_activation = tf.keras.layers.LeakyReLU(0.1)
droput_1 = 0.4
droput_2 = 0.3
droput_3 = 0.25

models = [
    tfm.Sequential([
        tfl.Dense(units=5, activation="softmax")
    ]),

    tfm.Sequential([
        tfl.Dense(units=100, activation=hidden_activation),
        tfl.Dense(units=5, activation="softmax")
    ]),

    tfm.Sequential([
        tfl.Dense(units=125, activation=hidden_activation),
        tfl.Dense(units=75, activation=hidden_activation),
        tfl.Dense(units=5, activation="softmax")
    ]),

    tfm.Sequential([
        tfl.Dense(units=150, activation=hidden_activation),
        tfl.Dense(units=100, activation=hidden_activation),
        tfl.Dense(units=50, activation=hidden_activation),
        tfl.Dense(units=5, activation="softmax")
    ]),

    tfm.Sequential([
        tfl.Conv2D(filters=8, kernel_size=(3, 3), activation=hidden_activation, padding="same"),
        tfl.MaxPool2D(),
        tfl.Conv2D(filters=16, kernel_size=(3, 3), activation=hidden_activation, padding="same"),
        tfl.MaxPool2D(),
        tfl.Conv2D(filters=32, kernel_size=(3, 3), activation=hidden_activation, padding="same"),
        tfl.MaxPool2D(),
        tfl.Flatten(),
        tfl.Dense(units=64, activation=hidden_activation),
        tfl.Dense(units=5, activation="softmax"),
    ]),

    tfm.Sequential([
        tfl.Conv2D(filters=16, kernel_size=(3, 3), activation=hidden_activation, padding="same"),
        tfl.MaxPool2D(),
        tfl.Conv2D(filters=32, kernel_size=(3, 3), activation=hidden_activation, padding="same"),
        tfl.MaxPool2D(),
        tfl.Conv2D(filters=64, kernel_size=(3, 3), activation=hidden_activation, padding="same"),
        tfl.MaxPool2D(),
        tfl.Flatten(),
        tfl.Dense(units=128, activation=hidden_activation),
        tfl.Dense(units=5, activation="softmax"),
    ]),

    tfm.Sequential([
        tfl.Conv2D(filters=8, kernel_size=(5, 5), activation=hidden_activation, padding="valid"),
        tfl.MaxPool2D(),
        tfl.Conv2D(filters=16, kernel_size=(3, 3), activation=hidden_activation, padding="valid"),
        tfl.MaxPool2D(),
        tfl.Conv2D(filters=32, kernel_size=(3, 3), activation=hidden_activation, padding="valid"),
        tfl.Flatten(),
        tfl.Dense(units=64, activation=hidden_activation),
        tfl.Dense(units=5, activation="softmax"),
    ]),

    tfm.Sequential([
        tfl.Conv2D(filters=16, kernel_size=(5, 5), activation=hidden_activation, padding="valid"),
        tfl.MaxPool2D(),
        tfl.Conv2D(filters=32, kernel_size=(3, 3), activation=hidden_activation, padding="valid"),
        tfl.MaxPool2D(),
        tfl.Conv2D(filters=64, kernel_size=(3, 3), activation=hidden_activation, padding="valid"),
        tfl.Flatten(),
        tfl.Dense(units=128, activation=hidden_activation),
        tfl.Dense(units=5, activation="softmax"),
    ]),

    tfm.Sequential([
        tfl.Conv2D(filters=8, kernel_size=(5, 5), activation=hidden_activation, padding="valid"),
        tfl.MaxPool2D(),
        tfl.Conv2D(filters=16, kernel_size=(3, 3), activation=hidden_activation, padding="valid"),
        tfl.MaxPool2D(),
        tfl.Conv2D(filters=32, kernel_size=(3, 3), activation=hidden_activation, padding="valid"),
        tfl.MaxPool2D(),
        tfl.Conv2D(filters=64, kernel_size=(3, 3), activation=hidden_activation, padding="valid"),
        tfl.Conv2D(filters=32, kernel_size=(1, 1), activation=hidden_activation, padding="same"),
        tfl.Conv2D(filters=5, kernel_size=(1, 1), activation="softmax", padding="same"),
        tfl.Reshape([5])
    ]),

    tfm.Sequential([
        tfl.Conv2D(filters=16, kernel_size=(5, 5), activation=hidden_activation, padding="valid"),
        tfl.MaxPool2D(),
        tfl.Conv2D(filters=32, kernel_size=(3, 3), activation=hidden_activation, padding="valid"),
        tfl.MaxPool2D(),
        tfl.Conv2D(filters=64, kernel_size=(3, 3), activation=hidden_activation, padding="valid"),
        tfl.MaxPool2D(),
        tfl.Conv2D(filters=128, kernel_size=(3, 3), activation=hidden_activation, padding="valid"),
        tfl.Conv2D(filters=64, kernel_size=(1, 1), activation=hidden_activation, padding="same"),
        tfl.Conv2D(filters=5, kernel_size=(1, 1), activation="softmax", padding="same"),
        tfl.Reshape([5])
    ]),

    tfm.Sequential([
        tfl.Dense(units=100, activation=hidden_activation),
        tfl.Dropout(droput_1),
        tfl.Dense(units=5, activation="softmax")
    ]),

    tfm.Sequential([
        tfl.Dense(units=125, activation=hidden_activation),
        tfl.Dropout(droput_1),
        tfl.Dense(units=75, activation=hidden_activation),
        tfl.Dropout(droput_2),
        tfl.Dense(units=5, activation="softmax")
    ]),

    tfm.Sequential([
        tfl.Dense(units=150, activation=hidden_activation),
        tfl.Dropout(droput_1),
        tfl.Dense(units=100, activation=hidden_activation),
        tfl.Dropout(droput_2),
        tfl.Dense(units=50, activation=hidden_activation),
        tfl.Dropout(droput_3),
        tfl.Dense(units=5, activation="softmax")
    ]),

    tfm.Sequential([
        tfl.Conv2D(filters=8, kernel_size=(3, 3), activation=hidden_activation, padding="same"),
        tfl.MaxPool2D(),
        tfl.Conv2D(filters=16, kernel_size=(3, 3), activation=hidden_activation, padding="same"),
        tfl.MaxPool2D(),
        tfl.Conv2D(filters=32, kernel_size=(3, 3), activation=hidden_activation, padding="same"),
        tfl.MaxPool2D(),
        tfl.Flatten(),
        tfl.Dropout(droput_1),
        tfl.Dense(units=64, activation=hidden_activation),
        tfl.Dropout(droput_2),
        tfl.Dense(units=5, activation="softmax"),
    ]),

    tfm.Sequential([
        tfl.Conv2D(filters=16, kernel_size=(3, 3), activation=hidden_activation, padding="same"),
        tfl.MaxPool2D(),
        tfl.Conv2D(filters=32, kernel_size=(3, 3), activation=hidden_activation, padding="same"),
        tfl.MaxPool2D(),
        tfl.Conv2D(filters=64, kernel_size=(3, 3), activation=hidden_activation, padding="same"),
        tfl.MaxPool2D(),
        tfl.Flatten(),
        tfl.Dropout(droput_1),
        tfl.Dense(units=128, activation=hidden_activation),
        tfl.Dropout(droput_2),
        tfl.Dense(units=5, activation="softmax"),
    ]),

    tfm.Sequential([
        tfl.Conv2D(filters=8, kernel_size=(5, 5), activation=hidden_activation, padding="valid"),
        tfl.MaxPool2D(),
        tfl.Conv2D(filters=16, kernel_size=(3, 3), activation=hidden_activation, padding="valid"),
        tfl.MaxPool2D(),
        tfl.Conv2D(filters=32, kernel_size=(3, 3), activation=hidden_activation, padding="valid"),
        tfl.Flatten(),
        tfl.Dropout(droput_1),
        tfl.Dense(units=64, activation=hidden_activation),
        tfl.Dropout(droput_2),
        tfl.Dense(units=5, activation="softmax"),
    ]),

    tfm.Sequential([
        tfl.Conv2D(filters=16, kernel_size=(5, 5), activation=hidden_activation, padding="valid"),
        tfl.MaxPool2D(),
        tfl.Conv2D(filters=32, kernel_size=(3, 3), activation=hidden_activation, padding="valid"),
        tfl.MaxPool2D(),
        tfl.Conv2D(filters=64, kernel_size=(3, 3), activation=hidden_activation, padding="valid"),
        tfl.Flatten(),
        tfl.Dropout(droput_1),
        tfl.Dense(units=128, activation=hidden_activation),
        tfl.Dropout(droput_2),
        tfl.Dense(units=5, activation="softmax"),
    ]),
]

model_names = [
    "baseline",
    "only_DENS_S", "only_DENS_M", "only_DENS_L",
    "CONV_DENS_1_S", "CONV_DENS_1_L", 
    "CONV_DENS_2_S", "CONV_DENS_2_L", 
    "only_CONV_S", "only_CONV_L",
    "only_DENS_S_dropout", "only_DENS_M_dropout", "only_DENS_L_dropout",
    "CONV_DENS_1_S_dropout", "CONV_DENS_1_L_dropout", 
    "CONV_DENS_2_S_dropout", "CONV_DENS_2_L_dropout"
]

model_data_set = [
    0,
    0, 0, 0,
    1, 1,
    1, 1,
    1, 1,
    0, 0, 0,
    1, 1,
    1, 1,
]

table_header = ["Total parameters", "Trainable parameters", "Non-trainable parameters", "Size", "Optimized size", 
                "Training time", "FLOPS", "Test accuracy", "Optimized test accuracy"]

if __name__ == "__main__":
    data_sets = [data_loaders.load_as_array(), data_loaders.load_as_images()]
    results = {}
    
    for model, model_name, data_set in zip(models, model_names, model_data_set):
        X_train, X_test, y_train, y_test = data_sets[data_set]
        results[model_name] = {}
        results_model = results[model_name]

        # get weights for the given seed
        model.build(X_train.shape)
        weights = model.get_weights()

        # get the best number of epochs based on validation data set
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, batch_size=16, verbose=2,
                            callbacks=[tfc.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=False)]).history
        model.set_weights(weights)
        epochs = len(history["loss"]) - 3
        
        # train on the whole train data set
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        train_start = time.time()
        model.fit(X_train, y_train, epochs=epochs, validation_split=0.0, batch_size=16, verbose=2)
        results_model["Training time"] = f"{time.time() - train_start:.2f} s"

        # evaluate the results
        results_model["Test accuracy"] = f"{model.evaluate(X_test, y_test, verbose=2)[1] * 100:.2f} \\%"

        # get the summary of the model
        model.summary(print_fn=lambda x, y=results_model: collect_model_summary(x, y))
        results_model["FLOPS"] = kf.get_flops(model, batch_size=1)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        results_file = open(f"models/{model_name}.tflite", "wb")
        results_file.write(tflite_model)
        results_file.close()
        results_model["Size"] = os.path.getsize(f"models/{model_name}.tflite")
        os.system(f'echo "const unsigned char model[] = {{" > models/{model_name}.h && cat models/{model_name}.tflite | xxd -i >> models/{model_name}.h && echo "}};" >> models/{model_name}.h && rm -f models/{model_name}.tflite')
        del tflite_model

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter.representative_dataset = lambda x=X_train: representative_dataset(x)
        tflite_model_opt = converter.convert()
        results_file = open(f"models/{model_name}.tflite", "wb")
        results_file.write(tflite_model_opt)
        results_file.close()
        results_model["Optimized size"] = os.path.getsize(f"models/{model_name}.tflite")
        os.system(f'echo "const unsigned char model[] = {{" > models/{model_name}_opt.h && cat models/{model_name}.tflite | xxd -i >> models/{model_name}_opt.h && echo "}};" >> models/{model_name}_opt.h && rm -f models/{model_name}.tflite')

        interpreter = tf.lite.Interpreter(model_content=tflite_model_opt)
        interpreter.allocate_tensors()
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]
        input_scale, input_zero_point = interpreter.get_output_details()[0]["quantization"]
        accuracy = 0
        for i, sample in enumerate(X_test):
            interpreter.set_tensor(input_index, np.expand_dims(sample / input_scale + input_zero_point, 0).astype(np.int8))
            interpreter.invoke()
            accuracy += np.argmax(y_test[i]) == np.argmax(interpreter.get_tensor(output_index)[0]) # rescaling is not needed
        results_model["Optimized test accuracy"] = f"{accuracy / X_test.shape[0] * 100:.2f} \\%"
        del tflite_model_opt

    with open("network/model_search_results.tex", "w") as results_file:
        print = functools.partial(print, file=results_file)
        row_end = "\\\\"
        backslash_underscore = "\\_"
        print("\\begin{table}[ht]", "\\tiny", "\\center", "\\begin{tabular}{ |c|c|c|c|c|c|c|c|c|c| }", sep="\n")        
        print("\\hline")

        print("& ", end="")
        for header in table_header[:-1]:
            print(f"\\thead{{{header.replace(' ', row_end)}}} & ", end="")
        print(f"\\thead{{{table_header[-1].replace(' ', row_end)}}} {row_end}")
        print("\\hline")

        for model_name in model_names:
            results_model = results[model_name]
            print(f"\\thead{{{model_name.replace('_', backslash_underscore)}}} & ", end="")
            for header in table_header[:-1]:
                print(f"{results_model[header]} & ", end="")
            print(f"{results_model[table_header[-1]]} {row_end}")

        print("\\hline")
        print("\\end{tabular}", "\\end{table}", sep="\n")
