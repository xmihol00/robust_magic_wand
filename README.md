# Robust Magic Wand
Project for the **Algorithms and Selected Topics Communications and Mobile Computing (Embedded Machine Learning)** course on TU Graz, in winter semester 2022/23. The goal was to create a magic wand using Arduino Nano 33 BLE Sense mounted on a 30 cm long stick, which would recognize spells from the Harry Potter movies. See the `report.pdf` file for detailed desrciption of the project.

## Requirements
1. Install the used python requirements using the following command:
```
pip install -r requirements.txt 
```

2. Download the **arduino-cli** from https://arduino.github.io/arduino-cli/0.21/installation/#download and extract the downloaded file to a directory already in the system ``PATH``.

3. Install the TensorFlow Lite library from https://github.com/tensorflow/tflite-micro-arduino-examples#github

## Repository structure
```
--|
  |-- arduino_rasterization/    Files for rasterizing spell stroke in to an image on Arduino and displaying as a plot.
  |
  |-- data/                     Files, from which the smaller data set is created.
  |
  |-- data_collection/          Files used for collecting training samples of different spells.
  |
  |-- data_large/               Files, from which the larger data set is created.
  |
  |-- final/                    Files with the final solution.
  |
  |-- model_search/             Files used for searching the best model architecture and hyperparameter tuning.
  |
  |-- recognition/              Files used for testing inference of different models on Arduiono.
  |
  |-- inference_time_echo.py    Script used to calculate the average inference time of recognition on Arduino.
  |
  |-- launch_arduino.sh         Script used to compile and upload programs to Arduino.
  |
  |-- pretty_serial_echo.py     Script used to show plots of spells casted at inference time.
  |
  |-- report.pdf                The final report, which describes this project in detail.
  |
  |-- serial_echo.py            Script used to echo the serial output of Arduino to stdout.
```

## Results
See this video https://drive.google.com/file/d/1aq8NN16VYL84Q72MNh-RHrPLKKY7SaSN/view?usp=sharing demonstrating the achieved results.
