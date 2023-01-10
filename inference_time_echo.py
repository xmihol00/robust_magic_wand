import serial
import re

PORT = "/dev/ttyACM0"
BAUD_RATE = 9600

arduino = serial.Serial(PORT, BAUD_RATE)

number_of_samples = 10
inference_time_sum = 0
collected_samples = 0

while True:
    line = arduino.readline()
    line = line.decode("utf-8")
    match = re.match(r"Inference time: (\d+.\d+) ms", line)
    if match:
        inference_time_sum += float(match.groups()[0])
        collected_samples += 1
        if collected_samples == number_of_samples:
            break
    print(line, end='')

print(f"Inference time average over {number_of_samples} runs is: {inference_time_sum / number_of_samples} ms.")