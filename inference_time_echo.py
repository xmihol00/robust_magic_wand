import serial
import re
import json
import sys

with open("inference_times.json", "r") as json_file:
    inference_times = json.load(json_file)

try:
    model_name = sys.argv[1]
except:
    print("Specify model name in the first argument", file=sys.stderr)
    exit(1)

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

inference_times[model_name] = inference_time_sum / number_of_samples

with open("inference_times.json", "w") as json_file:
    json.dump(inference_times, json_file, indent=4)

print(f"Inference time average over {number_of_samples} runs is: {inference_times[model_name]} ms.")