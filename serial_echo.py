import serial

PORT = "/dev/ttyACM0"
BAUD_RATE = 9600

arduino = serial.Serial(PORT, BAUD_RATE)

while True:
    line = arduino.readline()
    line = line.decode("utf-8")
    print(line, end='')