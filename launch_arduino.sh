
if [ $1 = "compile" ]
then
    echo "compiling..."
    arduino-cli compile --build-path ./$2build -b arduino:mbed_nano:nano33ble $2${2::-1}.ino
fi

if [ $1 = "upload" ]
then
    echo "uploading..."
    arduino-cli compile --build-path ./$2build -b arduino:mbed_nano:nano33ble $2${2::-1}.ino && arduino-cli upload -b arduino:mbed_nano:nano33ble -p /dev/ttyACM0 --input-dir ./$2build $2${2::-1}.ino
fi
