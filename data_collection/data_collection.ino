#ifdef BOARD1
	#include <Arduino_LSM9DS1.h>
#else
	#include "Arduino_BMI270_BMM150.h"
#endif

const float ACCELERATION_THRESHOLD = 3.0;
const unsigned SAMPLES_PER_SPELL = 119;

void setup()
{
    Serial.begin(9600);
    while (!Serial)
        ;

    if (!IMU.begin())
    {
        while (true)
            ;
    }
}

void loop()
{
    float x, y, z;
    while (true)
    {
        if (IMU.accelerationAvailable())
        {
            IMU.readAcceleration(x, y, z);
            if (fabs(x) + fabs(y) + fabs(z) >= ACCELERATION_THRESHOLD)
            {
                break;
            }
        }
    }

    for (unsigned i = 0; i < SAMPLES_PER_SPELL; )
    {        
        if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) // not waiting for magnetometr, too slow
        {
            IMU.readAcceleration(x, y, z);
            Serial.print(x, 4);
            Serial.print(' ');
            Serial.print(y, 4);
            Serial.print(' ');
            Serial.print(z, 4);
            Serial.print(' ');

            IMU.readGyroscope(x, y, z);
            Serial.print(x, 4);
            Serial.print(' ');
            Serial.print(y, 4);
            Serial.print(' ');
            Serial.print(z, 4);
            Serial.print(' ');

            IMU.readMagneticField(x, y, z);
            Serial.print(x, 4);
            Serial.print(' ');
            Serial.print(y, 4);
            Serial.print(' ');
            Serial.println(z, 4);

            i++;
        }
    }
    Serial.println();

    while (!Serial.available())
        ;
    Serial.read();
}