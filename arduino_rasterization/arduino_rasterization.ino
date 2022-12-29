#include <Arduino_LSM9DS1.h>

const float ACCELERATION_TRESHOLD = 2.25; // threshold of significant in G's

const unsigned IMAGE_HEIGHT = 80;
const unsigned IMAGE_WIDTH = 80;
const unsigned IMAGE_CENTER_SHIFT = 40;

const unsigned SAMPLES_PER_SPELL = 119;
const unsigned SAMPLES_DOUBLED = 119 << 1;
const unsigned SAMPLES_TRIPPELED = SAMPLES_PER_SPELL + SAMPLES_DOUBLED;
const float DELTA_T = 1.0f / SAMPLES_PER_SPELL;

float acceleration_average_x = 0.0f;
float acceleration_average_y = 0.0f;

float orientation_average_x = 0.0f;
float orientation_average_y = 0.0f;

float acceleration_data[SAMPLES_TRIPPELED] = {};
float gyroscope_data[SAMPLES_TRIPPELED] = {};
float magnetometr_data[SAMPLES_TRIPPELED] = {};
float orientation_data[SAMPLES_DOUBLED] = {};
float stroke_points[SAMPLES_DOUBLED] = {};

int samples_read = SAMPLES_PER_SPELL;

u_char image[IMAGE_HEIGHT * IMAGE_WIDTH] = {};

void average_acceleration()
{
    acceleration_average_x = 0.0f;
    acceleration_average_y = 0.0f;
    
    for (int i = 0; i < SAMPLES_TRIPPELED; i += 3)
    {
        acceleration_average_x += acceleration_data[i + 1];
        acceleration_average_y += acceleration_data[i + 2];
    }

    acceleration_average_x *= DELTA_T;
    acceleration_average_y *= DELTA_T;
}

void calculate_orientation()
{
    orientation_average_x = 0.0f;
    orientation_average_y = 0.0f;
    
    float previous_orientation_x = 0.0f;
    float previous_orientation_y = 0.0f;

    for (int i = 0, j = 0; j < SAMPLES_DOUBLED; i += 3, j += 2)
    {
        orientation_data[j] = previous_orientation_x + gyroscope_data[i + 1] * DELTA_T;
        orientation_data[j + 1] = previous_orientation_y + gyroscope_data[i + 2] * DELTA_T;

        previous_orientation_x = orientation_data[j];
        previous_orientation_y = orientation_data[j + 1];

        orientation_average_x += previous_orientation_x;    
        orientation_average_y += previous_orientation_y;            
    }

    orientation_average_x *= DELTA_T;
    orientation_average_y *= DELTA_T;
}

void create_stroke()
{
    float max = 0;

    float acceleration_magnitude = sqrtf((acceleration_average_x * acceleration_average_x) + (acceleration_average_y * acceleration_average_y));
    if (acceleration_magnitude < 0.0001f)
    {
        acceleration_magnitude = 0.0001f;
    }
    const float normalized_acceleration_x = acceleration_average_x / acceleration_magnitude;
    const float normalized_acceleration_y = acceleration_average_y / acceleration_magnitude;

    for (int i = 0; i < SAMPLES_DOUBLED; i += 2)
    {
        float normalized_x = (orientation_data[i] - orientation_average_x);
        float normalized_y = (orientation_data[i + 1] - orientation_average_y);

        float x = (-normalized_acceleration_x * normalized_x) + (-normalized_acceleration_y * normalized_y);
        float y = (normalized_acceleration_x * normalized_y) + (-normalized_acceleration_y * normalized_x);

        stroke_points[i] = x;
        stroke_points[i + 1] = y;

        float absolute = fabs(x);
        if (absolute > max)
        {
            max = absolute;
        }
        absolute = fabs(y);
        if (absolute > max)
        {
            max = absolute;
        }
    }

    float shift = 1 / max * IMAGE_CENTER_SHIFT;
    u_char color = 255 - SAMPLES_PER_SPELL + 1;

    for (int i = 0; i < SAMPLES_DOUBLED; i += 2)
    {        
        unsigned x = static_cast<unsigned>(roundf(stroke_points[i] * shift + IMAGE_CENTER_SHIFT));
        unsigned y = static_cast<unsigned>(roundf(stroke_points[i + 1] * shift + IMAGE_CENTER_SHIFT));

        x -= x == IMAGE_WIDTH;
        y -= y == IMAGE_HEIGHT;

        image[y * IMAGE_HEIGHT + x] = color++;
    }
}

void setup()
{
    Serial.begin(9600);
    while (!Serial)
        ;

    if (!IMU.begin())
    {
        Serial.println("Failed to initialize IMU!");
        while (1)
            ;
    }

    // print the header
    // Serial.println("aX,aY,aZ,gX,gY,gZ");
}

void loop()
{

    // wait for significant motion
    while (samples_read == SAMPLES_PER_SPELL)
    {
        if (IMU.accelerationAvailable())
        {
            // read the acceleration data
            float aX, aY, aZ;
            IMU.readAcceleration(aX, aY, aZ);

            // check if it's above the threshold
            if (fabs(aX) + fabs(aY) + fabs(aZ) >= ACCELERATION_TRESHOLD)
            {
                // reset the sample read count
                samples_read = 0;
                break;
            }
        }
    }

    // check if the all the required samples have been read since
    // the last time the significant motion was detected
    while (samples_read < SAMPLES_PER_SPELL)
    {
        // check if both new acceleration and gyroscope data is
        // available
        if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable())
        {
            // read the acceleration_average and gyroscope data
            int index = samples_read * 3;
            IMU.readAcceleration(acceleration_data[index], acceleration_data[index + 1], acceleration_data[index + 2]);
            IMU.readGyroscope(gyroscope_data[index], gyroscope_data[index + 1], gyroscope_data[index + 2]);
            IMU.readMagneticField(magnetometr_data[index], magnetometr_data[index + 1], magnetometr_data[index + 2]);

            samples_read++;
        }
    }

    average_acceleration();
    calculate_orientation();
    create_stroke();

    for (int i = 0; i < IMAGE_HEIGHT; i++)
    {
        for (int j = 0; j < IMAGE_WIDTH; j++)
        {
            int index = i * IMAGE_WIDTH + j;
            Serial.print(image[index]);
            Serial.print(' ');
            image[index] = 0;
        }
        Serial.println();
    }
    Serial.println();

    while (!Serial.available())
        ;
    
    Serial.read();
}