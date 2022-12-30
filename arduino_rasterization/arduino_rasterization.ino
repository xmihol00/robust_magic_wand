
#include <Arduino_LSM9DS1.h>
#include <limits>

const float ACCELERATION_TRESHOLD = 1.5;

const unsigned IMAGE_HEIGHT = 40;
const unsigned IMAGE_WIDTH = 40;
const unsigned IMAGE_INDEX = 39;

const unsigned SAMPLES_PER_SPELL = 119;
const unsigned SAMPLES_DOUBLED = 119 << 1;
const unsigned SAMPLES_TRIPPELED = SAMPLES_PER_SPELL + SAMPLES_DOUBLED;
const float DELTA_T = 1.0f / SAMPLES_PER_SPELL;

float acceleration_average_x, acceleration_average_y;
float orientation_average_x, orientation_average_y;
float min_x, min_y, max_x, max_y;

float acceleration_data[SAMPLES_TRIPPELED] = {};
float gyroscope_data[SAMPLES_TRIPPELED] = {};
float magnetometr_data[SAMPLES_TRIPPELED] = {};
float orientation_data[SAMPLES_DOUBLED] = {};
float stroke_points[SAMPLES_DOUBLED] = {};

u_char image[IMAGE_HEIGHT * IMAGE_WIDTH] = {};

void average_acceleration()
{
    acceleration_average_x = 0.0f;
    acceleration_average_y = 0.0f;
    
    for (unsigned i = 0; i < SAMPLES_TRIPPELED; i += 3)
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

    for (unsigned i = 0, j = 0; j < SAMPLES_DOUBLED; i += 3, j += 2)
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

void calculate_stroke()
{
    min_x = min_y = std::numeric_limits<float>::max();
    max_x = max_y = std::numeric_limits<float>::min();

    float acceleration_magnitude = sqrtf((acceleration_average_x * acceleration_average_x) + (acceleration_average_y * acceleration_average_y));
    if (acceleration_magnitude < 0.0001f)
    {
        acceleration_magnitude = 0.0001f;
    }
    const float normalized_acceleration_x = acceleration_average_x / acceleration_magnitude;
    const float normalized_acceleration_y = acceleration_average_y / acceleration_magnitude;

    for (unsigned i = 0; i < SAMPLES_DOUBLED; i += 2)
    {
        float normalized_orientation_x = (orientation_data[i] - orientation_average_x);
        float normalized_orientation_y = (orientation_data[i + 1] - orientation_average_y);

        float x = -normalized_acceleration_x * normalized_orientation_x - normalized_acceleration_y * normalized_orientation_y;
        float y = normalized_acceleration_x * normalized_orientation_y - normalized_acceleration_y * normalized_orientation_x;

        stroke_points[i] = x;
        stroke_points[i + 1] = y;

        if (x > max_x)
        {
            max_x = x;
        }
        else if (x < min_x)
        {
            min_x = x;
        }

        if (y > max_y)
        {
            max_y = y;
        }
        else if (y < min_y)
        {
            min_y = y;
        }
    }
}

void rasterize_stroke()
{
    float shift_x = 1 / (max_x - min_x) * IMAGE_INDEX;
    float shift_y = 1 / (max_y - min_y) * IMAGE_INDEX;
    u_char color = 255 - SAMPLES_PER_SPELL + 1;

    for (unsigned i = 0; i < SAMPLES_DOUBLED; i += 2)
    {        
        unsigned x = static_cast<unsigned>(roundf((stroke_points[i] - min_x) * shift_x));
        unsigned y = static_cast<unsigned>(roundf((stroke_points[i + 1] - min_y) * shift_y));

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
        Serial.println("Failed to initialize IMU.");
        while (true)
            ;
    }
}

void loop()
{
    while (true)
    {
        if (IMU.accelerationAvailable())
        {
            float x, y, z;
            IMU.readAcceleration(x, y, z);
            if (fabs(y) + fabs(z) >= ACCELERATION_TRESHOLD)
            {
                break;
            }
        }
    }

    for (unsigned i = 0; i < SAMPLES_TRIPPELED; )
    {
        if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable())
        {
            IMU.readAcceleration(acceleration_data[i], acceleration_data[i + 1], acceleration_data[i + 2]);
            IMU.readGyroscope(gyroscope_data[i], gyroscope_data[i + 1], gyroscope_data[i + 2]);
            IMU.readMagneticField(magnetometr_data[i], magnetometr_data[i + 1], magnetometr_data[i + 2]);

            i += 3;
        }
    }

    average_acceleration();
    calculate_orientation();
    calculate_stroke();
    rasterize_stroke();

    for (unsigned i = 0; i < IMAGE_HEIGHT; i++)
    {
        for (unsigned j = 0; j < IMAGE_WIDTH; j++)
        {
            unsigned index = i * IMAGE_WIDTH + j;
            Serial.print(image[index]);
            Serial.print(' ');
            image[index] = 0; // reset image
        }
        Serial.println();
    }
    Serial.println();

    // wait for command to collect another spell
    while (!Serial.available())
        ;
    
    Serial.read();
}
