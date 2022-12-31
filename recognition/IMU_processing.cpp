
#include "IMU_processing.h"

extern const unsigned SAMPLES_PER_SPELL;
extern const unsigned SAMPLES_DOUBLED;
extern const unsigned SAMPLES_TRIPPELED;
extern const float DELTA_T;

float acceleration_average_x, acceleration_average_y;
float orientation_average_x, orientation_average_y;
float min_x, min_y, max_x, max_y;

extern float acceleration_data[];
extern float gyroscope_data[];
extern float magnetometr_data[];
extern float orientation_data[];
extern float stroke_points[];

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
	min_x = min_y = numeric_limits<float>::max();
	max_x = max_y = numeric_limits<float>::min();

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
