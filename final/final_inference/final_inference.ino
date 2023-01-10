
#include <Arduino_LSM9DS1.h>
#include <limits>
#include <cstdint>

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model.h"

using namespace std;
using namespace tflite;

const float ACCELERATION_TRESHOLD = 2;
const unsigned PREPARATION_DELAY_MS = 2500;

const unsigned SAMPLES_PER_SPELL = 119;
const unsigned SAMPLES_DOUBLED = SAMPLES_PER_SPELL << 1;
const unsigned SAMPLES_TRIPPELED = SAMPLES_PER_SPELL + SAMPLES_DOUBLED;
const float DELTA_T = 1.0f / SAMPLES_PER_SPELL;

const unsigned NUMBER_OF_LABELS = 5;

const char* LABELS[NUMBER_OF_LABELS] = { "Alohomora", "Arresto Momentum", "Avada Kedavra", "Locomotor", "Revelio" };

const char* LABELS_PADDED[NUMBER_OF_LABELS] = { "Alohomora:        ", 
												"Arresto Momentum: ", 
												"Avada Kedavra:    ", 
												"Locomotor:        ", 
												"Revelio:          ", };

float acceleration_average_x, acceleration_average_y;
float angle_average_x, angle_average_y;
float min_x, min_y, max_x, max_y;

float acceleration_data[SAMPLES_TRIPPELED] = {};
float gyroscope_data[SAMPLES_TRIPPELED] = {};
float magnetometr_data[SAMPLES_TRIPPELED] = {};
float angles[SAMPLES_DOUBLED] = {};
float stroke_points[SAMPLES_DOUBLED] = {};

AllOpsResolver ops_resolver;
const Model *tf_model = nullptr;
MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input_tensor = nullptr;
TfLiteTensor *output_tensor = nullptr;

float inverse_input_scale = 0.0f;
float input_zero_point = 0.0f;
float output_scale = 0.0f;
float output_zero_point = 0.0f;

const unsigned TENSOR_ARENA_SIZE = 32 * 1024;
byte tensor_arena[TENSOR_ARENA_SIZE] __attribute__((aligned(16)));

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

void calculate_angle()
{
	angle_average_x = 0.0f;
	angle_average_y = 0.0f;

	float previous_angle_x = 0.0f;
	float previous_angle_y = 0.0f;

	for (unsigned i = 0, j = 0; j < SAMPLES_DOUBLED; i += 3, j += 2)
	{
		angles[j] = previous_angle_x + gyroscope_data[i + 1] * DELTA_T;
		angles[j + 1] = previous_angle_y + gyroscope_data[i + 2] * DELTA_T;

		previous_angle_x = angles[j];
		previous_angle_y = angles[j + 1];

		angle_average_x += previous_angle_x;
		angle_average_y += previous_angle_y;
	}

	angle_average_x *= DELTA_T;
	angle_average_y *= DELTA_T;
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
		float normalized_angle_x = (angles[i] - angle_average_x);
		float normalized_angle_y = (angles[i + 1] - angle_average_y);

		float x = -normalized_acceleration_x * normalized_angle_x - normalized_acceleration_y * normalized_angle_y;
		float y = normalized_acceleration_x * normalized_angle_y - normalized_acceleration_y * normalized_angle_x;

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

void load_stroke()
{
	float shift_x = 1.0f / (max_x - min_x);
	float shift_y = 1.0f / (max_y - min_y);

    for (unsigned i = 0; i < SAMPLES_DOUBLED; i += 2)
    {
        input_tensor->data.int8[i] = static_cast<int8_t>((stroke_points[i] - min_x) * shift_x * inverse_input_scale + input_zero_point);
        input_tensor->data.int8[i + 1] = static_cast<unsigned>((stroke_points[i + 1] - min_y) * shift_y * inverse_input_scale + input_zero_point);
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

	// get the TFL representation of the model byte array
	tf_model = GetModel(model);
	if (tf_model->version() != TFLITE_SCHEMA_VERSION)
	{
		Serial.println("Model schema mismatch.");
		while (true)
			;
	}

	interpreter = new MicroInterpreter(tf_model, ops_resolver, tensor_arena, TENSOR_ARENA_SIZE);
	interpreter->AllocateTensors();
	input_tensor = interpreter->input(0);
	output_tensor = interpreter->output(0);

	inverse_input_scale = 1 / input_tensor->params.scale;
	input_zero_point = input_tensor->params.zero_point;
	output_scale = output_tensor->params.scale;
	output_zero_point = output_tensor->params.zero_point;
}

void loop()
{
	Serial.println("Cast a spell.");

	while (true)
	{
		if (IMU.accelerationAvailable())
		{
			float x, y, z;
			IMU.readAcceleration(x, y, z);
			if (fabs(x) + fabs(y) + fabs(z) >= ACCELERATION_TRESHOLD)
			{
				break;
			}
		}
	}

	Serial.println("Capturing a spell...");

	for (unsigned i = 0; i < SAMPLES_TRIPPELED;)
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
	calculate_angle();
	calculate_stroke();
	load_stroke();

	TfLiteStatus invokeStatus = interpreter->Invoke();
	if (invokeStatus != kTfLiteOk)
	{
		Serial.println("Invoke failed.");
		while (true)
			;
	}

	int8_t best_score = INT8_MIN;
	unsigned best_label;

	for (unsigned i = 0; i < NUMBER_OF_LABELS; i++)
	{
		int8_t score = output_tensor->data.int8[i];

		Serial.print(LABELS_PADDED[i]);
		Serial.print((score - output_zero_point) * output_scale * 100.0f, 2);
		Serial.println(" %");

		if (score > best_score)
		{
			best_score = score;
			best_label = i;
		}
	}

	Serial.println(LABELS[best_label]);

    delay(PREPARATION_DELAY_MS);
    Serial.println();
}
