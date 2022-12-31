
#include <Arduino_LSM9DS1.h>
#include <limits>
#include <cstring>
#include <cstdint>

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model.h"

using namespace std;
using namespace tflite;

const float ACCELERATION_TRESHOLD = 1.5;

const unsigned IMAGE_HEIGHT = 40;
const unsigned IMAGE_WIDTH = 40;
const unsigned IMAGE_INDEX = 39;
const unsigned NUMNBER_OF_IMAGE_PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH;

const unsigned SAMPLES_PER_SPELL = 119;
const unsigned SAMPLES_DOUBLED = 119 << 1;
const unsigned SAMPLES_TRIPPELED = SAMPLES_PER_SPELL + SAMPLES_DOUBLED;
const float DELTA_T = 1.0f / SAMPLES_PER_SPELL;

const unsigned NUMBER_OF_LABELS = 5;
const char* LABELS[NUMBER_OF_LABELS] = { "Oh no! 'Avada Kedavra' RIP :(.", "Every small kid here can move things with 'Locomotor' :).", 
										 "Red light! 'Arresto Momentum' stop moving.", "You can't see it, 'Revelio', you can see it.", 
										 "'Alohomora' is not meant for stealing, get out!" };

float acceleration_average_x, acceleration_average_y;
float orientation_average_x, orientation_average_y;
float min_x, min_y, max_x, max_y;

float acceleration_data[SAMPLES_TRIPPELED] = {};
float gyroscope_data[SAMPLES_TRIPPELED] = {};
float magnetometr_data[SAMPLES_TRIPPELED] = {};
float orientation_data[SAMPLES_DOUBLED] = {};
float stroke_points[SAMPLES_DOUBLED] = {};

AllOpsResolver ops_resolver;
const Model *tf_model = nullptr;
MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input_tensor = nullptr;
TfLiteTensor *output_tensor = nullptr;
float input_scale = 0.0f;
float input_zero_point = 0.0f;
float output_scale = 0.0f;
float output_zero_point = 0.0f;

const unsigned TENSOR_ARENA_SIZE = 128 * 1024;
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

void load_stroke()
{
	float shift_x = 1.0f / (max_x - min_x) * IMAGE_INDEX;
	float shift_y = 1.0f / (max_y - min_y) * IMAGE_INDEX;
	float color = (255.0f - 2 * SAMPLES_PER_SPELL + 2.0f) / 255.0f / input_scale + input_zero_point;
	float color_increase = 2.0f / 255.0f / input_scale;

	for (unsigned i = 0; i < SAMPLES_DOUBLED; i += 2)
	{
		unsigned x = static_cast<unsigned>(roundf((stroke_points[i] - min_x) * shift_x));
		unsigned y = static_cast<unsigned>(roundf((stroke_points[i + 1] - min_y) * shift_y));

		input_tensor->data.int8[y * IMAGE_WIDTH + x] = static_cast<int8_t>(color);
		color += color_increase;
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

	input_scale = input_tensor->params.scale;
	input_zero_point = input_tensor->params.zero_point;
	output_scale = output_tensor->params.scale;
	output_zero_point = output_tensor->params.zero_point;
}

void loop()
{
	// reset the input tensor
	memset(input_tensor->data.int8, input_zero_point, NUMNBER_OF_IMAGE_PIXELS * sizeof(int8_t));

	Serial.println();
	Serial.println("Get your magic wand ready.");
	delay(2000);
	Serial.println("Now is the time to show off, perform a spell.");

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
	Serial.println("Someone is performing magic here...");

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
	Serial.println("Let's see how good of a magician are you...");

	average_acceleration();
	calculate_orientation();
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
		if (score > best_score)
		{
			best_score = score;
			best_label = i;
		}
	}

	Serial.println();
	Serial.println(LABELS[best_label]);
	delay(1000);
}
