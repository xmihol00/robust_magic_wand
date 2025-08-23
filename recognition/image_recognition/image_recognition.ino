#ifdef BOARD1
	#include <Arduino_LSM9DS1.h>
#else
	#include "Arduino_BMI270_BMM150.h"
#endif
#include <limits>
#include <cstring>

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model.h"

#define OPTIMIZED 1					// set to 1, when optimized model is loaded in model.h

#define CROPPED_INPUT 0				// set to 1, when model in model.h expects input of copped lenght (110 samples instead of 119)

#define REGULAR_OUTPUT 1			// set to 1 for basic output of the program

#define PERCENTAGE_OUTPUT 1			// set to 1 to enhance output with percentages of each class

#define INFERENCE_TIME_OUTPUT 1		// set to 1 to see the time it takes from having samples measured until prediction

#define FUNNY_OUTPUT 0				// set to 1 for funny output of the program

#define PRETTY_OUTPUT 1				// set to 1 to print info necessary to create output of the program as in the video, use the pretty_serial_echo.py to display it

using namespace std;
using namespace tflite;

const float ACCELERATION_TRESHOLD = 2.0;
const unsigned PREPARATION_DELAY_MS = 2500;

const unsigned IMAGE_HEIGHT = 20;
const unsigned IMAGE_WIDTH = 20;
const unsigned IMAGE_INDEX = IMAGE_HEIGHT - 1;
const unsigned NUMNBER_OF_IMAGE_PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH;

const unsigned SAMPLES_PER_SPELL = 119;
const unsigned SAMPLES_DOUBLED = SAMPLES_PER_SPELL << 1;
const unsigned SAMPLES_TRIPPELED = SAMPLES_PER_SPELL + SAMPLES_DOUBLED;
const unsigned CROPPED_SAMPLES_PER_SPELL = 110;
const unsigned CROPPED_SAMPLES_DOUBLED = CROPPED_SAMPLES_PER_SPELL << 1;
const unsigned FRONT_CROP_SAMPLES = 4;
const float DELTA_T = 1.0f / SAMPLES_PER_SPELL;

const unsigned NUMBER_OF_LABELS = 5;

#if (REGULAR_OUTPUT & !(FUNNY_OUTPUT)) | PRETTY_OUTPUT 
const char* LABELS[NUMBER_OF_LABELS] = { "Alohomora", "Arresto Momentum", "Avada Kedavra", "Locomotor", "Revelio" };
#endif

#if FUNNY_OUTPUT & !(PRETTY_OUTPUT)
const char* LABELS[NUMBER_OF_LABELS] = { "'Alohomora' is not meant for stealing, get out!", "Red light! 'Arresto Momentum' stop moving.", 
										 "Oh no! 'Avada Kedavra' RIP :(.", "Every small kid here can move things with 'Locomotor' :).", 
										 "You can't see it, 'Revelio', you can see it.",  };
#endif

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
float angles[SAMPLES_DOUBLED] = {};
float stroke_points[SAMPLES_DOUBLED] = {};

AllOpsResolver ops_resolver;
const Model *tf_model = nullptr;
MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input_tensor = nullptr;
TfLiteTensor *output_tensor = nullptr;
#if OPTIMIZED
float inverse_input_scale = 0.0f;
float input_zero_point = 0.0f;

	#if PERCENTAGE_OUTPUT | PRETTY_OUTPUT
	float output_scale = 0.0f;
	float output_zero_point = 0.0f;
	#endif
#endif

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
	float shift_x = 1.0f / (max_x - min_x) * IMAGE_INDEX;
	float shift_y = 1.0f / (max_y - min_y) * IMAGE_INDEX;

#if OPTIMIZED
	#if CROPPED_INPUT
		float color = (255.0f - CROPPED_SAMPLES_DOUBLED + 2.0f) / 255.0f * inverse_input_scale + input_zero_point;
		float color_increase = 2.0f / 255.0f * inverse_input_scale;

		for (unsigned i = FRONT_CROP_SAMPLES; i < CROPPED_SAMPLES_DOUBLED; i += 2)
		{
			unsigned x = static_cast<unsigned>(roundf((stroke_points[i] - min_x) * shift_x));
			unsigned y = static_cast<unsigned>(roundf((stroke_points[i + 1] - min_y) * shift_y));

			input_tensor->data.int8[y * IMAGE_WIDTH + x] = static_cast<int8_t>(color);
			color += color_increase;
		}
	#else
		float color = (255.0f - SAMPLES_DOUBLED + 2.0f) / 255.0f * inverse_input_scale + input_zero_point;
		float color_increase = 2.0f / 255.0f * inverse_input_scale;

		for (unsigned i = 0; i < SAMPLES_DOUBLED; i += 2)
		{
			unsigned x = static_cast<unsigned>(roundf((stroke_points[i] - min_x) * shift_x));
			unsigned y = static_cast<unsigned>(roundf((stroke_points[i + 1] - min_y) * shift_y));

			input_tensor->data.int8[y * IMAGE_WIDTH + x] = static_cast<int8_t>(color);
			color += color_increase;
		}
	#endif
#else
	#if CROPPED_INPUT
		float color = (255.0f - CROPPED_SAMPLES_DOUBLED + 2.0f) / 255.0f;
		float color_increase = 2.0f / 255.0f;

		for (unsigned i = FRONT_CROP_SAMPLES; i < CROPPED_SAMPLES_DOUBLED; i += 2)
		{
			unsigned x = static_cast<unsigned>(roundf((stroke_points[i] - min_x) * shift_x));
			unsigned y = static_cast<unsigned>(roundf((stroke_points[i + 1] - min_y) * shift_y));

			input_tensor->data.f[y * IMAGE_WIDTH + x] = color;
			color += color_increase;
		}
	#else
		float color = (255.0f - SAMPLES_DOUBLED + 2.0f) / 255.0f;
		float color_increase = 2.0f / 255.0f;

		for (unsigned i = 0; i < SAMPLES_DOUBLED; i += 2)
		{
			unsigned x = static_cast<unsigned>(roundf((stroke_points[i] - min_x) * shift_x));
			unsigned y = static_cast<unsigned>(roundf((stroke_points[i + 1] - min_y) * shift_y));

			input_tensor->data.f[y * IMAGE_WIDTH + x] = color;
			color += color_increase;
		}
	#endif
#endif

#if PRETTY_OUTPUT
	shift_x = 1.0f / (max_x - min_x);
	shift_y = 1.0f / (max_y - min_y);

	for (unsigned i = 0; i < SAMPLES_DOUBLED; i += 2)
	{
		Serial.print((stroke_points[i] - min_x) * shift_x, 4);
		Serial.print(" ");
		Serial.println((stroke_points[i + 1] - min_y) * shift_y, 4);
	}
#endif
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

#if OPTIMIZED
	inverse_input_scale = 1 / input_tensor->params.scale;
	input_zero_point = input_tensor->params.zero_point;

	#if PERCENTAGE_OUTPUT | PRETTY_OUTPUT
	output_scale = output_tensor->params.scale;
	output_zero_point = output_tensor->params.zero_point;
	#endif
#endif
}

void loop()
{
#if OPTIMIZED
	memset(input_tensor->data.int8, input_zero_point, NUMNBER_OF_IMAGE_PIXELS * sizeof(int8_t));
#else
	memset(input_tensor->data.f, 0, NUMNBER_OF_IMAGE_PIXELS * sizeof(float));
#endif

#if REGULAR_OUTPUT & !(PRETTY_OUTPUT) & !(FUNNY_OUTPUT)
	Serial.println("Cast a spell.");
#endif
#if FUNNY_OUTPUT & !(PRETTY_OUTPUT)
	Serial.println("Now is the time to perform a spell.");
#endif

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

#if REGULAR_OUTPUT & !(PRETTY_OUTPUT) & !(FUNNY_OUTPUT)
	Serial.println("Capturing a spell...");
#endif
#if FUNNY_OUTPUT & !(PRETTY_OUTPUT)
	Serial.println("Someone is performing magic here...");
#endif

	for (unsigned i = 0; i < SAMPLES_TRIPPELED;)
	{
		if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable())
		{
			IMU.readAcceleration(acceleration_data[i], acceleration_data[i + 1], acceleration_data[i + 2]);
			IMU.readGyroscope(gyroscope_data[i], gyroscope_data[i + 1], gyroscope_data[i + 2]);

			i += 3;
		}
	}
	
#if FUNNY_OUTPUT & !(PRETTY_OUTPUT)
	Serial.println("Let's see how good of a magician are you...");
#endif

#if INFERENCE_TIME_OUTPUT & !(PRETTY_OUTPUT)
	unsigned long inference_start = micros();
#endif

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

#if INFERENCE_TIME_OUTPUT & !(PRETTY_OUTPUT)
	unsigned long inference_end = micros();
	unsigned long inference_time = inference_end - inference_start;
	Serial.print("Inference time: ");
	Serial.print(inference_time * 0.001f, 3);
	Serial.println(" ms");
#endif

#if OPTIMIZED
	int8_t best_score = INT8_MIN;
	unsigned best_label;
#else
	float best_score = numeric_limits<float>::min();
	unsigned best_label;
#endif
	for (unsigned i = 0; i < NUMBER_OF_LABELS; i++)
	{
#if OPTIMIZED
		int8_t score = output_tensor->data.int8[i]; // the scaling does not have impact on argmax
#else
		float score = output_tensor->data.f[i];
#endif

#if PERCENTAGE_OUTPUT
		Serial.print(LABELS_PADDED[i]);
	#if OPTIMIZED
		Serial.print((score - output_zero_point) * output_scale * 100.0f, 2);
	#else
		Serial.print(score * 100.0f, 2);
	#endif
		Serial.println(" %");
#endif

		if (score > best_score)
		{
			best_score = score;
			best_label = i;
		}
	}

#if PERCENTAGE_OUTPUT & !(PRETTY_OUTPUT)
	Serial.println();
#endif
#if REGULAR_OUTPUT | PRETTY_OUTPUT | FUNNY_OUTPUT
	Serial.println(LABELS[best_label]);
#endif

#if (REGULAR_OUTPUT | FUNNY_OUTPUT) & !(PRETTY_OUTPUT)
	delay(PREPARATION_DELAY_MS);
	Serial.println();
#endif

#if PRETTY_OUTPUT
	while (!Serial.available())
	        ;
	    Serial.read();
#endif
}
