#pragma once


float *read_image(int size, const char* img_file);

float *read_weights(int out_channels, int in_channels, int kernel_size, const char* weight_file);

void print_weights(int out_channels, int in_channels, int kernel_size, float* weights);

void print_image(int channels, int rows, int cols, float* image);

float *read_bias(int out_channels, const char* bias_file);

void print_bias(int out_channels, float* bias);

int get_post_maxPool_size(int pool_size, int input_size);

float *read_weights_fc(int rows, int cols, const char* weight_file);

void *print_weights_fc(int rows, int cols, float* weights);

float *read_bias_fc(int rows, const char* weight_file);

void *print_bias_fc(int rows, float* bias);

void print_linear(int rows, float* vec);