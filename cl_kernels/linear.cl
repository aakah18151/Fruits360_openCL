__kernel void linear(__global int* in_features, __global int* out_features, __global float* input, __global float* weights, __global float* bias, __global float* out)
{

	int row = get_global_id(0);
	float sum = 0.0;
	float val;

	
	for(int i=0; i<*in_features; i++)
	{
		val = *(weights + row**in_features + i) * *(input + i);
		sum += val;
	}

	sum += *(bias + row);
  
  //printf("\n%f", sum);

	*(out + row) = sum;

}