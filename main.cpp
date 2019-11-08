#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <fstream>
#include <iostream>
#include <vector>

#include "Utils/utils.h"

static const char* input_imagePath = "test_v5.dat";
static const char* input_labelPath = "output8.txt";



int main()
{
	FILE *fp = fopen(input_imagePath, "r");
	FILE *fp1 = fopen(input_labelPath, "r");
	int* hInputlabel;
	int label = 99;
	hInputlabel = read_label(label, fp1);
	float* hInputImage;
	int count;
	float Acc=0;
	
	for(int z=0;z<99;z++)
	{
		
	int imgRows = 32;
	int imgCols = 32;
	int imgChannels = 3;
	

	hInputImage = read_image(imgRows, fp);
	
	
	//for(int y=0;y<99;y++)
	//{
		//std::cout<<hInputlabel[z]<<"\n";
	//}

	
	//print_image(imgChannels, imgRows, imgCols, hInputImage);


	/* ------------------------------------- OpenCL Setup Code ------------------------------------- */
	
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	std::vector<cl::Device> devices;
	platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

	cl::Context context(devices);

	cl::CommandQueue queue = cl::CommandQueue(context, devices[0]);
	
	/* ------------------------------------- OpenCL Setup Code ------------------------------------- */

	int in_channels, out_channels, kernel_size;

	/* ------------------------------------ Layer 1 Starts ------------------------------------ */
	in_channels = imgChannels;
	out_channels = 16;
	kernel_size = 3;

	// Read parameters
	static const char* c1_weights_file = "c1.txt";  
	static const char* c1_bias_file = "b1.txt";   
	
	float *c1_weights;
	float *c1_biases;
	
	

	c1_weights = read_weights(out_channels, in_channels, kernel_size, c1_weights_file);
	c1_biases = read_bias(out_channels, c1_bias_file);

	
//	print_weights(out_channels, in_channels, kernel_size, c1_weights);
	//print_bias(out_channels, c1_biases);
	
	
	float* c1_out;
	c1_out = new float [out_channels*imgRows*imgCols];

//	std::cout<<"Performing Convolution 1 "<<std::endl;
	
	try
	{
		cl::Buffer inputBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, in_channels*imgRows*imgCols*sizeof(float));
		cl::Buffer filterBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, in_channels*out_channels*kernel_size*kernel_size*sizeof(float));
		cl::Buffer biasBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, out_channels*sizeof(float));
		cl::Buffer outputBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, out_channels*imgRows*imgCols*sizeof(float));
		cl::Buffer in_channelsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
		cl::Buffer out_channelsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
		cl::Buffer kernelSizeBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
		cl::Buffer imgRowsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
		cl::Buffer imgColsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));

		queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, in_channels*imgRows*imgCols*sizeof(float), hInputImage);
		queue.enqueueWriteBuffer(filterBuffer, CL_TRUE, 0, in_channels*out_channels*kernel_size*kernel_size*sizeof(float), c1_weights);
		queue.enqueueWriteBuffer(biasBuffer, CL_TRUE, 0, out_channels*sizeof(float), c1_biases);
		queue.enqueueWriteBuffer(outputBuffer, CL_TRUE, 0, out_channels*imgRows*imgCols*sizeof(float), c1_out);
		queue.enqueueWriteBuffer(in_channelsBuffer, CL_TRUE, 0, sizeof(int), &in_channels);
		queue.enqueueWriteBuffer(out_channelsBuffer, CL_TRUE, 0, sizeof(int), &out_channels);
		queue.enqueueWriteBuffer(kernelSizeBuffer, CL_TRUE, 0, sizeof(int), &kernel_size);
		queue.enqueueWriteBuffer(imgRowsBuffer, CL_TRUE, 0, sizeof(int), &imgRows);
		queue.enqueueWriteBuffer(imgColsBuffer, CL_TRUE, 0, sizeof(int), &imgCols);

		std::ifstream sourceFile("cl_kernels/conv.cl");
      std::string sourceCode(
         std::istreambuf_iterator<char>(sourceFile),
         (std::istreambuf_iterator<char>()));
      cl::Program::Sources source(1,
         std::make_pair(sourceCode.c_str(),
         sourceCode.length() + 1));

     	cl::Program program = cl::Program(context, source);

     	program.build(devices);
     	
     	cl::Kernel kernel(program, "convolution");

     	kernel.setArg(0, out_channelsBuffer);
     	kernel.setArg(1, in_channelsBuffer);
     	kernel.setArg(2, kernelSizeBuffer);
     	kernel.setArg(3, inputBuffer);
     	kernel.setArg(4, filterBuffer);
     	kernel.setArg(5, biasBuffer);
     	kernel.setArg(6, outputBuffer);
     	kernel.setArg(7, imgRowsBuffer);
     	kernel.setArg(8, imgColsBuffer);

     	cl::NDRange global(imgRows, imgCols);
     	cl::NDRange local(1,1);
     	queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);

     	// Read data back
     queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, out_channels*imgRows*imgCols*sizeof(float), c1_out);
	 // for(unsigned int i=0;i<50;i++)
	 // std::cout<<"c1_out  "<<i<<"  "<<*(c1_out+i)<<"\n";
	}
	catch(cl::Error error)
	{
		std::cout << error.what() << "(" << error.err() << ")" <<std::endl;
	}

	/* ------------------------------------ Layer 1 Ends ------------------------------------ */

	/* ------------------------------------ Layer 2 Starts ------------------------------------ */
	
	in_channels = 16;
	out_channels = 16;
	kernel_size = 3;
	imgRows = 32;
	imgCols = 32;

	// Read parameters
	static const char* c2_weights_file = "c2.txt";  // Change this
	static const char* c2_bias_file = "b2.txt";  // Change this

	float *c2_weights;
	float *c2_biases;
	
   


	c2_weights = read_weights(out_channels, in_channels, kernel_size, c2_weights_file);
	c2_biases = read_bias(out_channels, c2_bias_file);
	
	//print_weights(out_channels, in_channels, kernel_size, c2_weights);
	//print_bias(out_channels, c2_biases);

	// Allocate space for output
	float* c2_out;
	c2_out = new float [out_channels*imgRows*imgCols];
	
	//std::cout<<"Performing Convolution 2 "<<std::endl;

	try
	{
		cl::Buffer inputBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, in_channels*imgRows*imgCols*sizeof(float));
		cl::Buffer filterBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, in_channels*out_channels*kernel_size*kernel_size*sizeof(float));
		cl::Buffer biasBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, out_channels*sizeof(float));
		cl::Buffer outputBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, out_channels*imgRows*imgCols*sizeof(float));
		cl::Buffer in_channelsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
		cl::Buffer out_channelsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
		cl::Buffer kernelSizeBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
		cl::Buffer imgRowsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
		cl::Buffer imgColsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));

		queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, in_channels*imgRows*imgCols*sizeof(float), c1_out);
		queue.enqueueWriteBuffer(filterBuffer, CL_TRUE, 0, in_channels*out_channels*kernel_size*kernel_size*sizeof(float), c2_weights);
		queue.enqueueWriteBuffer(biasBuffer, CL_TRUE, 0, out_channels*sizeof(float), c2_biases);
		queue.enqueueWriteBuffer(outputBuffer, CL_TRUE, 0, out_channels*imgRows*imgCols*sizeof(float), c2_out);
		queue.enqueueWriteBuffer(in_channelsBuffer, CL_TRUE, 0, sizeof(int), &in_channels);
		queue.enqueueWriteBuffer(out_channelsBuffer, CL_TRUE, 0, sizeof(int), &out_channels);
		queue.enqueueWriteBuffer(kernelSizeBuffer, CL_TRUE, 0, sizeof(int), &kernel_size);
		queue.enqueueWriteBuffer(imgRowsBuffer, CL_TRUE, 0, sizeof(int), &imgRows);
		queue.enqueueWriteBuffer(imgColsBuffer, CL_TRUE, 0, sizeof(int), &imgCols);

		std::ifstream sourceFile("cl_kernels/conv.cl");
      std::string sourceCode(
         std::istreambuf_iterator<char>(sourceFile),
         (std::istreambuf_iterator<char>()));
      cl::Program::Sources source(1,
         std::make_pair(sourceCode.c_str(),
         sourceCode.length() + 1));

     	cl::Program program = cl::Program(context, source);

     	program.build(devices);
     	
     	cl::Kernel kernel(program, "convolution");

     	kernel.setArg(0, out_channelsBuffer);
     	kernel.setArg(1, in_channelsBuffer);
     	kernel.setArg(2, kernelSizeBuffer);
     	kernel.setArg(3, inputBuffer);
     	kernel.setArg(4, filterBuffer);
     	kernel.setArg(5, biasBuffer);
     	kernel.setArg(6, outputBuffer);
     	kernel.setArg(7, imgRowsBuffer);
     	kernel.setArg(8, imgColsBuffer);

     	cl::NDRange global(imgCols, imgRows);
     	cl::NDRange local(1, 1);
     	queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);

     	// Read data back
     	queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, out_channels*imgRows*imgCols*sizeof(float), c2_out);
		
		//for(int i=0;i<out_channels*imgRows*imgCols;i++)
		//std::cout<<"c2_out"<<" "<<" "<<*(c2_out+i)<<"\n";

	}
	catch(cl::Error error)
	{
		std::cout << error.what() << "(" << error.err() << ")" <<std::endl;
	}
	//print_image(out_channels, imgRows, imgCols, c2_out);

	/* ------------------------------------ MaxPool 2D Starts ------------------------------------ */

	int channels, pool_size, outImgRows, outImgCols;
	channels = out_channels;
	imgRows = 32;
	imgCols = 32;
	pool_size = 2;

	outImgRows = get_post_maxPool_size(pool_size, imgRows);
	outImgCols = get_post_maxPool_size(pool_size, imgCols);

	float* c3_out;
 
   float * c3_out_new;
   c3_out_new = new float [channels*outImgRows*outImgCols];
	c3_out = new float [channels*outImgRows*outImgCols];
	
	//std::cout<<"Performing Max Pool 2D"<<std::endl;

	try
	{
		cl::Buffer inputBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, channels*imgRows*imgCols*sizeof(float));
		cl::Buffer outputBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, channels*outImgRows*outImgCols*sizeof(float));
		cl::Buffer channelsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
		cl::Buffer poolSizeBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
		cl::Buffer inDimBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
		cl::Buffer outDimBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));

		queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, channels*imgRows*imgCols*sizeof(float), c2_out);
		queue.enqueueWriteBuffer(outputBuffer, CL_TRUE, 0, channels*outImgRows*outImgCols*sizeof(float), c3_out);
		queue.enqueueWriteBuffer(channelsBuffer, CL_TRUE, 0, sizeof(int), &channels);
		queue.enqueueWriteBuffer(poolSizeBuffer, CL_TRUE, 0, sizeof(int), &pool_size);
		queue.enqueueWriteBuffer(inDimBuffer, CL_TRUE, 0, sizeof(int), &imgRows);
		queue.enqueueWriteBuffer(outDimBuffer, CL_TRUE, 0, sizeof(int), &outImgRows);

		std::ifstream sourceFile("cl_kernels/max_pool2d.cl");
      std::string sourceCode(
         std::istreambuf_iterator<char>(sourceFile),
         (std::istreambuf_iterator<char>()));
      cl::Program::Sources source(1,
         std::make_pair(sourceCode.c_str(),
         sourceCode.length() + 1));

     	cl::Program program = cl::Program(context, source);

     	program.build(devices);
     	
     	cl::Kernel kernel(program, "max_pool2d");

     	kernel.setArg(0, channelsBuffer);
     	kernel.setArg(1, inDimBuffer);
     	kernel.setArg(2, poolSizeBuffer);
     	kernel.setArg(3, outDimBuffer);
     	kernel.setArg(4, inputBuffer);
     	kernel.setArg(5, outputBuffer);

     	cl::NDRange global(outImgRows, outImgCols);
     	cl::NDRange local(1, 1);
     	queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);

     	queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, channels*outImgRows*outImgCols*sizeof(float), c3_out);
      //for(int i=0;i<channels*outImgRows*outImgCols;i++)
	    //std::cout<<"maxpool_out"<<" "<<i<<" "<<*(c3_out+i)<<"\n";
         
         
         
         int j=0, start=0, end=4080;
         
         for(int i=start; i<=end&&j<4096;i=i+16,j++)
         {
           c3_out_new[i]=c3_out[j];
           if(i==end)
           {
           i=start-15;
           end=end+1;
           start++;
           }
           }
      //for(int i=0;i<4096;i++)
	    //std::cout<<"Flatten :"<<" "<<i+1<<" "<<c3_out_new[i]<<"\n";   
      //std::cout<<c3_out_new[i]<<"\n";     
	}
	catch(cl::Error error)
	{
		std::cout << error.what() << "(" << error.err() << ")" <<std::endl;
	}

  //	print_image(channels, outImgRows, outImgCols, c3_out);
	/* ------------------------------------ Max Pool 2D Ends ------------------------------------ */

	
	/* ------------------------------------ Final Layer Starts ------------------------------------ */
	
	int in_features,out_features;	
	in_features = channels*outImgRows*outImgCols;
	out_features = 50;
////	std::cout<<"INFEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE"<<in_features;
	//Read parameters
	const char* f3_weights_file = "p.txt";
	const char* f3_bias_file = "pb.txt";


	float* f3_weights;
	float* f3_biases;

	f3_weights = read_weights_fc(out_features, in_features, f3_weights_file);
	f3_biases = read_bias_fc(out_features, f3_bias_file);
   
  //print_weights(out_channels, in_channels, kernel_size, c1_weights);
	//print_bias(out_channels, c1_biases);

	float* c21_out;
	c21_out = new float [out_features];

	//print_weights_fc(out_features, in_features, f3_weights);
	//print_bias_fc(out_features, f3_biases);

	//std::cout<<"Performing Fully Connected 3"<<std::endl;
	
	try
	{
		cl::Buffer inputBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, in_features*sizeof(float));
		cl::Buffer outputBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, out_features*sizeof(float));
		cl::Buffer weightsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, in_features*out_features*sizeof(float));
		cl::Buffer biasesBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, out_features*sizeof(float));
		cl::Buffer inFeaturesBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
		cl::Buffer outFeaturesBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));

		queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, in_features*sizeof(float), c3_out_new);
		queue.enqueueWriteBuffer(outputBuffer, CL_TRUE, 0, out_features*sizeof(float), c21_out);
		queue.enqueueWriteBuffer(weightsBuffer, CL_TRUE, 0, in_features*out_features*sizeof(float), f3_weights);
		queue.enqueueWriteBuffer(biasesBuffer, CL_TRUE, 0, out_features*sizeof(float), f3_biases);
		queue.enqueueWriteBuffer(inFeaturesBuffer, CL_TRUE, 0, sizeof(int), &in_features);
		queue.enqueueWriteBuffer(outFeaturesBuffer, CL_TRUE, 0, sizeof(int), &out_features);

		std::ifstream sourceFile("cl_kernels/linear.cl");
      std::string sourceCode(
         std::istreambuf_iterator<char>(sourceFile),
         (std::istreambuf_iterator<char>()));
      cl::Program::Sources source(1,
         std::make_pair(sourceCode.c_str(),
         sourceCode.length() + 1));

     	cl::Program program = cl::Program(context, source);

     	program.build(devices);
     	
     	cl::Kernel kernel(program, "linear");

     	kernel.setArg(0, inFeaturesBuffer);
     	kernel.setArg(1, outFeaturesBuffer);
     	kernel.setArg(2, inputBuffer);
     	kernel.setArg(3, weightsBuffer);
     	kernel.setArg(4, biasesBuffer);
     	kernel.setArg(5, outputBuffer);

     	cl::NDRange global(out_features,1);
     	cl::NDRange local(1, 1);
     	queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);

     	queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, out_features*sizeof(float), c21_out);
      
//for(int i=0;i<out_features;i++)
	//std::cout<<"final_out"<<i+1<<"   "<<*(c21_out+i)<<"\n";
	}
	catch(cl::Error error)
	{
		std::cout << error.what() << "(" << error.err() << ")" <<std::endl;
	}
    
	count=print_linear(out_features, c21_out,z);
	// std::cout<<"IN MAIN COUNT "<<count<<"\n";
	/* ------------------------------------ Fully Connected 2 Ends ------------------------------------ */

	
	}
	//*fp.close();
	 
	 Acc= (count*100)/99;
	
	std::cout<<"\nACCURACY =  "<<Acc<<" %"<<"\n"<<"\n";
	return 0;
}
