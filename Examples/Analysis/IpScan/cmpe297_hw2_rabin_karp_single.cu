// RabinKarpStringMatching.cpp : Defines the entry point for the console application.
//

#include<stdio.h>
#include<iostream>
using namespace std;

#define cudaAssert(f) \
		do {	\
			cudaError_t err=f;\
			if(err != cudaSuccess) { \
				fprintf(stderr,"cudaError at %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));\
				exit(-1);\
			}\
		}while(0)

/* Compare the input string with the pattern, starting from the offset upto the pattern length */
__device__ int memCmpDev(char *input, char *pattern, int offset,int N,int M)
{
	bool result = true;
	int j = 0;
	for (int i = offset; i < offset + M && result; i++)
	{
		if (input[i] != pattern[j++]) result = false;
	}
	return !result;
}

__global__ void findIfExistsCu(char* input, int  input_length, char* pattern, int pattern_length,int patHash,int* result,int *runtime)
{ 
	int startTime = clock();
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if(x<=input_length-pattern_length)
	{
		int hy,i;
		for(int hy=i=0;i<pattern_length;i++)
			hy = (hy * 256 + input[i+x]) % 997;
		if(hy == patHash && memCmpDev(input,pattern,x,input_length,pattern_length) == 0)
			result[x]=1;
	}
	int stopTime = clock();
	runtime[x] = int(stopTime - startTime);
}

int main()
{
	char input[] = "HEABAL";
	char pattern[] = "AB";
	int pattern_length = 2;
	int patHash = 0;
	int input_length = 6;
	char* d_input;
	char* d_pattern;
	int* d_result;
	int* result;
	for (int i = 0; i < pattern_length; i++)
	{
		patHash = (patHash * 256 + pattern[i]) % 997;
	}
	result = (int *) malloc((input_length)*sizeof(int));
	cudaAssert(cudaMalloc((void **)&d_input, input_length * sizeof(char)));
	cudaAssert(cudaMalloc((void **)&d_pattern, pattern_length * sizeof(char)));
	cudaAssert(cudaMalloc((void **)&d_result,(input_length)*sizeof(int)));
	cudaAssert(cudaMemcpy(d_input, input, input_length * sizeof(char), cudaMemcpyHostToDevice));

	cudaAssert(cudaMemcpy(d_pattern, pattern, pattern_length * sizeof(char), cudaMemcpyHostToDevice));
	cudaAssert(cudaMemset(d_result,0,(input_length)*sizeof(int)));
	dim3 block(input_length-pattern_length+1);
	dim3 grid(1);
	
	int* runtime_gpu;
	int runtime_size = (input_length-pattern_length+1)* sizeof(int);
	int* runtime = (int*)malloc(runtime_size);
	memset(runtime, 0, runtime_size);
	cudaMalloc((void**)&runtime_gpu, runtime_size);

	findIfExistsCu <<<grid,block>>> (d_input,input_length,d_pattern,pattern_length,patHash,d_result,runtime_gpu);
	cudaMemcpy(runtime, runtime_gpu, runtime_size, cudaMemcpyDeviceToHost);
	cudaAssert(cudaThreadSynchronize());

	int elapsed_time = 0;
	for(int i = 0; i <runtime_size; i++)
		elapsed_time += runtime[i];
	printf("Total cycles: %d \n", elapsed_time);
	elapsed_time = elapsed_time / (824);
	printf("Kernel Execution Time: %d us\n", elapsed_time);

	cudaMemcpy(result, d_result, (input_length)*sizeof(int), cudaMemcpyDeviceToHost);

	cout<<"Searching for a single pattern in a single string"<<endl;
	cout<<"Print at which position the pattern was found"<<endl;
	cout<<"Input string = "<<input<<endl;
	cout<<"pattern="<<pattern<<endl;
	for(int i=0;i<input_length;i++)
		cout << "Pos:"<<i<<" Result: "<< result[i]<<" "<<endl;
	return 0;
}

