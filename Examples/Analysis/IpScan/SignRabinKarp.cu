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

__global__ void findIfExistsCu(char* input, int  N, char* pattern, int M,int patHash,int* result)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if(x<=N-M)
	{
		int hy,i;
		for(int hy=i=0;i<M;i++)
			hy = (hy * 256 + input[i+x]) % 997;
		if(hy == patHash && memCmpDev(input,pattern,x,N,M) == 0)
			result[x]=1;
	}
}

int main()
{
	char input[] = "HEABAL";
	char pattern[] = "AB";
	int M = 2;
	int patHash = 0;
	int N = 6;
	char* d_input;
	char* d_pattern;
	int* d_result;
	int* result;
	for (int i = 0; i < M; i++)
	{
		patHash = (patHash * 256 + pattern[i]) % 997;
	}		
	result = (int *) malloc((N-M)*sizeof(int));
	cudaAssert(cudaMalloc((void **)&d_input, N * sizeof(char)));
	cudaAssert(cudaMalloc((void **)&d_pattern, M * sizeof(char)));
	cudaAssert(cudaMalloc((void **)&d_result,(N-M)*sizeof(int)));
	cudaAssert(cudaMemcpy(d_input, input, N * sizeof(char), cudaMemcpyHostToDevice));
	cudaAssert(cudaMemcpy(d_pattern, pattern, M * sizeof(char), cudaMemcpyHostToDevice));
	cudaAssert(cudaMemset(d_result,0,(N - M)*sizeof(int)));
	dim3 block(N, 0, 0);
	dim3 grid(1, 0, 0);
	findIfExistsCu <<<grid, block>>> (d_input,N,d_pattern,M,patHash,d_result);
	cudaAssert(cudaThreadSynchronize());
	cudaMemcpy(result, d_result, (N-M)*sizeof(int), cudaMemcpyDeviceToHost);
	
	for(int i=0;i<=N-M;i++)
		cout << result[i]<<" ";
	return 0;
}
