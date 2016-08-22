// RabinKarpStringMatching.cpp : Defines the entry point for the console application.
//

#include<stdio.h>
#include<iostream>
using namespace std;

__device__ int memCmpDev(char *input, char *pattern, int offset,int N,int M)
{
	if (threadIdx.x == offset)
	{
		if (N - offset < M) return -1;
		bool result = true;
		int j = 0;
		for (int i = offset; i < offset + M && result; i++)
		{
			if (input[i] != pattern[j++]) result = false;
		}
		return !result;
	}
}

__global__ void findIfExistsCu(char input[], int  N, char pattern[], int M,int RM,int inputHash,int patHash,int* result)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if (threadIdx.x==0 && inputHash == patHash && memCmpDev(input, pattern, 0, N, M) == 0) 
		{int x = 0; result = &x;}
	else
	{
		if (x >= M && N - M >= M)
		{
			inputHash = (inputHash + 997 - (input[x - M] * RM) % 997) % 997;
			inputHash = (inputHash * 256 + input[x]) % 997;
			if (inputHash == patHash && memCmpDev(input, pattern, x - M + 1, N, M) == 0)
				{
				int y = x-1;
				result= &y;}
		}
	}
}
int main()
{
	char input[] = "HEABAL";
	char pattern[] = "AB";
	int M = 2;
	int patHash = 0;
	int initInputHash = 0;
	int N = 6;
	char* d_input;
	int* d_N;
	char* d_pattern;
	int* d_M;
	int* d_RM;
	int* d_initInputHash;
	int* d_patHash;
	int* d_result;
	int result = -1;
	int RM = 1;
	for (int i = 1; i <= M - 1; i++)
		RM = (256 * RM) % 997;
	if (N >= M)
	{
		for (int i = 0; i < M; i++)
		{
			initInputHash = (initInputHash * 256 + input[i]) % 997;
			patHash = (patHash * 256 + pattern[i]) % 997;
		}		
		cudaMalloc((void **)&d_input, N * sizeof(char));
		cudaMalloc((void **)&d_N, sizeof(int));
		cudaMalloc((void **)&d_pattern, M * sizeof(char));
		cudaMalloc((void **)&d_M, sizeof(int));
		cudaMalloc((void **)&d_RM, sizeof(int));
		cudaMalloc((void **)&d_initInputHash, sizeof(int));
		cudaMalloc((void **)&d_patHash, sizeof(int));
		cudaMalloc((void **)&d_result, sizeof(int));
		cudaMemcpy(d_input, input, N * sizeof(char), cudaMemcpyHostToDevice);
	//	cudaMemcpy(d_N, N,sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_pattern, pattern, M * sizeof(char), cudaMemcpyHostToDevice);
	//	cudaMemcpy(d_RM,RM, sizeof(int), cudaMemcpyHostToDevice);
	//	cudaMemcpy(d_M, M, sizeof(int), cudaMemcpyHostToDevice);
	//	cudaMemcpy(d_initInputHash, initInputHash, sizeof(int), cudaMemcpyHostToDevice);
	//	cudaMemcpy(d_patHash, patHash, sizeof(int), cudaMemcpyHostToDevice);
		dim3 block(N, 0, 0);
		dim3 grid(1, 0, 0);
		findIfExistsCu <<<grid, block>>> (d_input,N,d_pattern,M,RM,initInputHash,patHash,d_result);
		cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
	}
	cout << result;
	return 0;
}