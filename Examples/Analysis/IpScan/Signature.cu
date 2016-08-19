#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

#define cudaAssert(f) \
	do {	\
		cudaError_t err=f;\
		if(err != cudaSuccess) { \
			fprintf(stderr,"cudaError at %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));\
			exit(-1);\
		}\
	}while(0)

__global__ void shiftOrGPU(const char* T, const char *P, const int n,
		const int m, const int *bmBc, const int *preComp, bool *result,  int* runtime_gpu) 
{
	clock_t start_time = clock();
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;

	if(x <= n-m && preComp[x]==x) {
		bool found  = true;

		for(int i=0;i<m;++i)
		{
			if(P[i]!=T[x+i]) {
				found = false;
				break;
			}
		}
		if(found)
		{
			printf("The pattern was found at thread Id %d",x);
			result[x] = true;
		}
	}
	clock_t stop_time = clock();
	runtime_gpu[x]=(int)(stop_time - start_time);
}

void preComputeShifts(char* T, int n, int m, int *bmBc,  int *preComp)
{
	int i = 0;
	while(i<=n-m)
	{
		i += bmBc[T[i+m]];
		printf("value of i in preCompShift %d \n",i);
		preComp[i]=i;
	}
} 

void preBmBc(char *P,int m, int bmBc[])
{
	int i;
	for(i=0; i<255; i++)
		bmBc[i]=m+1;
	for(i=0;i<m;i++)
		bmBc[i]=m-i;
}

int main(void)
{
	char *h_text = "ABHELO";
	char *h_pattern = "AB";
	int lenp = 3;
	int lent = 7;
	char *d_text;
	char *d_pattern;
	bool *d_result;
	int* d_bmBc;
	int *d_preComp;
	int bmBc[255];
	bool * h_result[7];
	int *h_preComp = (int*) malloc(7 * sizeof(int));

	preBmBc(h_pattern, lenp, bmBc);
	preComputeShifts(h_text, lent, lenp, bmBc, h_preComp);

	cudaMalloc((void**)&d_text,lent*sizeof(char));
	cudaMalloc((void**)&d_pattern,lenp*sizeof(char));
	cudaMalloc((void**)&d_result,lent*sizeof(bool));
	cudaMalloc((void**)&d_bmBc,255*sizeof(int));
	cudaMalloc((void**)&d_preComp,lent*sizeof(int));
	cudaMemcpy(d_text, h_text,lent*sizeof(char),cudaMemcpyHostToDevice);
	cudaMemcpy(d_pattern, h_pattern,lenp*sizeof(char),cudaMemcpyHostToDevice);
	cudaMemcpy(d_bmBc,bmBc,255*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_preComp, h_preComp, lent*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemset(d_result, false, lent*sizeof(bool));
	
	float time;
	cudaEvent_t start, stop;

	cudaAssert( cudaEventCreate(&start) );
	cudaAssert( cudaEventCreate(&stop) );
	cudaAssert( cudaEventRecord(start, 0) );

	 int* runtime_gpu;
		int runtime_size = 1 * 8 * sizeof(int);
	    int* runtime = (int*)malloc(runtime_size);
	    memset(runtime, 0, runtime_size);
	    cudaMalloc((void**)&runtime_gpu, runtime_size);

	shiftOrGPU<<<1,8>>>(d_text, d_pattern, lent, lenp, d_bmBc, d_preComp, d_result,runtime_gpu);
	cudaAssert(cudaThreadSynchronize());
	
	int elapsed_time = 0;
	    for(int i = 0; i < 8; i++)
	            elapsed_time += runtime[i];
	    elapsed_time = elapsed_time / (824 * 10^6);
	    printf("Kernel Execution Time: %d ms\n", elapsed_time/1000);

	cudaAssert( cudaEventRecord(stop, 0) );
	cudaAssert( cudaEventSynchronize(stop) );
	cudaAssert( cudaEventElapsedTime(&time, start, stop) );

	printf("Time to generate:  %3.1f ms \n", time);

	cudaMemcpy(h_result, d_result, lent*sizeof(bool),cudaMemcpyDeviceToHost);

	// cudaDeviceReset causes the driver to clean up all state. While
		// not mandatory in normal operation, it is good practice.  It is also
		// needed to ensure correct operation when the application is being
		// profiled. Calling cudaDeviceReset causes all profile data to be
		// flushed before the application exits
	cudaAssert(cudaDeviceReset());
	for(int i=0;i<7;i++)
		;
	//	cout<<h_result[i]<<" ";
}