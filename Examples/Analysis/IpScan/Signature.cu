#include <stdio.h>
#include <iostream>

#include <cuda_runtime.h>
using namespace std;
__global__ void shiftOrGPU(const char* T, const char *P, const int n,
		const int m, const int *bmBc, const int *preComp, bool *result) 
{
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
	shiftOrGPU<<<1,8>>>(d_text, d_pattern, lent, lenp, d_bmBc, d_preComp, d_result);
	cudaMemcpy(h_result, d_result, lent*sizeof(bool),cudaMemcpyDeviceToHost);
	
	for(int i=0;i<7;i++)
		;
	//	cout<<h_result[i]<<" ";
}