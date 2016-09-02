#include<stdio.h>

__global__ vecAdd(int *a,int *b,int *c)
{
	int i = threadIdx.x;
	c[i]=a[i]+b[i];
}

int main()
{
	int a[10] = {0,1,2,3,4,5,6,7,8,9};
	int b[10] = {0,1,2,3,4,5,6,7,8,9};
	int *d_a, *d_b, *d_c;
	cudaMalloc((void**)&d_a,10 * sizeof(int));
	cudaMalloc((void**)&d_b, 10 * sizeof(int));
	cudaMalloc((void**)&d_c, 10 * sizeof(int));
	cudaMemcpy(d_a,a,cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,b,cudaMemcpyHostToDevice);
	vecAdd<<<1,10>>>(a,b,c);
	cudaMemcpy(c,d_c,cudaMemcpyDeviceToHost);
	for(int i=0;i<10;i++)
		printf("%d",c[i]);
}

