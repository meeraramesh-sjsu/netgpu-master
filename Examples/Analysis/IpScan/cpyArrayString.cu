#include <stdio.h>
#include <iostream>

#define max_text_length 12;
__device__ int memCmpDev(char *input, char *pattern, int offset,int M)
{
	bool result = true;
	int j = 0;
	for (int i = offset; i < offset + M && result; i++)
	{
		if (input[i] != pattern[j++]) result = false;
	}
	return !result;
}

__global__ void findIfExistsCu(char* d_input,int inputLen,char* pattern,int * indexes,int num_strings,int * patHash,int* d_result)
{
	int startTime = clock();
	int x = threadIdx.x + blockIdx.x * blockDim.x;
 
 	for(int i=0;i<num_strings;i++){
    int M = indexes[2*i+1] - indexes[2*i];
  
	if(x<=inputLen-M)
	{
	int hy,j;
   	for(hy=j=0;j<M;j++)
	hy = (hy * 256 + d_input[j+x]) % 997;
	
	if(hy == patHash[i] && memCmpDev(d_input,pattern,x,M) == 0)
	d_result[i]=1;
	}
	}
	//runtime[x] = int(stopTime - startTime);
}

void calcPatHash(char *tmp[], int *patHash, int numStr)
{
 for(int i=0;i<numStr;i++)
 {
 int index=0;
 while(tmp[i][index]!='\0')
 {
 patHash[i] = (patHash[i] * 256 + tmp[i][index]) % 997;
 index++;
 }
 }
}

int main(){

  char input[] = "absome textcd";
  int inputLen = 13;
  char *d_input;
 int max_text_length;
 int num_str = 3;
 char *tmp[num_str];
 int *patHash;
 int* d_result;
 int* result;
 
 patHash = (int*) malloc(num_str * sizeof(int));
 tmp[0] = (char*) malloc(max_text_length*sizeof(char));
 tmp[1] = (char*) malloc(max_text_length*sizeof(char));
 tmp[2] = (char*) malloc(max_text_length*sizeof(char));
 result = (int *) malloc((num_str)*sizeof(int));
 
 tmp[0] = "some text";
 tmp[1] = "rand txt";
 tmp[2] = "text";

 int stridx[2*num_str];
 int *d_stridx;
 stridx[0] = 0;
 stridx[1] = 9;
 stridx[2] = 9;
 stridx[3] = 17;
 stridx[4] = 17;
 stridx[5] = 21;
 
 char *a, *d_a;
 a = (char *)malloc(num_str*max_text_length*sizeof(char));
 //flatten
 int subidx = 0;
 for(int i=0;i<num_str;i++)
 {
   for (int j=stridx[2*i]; j<stridx[2*i+1]; j++)
     a[j] = tmp[i][subidx++];
   subidx = 0;
 }
 
calcPatHash(tmp,patHash,numStr);
cudaMalloc((void**)&d_a,num_str*max_text_length*sizeof(char));
cudaMemcpy(d_a, a,num_str*max_text_length*sizeof(char),cudaMemcpyHostToDevice);
cudaMalloc((void**)&d_stridx,num_str*2*sizeof(int));
cudaMemcpy(d_stridx, stridx,2*num_str*sizeof(int),cudaMemcpyHostToDevice);
cudaMalloc((void **)&d_input,inputLen * sizeof(char));
cudaMalloc((void **)&d_result,numstr*sizeof(int));
cudaMemcpy(d_input, input, inputLen * sizeof(char), cudaMemcpyHostToDevice);

dim3 block(inputLen);
dim3 grid(1);

 findIfExistsCu<<<grid,block>>>(d_input,inputLen,d_a,d_stridx,num_str,patHash,d_result);
 cudaDeviceSynchronize();

cudaMemcpy(result, d_result, num_str*sizeof(int), cudaMemcpyDeviceToHost);

	for(int i=0;i<numstr;i++)
		cout << result[i]<<" ";
		
}