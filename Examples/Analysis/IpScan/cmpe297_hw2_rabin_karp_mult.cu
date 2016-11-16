#include <stdio.h>
#include<vector>
#include <iostream>

using namespace std;

__device__ int memCmpDev(char *input, char *pattern,int *indexes,int inputStart,int offset,int M)
{
		bool result = true;
		int j = indexes[2*inputStart];
		for (int i = offset; i < offset + M && result; i++)
		{
			if (input[i] != pattern[j++]) result = false;
		}
		return !result;
}

__global__ void findIfExistsCu(char* d_input,int input_length,char* pattern,int * indexes,int num_strings,int * patHash,int* d_result,int *runtime)
{
		int startTime = clock64();
		int x = threadIdx.x + blockIdx.x * blockDim.x;
	 
	 	for(int i=0;i<num_strings;i++){
	    int M = indexes[2*i+1] - indexes[2*i];
	  
		if(x<=input_length-M)
		{
		int hy,j;
	   	for(hy=j=0;j<M;j++)
		hy = (hy * 256 + d_input[j+x]) % 997;
		if(hy == patHash[i] && memCmpDev(d_input,pattern,indexes,i,x,M) == 0)
		d_result[i]=1;
		}
		}
		int stopTime = clock64();
		runtime[x]=int(stopTime - startTime);
}

void calcPatHash(vector<string> tmp, int *patHash, int numStr)
{
		for(int i=0;i<numStr;i++)
		{
		for(int index=0;index<(tmp[i].size());index++)
		{
		patHash[i] = (patHash[i] * 256 + tmp[i][index]) % 997;
		}
		}
}

int main()
{
		string input = "absome textcd";
		int input_length = input.size();
		char *d_input;

		vector<string> tmp;
		tmp.push_back("some text");
		tmp.push_back("ab");
		tmp.push_back("text");
		int *patHash;
		int *d_patHash;
		int* d_result;
		int* result;

		int num_str = tmp.size();
		patHash = (int*) calloc(num_str,sizeof(int));
		result = (int*) calloc(num_str,sizeof(int));

		int stridx[2*num_str];
		memset(stridx,0,2*num_str);
		int *d_stridx;
		for(int i=0,j=0,k=0;i<2*num_str;i+=2)
		{
		stridx[i]= k;
		stridx[i+1]= stridx[i]+tmp[j++].size();
		k=stridx[i+1];
		}

		char *pattern, *d_pattern;
		pattern = (char *)malloc(stridx[2*num_str - 1]*sizeof(char));
		//flatten
		int subidx = 0;
		for(int i=0;i<num_str;i++)
		{
		for (int j=stridx[2*i]; j<stridx[2*i+1]; j++)
		{
		pattern[j] = tmp[i][subidx++];
		} 
		subidx = 0;
		}

		int* runtime_gpu;
		int runtime_size = (input_length)* sizeof(int);
		int* runtime = (int*)malloc(runtime_size);
		memset(runtime, 0, runtime_size);
		cudaMalloc((void**)&runtime_gpu, runtime_size);


		calcPatHash(tmp,patHash,num_str);
		cudaMalloc((void**)&d_pattern,stridx[2*num_str - 1]*sizeof(char));
		cudaMemcpy(d_pattern, pattern, stridx[2*num_str - 1]*sizeof(char),cudaMemcpyHostToDevice);
		cudaMalloc((void**)&d_stridx,num_str*2*sizeof(int));
		cudaMemcpy(d_stridx, stridx,2*num_str*sizeof(int),cudaMemcpyHostToDevice);
		cudaMalloc((void **)&d_input,input_length * sizeof(char));
		cudaMalloc((void **)&d_result,num_str*sizeof(int));
		cudaMemcpy(d_input, input.c_str(), input_length * sizeof(char), cudaMemcpyHostToDevice);
		cudaMalloc((void **)&d_patHash, num_str * sizeof(int));
		cudaMemcpy(d_patHash,patHash,num_str * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemset(d_result, 0 , num_str * sizeof(int));

		dim3 block(input_length);
		dim3 grid(1);

		findIfExistsCu<<<grid,block>>>(d_input,input_length,d_pattern,d_stridx,num_str,d_patHash,d_result,runtime_gpu);
		cudaMemcpy(runtime, runtime_gpu, runtime_size, cudaMemcpyDeviceToHost);
		cudaThreadSynchronize();

		int elapsed_time = 0;
		for(int i = 0; i <runtime_size; i++)
		elapsed_time += runtime[i];
		printf("Total cycles: %d \n", elapsed_time);
		elapsed_time = elapsed_time / (824);
		printf("Kernel Execution Time: %d us\n", elapsed_time);


		cudaMemcpy(result, d_result, num_str*sizeof(int), cudaMemcpyDeviceToHost);
		cout<<"Searching for multiple patterns in input sequence"<<endl;
		for(int i=0;i<num_str;i++)
		{
		if(result[i]) cout<<tmp[i]<<" was found"<<endl;
		}				
}
