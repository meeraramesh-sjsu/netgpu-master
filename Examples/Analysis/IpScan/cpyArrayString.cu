#include <stdio.h>
#include <iostream>

using namespace std;

__device__ int memCmpDev(char *input, char *pattern,int *indexes,int i,int offset,int M)
{
	bool result = true;
	int j = indexes[2*i];
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
	
	if(hy == patHash[i] && memCmpDev(d_input,pattern,indexes,i,x,M) == 0)
	d_result[i]=1;
	}
	}
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

string input = "absome textcd";
int inputLen = input.size();
char *d_input;
 
 //char *tmp[num_str];
 vector<string> tmp;
 tmp.push_back("some text");
 tmp.push_back("ab");
 tmp.push_back("text");
 
 int *patHash;
int *d_patHash;
 int* d_result;
 int* result;
 //int max_text_length = 12;
int num_str = tmp.size();

 patHash = (int*) malloc(num_str * sizeof(int));
 /*tmp[0] = (char*) malloc(max_text_length*sizeof(char));
 tmp[1] = (char*) malloc(max_text_length*sizeof(char));
 tmp[2] = (char*) malloc(max_text_length*sizeof(char));*/
 /*tmp[0] = "some text";
 tmp[1] = "ab";
 tmp[2] = "text";*/
 
 result = (int*) malloc((num_str)*sizeof(int));
 
 memset(result,0,num_str*sizeof(int));

 int stridx[2*num_str];
 memset(stridx,0,2*num_str);
 int *d_stridx;
 for(int i=0,j=0;i<2*num_str;i+=2)
 {
 stridx[i]= stridx[i+1];
 stridx[i+1]= stridx[i]+tmp[j++].size();
 }
 /*stridx[0] = 0;
 stridx[1] = 9;
 stridx[2] = 9;
 stridx[3] = 11;
 stridx[4] = 11;
 stridx[5] = 15;*/
 
 char *a, *d_a;
 a = (char *)malloc(stridx[2*num_str - 1]*sizeof(char));
 //flatten
 int subidx = 0;
 for(int i=0;i<num_str;i++)
 {
   for (int j=stridx[2*i]; j<stridx[2*i+1]; j++)
     a[j] = tmp[i][subidx++];
   subidx = 0;
 }
 
calcPatHash(tmp,patHash,num_str);
cudaMalloc((void**)&d_a,stridx[2*num_str - 1]*sizeof(char));
cudaMemcpy(d_a, a, stridx[2*num_str - 1]*sizeof(char),cudaMemcpyHostToDevice);
cudaMalloc((void**)&d_stridx,num_str*2*sizeof(int));
cudaMemcpy(d_stridx, stridx,2*num_str*sizeof(int),cudaMemcpyHostToDevice);
cudaMalloc((void **)&d_input,inputLen * sizeof(char));
cudaMalloc((void **)&d_result,num_str*sizeof(int));
cudaMemcpy(d_input, input.c_str(), inputLen * sizeof(char), cudaMemcpyHostToDevice);
cudaMalloc((void **)&d_patHash, num_str * sizeof(int));
cudaMemcpy(d_patHash,patHash,num_str * sizeof(int), cudaMemcpyHostToDevice);
cudaMemset(d_result, 0 , num_str * sizeof(int));

dim3 block(inputLen);
dim3 grid(1);

 findIfExistsCu<<<grid,block>>>(d_input,inputLen,d_a,d_stridx,num_str,d_patHash,d_result);
 cudaDeviceSynchronize();

cudaMemcpy(result, d_result, num_str*sizeof(int), cudaMemcpyDeviceToHost);

cout<<"Searching for multiple patterns in input sequence"<<endl;
	for(int i=0;i<num_str;i++)
{
if(result[i]) cout<<tmp[i]<<" was found"<<endl;
}		
		
}
