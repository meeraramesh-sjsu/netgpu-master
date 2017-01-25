#include<stdio.h>
#include<string>
#include<iostream>
using namespace std;

#define statesrow 11
#define chars 256
int states = 0;
int gotofn[statesrow][chars];	

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %dn", cudaGetErrorString(code), file, line);
   }
}

int buildGoto(string arr[],int k)
{
int states = 1;
memset(gotofn,0,sizeof(gotofn));
for(int i=0;i<k;i++)
{
	string temp = arr[i];
	int currentState = 0;
	int ch = 0;
	
	for(int j=0;;j++) {
	ch = temp[j];	
		
	if(gotofn[currentState][ch] == 0)
	gotofn[currentState][ch] = states++;
	
	if(j==temp.size()-1) {
	gotofn[currentState][ch] |= ((1<<i)<<16);
	break;
	}
		
	currentState = gotofn[currentState][ch] & 0x0000FFFF;
	
	}
}
return states;	
}

__global__ void kernelFn(int* gotofn, size_t pitch,char *input,int *result)
{
__shared__ int stateszero[256];
__shared__ char s_input[256];

int threadIndex = threadIdx.x + blockIdx.x * blockDim.x; 
//Copying the state 0 information to shared memory
stateszero[threadIdx.x] = gotofn[threadIdx.x];
s_input[threadIdx.x] = input[threadIndex];
__syncthreads();

int pos=threadIdx.x;
char ch = s_input[pos++];
int nextState = stateszero[ch];
int currState = nextState & 0x0000FFFF;
if(blockIdx.x == 0) printf("ch=%c nextState=%d currState=%d \n",ch,nextState,currState);
if(currState!=0) {

nextState >>= 16;
int outputMatch = nextState & 0x0000FFFF;
if(outputMatch > 0) result[blockIdx.x]= outputMatch;
//printf("currstate %d output %d",currState,outputMatch);
while(currState !=0 && pos<256) {
ch = s_input[pos++];
nextState = gotofn[currState*256 + ch];
currState = nextState & 0x0000FFFF;
nextState >>= 16;
outputMatch = nextState & 0x0000FFFF;
//printf("Output Match %d ",outputMatch);
if(outputMatch > 0) result[blockIdx.x] = outputMatch;
}
}

/*Debug Code
if(threadIdx.x==0 && blockIdx.x==0)
{
printf("In Kernel");
    		for(int i=0;i<11;i++)
    		{
    			printf("state %d Children: ",i);
    			for(int j=0;j<255;j++)
    			{
    				if(gotofn[i*256 + j] != 0)
    					printf("state %d outputVec %d",gotofn[i*256 + j]&0x0000FFFF,(gotofn[i*256 + j]&0xFFFF0000)>>16);
    			}
    			printf("\n");
    		}
} */

}

int main()
{
string arr[]={"he","she","hers","his"};

char *input = "qwertyuheshehershisjxcvbnm1qwertyuiopasdfghjklzxcvbnm2qwertyuiopasdfghjklzxcvbnm3qwertyuiopasdfghjklzxcvbnm4qwertyuiopasdfghjklzxcvbnm5qwertyuiopasdfghjklzxcvbnm6qwertyuiopasdfghjklzxcvbnm7qwertyuiopasdfghjklzxcvbnm8qwertyuiopasdfghjklzxcvbnm9qwertyuiopaaaqwertyuheshehershisjxcvbnm1qwertyuiopasdfghjklzxcvbnm2qwertyuiopasdfghjklzxcvbnm3qwertyuiopasdfghjklzxcvbnm4qwertyuiopasdfghjklzxcvbnm5qwertyuiopasdfghjklzxcvbnm6qwertyuiopasdfghjklzxcvbnm7qwertyuiopasdfghjklzxcvbnm8qwertyuiopasdfghjklzxcvbnm9qwertyuiopaaaqwertyuheshemershisjxcvbnm1qwertyuiopasdfghjklzxcvbnm2qwertyuiopasdfghjklzxcvbnm3qwertyuiopasdfghjklzxcvbnm4qwertyuiopasdfghjklzxcvbnm5qwertyuiopasdfghjklzxcvbnm6qwertyuiopasdfghjklzxcvbnm7qwertyuiopasdfghjklzxcvbnm8qwertyuiopasdfghjklzxcvbnm9qwertyuiopaaaqwertyuabchimerslisjxcvbnm1qwertyuiopasdfghjklzxcvbnm2qwertyuiopasdfghjklzxcvbnm3qwertyuiopasdfghjklzxcvbnm4qwertyuiopasdfghjklzxcvbnm5qwertyuiopasdfghjklzxcvbnm6qwertyuiopasdfghjklzxcvbnm7qwertyuiopasdfghjklzxcvbnm8qwertyuiopasdfghjklzxcvbnm9qwertyuiopaaa";

states = buildGoto(arr,4);
cout<<"states="<<states<<endl;
int *d_gotofn;
size_t pitch;
char* d_input;
size_t N = strlen(input)/256;
cout<<"N= "<<N<<endl;
int * result = (int*)malloc(N *sizeof(int));
memset(result,0,N *sizeof(int));
int * d_result;
gpuErrchk(cudaMallocPitch(&d_gotofn,&pitch,chars * sizeof(int),states));
gpuErrchk(cudaMemcpy2D(d_gotofn,pitch,gotofn,chars * sizeof(int),chars * sizeof(int),states,cudaMemcpyHostToDevice));
gpuErrchk(cudaMalloc(&d_input,strlen(input)*sizeof (char)));
gpuErrchk(cudaMalloc(&d_result,N *sizeof (int)));
gpuErrchk(cudaMemcpy(d_input,input,strlen(input)*sizeof (char),cudaMemcpyHostToDevice));
gpuErrchk(cudaMemset(d_result,0,N*sizeof (int)));
dim3 block(256);
dim3 grid(N);
kernelFn<<<grid,block>>>(d_gotofn,pitch,d_input,d_result);
cudaDeviceSynchronize();
cout <<"Last Error= "<<  cudaGetErrorString(cudaGetLastError())<<endl;
gpuErrchk(cudaMemcpy(result,d_result,N *sizeof (int),cudaMemcpyDeviceToHost));
for(int i=0;i<N;i++)
cout<<"output"<<i<<" "<<result[i]<<" ";
return 0;
}
