/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

 */

#ifndef AnalysisSkeleton_h
#define AnalysisSkeleton_h

#include <inttypes.h>
#include <iostream>
#include <vector>
#include <string>
#include <ctime>
#include <fstream>

using namespace std;
//#include <cuda.h>
//#include <cuda_runtime.h>
#include "/usr/local/cuda/include/cuda.h"
#include "/usr/local/cuda/include/cuda_runtime.h"

#include "../Util.h"
#include "../Common/PacketBuffer.h"

/* Libraries */
#include "Libs/Host/GpuMemoryManagement/BMMS.h"
#include "Libs/Gpu/CudaSafeAccess.h"
#include "Libs/Gpu/CudaSharedMemoryOperations.h"
#include "Libs/Gpu/Endianness.h" 
#include "Libs/Gpu/InlineFiltering.h"
#include "Libs/Gpu/Protocols.h"
#include "AnalysisState.h"

//Include ppp syncblocks counters
#ifdef __CUDACC__
#include ".syncblocks_counters.ppph"
#endif

//Checkings
#include "Checkings.h"

/* Including MACROS */
#include "Libs/Gpu/Macros/General.h"
#include "Libs/Gpu/Macros/Mining.h"
#include "Libs/Gpu/Macros/Filtering.h"
#include "Libs/Gpu/Macros/Operations.h"
#include "Libs/Gpu/Macros/Hooks.h"
#include "Libs/Gpu/Macros/Util.h"



/* Base blank class AnalysisSkeleton definition */
class AnalysisSkeleton {

public:

private:

};

#ifdef __CUDACC__

/**** Forward declaration prototypes ****/
#define statesrow 550000

int gotofn[550000][256];
int output[statesrow];

template<typename T,typename R>
__global__ void COMPOUND_NAME(IpScan,KernelAnalysis)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results,analysisState_t state,int *gotofn,int *d_result,int *d_output);

template<typename T,typename R>
__device__  void COMPOUND_NAME(IpScan,mining)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results, analysisState_t state);

template<typename T,typename R>
__device__  void COMPOUND_NAME(IpScan,filtering)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results, analysisState_t state);

template<typename T,typename R>
__device__  void COMPOUND_NAME(IpScan,analysis)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results,analysisState_t state,int *gotofn, int *d_result);

template<typename T,typename R>
__device__  void COMPOUND_NAME(IpScan,operations)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results,analysisState_t state);

template<typename R>
void COMPOUND_NAME(IpScan,hooks)(PacketBuffer *packetBuffer, R* results, analysisState_t state, int64_t* auxBlocks,int *d_result);

/**** Module loader ****/
#include ".dmodule.ppph"

//GoTO function used for AhoCorasick Algorithm. Using this function the next State to be taken is determined.
double timeTaken = 0;
int buildGoto(vector<string> arr)
{
	struct timeval startTV, endTV;
	gettimeofday(&startTV, NULL);

	int states = 1;
	memset(gotofn,0,sizeof(gotofn));
	for(int i=0;i<arr.size();i++)
	{
		string temp = arr[i];
		int currentState = 0;
		int ch = 0;

		for(int j=0;j<temp.size();j++) {
			ch = temp[j];

			if(gotofn[currentState][ch] == 0)
				gotofn[currentState][ch] = states++;

			/*	if(j==temp.size()-1) {
	gotofn[currentState][ch] |= ((1<<i)<<16);
	break;*/
			currentState = gotofn[currentState][ch];
		}

		output[currentState] = i;
	}

	gettimeofday(&endTV, NULL);
	timeTaken = endTV.tv_sec * 1e6 + endTV.tv_usec - (startTV.tv_sec * 1e6 + startTV.tv_usec);
	return states;
}

//default Kernel 
template<typename T,typename R>
__global__ void COMPOUND_NAME(IpScan,KernelAnalysis)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results, analysisState_t state, int* gotofn, int *result,int *d_output){
	state.blockIterator = blockIdx.x;
	COMPOUND_NAME(IpScan,mining)(GPU_buffer, GPU_data, GPU_results, state);
	__syncthreads();	

	state.blockIterator = blockIdx.x;
	COMPOUND_NAME(IpScan,filtering)(GPU_buffer, GPU_data, GPU_results, state);
	__syncthreads();	

	/* Analysis implementation*/
	COMPOUND_NAME(IpScan,analysis)(GPU_buffer, GPU_data, GPU_results, state, gotofn, result, d_output);

	/* If there are SYNCBLOCKS barriers do not put Operations function call here */
#if __SYNCBLOCKS_COUNTER == 0 && __SYNCBLOCKS_PRECODED_COUNTER == 0
	COMPOUND_NAME(IpScan,operations)(GPU_buffer, GPU_data, GPU_results, state);
#endif

}

/**** Launch wrapper ****/
//default Launch Wrapper for Analysis not using Windows 
template<typename T,typename R>
void COMPOUND_NAME(IpScan,launchAnalysis_wrapper)(PacketBuffer* packetBuffer, packet_t* GPU_buffer){

	analysisState_t state;
	T *GPU_data;
	R *GPU_results, *results;
	int64_t *auxBlocks;
	int * d_result;

	if(packetBuffer != NULL){

		memset(&state,0,sizeof(state));
		//TODO: CONDITIONAL PINNED MEMORY && STREAMS

		/*** Host memory allocation***/
		//results = (R*)malloc(sizeof(R)*MAX_BUFFER_PACKETS);	
		cudaAssert(cudaHostAlloc((void**)&results,sizeof(R)*MAX_BUFFER_PACKETS,0));

		//auxBlocks = (int64_t*)malloc(sizeof(int64_t)*MAX_BUFFER_PACKETS/ANALYSIS_TPB);	
		cudaAssert(cudaHostAlloc((void**)&auxBlocks,sizeof(int64_t)*MAX_BUFFER_PACKETS,0));

		/*** GPU memory allocation***/
		BMMS::mallocBMMS((void**)&GPU_data,ARRAY_SIZE(T));
		BMMS::mallocBMMS((void**)&GPU_results,ARRAY_SIZE(R));
		BMMS::mallocBMMS((void**)&state.GPU_aux,ARRAY_SIZE(T));  //Auxiliary array
		BMMS::mallocBMMS((void**)&state.GPU_auxBlocks,2*sizeof(int64_t)*MAX_BUFFER_PACKETS);
		BMMS::mallocBMMS((void**)&state.inputs.GPU_extendedParameters,sizeof(int64_t)*MAX_INPUT_EXTENDED_PARAMETERS);
		BMMS::mallocBMMS((void**)&state.GPU_codeRequiresWLR,ARRAY_SIZE(uint32_t)); //Op Code Exec Flags

		/*** MEMSET 0 GPU arrays ***/
		cudaAssert(cudaMemset(GPU_data,0,ARRAY_SIZE(T)));	
		cudaAssert(cudaMemset(GPU_results,0,ARRAY_SIZE(R)));	
		cudaAssert(cudaMemset(state.GPU_aux,0,ARRAY_SIZE(T)));	
		cudaAssert(cudaMemset(state.GPU_auxBlocks,0,2*sizeof(int64_t)*MAX_BUFFER_PACKETS));
		cudaAssert(cudaMemset(state.GPU_codeRequiresWLR,0,ARRAY_SIZE(uint32_t)));

		//N is the number of packets, same as number of blocks
		size_t N = 260;

		/*** KERNEL DIMS ***/
		//dim3 block(ANALYSIS_TPB);		 			//Threads Per Block (1D)
		//dim3 grid(MAX_BUFFER_PACKETS/ANALYSIS_TPB);		 	//Grid size (1D)
		//dim3  block(10);
		//dim3 grid(1);
		dim3 block(256);
		dim3 grid(260);


		//Set state number of blocks and last Packet position
		state.windowState.totalNumberOfBlocks = MAX_BUFFER_PACKETS;
		state.windowState.hasReachedWindowLimit = true;
		state.lastPacket = packetBuffer->getNumOfPackets(); 
		state.windowState.windowStartTime= packetBuffer->getPacket(0)->timestamp;
		state.windowState.windowEndTime= packetBuffer->getPacket(packetBuffer->getNumOfPackets()-1)->timestamp;

		vector<string> tmp;

		string line;
		  ifstream myfile("/home/meera/gpudir/netgpu-master/src/Analysis/Pattern/patterns50.cpp");
		  if (myfile)  // same as: if (myfile.good())
		    {
		    while (getline( myfile, line ))  // same as: while (getline( myfile, line ).good())
		      {
		    	tmp.push_back(line);
		      }
		    myfile.close();
		    }
		  else cout << "fooey\n";
		  cout<<"Number of patterns = "<<tmp.size()<<endl;
		/*tmp.push_back("ebf55eb499cd2180fc21750458eb5190");
		tmp.push_back("2421b44ee90600b43ecd21b44fbacc01cd21726eb8023dba9e00cd218bd8b80057cd2183f90074dfb8024233d233c9cd21a3d401b8004233c933d2cd21b43f8b0ed4018b167a02cd21");
		tmp.push_back("eb2b905a45cd602ec606250601902e803e2606008d3e08060e07755e2ec606260605902ec6062b06ff90eb4e902ec6062b060090b435b060cd21bb000126817f");
		tmp.push_back("5a45cd602ec606250601902e803e2606");
		tmp.push_back("81c91f00cd21b43ecd215a1f59b443b0");
		tmp.push_back("35b060cd21bb000126817f035a4574c0");
		tmp.push_back("1c0226803de8742db99f0183ee03f3a4");
		tmp.push_back("a483eb0426891e020026c7060000f5e9bfcfcfc53690");
		tmp.push_back("5657b800b88ed8bb00008a073c307502b04f88074343");
		tmp.push_back("2ec6062b060090b435b060cd21bb0001");
		tmp.push_back("521eb8023dcd2193b43f33c98ed941ba");
		tmp.push_back("2e300547e2fab8dd4bcd213d34127503");
		tmp.push_back("0f83c61890b9d9062e3004fec046e2f8");
		tmp.push_back("803eb801287326803eb7010977e6b403b009bb03018a2eb8018a0eb701b600b202cd13fe06b701eb");
		tmp.push_back("0733c98bd1b802422e8b1e390f9cfa2eff1ee80dc38becb80057e8ebffbb630f890f895702e8c802");
		tmp.push_back("b80101e8af005b5803c1f7d832e403c8b440cd210e1f72152bc8751133d2b80042cd21ba9b02b903");
		tmp.push_back("c8751133d2b80042cd21ba9b02b90300b440cd21595ab80157cd21b43ecd21e92cffb003cf2a2e2a");
		tmp.push_back("c8751133d2b80042cd21ba9b02b90300b440cd21595ab80157cd21b43ecd21e92cffb003cf2a2e43");
		tmp.push_back("0805fa26a3900026891e9200fbc39c2eff1e0205c3b8004233c933d2e8efffc3b43ee8e9ffc3a11d");
		tmp.push_back("00c353ffd65b8f060f00b440b94f02ba4f02cd2133c9b8004233d2cd21baa304b44059cd21b80157");
		tmp.push_back("02cd2133c9b8004233d2cd21baa304b44059cd21b801575a59cd21b43ecd21585a1f59cd215a1fb8");
		tmp.push_back("b9a5008d960000cd21b440b944028d969303cd21b8004233c933d2cd21b4408d96cf02b91a00cd21");
		tmp.push_back("b8004233c933d2cd21b4408d96cf02b91a00cd21b43ecd21c3b003cfb82435cd215306b4258d96f0");
		tmp.push_back("be9303b94402e867feb440b9a5008d960000cd21b440b944028d969303cd21b8004233c933d2cd21");
		tmp.push_back("21b8024233c999cd21b4408d960301b92202cd21b801578b8e80038b968203cd21b43ecd21b5008a");
		tmp.push_back("5133c9e85200b002e84300b4408d96960359cd21b8024233c999cd21b4408d960301b92202cd21b8");
		tmp.push_back("33f6bb0c00b905008a0704148842f64346e2f5c642f600c7");
		tmp.push_back("680001501e06ba44008ec226ah13b0600017423be000189f7b96701f3a4061fb82135cd213e891e48013e8c064a01b82125ba5901cd21071fbf00febe4c01");
		tmp.push_back("Hello");
		tmp.push_back("how");
		tmp.push_back("are");
		tmp.push_back("you");*/

		/*Pattern matching starts*/
		/*	 vector<string> tmp;
		 tmp.push_back("Hello");
		 tmp.push_back("how");
		 tmp.push_back("are");
		 tmp.push_back("you");*/

		int chars = 256;
		memset(gotofn,0,sizeof(gotofn));
		int states = buildGoto(tmp);
		cout<<"total packets= "<<state.lastPacket<<endl;

		int *d_gotofn;
		int *d_output;
		size_t pitch;
		float time;
		cudaEvent_t start, stop;
		int * result = (int*)malloc(N *sizeof(int));
		memset(result,0,N *sizeof(int));

		cudaAssert(cudaMallocPitch(&d_gotofn,&pitch,chars * sizeof(int),states));
		cudaAssert(cudaMemcpy2D(d_gotofn,pitch,gotofn,chars * sizeof(int),chars * sizeof(int),states,cudaMemcpyHostToDevice));

		//Getting the device pointer for the pinned memory

		cudaAssert(cudaMalloc(&d_result,N *sizeof (int)));
		cudaAssert(cudaMemset(d_result,0,N*sizeof (int)));
		cudaAssert(cudaMalloc(&d_output,states * sizeof(int)));
		cudaAssert(cudaMemcpy(d_output,output,states * sizeof(int),cudaMemcpyHostToDevice));

		cudaAssert( cudaEventCreate(&start) );
		cudaAssert( cudaEventCreate(&stop) );
		//Records an event
		cudaAssert( cudaEventRecord(start, 0) );
		//cudaEventRecord is aynchronous, to make sure the event is recorded, below command used
		cudaAssert( cudaEventSynchronize(start));

		DEBUG(STR(IpScan)"> Throwing Kernel with default implementation.");
		DEBUG(STR(IpScan)"> Parameters -> gridDim:%d",grid.x);

		COMPOUND_NAME(IpScan,KernelAnalysis)<<<grid,block>>>(GPU_buffer,GPU_data,GPU_results,state,d_gotofn,d_result,d_output);
		cudaAssert(cudaThreadSynchronize());

		cudaAssert( cudaEventRecord(stop, 0) );
		cudaAssert( cudaEventSynchronize(stop) );
		cudaAssert( cudaEventElapsedTime(&time, start, stop) );

		printf("Time to generate:  %3.1f ms \n", time);

		/*** Copy results & auxBlocks arrays ***/
		cudaAssert(cudaMemcpy(results,GPU_results,MAX_BUFFER_PACKETS*sizeof(R),cudaMemcpyDeviceToHost));
		cudaAssert(cudaMemcpy(auxBlocks,state.GPU_auxBlocks,sizeof(int64_t)*MAX_BUFFER_PACKETS,cudaMemcpyDeviceToHost));
		cudaAssert(cudaMemcpy(result,d_result,N *sizeof (int),cudaMemcpyDeviceToHost));
		cudaAssert(cudaThreadSynchronize());

		/*** FREE GPU DYNAMIC MEMORY ***/
		BMMS::freeBMMS(GPU_data);
		BMMS::freeBMMS(GPU_results);
		BMMS::freeBMMS(state.GPU_aux);
		BMMS::freeBMMS(state.GPU_auxBlocks);
		BMMS::freeBMMS(state.inputs.GPU_extendedParameters);
		BMMS::freeBMMS(state.GPU_codeRequiresWLR);

		/*** LAUNCH HOOK (Host function) ***/

		/*printf("Printing the multiple pattern result array \n");
		for(int i=0;i<num_str;i++)
			cout<<result[i]<<" ";*/

		cout<<endl;
		//Launch hook (or preHook if window is set)
		COMPOUND_NAME(IpScan,hooks)(packetBuffer, results, state,auxBlocks,result);
		//Frees results
		cudaAssert(cudaFreeHost(results));
		//free(results);


		cout<<"Time taken for GOTO "<<timeTaken<<" us"<<endl;
	}
}


#endif // __CUDACC__


#endif // AnalysisSkeleton_h





