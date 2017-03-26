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
int* output;

template<typename T,typename R>
__global__ void COMPOUND_NAME(ANALYSIS_NAME,KernelAnalysis)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results,analysisState_t state,int *gotofn,int *d_result,int *d_output);

template<typename T,typename R>
__device__  void COMPOUND_NAME(ANALYSIS_NAME,mining)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results, analysisState_t state);

template<typename T,typename R>
__device__  void COMPOUND_NAME(ANALYSIS_NAME,filtering)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results, analysisState_t state);

template<typename T,typename R>
__device__  void COMPOUND_NAME(ANALYSIS_NAME,analysis)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results,analysisState_t state,int *gotofn, int *d_result);

template<typename T,typename R>
__device__  void COMPOUND_NAME(ANALYSIS_NAME,operations)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results,analysisState_t state);

template<typename R>
void COMPOUND_NAME(ANALYSIS_NAME,hooks)(PacketBuffer *packetBuffer, R* results, analysisState_t state, int64_t* auxBlocks,int *d_result);

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
printf("Number of states in gotofn = %d",states);	
return states;
}

//default Kernel 
template<typename T,typename R>
__global__ void COMPOUND_NAME(ANALYSIS_NAME,KernelAnalysis)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results, analysisState_t state, int* gotofn, int *result,int *d_output){
	state.blockIterator = blockIdx.x;
	COMPOUND_NAME(ANALYSIS_NAME,mining)(GPU_buffer, GPU_data, GPU_results, state);
	__syncthreads();	

	state.blockIterator = blockIdx.x;
	COMPOUND_NAME(ANALYSIS_NAME,filtering)(GPU_buffer, GPU_data, GPU_results, state);
	__syncthreads();	

	/* Analysis implementation*/
	COMPOUND_NAME(ANALYSIS_NAME,analysis)(GPU_buffer, GPU_data, GPU_results, state, gotofn, result, d_output);

	/* If there are SYNCBLOCKS barriers do not put Operations function call here */
#if __SYNCBLOCKS_COUNTER == 0 && __SYNCBLOCKS_PRECODED_COUNTER == 0
	COMPOUND_NAME(ANALYSIS_NAME,operations)(GPU_buffer, GPU_data, GPU_results, state);
#endif

}

/**** Launch wrapper ****/
//default Launch Wrapper for Analysis not using Windows 
template<typename T,typename R>
void COMPOUND_NAME(ANALYSIS_NAME,launchAnalysis_wrapper)(PacketBuffer* packetBuffer, packet_t* GPU_buffer,int numberOfPatterns){

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
		printf("%d numberOfPatterns= ",numberOfPatterns);
		char* str = (char* ) malloc(sizeof(int));
		sprintf(str,"%d",numberOfPatterns);
		printf("%s",str);
		string temp(str);
		string fileName = "/home/meera/gpudir/netgpu-master/src/Analysis/Pattern/patterns" + temp + ".cpp";
		string line;

		  ifstream myfile(fileName.c_str());

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
	
		int chars = 256;
		memset(gotofn,0,sizeof(gotofn));

		cudaAssert(cudaHostAlloc((void**) &output, statesrow * sizeof(int), cudaHostAllocMapped));
		DEBUG(STR(ANALYSIS_NAME)"> Allocated Output Array in Pinned memory, size = %d Bytes",statesrow * sizeof(int));

		int states = buildGoto(tmp);
		int* array;

		//	printf("CUDA HOST ALLOC \n");
		cudaAssert(cudaHostAlloc((void**) &array, states * 256 * sizeof(int), cudaHostAllocMapped));

		DEBUG(STR(ANALYSIS_NAME)"> Allocated Host Array in Pinned memory, size = %d Bytes",states * 256 * sizeof(int));

		for(int i=0;i<states;i++)
			for(int j=0;j<256;j++)
		array[i * 256 + j ] = gotofn[i][j];

		cout<<"total packets= "<<state.lastPacket<<endl;

		int *d_gotofn;
		int *d_output;
		size_t pitch;
		float time;
		cudaEvent_t start, stop;
		int * result;
		cudaAssert(cudaHostAlloc((void**) &result, N * sizeof(int), cudaHostAllocMapped));

		memset(result,0,N *sizeof(int));

		//cudaAssert(cudaMallocPitch(&d_gotofn,&pitch,chars * sizeof(int),states));
		//cudaAssert(cudaMemcpy2D(d_gotofn,pitch,gotofn,chars * sizeof(int),chars * sizeof(int),states,cudaMemcpyHostToDevice));

		//Getting the device pointer for the pinned memory
		cudaAssert(cudaHostGetDevicePointer(&d_gotofn, array, 0));
		DEBUG(STR(ANALYSIS_NAME)"Set a device pointer to point to the array");

		cudaAssert(cudaHostGetDevicePointer(&d_result, result, 0));
		cudaAssert(cudaMemset(d_result,0,N*sizeof (int)));
		cudaAssert(cudaHostGetDevicePointer(&d_output, output, 0));

		cudaAssert( cudaEventCreate(&start) );
		cudaAssert( cudaEventCreate(&stop) );
		//Records an event
		cudaAssert( cudaEventRecord(start, 0) );
		//cudaEventRecord is aynchronous, to make sure the event is recorded, below command used
		cudaAssert( cudaEventSynchronize(start));

		DEBUG(STR(ANALYSIS_NAME)"> Throwing Kernel with default implementation.");
		DEBUG(STR(ANALYSIS_NAME)"> Parameters -> gridDim:%d",grid.x);

		COMPOUND_NAME(ANALYSIS_NAME,KernelAnalysis)<<<grid,block>>>(GPU_buffer,GPU_data,GPU_results,state,d_gotofn,d_result,d_output);
		cudaAssert(cudaThreadSynchronize());

		cudaAssert( cudaEventRecord(stop, 0) );
		cudaAssert( cudaEventSynchronize(stop) );
		cudaAssert( cudaEventElapsedTime(&time, start, stop) );

		printf("Time to generate:  %3.1f ms \n", time);

		/*** Copy results & auxBlocks arrays ***/
		cudaAssert(cudaMemcpy(results,GPU_results,MAX_BUFFER_PACKETS*sizeof(R),cudaMemcpyDeviceToHost));
		cudaAssert(cudaMemcpy(auxBlocks,state.GPU_auxBlocks,sizeof(int64_t)*MAX_BUFFER_PACKETS,cudaMemcpyDeviceToHost));
		//cudaAssert(cudaMemcpy(result,d_result,N *sizeof (int),cudaMemcpyDeviceToHost));
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
		COMPOUND_NAME(ANALYSIS_NAME,hooks)(packetBuffer, results, state,auxBlocks,result);
		//Frees results
		cudaAssert(cudaFreeHost(results));
		//free(results);


		cout<<"Time taken for GOTO "<<timeTaken<<" us"<<endl;
	}
}


#endif // __CUDACC__


#endif // AnalysisSkeleton_h





