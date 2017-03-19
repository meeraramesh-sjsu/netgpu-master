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

template<typename T,typename R>
__global__ void COMPOUND_NAME(ANALYSIS_NAME,KernelAnalysis)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results,analysisState_t state,char* pattern,int * indexes,int num_strings,int * patHash,int *d_result);

template<typename T,typename R>
__device__  void COMPOUND_NAME(ANALYSIS_NAME,mining)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results, analysisState_t state,char* pattern,int * indexes,int num_strings,int * patHash);

template<typename T,typename R>
__device__  void COMPOUND_NAME(ANALYSIS_NAME,filtering)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results, analysisState_t state);

template<typename T,typename R>
__device__  void COMPOUND_NAME(ANALYSIS_NAME,analysis)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results,analysisState_t state,int *d_result);

template<typename T,typename R>
__device__  void COMPOUND_NAME(ANALYSIS_NAME,operations)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results,analysisState_t state);

template<typename R>
void COMPOUND_NAME(ANALYSIS_NAME,hooks)(PacketBuffer *packetBuffer, R* results, analysisState_t state, int64_t* auxBlocks,int *d_result,char *pattern,int *stridx);

/**** Module loader ****/
#include ".dmodule.ppph"

/*
*** Kernel Prototypes ***
//Predefined Code kernels
#define ITERATOR__ 0
#include "PredefinedExtraKernel.def"

#define ITERATOR__ 1
#include "PredefinedExtraKernel.def"

#define ITERATOR__ 2 
#include "PredefinedExtraKernel.def"

#define ITERATOR__ 3
#include "PredefinedExtraKernel.def"

#define ITERATOR__ 4
#include "PredefinedExtraKernel.def"

#define ITERATOR__ 5
#include "PredefinedExtraKernel.def"

#define ITERATOR__ 6
#include "PredefinedExtraKernel.def"

#define ITERATOR__ 7
#include "PredefinedExtraKernel.def"

#define ITERATOR__ 8
#include "PredefinedExtraKernel.def"

#define ITERATOR__ 9
#include "PredefinedExtraKernel.def"

#define ITERATOR__ 10
#include "PredefinedExtraKernel.def"

#define ITERATOR__ 11
#include "PredefinedExtraKernel.def"

#define ITERATOR__ 12
#include "PredefinedExtraKernel.def"

#define ITERATOR__ 13
#include "PredefinedExtraKernel.def"

#define ITERATOR__ 14 
#include "PredefinedExtraKernel.def"

#define ITERATOR__ 15
#include "PredefinedExtraKernel.def"

//User kernels
#define ITERATOR__ 0
#include "UserExtraKernel.def"

#define ITERATOR__ 1 
#include "UserExtraKernel.def"

#define ITERATOR__ 2
#include "UserExtraKernel.def"

#define ITERATOR__ 3
#include "UserExtraKernel.def"

#define ITERATOR__ 4
#include "UserExtraKernel.def"

#define ITERATOR__ 5
#include "UserExtraKernel.def"

#define ITERATOR__ 6
#include "UserExtraKernel.def"

#define ITERATOR__ 7
#include "UserExtraKernel.def"

#define ITERATOR__ 8
#include "UserExtraKernel.def"

#define ITERATOR__ 9
#include "UserExtraKernel.def"

#define ITERATOR__ 10
#include "UserExtraKernel.def"

#define ITERATOR__ 11
#include "UserExtraKernel.def"

#define ITERATOR__ 12
#include "UserExtraKernel.def"

#define ITERATOR__ 13
#include "UserExtraKernel.def"

#define ITERATOR__ 14
#include "UserExtraKernel.def"

#define ITERATOR__ 15
#include "UserExtraKernel.def"
*/

/* END OF EXTRA KERNELS */
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

//default Kernel 
template<typename T,typename R>
__global__ void COMPOUND_NAME(ANALYSIS_NAME,KernelAnalysis)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results, analysisState_t state,char* pattern,int * indexes,int num_strings,int * patHash,int *d_result){
	state.blockIterator = blockIdx.x;
	COMPOUND_NAME(ANALYSIS_NAME,mining)(GPU_buffer, GPU_data, GPU_results, state,pattern,indexes,num_strings,patHash);
	__syncthreads();	

	state.blockIterator = blockIdx.x;
	COMPOUND_NAME(ANALYSIS_NAME,filtering)(GPU_buffer, GPU_data, GPU_results, state);
	__syncthreads();	

	/* Analysis implementation*/
	COMPOUND_NAME(ANALYSIS_NAME,analysis)(GPU_buffer, GPU_data, GPU_results, state, d_result);

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
		cudaAssert(cudaThreadSynchronize());

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

		DEBUG(STR(ANALYSIS_NAME)"> Throwing Kernel with default implementation.");
		DEBUG(STR(ANALYSIS_NAME)"> Parameters -> gridDim:%d",grid.x);

		float time;
		cudaEvent_t start, stop;

		cudaAssert( cudaEventCreate(&start) );
		cudaAssert( cudaEventCreate(&stop) );
		cudaAssert( cudaEventRecord(start, 0) );

		/*Pattern matching starts*/
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

		 int *patHash;
		 int *d_patHash;

		 int num_str = tmp.size();

		 //patHash = (int*) calloc(num_str,sizeof(int));
		 cudaAssert(cudaHostAlloc((void**) &patHash, num_str * sizeof(int), cudaHostAllocMapped));
		 memset(patHash,0,num_str);

		 int* stridx;
		 cudaAssert(cudaHostAlloc((void**) &stridx, 2*num_str * sizeof(int), cudaHostAllocMapped));

		 memset(stridx,0,2*num_str);
		 int *d_stridx;

		 for(int i=0,j=0,k=0;i<2*num_str;i+=2)
		 {
		 stridx[i]= k;
		 stridx[i+1]= stridx[i]+tmp[j++].size();
		 k=stridx[i+1];
		 }

		 char *a, *d_a;
		 int *d_result;
		 //a = (char *)malloc(stridx[2*num_str - 1]*sizeof(char));
		 cudaAssert(cudaHostAlloc((void**) &a, stridx[2*num_str - 1] * sizeof(char), cudaHostAllocMapped));

		 //flatten
		 int subidx = 0;
		 for(int i=0;i<num_str;i++)
		 {
		 for (int j=stridx[2*i]; j<stridx[2*i+1]; j++)
		 {
		     a[j] = tmp[i][subidx++];
		}
		 subidx = 0;
		}

		calcPatHash(tmp,patHash,num_str);
		//cudaMalloc((void**)&d_a,stridx[2*num_str - 1]*sizeof(char));
		//cudaMemcpy(d_a, a, stridx[2*num_str - 1]*sizeof(char),cudaMemcpyHostToDevice);
		//cudaMalloc((void**)&d_stridx,num_str*2*sizeof(int));
		//cudaMemcpy(d_stridx, stridx,2*num_str*sizeof(int),cudaMemcpyHostToDevice);
		//cudaMalloc((void **)&d_patHash, num_str * sizeof(int));
		//cudaMemcpy(d_patHash,patHash,num_str * sizeof(int), cudaMemcpyHostToDevice);
		//cudaMalloc((void**)&d_result,num_str * sizeof(int));
		//cudaMemset(d_result,0,num_str*sizeof(int));
		cudaAssert(cudaHostGetDevicePointer(&d_a, a, 0));
		cudaAssert(cudaHostGetDevicePointer(&d_stridx, stridx, 0));
		cudaAssert(cudaHostGetDevicePointer(&d_patHash, patHash, 0));

		int *result;
		//result = (int*)malloc(num_str * sizeof(int));
		 cudaAssert(cudaHostAlloc((void**) &result,num_str * sizeof(int), cudaHostAllocMapped));

		memset(result,0,num_str*sizeof(int));
		cudaAssert(cudaHostGetDevicePointer(&d_result, result, 0));
		//char* pattern,int * indexes,int num_strings,int * patHash add to kernel
		/*Pattern matching ends*/

		COMPOUND_NAME(ANALYSIS_NAME,KernelAnalysis)<<<grid,block>>>(GPU_buffer,GPU_data,GPU_results,state,d_a,d_stridx,num_str,d_patHash,d_result);
		cudaAssert(cudaThreadSynchronize());

		cudaAssert( cudaEventRecord(stop, 0) );
		cudaAssert( cudaEventSynchronize(stop) );
		cudaAssert( cudaEventElapsedTime(&time, start, stop) );

		printf("Time to generate:  %3.1f ms \n", time);

		/*** Copy results & auxBlocks arrays ***/
		cudaAssert(cudaMemcpy(results,GPU_results,MAX_BUFFER_PACKETS*sizeof(R),cudaMemcpyDeviceToHost));
		cudaAssert(cudaMemcpy(auxBlocks,state.GPU_auxBlocks,sizeof(int64_t)*MAX_BUFFER_PACKETS,cudaMemcpyDeviceToHost));
		//cudaAssert(cudaMemcpy(result,d_result,num_str*sizeof(int),cudaMemcpyDeviceToHost));
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
		COMPOUND_NAME(ANALYSIS_NAME,hooks)(packetBuffer, results, state,auxBlocks,result,a,stridx);
		//Frees results
		cudaAssert(cudaFreeHost(results));
		//free(results);
	}
}


#endif // __CUDACC__


#endif // AnalysisSkeleton_h





