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
__global__ void COMPOUND_NAME(ANALYSIS_NAME,KernelAnalysis)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results,analysisState_t state);

template<typename T,typename R>
__device__  void COMPOUND_NAME(ANALYSIS_NAME,mining)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results, analysisState_t state);

template<typename T,typename R>
__device__  void COMPOUND_NAME(ANALYSIS_NAME,filtering)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results, analysisState_t state);

template<typename T,typename R>
__device__  void COMPOUND_NAME(ANALYSIS_NAME,analysis)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results,analysisState_t state);

template<typename T,typename R>
__device__  void COMPOUND_NAME(ANALYSIS_NAME,operations)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results,analysisState_t state);

void kernel_wrapper(int *a,int *b);
template<typename R>
void COMPOUND_NAME(ANALYSIS_NAME,hooks)(PacketBuffer *packetBuffer, R* results, analysisState_t state, int64_t* auxBlocks); 

/**** Module loader ****/
#include ".dmodule.ppph"

/**** Kernel Prototypes ****/
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


/* END OF EXTRA KERNELS */


#if HAS_WINDOW == 1

//default Windowed Kernel
template<typename T,typename R>
__global__ void COMPOUND_NAME(ANALYSIS_NAME,KernelAnalysis)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results, analysisState_t state){

	/*** MINING & PREANALYSIS FILTERING ***/
	//Setting block iterator to my block by default
	state.blockIterator = blockIdx.x;

//Commented below parts	
	//Looping when array is larger than grid dimensions 
	while( state.blockIterator < state.windowState.totalNumberOfBlocks ){
		//If not previously Mined & prefiltered
		if(state.blockIterator >= state.windowState.blocksPreviouslyMined){	
 			COMPOUND_NAME(ANALYSIS_NAME,mining)(GPU_buffer, GPU_data, GPU_results, state);
			__syncthreads();	
//
			COMPOUND_NAME(ANALYSIS_NAME,filtering)(GPU_buffer, GPU_data, GPU_results, state);
			__syncthreads();	
		}
		state.blockIterator += gridDim.x;
	}
//
//	/*** ANALYSIS OPERATIONS ***/
	state.blockIterator = blockIdx.x;

	//Note that no loop used in here. Loop should be implemented in the Analysis code or used predefined analysis code
	COMPOUND_NAME(ANALYSIS_NAME,analysis)(GPU_buffer, GPU_data, GPU_results, state);
	__syncthreads();	

	/*** POST ANALYSIS OPERATIONS ***/

	/* If there are SYNCBLOCKS barriers do not put Operations function call here */
	#if __SYNCBLOCKS_COUNTER == 0 && __SYNCBLOCKS_PRECODED_COUNTER == 0
		COMPOUND_NAME(ANALYSIS_NAME,operations)(GPU_buffer, GPU_data, GPU_results, state);
	#endif 
}

//default Launch Wrapper for Analysis with Windows

/*************************************************************************************************************************/
/* 															 */
/* These wrapper assumes that the number of analysis input elements is equal to the number of output elements		 */
/* To improve these, use an specialization of COMPOUND_NAME(ANALYSIS_NAME,launchAnalysis_wrapper), as in Histogramer LIB */
/* 				 	----------									 */
/*				Nin -> | ANALYSIS | -> Nout 								 */
/* 				 	----------									 */
/*   				    Where Nin == Nout									 */
/* It also assumes that Input type and output type are the same 							 */
/*************************************************************************************************************************/
 
template<typename T,typename R>
void COMPOUND_NAME(ANALYSIS_NAME,launchAnalysis_wrapper)(PacketBuffer* packetBuffer, packet_t* GPU_buffer){

	/* WINDOW flag and state static vars*/
	//TODO: stupid static flag to avoid NVCC compiler BUG with static template-type var declaration&initialization in the same line
	static bool init=true;
	static analysisState_t state; //Analysis State
	static int spreadFactor; //Spread factor
	
	/* GPU_data static var note that T and R are the same type */
	static T *GPU_data;
	
	/* Automatic vars*/
	R *GPU_results; //Results pointer
	
	//If a NULL buffer & window is empty return
	if(packetBuffer == NULL && GPU_data == NULL)
		return;
	
	
	#if WINDOW_TYPE == PACKET_WINDOW 
		/* PACKET LIMIT */
		if(state.windowState.hasReachedWindowLimit || init || GPU_data == NULL){
			//TODO: uncomment this when CUDA BUG accessing large arrays is solved.
			//((WINDOW_LIMIT)%MAX_BUFFER_PACKETS == 0)? spreadFactor =(WINDOW_LIMIT)/MAX_BUFFER_PACKETS :spreadFactor =((WINDOW_LIMIT)/MAX_BUFFER_PACKETS)+1;
			((WINDOW_LIMIT)%MAX_BUFFER_PACKETS == 0)? spreadFactor =(WINDOW_LIMIT)/MAX_BUFFER_PACKETS :spreadFactor =((WINDOW_LIMIT)/MAX_BUFFER_PACKETS)-1;

		}

	#elif WINDOW_TYPE == TIME_WINDOW
		/* TIME LIMIT */
		if(state.windowState.hasReachedWindowLimit || init || GPU_data == NULL){
			spreadFactor = 2; //TODO: File configurable
		}
	#else
		#error Incorrect WINDOW_TYPE value or value not defined.
	#endif

	//TODO: Stupid assignation due NVCC compiler BUG
	if(init){
		GPU_data = NULL;
		init = false;
	}
	
	if(GPU_data == NULL){
		//First time or end of window reached
		//Init state state
		memset(&state,0,sizeof(state));
	}
		
	//window Start time
	//TODO: CONDITIONAL PINNED MEMORY && STREAMS

	/*---------------- MEMORY ALLOCATION ----------------*/

	if(GPU_data == NULL){
		//Initialization
		BMMS::mallocBMMS((void**)&GPU_data,ARRAY_SIZE(T)*spreadFactor);
		cudaAssert(cudaMemset(GPU_data,0,ARRAY_SIZE(T)*spreadFactor));	
		//Set windowStartTime
		state.windowState.windowStartTime= packetBuffer->getPacket(0)->timestamp;
	}else{
		//Check for space and set spreadFactor
		if((state.windowState.totalNumberOfBlocks + BUFFER_BLOCKS) > spreadFactor*BUFFER_BLOCKS){
			//Set spreadFactor to new value	
			int previousSpreadFactor = spreadFactor;
			spreadFactor+= 2; //TODO: file configurable

			DEBUG2("Reallocating GPU_data. New spreadFactor:%d",spreadFactor);

			//Reallocate GPU_data
			T* dummy = GPU_data;
			BMMS::mallocBMMS((void**)&GPU_data,ARRAY_SIZE(T)*spreadFactor);
			cudaAssert(cudaMemset(GPU_data,0,ARRAY_SIZE(T)*spreadFactor));	
			cudaAssert(cudaMemcpy(GPU_data,dummy,ARRAY_SIZE(T)*previousSpreadFactor,cudaMemcpyDeviceToDevice));
			BMMS::freeBMMS(dummy);
		}	
	}

	/*** GPU memory allocation rest of the arrays ***/
	BMMS::mallocBMMS((void**)&GPU_results,ARRAY_SIZE(T)*spreadFactor);
	BMMS::mallocBMMS((void**)&state.inputs.GPU_extendedParameters,sizeof(int64_t)*MAX_INPUT_EXTENDED_PARAMETERS);
	BMMS::mallocBMMS((void**)&state.GPU_aux,ARRAY_SIZE(T)*spreadFactor);  //Auxiliary array
	BMMS::mallocBMMS((void**)&state.GPU_auxBlocks,2*spreadFactor*ARRAY_SIZE(int64_t)/ANALYSIS_TPB); //Aux blocks
	BMMS::mallocBMMS((void**)&state.GPU_codeRequiresWLR,ARRAY_SIZE(uint32_t)/ANALYSIS_TPB); //Op Code Exec Flags

	/*** MEMSET 0 GPU arrays ***/
	cudaAssert(cudaMemset(GPU_results,0,ARRAY_SIZE(T)*spreadFactor));
	cudaAssert(cudaMemset(state.GPU_aux,0,ARRAY_SIZE(T)*spreadFactor)); 
	cudaAssert(cudaMemset(state.GPU_auxBlocks,0,2*spreadFactor*ARRAY_SIZE(int64_t)/ANALYSIS_TPB));
	cudaAssert(cudaMemset(state.GPU_codeRequiresWLR,0,ARRAY_SIZE(uint32_t)/ANALYSIS_TPB));
	cudaAssert(cudaThreadSynchronize());

	
	/*** Set (current) windowEndTime ***/
	if(packetBuffer != NULL) //Skip NULL buffer
		state.windowState.windowEndTime= packetBuffer->getPacket(packetBuffer->getNumOfPackets()-1)->timestamp;

	/*** Check if window limit has been reached & set flag and modify state***/
	if(packetBuffer == NULL){
	
		state.windowState.blocksPreviouslyMined = state.windowState.totalNumberOfBlocks;
		state.windowState.totalNumberOfBlocks = state.windowState.blocksPreviouslyMined;

		//Set WLR flag 
		state.windowState.hasReachedWindowLimit = true;
	}else if(packetBuffer->getFlushFlag()){
		
		/*** Adding new elements info to analysis state ***/
		state.windowState.blocksPreviouslyMined = state.windowState.totalNumberOfBlocks;
		state.windowState.totalNumberOfBlocks = state.windowState.blocksPreviouslyMined + BUFFER_BLOCKS;
		state.lastPacket = packetBuffer->getNumOfPackets()+state.windowState.blocksPreviouslyMined*ANALYSIS_TPB; 

		//Set WLR flag 
		state.windowState.hasReachedWindowLimit = true;
	}else{
	
		/*** Adding new elements info to analysis state ***/
		state.windowState.blocksPreviouslyMined = state.windowState.totalNumberOfBlocks;
		state.windowState.totalNumberOfBlocks = state.windowState.blocksPreviouslyMined + BUFFER_BLOCKS;
		state.lastPacket = packetBuffer->getNumOfPackets()+state.windowState.blocksPreviouslyMined*ANALYSIS_TPB; 


		//Set WLR flag 
		#if WINDOW_TYPE == PACKET_WINDOW 
			state.windowState.hasReachedWindowLimit = hasReachedPacketLimitWindow(state.windowState.totalNumberOfBlocks*ANALYSIS_TPB, WINDOW_LIMIT);
		#elif WINDOW_TYPE == TIME_WINDOW
			state.windowState.hasReachedWindowLimit = hasReachedTimeLimitWindow(start,packetBuffer->getPacket(packetBuffer->getNumOfPackets()-1)->timestamp,WINDOW_LIMIT);
	
		#endif	
	}
	
	/*---------------- KERNEL LAUNCHING ----------------*/

	/*** Calculate KERNEL DIMS ***/
	dim3 block(ANALYSIS_TPB);
	dim3 grid;		

	if(state.windowState.totalNumberOfBlocks*ANALYSIS_TPB > CUDA_MAX_THREADS ){ 
	
		//Exceeds limitation of cuda maximum thread number (65536 currently)
		if((CUDA_MAX_THREADS/ANALYSIS_TPB) <= CUDA_MAX_BLOCKS_PER_DIM){
			grid.x = (CUDA_MAX_THREADS/ANALYSIS_TPB);
			DEBUG("--> #thread limitation");
		} else{
			DEBUG("--> #thread and #block limitation");
			grid.x = CUDA_MAX_BLOCKS_PER_DIM;
		}
	
	}else if( state.windowState.totalNumberOfBlocks > CUDA_MAX_BLOCKS_PER_DIM){
		//Exceeds limitation of cuda "maximum blocks per dimension"
		grid.x = CUDA_MAX_BLOCKS_PER_DIM;
		DEBUG("--> #blocks limitation");
	}else{
		grid.x = state.windowState.totalNumberOfBlocks;
	}

	/*** KERNEL CALLS ***/
	//Debug
	DEBUG(STR(ANALYSIS_NAME)"> Throwing Kernel in a WINDOWED analysis, with default implementation.");
	DEBUG(STR(ANALYSIS_NAME)"> Parameters -> gridDim:%d, Total number of blocks:%d, Blocks already mined: %d, Has reached window limit: %d, Last packet index: %d",grid.x,state.windowState.totalNumberOfBlocks,state.windowState.blocksPreviouslyMined,state.windowState.hasReachedWindowLimit,state.lastPacket);
	COMPOUND_NAME(ANALYSIS_NAME,KernelAnalysis)<<<grid,block>>>(GPU_buffer,GPU_data,GPU_results,state);
	cudaAssert(cudaThreadSynchronize());

	/*EXTRA KERNEL CALLS */
	
	/*Predefined Analysis Extra Kernels calls*/
	#define ITERATOR__ 0
	#include "PredefinedExtraKernelCall.def"
	
	#define ITERATOR__ 1
	#include "PredefinedExtraKernelCall.def"

	#define ITERATOR__ 2
	#include "PredefinedExtraKernelCall.def"

	#define ITERATOR__ 3
	#include "PredefinedExtraKernelCall.def"

	#define ITERATOR__ 4
	#include "PredefinedExtraKernelCall.def"

	#define ITERATOR__ 5
	#include "PredefinedExtraKernelCall.def"

	#define ITERATOR__ 6
	#include "PredefinedExtraKernelCall.def"

	#define ITERATOR__ 7
	#include "PredefinedExtraKernelCall.def"

	#define ITERATOR__ 8
	#include "PredefinedExtraKernelCall.def"

	#define ITERATOR__ 9

	#define ITERATOR__ 8
	#include "PredefinedExtraKernelCall.def"

	#define ITERATOR__ 9
	#include "PredefinedExtraKernelCall.def"

	#define ITERATOR__ 10
	#include "PredefinedExtraKernelCall.def"

	#define ITERATOR__ 11
	#include "PredefinedExtraKernelCall.def"

	#define ITERATOR__ 12
	#include "PredefinedExtraKernelCall.def"

	#define ITERATOR__ 13
	#include "PredefinedExtraKernelCall.def"

	#define ITERATOR__ 14
	#include "PredefinedExtraKernelCall.def"

	#define ITERATOR__ 15
	#include "PredefinedExtraKernelCall.def"


	/*Userdefined Extra Kernels calls*/
	#define ITERATOR__ 0
	#include "UserExtraKernelCall.def"
	
	#define ITERATOR__ 1
	#include "UserExtraKernelCall.def"

	#define ITERATOR__ 2
	#include "UserExtraKernelCall.def"

	#define ITERATOR__ 3
	#include "UserExtraKernelCall.def"

	#define ITERATOR__ 4
	#include "UserExtraKernelCall.def"

	#define ITERATOR__ 5
	#include "UserExtraKernelCall.def"

	#define ITERATOR__ 6
	#include "UserExtraKernelCall.def"

	#define ITERATOR__ 7
	#include "UserExtraKernelCall.def"

	#define ITERATOR__ 8
	#include "UserExtraKernelCall.def"

	#define ITERATOR__ 9
	#include "UserExtraKernelCall.def"

	#define ITERATOR__ 10
	#include "UserExtraKernelCall.def"

	#define ITERATOR__ 11
	#include "UserExtraKernelCall.def"

	#define ITERATOR__ 12
	#include "UserExtraKernelCall.def"

	#define ITERATOR__ 13
	#include "UserExtraKernelCall.def"

	#define ITERATOR__ 14
	#include "UserExtraKernelCall.def"

	#define ITERATOR__ 15
	#include "UserExtraKernelCall.def"


	/*** END OF EXTRA KERNEL CALLS ***/


	/*** FREE GPU DYNAMIC MEMORY ***/
	BMMS::freeBMMS(GPU_data);
	BMMS::freeBMMS(state.GPU_aux);
	GPU_data = GPU_results;
	BMMS::freeBMMS(state.inputs.GPU_extendedParameters);
	BMMS::freeBMMS(state.GPU_codeRequiresWLR);

	if(state.windowState.hasReachedWindowLimit){
		R *results;
		int64_t *auxBlocks;
		//results = (R*)malloc(ARRAY_SIZE(R)*spreadFactor);
        	cudaAssert(cudaHostAlloc((void**)&results,ARRAY_SIZE(R)*spreadFactor,0));
		//auxBlocks = (int64_t*)malloc(ARRAY_SIZE(int64_t)*spreadFactor/ANALYSIS_TPB);
        	cudaAssert(cudaHostAlloc((void**)&auxBlocks,ARRAY_SIZE(uint64_t)*spreadFactor,0));

		/*** Copy results & auxBlocks arrays ***/
		cudaAssert(cudaMemcpy(results,GPU_results,ARRAY_SIZE(T)*spreadFactor,cudaMemcpyDeviceToHost));
		cudaAssert(cudaMemcpy(auxBlocks,state.GPU_auxBlocks,ARRAY_SIZE(int64_t)*spreadFactor/ANALYSIS_TPB,cudaMemcpyDeviceToHost));
		cudaAssert(cudaThreadSynchronize());
		
		/*** LAUNCH HOOK (Host function) ***/		
		COMPOUND_NAME(ANALYSIS_NAME,hooks)(packetBuffer, results, state,auxBlocks);

		//Frees host results arrays
		//free(results);
		cudaAssert(cudaFreeHost(results));
		//free(auxBlocks);
		cudaAssert(cudaFreeHost(auxBlocks));

		//Frees GPU_data (last results) and points GPU_data static var to NULL to reallocate
		BMMS::freeBMMS(GPU_data);
		GPU_data = NULL;
	}
	BMMS::freeBMMS(state.GPU_auxBlocks);	


}




#else //#if HAS_WINDOW == 1

//default Kernel 
template<typename T,typename R>
	__global__ void COMPOUND_NAME(ANALYSIS_NAME,KernelAnalysis)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results, analysisState_t state){
	state.blockIterator = blockIdx.x;
	COMPOUND_NAME(ANALYSIS_NAME,mining)(GPU_buffer, GPU_data, GPU_results, state);
	__syncthreads();	

	state.blockIterator = blockIdx.x;
	COMPOUND_NAME(ANALYSIS_NAME,filtering)(GPU_buffer, GPU_data, GPU_results, state);
	__syncthreads();	

	/* Analysis implementation*/
	COMPOUND_NAME(ANALYSIS_NAME,analysis)(GPU_buffer, GPU_data, GPU_results, state);

	/* If there are SYNCBLOCKS barriers do not put Operations function call here */
	#if __SYNCBLOCKS_COUNTER == 0 && __SYNCBLOCKS_PRECODED_COUNTER == 0
		COMPOUND_NAME(ANALYSIS_NAME,operations)(GPU_buffer, GPU_data, GPU_results, state);
	#endif

}


/**** Launch wrapper ****/
//default Launch Wrapper for Analysis not using Windows 

template<typename T,typename R>
void COMPOUND_NAME(ANALYSIS_NAME,launchAnalysis_wrapper)(PacketBuffer* packetBuffer, packet_t* GPU_buffer){

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
        	cudaAssert(cudaHostAlloc((void**)&auxBlocks,sizeof(int64_t)*MAX_BUFFER_PACKETS/ANALYSIS_TPB,0));

		/*** GPU memory allocation***/
		BMMS::mallocBMMS((void**)&GPU_data,ARRAY_SIZE(T));
		BMMS::mallocBMMS((void**)&GPU_results,ARRAY_SIZE(R));
		BMMS::mallocBMMS((void**)&state.GPU_aux,ARRAY_SIZE(T));  //Auxiliary array
		BMMS::mallocBMMS((void**)&state.GPU_auxBlocks,2*sizeof(int64_t)*MAX_BUFFER_PACKETS/ANALYSIS_TPB);
		BMMS::mallocBMMS((void**)&state.inputs.GPU_extendedParameters,sizeof(int64_t)*MAX_INPUT_EXTENDED_PARAMETERS);
		BMMS::mallocBMMS((void**)&state.GPU_codeRequiresWLR,ARRAY_SIZE(uint32_t)/ANALYSIS_TPB); //Op Code Exec Flags

		/*** MEMSET 0 GPU arrays ***/
		cudaAssert(cudaMemset(GPU_data,0,ARRAY_SIZE(T)));	
		cudaAssert(cudaMemset(GPU_results,0,ARRAY_SIZE(R)));	
		cudaAssert(cudaMemset(state.GPU_aux,0,ARRAY_SIZE(T)));	
		cudaAssert(cudaMemset(state.GPU_auxBlocks,0,2*sizeof(int64_t)*MAX_BUFFER_PACKETS/ANALYSIS_TPB));	
		cudaAssert(cudaMemset(state.GPU_codeRequiresWLR,0,ARRAY_SIZE(uint32_t)/ANALYSIS_TPB));
		cudaAssert(cudaThreadSynchronize());
		
		/*** KERNEL DIMS ***/
		//dim3 block(ANALYSIS_TPB);		 			//Threads Per Block (1D)
		//dim3 grid(MAX_BUFFER_PACKETS/ANALYSIS_TPB);		 	//Grid size (1D)
		//dim3  block(10);
		//dim3 grid(1);
		dim3 block(96);
		dim3 grid(237);
		//Set state number of blocks and last Packet position
		state.windowState.totalNumberOfBlocks = MAX_BUFFER_PACKETS;
		state.windowState.hasReachedWindowLimit = true;
		state.lastPacket = packetBuffer->getNumOfPackets(); 
		state.windowState.windowStartTime= packetBuffer->getPacket(0)->timestamp;
		state.windowState.windowEndTime= packetBuffer->getPacket(packetBuffer->getNumOfPackets()-1)->timestamp;

		DEBUG(STR(ANALYSIS_NAME)"> Throwing Kernel with default implementation.");
		DEBUG(STR(ANALYSIS_NAME)"> Parameters -> gridDim:%d",grid.x);
	
		/*** KERNEL CALLS ***/
		COMPOUND_NAME(ANALYSIS_NAME,KernelAnalysis)<<<grid,block>>>(GPU_buffer,GPU_data,GPU_results,state);
		cudaAssert(cudaThreadSynchronize());

		/*EXTRA KERNEL CALLS */
	
		/*Predefined Analysis Extra Kernels calls*/
		#define ITERATOR__ 0
		#include "PredefinedExtraKernelCall.def"
	
		#define ITERATOR__ 1
		#include "PredefinedExtraKernelCall.def"

		#define ITERATOR__ 2
		#include "PredefinedExtraKernelCall.def"

		#define ITERATOR__ 3
		#include "PredefinedExtraKernelCall.def"

		#define ITERATOR__ 4
		#include "PredefinedExtraKernelCall.def"

		#define ITERATOR__ 5
		#include "PredefinedExtraKernelCall.def"

		#define ITERATOR__ 6
		#include "PredefinedExtraKernelCall.def"

		#define ITERATOR__ 7
		#include "PredefinedExtraKernelCall.def"

		#define ITERATOR__ 8
		#include "PredefinedExtraKernelCall.def"

		#define ITERATOR__ 9
		#include "PredefinedExtraKernelCall.def"

		#define ITERATOR__ 10
		#include "PredefinedExtraKernelCall.def"

		#define ITERATOR__ 11
		#include "PredefinedExtraKernelCall.def"

		#define ITERATOR__ 12
		#include "PredefinedExtraKernelCall.def"

		#define ITERATOR__ 13
		#include "PredefinedExtraKernelCall.def"

		#define ITERATOR__ 14
		#include "PredefinedExtraKernelCall.def"

		#define ITERATOR__ 15
		#include "PredefinedExtraKernelCall.def"


		/*Userdefined Extra Kernels calls*/
		#define ITERATOR__ 0
		#include "UserExtraKernelCall.def"
	
		#define ITERATOR__ 1
		#include "UserExtraKernelCall.def"

		#define ITERATOR__ 2
		#include "UserExtraKernelCall.def"

		#define ITERATOR__ 3
		#include "UserExtraKernelCall.def"

		#define ITERATOR__ 4
		#include "UserExtraKernelCall.def"

		#define ITERATOR__ 5
		#include "UserExtraKernelCall.def"

		#define ITERATOR__ 6
		#include "UserExtraKernelCall.def"

		#define ITERATOR__ 7
		#include "UserExtraKernelCall.def"

		#define ITERATOR__ 8
		#include "UserExtraKernelCall.def"

		#define ITERATOR__ 9
		#include "UserExtraKernelCall.def"

		#define ITERATOR__ 10
		#include "UserExtraKernelCall.def"

		#define ITERATOR__ 11
		#include "UserExtraKernelCall.def"

		#define ITERATOR__ 12
		#include "UserExtraKernelCall.def"

		#define ITERATOR__ 13
		#include "UserExtraKernelCall.def"

		#define ITERATOR__ 14
		#include "UserExtraKernelCall.def"

		#define ITERATOR__ 15
		#include "UserExtraKernelCall.def"

		/*** END OF EXTRA KERNEL CALLS ***/

		/*** Copy results & auxBlocks arrays ***/
		cudaAssert(cudaMemcpy(results,GPU_results,MAX_BUFFER_PACKETS*sizeof(R),cudaMemcpyDeviceToHost));
		cudaAssert(cudaMemcpy(auxBlocks,state.GPU_auxBlocks,MAX_BUFFER_PACKETS/ANALYSIS_TPB*sizeof(int64_t),cudaMemcpyDeviceToHost));
		cudaAssert(cudaThreadSynchronize());

		/*** FREE GPU DYNAMIC MEMORY ***/
		BMMS::freeBMMS(GPU_data);
		BMMS::freeBMMS(GPU_results);
		BMMS::freeBMMS(state.GPU_aux);
		BMMS::freeBMMS(state.GPU_auxBlocks);
		BMMS::freeBMMS(state.inputs.GPU_extendedParameters);
		BMMS::freeBMMS(state.GPU_codeRequiresWLR);
	
		/*** LAUNCH HOOK (Host function) ***/
	
		//Launch hook (or preHook if window is set)
		COMPOUND_NAME(ANALYSIS_NAME,hooks)(packetBuffer, results, state,auxBlocks);
		
		//Frees results
		cudaAssert(cudaFreeHost(results));
		//free(results);
	}
}

#endif //#if HAS_WINDOW == 1

#endif // __CUDACC__


#endif // AnalysisSkeleton_h





