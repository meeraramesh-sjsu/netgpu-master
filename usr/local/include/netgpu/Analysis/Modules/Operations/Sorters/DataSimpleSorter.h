/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#warning Data SimpleSorter loaded

#ifdef __CUDACC__

#ifndef CALLNUMBER
	//Using predefined Kernel Counter to create unique identifiers for each
	// call to simpleSorter (for multiple call support)
	#define CALLNUMBER _PRECODED_COUNTER
#endif


//CALL MACRO 
#define SSORT_DATA(...)\
	IF_OPERATION_HAS_PERMISSION() \
		COMPOUND_NAME(simpleSortData,CALLNUMBER)(GPU_data, state,##__VA_ARGS__)

//Sorter __device__ function
template<typename T>
__device__ __inline__ void COMPOUND_NAME(simpleSortData,CALLNUMBER)(T* GPU_data, analysisState_t state, int64_t* GPU_auxBlocks=NULL)
{
	SET_EXTENDED_PARAMETER(4,(GPU_auxBlocks - (int64_t*)NULL))

	//Total Synchronization
	#include "../../PrecodedSyncblocks.def"

	IF_OPERATION_HAS_PERMISSION(){

		unsigned int i,blockIndex,pos;
		int cmpVal;

		__shared__ T blockItems[ANALYSIS_TPB];
		__shared__ T myValues[ANALYSIS_TPB];
		__shared__ T nullElement;
		__shared__ int64_t* GPU_auxBlocks;

		//Memset nullElement
		if(threadIdx.x ==0){
			cudaSharedMemset(&nullElement,0);
			GPU_auxBlocks = GET_EXTENDED_PARAMETER(4)+(int64_t*)NULL;
		}
		SYNCTHREADS();		

		state.blockIterator = blockIdx.x;

		while( state.blockIterator < state.windowState.totalNumberOfBlocks ){

			//Set start 
			blockIndex = state.blockIterator; 
	
			if(GPU_auxBlocks != NULL){
				if(threadIdx.x<GPU_auxBlocks[state.blockIterator])
					myValues[threadIdx.x] = GPU_data[threadIdx.x+blockIdx.x*blockDim.x];
				else
					myValue = nullElement;
			}else
				myValues[threadIdx.x] = GPU_data[threadIdx.x+blockIdx.x*blockDim.x];	

			//Mine first block
			blockItems[threadIdx.x] = myValues[threadIdx.x];
			SYNCTHREADS();		

			//Set 0 pos
			pos =0;
		
			//Loop for all blocks	
			do{	
				//Mine next block
				if(blockIdx.x != blockIndex ){
					if(GPU_auxBlocks == NULL){
						blockItems[threadIdx.x] = GPU_data[threadIdx.x+blockIndex*blockDim.x];
					}else{
						if(threadIdx.x < GPU_auxBlocks[blockIndex])
							blockItems[threadIdx.x] = GPU_data[threadIdx.x+blockIndex*blockDim.x]; 
						else
							blockItems[threadIdx.x] = nullElement; 
					}
				}
				SYNCTHREADS();

				//Count position elements below 
				for(i=0;i<ANALYSIS_TPB;i++){
					cmpVal = cudaSharedMemcmp(&blockItems[i],&myValues[threadIdx.x]);
					if(cmpVal<0 || (cmpVal==0 && (threadIdx.x+state.blockIterator*blockDim.x > i+blockIndex*blockDim.x)))
						pos++;
	
	
				}	
				SYNCTHREADS();

				//Next block index (circular)
				blockIndex++;
				if(blockIndex == state.windowState.totalNumberOfBlocks)
					blockIndex = 0;
	
			}while(blockIndex != state.blockIterator);

			//Fill array
			if(cudaSharedIsNotNull(&myValues[threadIdx.x])){
				state.GPU_aux[pos] = myValues[threadIdx.x];
			}
			SYNCTHREADS();

			//Loop iterator
			state.blockIterator += gridDim.x;
		}
	}//Operation has permission

	//TotalSynchronization
	#include "../../PrecodedSyncblocks.def"

	IF_OPERATION_HAS_PERMISSION(){
		state.blockIterator = blockIdx.x;

		while( state.blockIterator < state.windowState.totalNumberOfBlocks ){

			GPU_data[threadIdx.x+blockDim.x*state.blockIterator] = state.GPU_aux[threadIdx.x+blockDim.x*state.blockIterator];
			//Loop iterator
			state.blockIterator += gridDim.x;
		}
	} //Operation has permission
	SYNCTHREADS();
}

#endif //ifdef CUDACC


