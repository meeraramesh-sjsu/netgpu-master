/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#warning Results SimpleSorter loaded

#ifdef __CUDACC__

//Uses GPU_auxBlocks array to minimize number of global memory accesses.
// (precoded analysis fille this array with the number of items en each block)
#define OPTIMIZE_USING_GPU_AUXBLOCKS 1

//CALL MACRO 
#define SSORT_RESULTS()\
	COMPOUND_NAME(simpleSortResults,0)<OPTIMIZE_USING_GPU_AUXBLOCKS>(GPU_results,state)

#define SSORT_RESULTS_NO_OPTIMIZE()\
	COMPOUND_NAME(simpleSortResults,0)<0>(GPU_results,state)

//Dummy function, will be erased at compilation time
//do not erase
template<int optimize, typename T>
__device__ __inline__ void COMPOUND_NAME(simpleSortResults,0)(T* GPU_results, analysisState_t state)
{
	int i,elements;

	SET_EXTENDED_PARAMETER(4,optimize);
	if(POS == 0){
		state.GPU_auxBlocks[state.windowState.totalNumberOfBlocks+state.blockIterator] =0;
		for(i=0,elements=0;i<state.windowState.totalNumberOfBlocks;i++){
			elements+=state.GPU_auxBlocks[i];
		}
		state.GPU_auxBlocks[state.windowState.totalNumberOfBlocks] = elements;
	}
	//Total Synchronization
	#include "../../PrecodedSyncblocks.def"
		
	int i,blockIndex,pos;
	int cmpVal;

	__shared__ T blockItems[ANALYSIS_TPB];
	__shared__ T myValues[ANALYSIS_TPB];
	__shared__ T nullElement;
	__shared__ int optimize;

	IF_OPERATION_HAS_PERMISSION(){
		//Memset nullElement
		if(threadIdx.x ==0){
			cudaSharedMemset(&nullElement,0);
			optimize = GET_EXTENDED_PARAMETER(4);
		}
		SYNCTHREADS();		

		state.blockIterator = blockIdx.x;

		while( state.blockIterator < state.windowState.totalNumberOfBlocks ){

			//Set start 
			blockIndex = state.blockIterator; 
	
			if(optimize){
				if((threadIdx.x+state.blockIterator*blockDim.x) < state.GPU_auxBlocks[state.windowState.totalNumberOfBlocks])
					myValues[threadIdx.x] = GPU_results[threadIdx.x+blockIdx.x*blockDim.x];
				else
					myValues[threadIdx.x] = nullElement;
			}else
				myValues[threadIdx.x] = GPU_results[threadIdx.x+blockIdx.x*blockDim.x];	

			//Mine first block
			blockItems[threadIdx.x] = myValues[threadIdx.x];
			SYNCTHREADS();		

			//Set 0 pos
			pos =0;
		
			//Loop for all blocks	
			do{	
				//Mine next block
				if(blockIdx.x != blockIndex ){
					if(optimize==0){
						blockItems[threadIdx.x] = GPU_results[threadIdx.x+blockIndex*blockDim.x];
					}else{
						if((threadIdx.x+blockIndex*blockDim.x) < state.GPU_auxBlocks[state.windowState.totalNumberOfBlocks])
							blockItems[threadIdx.x] = GPU_results[threadIdx.x+blockIndex*blockDim.x]; 
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
				((T*)state.GPU_aux)[pos] = myValues[threadIdx.x];
			}
			SYNCTHREADS();

			//Loop iterator
			state.blockIterator += gridDim.x;
		}
	}//Operation has permission

	//TotalSynchronization
	#include "../../PrecodedSyncblocks.def"
	__shared__ int numOfBlockElements;

	if(threadIdx.x==0){	
		numOfBlockElements = state.GPU_auxBlocks[state.windowState.totalNumberOfBlocks];
		state.GPU_auxBlocks[0] = numOfBlockElements;
	}
	SYNCTHREADS();

	IF_OPERATION_HAS_PERMISSION(){
		state.blockIterator = blockIdx.x;

		while( state.blockIterator < state.windowState.totalNumberOfBlocks ){
			
			if((state.blockIterator*blockDim.x+threadIdx.x)<numOfBlockElements)
				GPU_results[threadIdx.x+blockDim.x*state.blockIterator] = ((T*)state.GPU_aux)[threadIdx.x+blockDim.x*state.blockIterator];
			//Loop iterator
			state.blockIterator += gridDim.x;
		}
	} //Operation has permission
	SYNCTHREADS();
}

#endif //ifdef CUDACC


