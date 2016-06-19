/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#include "Thresholds.h"
#warning "Rate analysis"

/*CHECK OF DATA_ELEMENT REDEFINITION AT THE END OF THE FILE */

#ifdef __CUDACC__


/***** RATE THRESHOLD ANALYSIS *****/
#define RATETHRESHOLD_ANALYSIS(a) \
	COMPOUND_NAME(ANALYSIS_NAME,preDefinedAnalysisCodeRateThreshold)(GPU_buffer,GPU_data,GPU_results,state,a);\
	__syncthreads()

template<typename T,typename R> __device__ __inline__ void COMPOUND_NAME(ANALYSIS_NAME,preDefinedAnalysisCodeRateThreshold)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results, analysisState_t state,int thresHold){

	/*Count elements inside block */
	int i,counter,futurePosition;

	__shared__ T elements[ANALYSIS_TPB];
	__shared__ bool leaders[ANALYSIS_TPB];
		
	state.blockIterator = blockIdx.x;

	//Looping when array is larger than grid dimensions 
	while( state.blockIterator < state.windowState.totalNumberOfBlocks ){
	
		//Mine
		elements[threadIdx.x] = cudaSafeGet(&GPU_data[state.blockIterator*blockDim.x+threadIdx.x]); 

		if(cudaSharedIsNotNull(&elements[threadIdx.x].user)) 
			leaders[threadIdx.x] = true;
		else
			leaders[threadIdx.x] = false;
		
		SYNCTHREADS();


		T myValue = elements[threadIdx.x];
		
		if(elements[threadIdx.x].counter  == 0)
			elements[threadIdx.x].counter = 1;
		
		/* Count equal elements*/
		for(i=threadIdx.x,counter = 0;i<blockDim.x;i++){
			if(cudaSharedMemcmp(&elements[i].user,&elements[threadIdx.x].user)==0)
					counter+=elements[i].counter;
			
		}	

		/* Determine if is Leader*/
		if(leaders[threadIdx.x]){
			for(i=(threadIdx.x-1);i>=0;i--){
				if(cudaSharedMemcmp(&elements[i].user,&elements[threadIdx.x].user) == 0){ 
					leaders[threadIdx.x] =  false;
					break;
				}	
			}
		}
		SYNCTHREADS();

		/* Determine future position inside block*/
		if(leaders[threadIdx.x]){

			for(i=(threadIdx.x-1),futurePosition=0;i>=0;i--){
				if(leaders[i])
					futurePosition++;	
			}		

		}


		/* Save partial result */	
		if(leaders[threadIdx.x]){
			myValue.counter = counter;
			GPU_data[futurePosition+state.blockIterator*blockDim.x] = myValue;
		}

		//Wait for all blocks to perform calc.
		counter=0;
		if(threadIdx.x == 0){
			//Calculate number of elements en each block and place it in Position nº 0 of each block on results vector
			for(int i=0;i<blockDim.x;i++){
				if(leaders[i])
					counter++;				
			}
			state.outputs.GPU_auxBlocks[state.windowState.totalNumberOfBlocks + state.blockIterator] = counter;
			
			if(state.blockIterator == 0)
				state.inputs.GPU_extendedParameters[0] = thresHold;
		}
		SYNCTHREADS();

		//Iterate
		state.blockIterator += gridDim.x;
	}


	/*** Sync ALL BLOCKS ***/
	SYNCBLOCKS_PRECODED();
	#include "../../../IncCounter.def"

	int counter, i,blockIndex, futurePosition;	
	blockIndex = blockIdx.x;

	T myValue;	
	__shared__ bool leaders[ANALYSIS_TPB];
	__shared__ T blockElements[ANALYSIS_TPB];
	__shared__ int numOfElementsPerBlock[MAX_BUFFER_PACKETS/ANALYSIS_TPB]; 
	__shared__ int threshold;
	__shared__ int step;
	__shared__ struct timeval windowStartTime;

	state.blockIterator = blockIdx.x;

	if(threadIdx.x == (ANALYSIS_TPB-1)){
		threshold = state.inputs.GPU_extendedParameters[0]; 
	}
	if(threadIdx.x == (ANALYSIS_TPB-2)){
		if(state.windowState.windowStartTime)
			windowStartTime = *state.windowState.windowStartTime;
		else
			//NO WINDOWED ANALYSIS
			windowStartTime = GPU_buffer[0].timestamp; 
				
	}
	
	//Looping when array is larger than grid dimensions 
	while( state.blockIterator < state.windowState.totalNumberOfBlocks ){
	
		step = state.blockIterator/(MAX_BUFFER_PACKETS/ANALYSIS_TPB)+1;
		//Mine my block items, num of elements

		if(threadIdx.x == 0){
			numOfElementsPerBlock[0] = state.outputs.GPU_auxBlocks[state.windowState.totalNumberOfBlocks+state.blockIterator];
		}
		SYNCTHREADS();

		if(threadIdx.x < numOfElementsPerBlock[0]){ //Save as many global memory accesses as possible
			blockElements[threadIdx.x] = cudaSafeGet(&GPU_data[state.blockIterator*blockDim.x+threadIdx.x]);
		}else{
			cudaSharedMemset(&blockElements[threadIdx.x],0);	
		}

		//Set leader flags appropiate to content of myValue	
		if(cudaSharedIsNotNull(&blockElements[threadIdx.x].user))
			leaders[threadIdx.x] = true;
		else
			leaders[threadIdx.x] = false;
	
		//Set myValue and counters
		myValue = blockElements[threadIdx.x];	
		counter = blockElements[threadIdx.x].counter;	

		blockIndex = ((state.blockIterator-1)<0)?state.windowState.totalNumberOfBlocks-1:state.blockIterator-1;

		//Loop for all blocks
		while(blockIndex != state.blockIterator){
			
			//Do not erase this	
			SYNCTHREADS();

			if(blockIndex == state.blockIterator)
				continue;
			
			bool hasLeaders = false;
			for(i=0;i<blockDim.x;i++){
				if(leaders[i]){
					hasLeaders = true;
					break;
				}
			} 
		
			if(!hasLeaders)
				break;

			//auxiliary var to use on circular decrement/increment
			int auxStep;			

			//Mine to numOfElements 
			if(!(blockIndex >= (step*(MAX_BUFFER_PACKETS/ANALYSIS_TPB)) && blockIndex <((step+1)*(MAX_BUFFER_PACKETS/ANALYSIS_TPB)))){
				auxStep = 	((step-1)<0)? state.windowState.totalNumberOfBlocks/(MAX_BUFFER_PACKETS/ANALYSIS_TPB)-1:step -1;
				if(threadIdx.x < (MAX_BUFFER_PACKETS/ANALYSIS_TPB)){
					numOfElementsPerBlock[threadIdx.x] = state.outputs.GPU_auxBlocks[state.windowState.totalNumberOfBlocks+(auxStep*(MAX_BUFFER_PACKETS/ANALYSIS_TPB))+threadIdx.x];
				}else{
					cudaSharedMemset(&blockElements[threadIdx.x],0);	
				}
			}	
			SYNCTHREADS();

			if(blockIndex >= (step*(MAX_BUFFER_PACKETS/ANALYSIS_TPB)) && threadIdx.x == 0){
				step = 	((step-1)<0)? state.windowState.totalNumberOfBlocks/(MAX_BUFFER_PACKETS/ANALYSIS_TPB)-1:step -1;
			}
			SYNCTHREADS();

			auxStep = 	((step+1)>state.windowState.totalNumberOfBlocks/(MAX_BUFFER_PACKETS/ANALYSIS_TPB))? 0:step+1;

			//Mine blockElements 
			if(threadIdx.x < numOfElementsPerBlock[blockIndex - ((auxStep)*(MAX_BUFFER_PACKETS/ANALYSIS_TPB))]){
				blockElements[threadIdx.x] = cudaSafeGet(&GPU_data[blockIndex*blockDim.x+threadIdx.x]);
			}else{
				cudaSharedMemset(&blockElements[threadIdx.x],0);	
			}
			SYNCTHREADS();
			

			if(leaders[threadIdx.x]){

			for(i=0;i<numOfElementsPerBlock[blockIndex - ((auxStep)*(MAX_BUFFER_PACKETS/ANALYSIS_TPB))];i++){

					if(cudaSharedMemcmp(&myValue.user,&blockElements[i].user) == 0){ //WARN: COMAPRING NOT SHARED ELEMENT
					
						if(blockIndex> state.blockIterator)
							counter += blockElements[i].counter;
						else
							leaders[threadIdx.x] = false;
					}	
				}
	
			}

			blockIndex = ((blockIndex-1)<0)?state.windowState.totalNumberOfBlocks-1:blockIndex-1;
		}


		//Calculate rate
		float rate;

		if(threshold>0){
			if(leaders[threadIdx.x]){
				long time_ms= cudaTimevaldiff(windowStartTime,GPU_buffer[(state.lastPacket-(state.windowState.blocksPreviouslyMined*ANALYSIS_TPB))-1].timestamp);
				rate = (float)((float)counter /(float)time_ms);

				if(rate*1000 < threshold)
					leaders[threadIdx.x] = false;
		
			}
		}
		SYNCTHREADS(); 

		/* Determine future position inside block*/
		if(leaders[threadIdx.x]){
			for(i=(threadIdx.x-1),futurePosition=0;i>=0;i--){
				if(leaders[i])
					futurePosition++;	
			}		
	
		}

		if(leaders[threadIdx.x]){
				myValue.counter = counter;	
				myValue.rate = rate*1000;
				GPU_results[state.blockIterator*blockDim.x+futurePosition]= myValue;
		}

		if(threadIdx.x == 0){
			for(i=0,counter=0;i<blockDim.x;i++){
				if(leaders[i])
					counter++;
			}
			state.outputs.GPU_auxBlocks[state.blockIterator] = counter;

		
		}	

		//Iterate
		state.blockIterator += gridDim.x;
	}




	#if NUM_OF_USER_SYNBLOCKS == 0
	COMPOUND_NAME(ANALYSIS_NAME,postAnalysisOperationsImplementation)(GPU_buffer, GPU_data, GPU_results,state);
	#endif 

}


/***** END OF RATETHRESHOLD ANALYSIS *****/


#endif //__CUDACC__

/* Redefine DATA_ELEMENT */
#undef DATA_ELEMENT
#define DATA_ELEMENT GPU_data[POS].user 


