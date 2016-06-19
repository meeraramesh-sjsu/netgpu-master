#include "Advanced.h" //Include your modified(from Template) .h

/*
	Example of how to program CUDA code directly using SYNCBLOCKS() MACRO. 
	No networking related code is exposed, just SYNCBLOCKS() example

	If you desire, check cpp to see MACRO expansions output 
	by doing (after at least 1 compilation; ppph files must exist):
	
	cpp -D __CUDACC__ -I . advanced.cu > cppOutput
	
	Search for kernel: AdvancedExample_KernelAnalysis_0, and it's device function
*/

template<typename T,typename R>
__device__ void COMPOUND_NAME(ANALYSIS_NAME,mining)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results,analysisState_t state){
	
	//DATA_ELEMENT in this case => GPU_DATA[POS]
	DATA_ELEMENT = state.blockIterator; //In this case, as we don't use windows is == blockIdx.x

}

template<typename T,typename R>
__device__ void COMPOUND_NAME(ANALYSIS_NAME,filtering)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results,analysisState_t state){
	
	
}

template<typename T,typename R>
__device__ void COMPOUND_NAME(ANALYSIS_NAME,analysis)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results,analysisState_t state){

	/*
		We will SYNCHRONIZE all threads to avoid race conditions in GPU_data access and move
		DATA_ELEMENT to the next block (circular move)
	*/
	
	//Total synchronization
	SYNCBLOCKS();

	//Declaring variables	
	int nextBlock;

	/*
	  	Loop for large windows (in this case it won't be necessary, as we do not use windows, but
	  	is better to ALWAYS implement it) 
	*/

	//Setting initial state.blockIterator
	state.blockIterator = blockIdx.x;

	while( state.blockIterator < state.windowState.totalNumberOfBlocks ){
		
		//Calculate nextBlock (circular)
		nextBlock = ( state.blockIterator == (state.windowState.totalNumberOfBlocks-1))?0:state.blockIterator+1;

		//Stupid move. Note that we can do that because blocks are Synchronized
		//If SYNCBLOCKS was not called => Race condition in GPU_data access.

		GPU_results[state.blockIterator*blockDim.x+threadIdx.x] = GPU_data[nextBlock*blockDim.x+threadIdx.x]; 		

	
		//Add gridDim.x to next iteration
		state.blockIterator += gridDim.x;	
	}
}

template<typename T,typename R>
__device__ void COMPOUND_NAME(ANALYSIS_NAME,operations)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results,analysisState_t state){
	
	/* NEVER CALLED, as not explicitily called in analysis section!!! */
}

template<typename R>
void COMPOUND_NAME(ANALYSIS_NAME,hooks)(PacketBuffer* packetBuffer,R* results, analysisState_t state, int64_t* auxBlocks){
	
        /* HOST CODE: Implement HOOKS code here. Use predefined hooks or define new ones. */

	//Printing results        

	/* Uncomment this to see output (very verbose)*/
#if 0 
        int i,j;
        fprintf(stderr, "Example(class name: %s)\n",STR(ANALYSIS_NAME));

        for(i=0;i<state.windowState.totalNumberOfBlocks;i++){
		for(j=0;j<ANALYSIS_TPB;j++)
			fprintf(stderr,"%d -> Block: %d Element:%d Value: %d\n",i*ANALYSIS_TPB+j,i,j,results[i*ANALYSIS_TPB+j]);		
	}	
#endif
}


