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
#include<algorithm>
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
__global__ void COMPOUND_NAME(ANALYSIS_NAME,KernelAnalysis)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results,analysisState_t state,int *d_result,char* d_pattern, int* d_stridx, int* d_SHIFT,
		int* d_PREFIX_value, int* d_PREFIX_index, int* d_PREFIX_size,int m,int prefixPitch);

template<typename T,typename R>
__device__  void COMPOUND_NAME(ANALYSIS_NAME,mining)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results, analysisState_t state);

template<typename T,typename R>
__device__  void COMPOUND_NAME(ANALYSIS_NAME,filtering)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results, analysisState_t state);

template<typename T,typename R>
__device__  void COMPOUND_NAME(ANALYSIS_NAME,analysis)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results,analysisState_t state,int *d_result,char* d_pattern, int* d_stridx, int* d_SHIFT,
		int* d_PREFIX_value, int* d_PREFIX_index, int* d_PREFIX_size,int m,int prefixPitch);

template<typename T,typename R>
__device__  void COMPOUND_NAME(ANALYSIS_NAME,operations)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results,analysisState_t state);

template<typename R>
void COMPOUND_NAME(ANALYSIS_NAME,hooks)(PacketBuffer *packetBuffer, R* results, analysisState_t state, int64_t* auxBlocks,int *d_result,std::vector<string> pattern);

/**** Module loader ****/
#include ".dmodule.ppph"

double timeTaken = 0;

unsigned int determine_shiftsize(int alphabet) {

	//the maximum size of the hash value of the B-size suffix of the patterns for the Wu-Manber algorithm
	if (alphabet == 2)
		return 22; // 1 << 2 + 1 << 2 + 1 + 1

	else if (alphabet == 4)
		return 64; // 3 << 2 + 3 << 2 + 3 + 1

	else if (alphabet == 8)
		return 148; // 7 << 2 + 7 << 2 + 7 + 1

	else if (alphabet == 20)
		return 400; // 19 << 2 + 19 << 2 + 19 + 1

	else if (alphabet == 128)
		return 2668; // 127 << 2 + 127 << 2 + 127 + 1

	else if (alphabet == 256)
		return 5356; //304 << 2 + 304 << 2 + 304 + 1

	else if (alphabet == 512)
		return 10732; //560 << 2 + 560 << 2 + 560 + 1

	else if (alphabet == 1024)
		return 21484; //1072 << 2 + 1072 << 2 + 1072 + 1

	else {
		printf("The alphabet size is not supported by wu-manber\n");
		exit(1);
	}
}

void preprocessing(vector<string> pattern,int* SHIFT,int shiftsize,int* PREFIX_value,int* PREFIX_index,int* PREFIX_size,int m) {

	struct timeval startTV, endTV;
	gettimeofday(&startTV, NULL);


	unsigned int j, q, hash;

	size_t shiftlen, prefixhash;
	int p_size = pattern.size();
	int m_nBitsInShift = 2;
	for (j = 0; j < p_size; ++j) {

		//add last 3-character subpattern

		hash = pattern[j][m - 2 - 1]; // bring in offsets of X in pattern j
		hash <<= m_nBitsInShift;
		hash += pattern[j][m - 1 - 1];
		hash <<= m_nBitsInShift;
		hash += pattern[j][m - 1];

		SHIFT[hash] = 0;

		//calculate the hash of the prefixes for each pattern

		prefixhash = pattern[j][0];
		prefixhash <<= m_nBitsInShift;
		prefixhash += pattern[j][1];

		PREFIX_value[hash * p_size + PREFIX_size[hash]] = prefixhash;
		PREFIX_index[hash * p_size + PREFIX_size[hash]] = j;
		PREFIX_size[hash]++;

	}
	gettimeofday(&endTV, NULL);
	timeTaken = endTV.tv_sec * 1e6 + endTV.tv_usec - (startTV.tv_sec * 1e6 + startTV.tv_usec);

}

struct length {
	bool operator() ( const string& a, const string& b )
	{
		return a.size() < b.size();
	}
};


//default Kernel 
template<typename T,typename R>
__global__ void COMPOUND_NAME(ANALYSIS_NAME,KernelAnalysis)(packet_t* GPU_buffer, T* GPU_data, R* GPU_results, analysisState_t state,int *result, char* d_pattern, int* d_stridx, int* d_SHIFT,
		int* d_PREFIX_value, int* d_PREFIX_index, int* d_PREFIX_size,int m,int prefixPitch){
	state.blockIterator = blockIdx.x;
	COMPOUND_NAME(ANALYSIS_NAME,mining)(GPU_buffer, GPU_data, GPU_results, state);
	__syncthreads();	

	state.blockIterator = blockIdx.x;
	COMPOUND_NAME(ANALYSIS_NAME,filtering)(GPU_buffer, GPU_data, GPU_results, state);
	__syncthreads();	

	/* Analysis implementation*/
	COMPOUND_NAME(ANALYSIS_NAME,analysis)(GPU_buffer, GPU_data, GPU_results, state, result, d_pattern, d_stridx, d_SHIFT,
			d_PREFIX_value, d_PREFIX_index, d_PREFIX_size, m, prefixPitch);

	/* If there are SYNCBLOCKS barriers do not put Operations function call here */
#if __SYNCBLOCKS_COUNTER == 0 && __SYNCBLOCKS_PRECODED_COUNTER == 0
	COMPOUND_NAME(ANALYSIS_NAME,operations)(GPU_buffer, GPU_data, GPU_results, state);
#endif

}

/**** Launch wrapper ****/
//default Launch Wrapper for Analysis not using Windows 
template<typename T,typename R>
void COMPOUND_NAME(ANALYSIS_NAME,launchAnalysis_wrapper)(PacketBuffer* packetBuffer, packet_t* GPU_buffer,int noOfPatterns){

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
		dim3 grid(288);
		size_t N = 288;

		//Set state number of blocks and last Packet position
		state.windowState.totalNumberOfBlocks = MAX_BUFFER_PACKETS;
		state.windowState.hasReachedWindowLimit = true;
		state.lastPacket = packetBuffer->getNumOfPackets(); 
		state.windowState.windowStartTime= packetBuffer->getPacket(0)->timestamp;
		state.windowState.windowEndTime= packetBuffer->getPacket(packetBuffer->getNumOfPackets()-1)->timestamp;

		DEBUG(STR(ANALYSIS_NAME)"> Throwing Kernel with default implementation.");
		DEBUG(STR(ANALYSIS_NAME)"> Parameters -> gridDim:%d",grid.x);

		int n = 14;
		int p_size = 3;
		int alphabet = 256;

		
		string fileName = "/home/meera/gpudir/netgpu-master/src/Analysis/Pattern/patterns" + patch::to_string(noOfPatterns) + ".cpp";
		vector<string> tmp;
		string line;
		std::ifstream myfile(fileName.c_str());

		if(!myfile) //Always test the file open.
		{
			std::cout<<"Error opening pattern file"<< std::endl;
			return;
		}
		while (std::getline(myfile, line))
		{
			tmp.push_back(line);
		}


		int m = (*min_element(tmp.begin(),tmp.end(),length())).size();
		p_size = tmp.size();
		int B = 3;
		int stridx[2*p_size];
		memset(stridx,0,2*p_size);

		//Giving value for stride
		for(int i=0,j=0,k=0;i<2*p_size;i+=2)
		{
			stridx[i]= k;
			stridx[i+1]= stridx[i]+tmp[j++].size();
			k=stridx[i+1];
		}

		char* pattern2 =  (char *)malloc(stridx[2*p_size - 1]*sizeof(char));

		//flatten
		int subidx = 0;
		for(int i=0;i<p_size;i++)
		{
			for (int j=stridx[2*i]; j<stridx[2*i+1]; j++)
			{
				pattern2[j] = tmp[i][subidx++];
			}
			subidx = 0;
		}

		int shiftsize = determine_shiftsize(alphabet);



		int *SHIFT = (int *) malloc(shiftsize * sizeof(int)); //shiftsize = maximum hash value of the B-size suffix of the patterns

		//The hash value of the B'-character prefix of a pattern
		int *PREFIX_value = (int *) malloc(shiftsize * p_size * sizeof(int)); //The possible prefixes for the hash values.

		//The pattern number
		int *PREFIX_index = (int *) malloc(shiftsize * p_size * sizeof(int));

		//How many patterns with the same prefix hash exist
		int *PREFIX_size = (int *) malloc(shiftsize * sizeof(int));

		for (int i = 0; i < shiftsize; i++) {

			//*( *SHIFT + i ) = m - B + 1;
			SHIFT[i] = m - B + 1;
			PREFIX_size[i] = 0;
		}



		preprocessing(tmp,SHIFT,shiftsize,PREFIX_value,PREFIX_index,PREFIX_size,m);
		char *d_pattern;
		int *d_stridx;

		int *d_SHIFT;
		int *d_PREFIX_value;
		int *d_PREFIX_index;
		int *d_PREFIX_size;

		int * result = (int*)malloc(N *sizeof(int));
		memset(result,0,N *sizeof(int));
		int * d_result;
		cudaAssert(cudaMalloc(&d_result,N *sizeof (int)));
		cudaAssert(cudaMemset(d_result,0,N*sizeof (int)));

		//Allocating device memory
		cudaAssert(cudaMalloc((void **)&d_SHIFT,shiftsize * sizeof(int)));
		//cudaAssert(cudaMalloc((void **)&d_PREFIX_value,shiftsize * p_size * sizeof(int)));
		//cudaAssert(cudaMalloc((void **)&d_PREFIX_index,shiftsize * p_size * sizeof(int)));
		//cudaAssert(cudaMalloc((void **)&d_PREFIX_size,shiftsize * sizeof(int)));
		//cudaAssert(cudaMalloc((void **)&d_pattern,strlen(pattern2) * sizeof(char)));
		cudaAssert(cudaMalloc((void **)&d_stridx, 2*p_size * sizeof(int)));

		//Copy Device Vectors
		cudaAssert(cudaMemcpy(d_SHIFT,SHIFT,shiftsize * sizeof(int), cudaMemcpyHostToDevice));
		///cudaAssert(cudaMemcpy(d_PREFIX_value,PREFIX_value,shiftsize * p_size * sizeof(int), cudaMemcpyHostToDevice));
		//cudaAssert(cudaMemcpy(d_PREFIX_index,PREFIX_index,shiftsize * p_size * sizeof(int), cudaMemcpyHostToDevice));
		//cudaAssert(cudaMemcpy(d_PREFIX_size,PREFIX_size,shiftsize * sizeof(int), cudaMemcpyHostToDevice));
		//cudaAssert(cudaMemcpy(d_pattern,pattern2,strlen(pattern2) * sizeof(char),cudaMemcpyHostToDevice));
		cudaAssert(cudaMemcpy(d_stridx,stridx,2*p_size * sizeof(int), cudaMemcpyHostToDevice));

		float time;
		cudaEvent_t start, stop;

		cudaAssert( cudaEventCreate(&start) );
		cudaAssert( cudaEventCreate(&stop) );
		cudaAssert( cudaEventRecord(start, 0) );


		COMPOUND_NAME(ANALYSIS_NAME,KernelAnalysis)<<<grid,block>>>(GPU_buffer,GPU_data,GPU_results,state,d_result,d_pattern,d_stridx, d_SHIFT,
				d_PREFIX_value, d_PREFIX_index, d_PREFIX_size, m, p_size);
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
		COMPOUND_NAME(ANALYSIS_NAME,hooks)(packetBuffer, results, state,auxBlocks,result,tmp);
		//Frees results
		cudaAssert(cudaFreeHost(results));
		//free(results);


		cout<<"Time taken for preprocessing "<<timeTaken<<" us"<<endl;
	}
}


#endif // __CUDACC__


#endif // AnalysisSkeleton_h





