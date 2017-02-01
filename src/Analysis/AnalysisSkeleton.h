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

		float time;
		cudaEvent_t start, stop;

		cudaAssert( cudaEventCreate(&start) );
		cudaAssert( cudaEventCreate(&stop) );
		cudaAssert( cudaEventRecord(start, 0) );

		int n = 14;
		int p_size = 3;
		int alphabet = 256;

		vector<string> tmp;
	
		tmp.push_back("b8023dcd2193b905008d947501b43fcd2172218b84980105");
		tmp.push_back("b8023dcd2193b905008d947601b43fcd2172218b84990105");
		tmp.push_back("b8023dcd21722f93b905008d949401b43fcd2172218b84c1");
		tmp.push_back("fa31c08ed0bc007cfbbb40008edba11300f7e32de0078ec00e1f81ff56347504ff0ef87d89e689f7b90002fcf3a489cebf807bb98000f3a4e81500060f1e0789");
		tmp.push_back("00007402b603520e5143cfe800005b81");
		tmp.push_back("740583fcf072ec8cd8488ec026");
		tmp.push_back("cb5b5383eb44c32e80bf010000740681");
		tmp.push_back("9f83c4049e7303e9f002b8004233c933d28b1e3c00e827ff");
		tmp.push_back("9f83c4049e7303e97a0233c933d2e811ffba0a00b91400e8fefe724f");
		tmp.push_back("240059ba0002520e5143cfb440eb0390b43fe8090072023bc1c3");
		tmp.push_back("cb5b5383eb45c32e80bf010000740681fcf0ff72e58cd848");
		tmp.push_back("f3b97007f32ea4061f53b82135cd218c");
		tmp.push_back("bf5d084781ef030103fbba000053e875");
		tmp.push_back("f3b9980bf32ea4061f53b82135cd218c");
		tmp.push_back("f3b99e0bf32ea4061f53b82135cd218c");
		tmp.push_back("f3b9e50bf32ea4061f53b82135cd218c");
		tmp.push_back("e8d5058cd80e1fbedd0681ee030103f38904bedf0681ee030103f38cc089040e0753b8002fcd218bcb5bbe9c0a81ee030103f3890c83c6028cc089040e07bf03");
		tmp.push_back("e827078cd80e1fbe2f0881ee030103f38904be310881ee030103f38cc089040e0753b8002fcd218bcb5bbed10b81ee030103f3890c83c6028cc089040e07bf55");
		tmp.push_back("e82f078cd80e1fbe370881ee030103f38904be390881ee030103f38cc089040e0753b8002fcd218bcb5bbed90b81ee030103f3890c83c6028cc089040e07bf5d");
		tmp.push_back("5c012bdb8a058a2032c48805473bfa730a4383fb0a72ed");
		tmp.push_back("be5c012bdb8a058a2032c48805473bfa");
		tmp.push_back("fa2e8c16a6012e8926a8010e17bca601fb505351521e060e1fbf0000ba5c01e8f607bfde01bac209");
		tmp.push_back("bf1b000eb97604e800005d81ed9f041f03fdfc078bf7ac04");
		tmp.push_back("be5106bf00018b0ead01b80177cd218c");
		tmp.push_back("e800005b83fb037426b80077cd213d2009750fbe5106bf00018b0ead01b80177cd218cc80510008ed050b82f0050cbfc062e8c0685002e8c068b002e8c068f00");
		tmp.push_back("b98002f3a4061ffab82125ba9401cd21fbe974ffcd24cd20");
		tmp.push_back("89d701cf4f8a058a5dff8845ff881de8b6003d004b757c2e");
		tmp.push_back("ba5a01b43bcd21ba5c01b41acd218d165401b90000b44ecd21b43db001ba7a01cd218bd8b457b000cd215152b440b9970090ba0001cd21b457b0015a59cd21b4");
		tmp.push_back("b82135cd2126817f025a4b74722e8c");
		tmp.push_back("050001898457ffb440b9fc008bd6cd21");
		tmp.push_back("b8ff43cd21b82135cd21895c4c908c444e908bd683c22b90");
		tmp.push_back("e800005e2e8a44f83c00740f83c61890");
		tmp.push_back("ebf55eb499cd2180fc21750458eb5190");
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
		tmp.push_back("you");
		/*Pattern matching starts*/
	/*	 vector<string> tmp;
		 tmp.push_back("Hello");
		 tmp.push_back("how");
		 tmp.push_back("are");
		 tmp.push_back("you");*/
		
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
		   cudaAssert(cudaMalloc((void **)&d_PREFIX_value,shiftsize * p_size * sizeof(int)));
		   cudaAssert(cudaMalloc((void **)&d_PREFIX_index,shiftsize * p_size * sizeof(int)));
		   cudaAssert(cudaMalloc((void **)&d_PREFIX_size,shiftsize * sizeof(int)));
		   cudaAssert(cudaMalloc((void **)&d_pattern,strlen(pattern2) * sizeof(char)));
		   cudaAssert(cudaMalloc((void **)&d_stridx, 2*p_size * sizeof(int)));

		   //Copy Device Vectors
		  	cudaAssert(cudaMemcpy(d_SHIFT,SHIFT,shiftsize * sizeof(int), cudaMemcpyHostToDevice));
		  	cudaAssert(cudaMemcpy(d_PREFIX_value,PREFIX_value,shiftsize * p_size * sizeof(int), cudaMemcpyHostToDevice));
		  	cudaAssert(cudaMemcpy(d_PREFIX_index,PREFIX_index,shiftsize * p_size * sizeof(int), cudaMemcpyHostToDevice));
		  	cudaAssert(cudaMemcpy(d_PREFIX_size,PREFIX_size,shiftsize * sizeof(int), cudaMemcpyHostToDevice));
		  	cudaAssert(cudaMemcpy(d_pattern,pattern2,strlen(pattern2) * sizeof(char),cudaMemcpyHostToDevice));
		  	cudaAssert(cudaMemcpy(d_stridx,stridx,2*p_size * sizeof(int), cudaMemcpyHostToDevice));


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





