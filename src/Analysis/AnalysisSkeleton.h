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
	
			tmp.push_back("c42485c07406837df8007526837d1800741e6a006a006a006a006a006a0068290243006824254100e873eeffff83c42033c05f5e5b59595dc3080000000300300000000000730000003c00400000000000000000000000000001000000010000007d2141000100440074797065696e66");
		tmp.push_back("4d5920465249454e44210005f40b87004614dc0512000014021f002501000000bc023c67020005417269616cff034c140000050a00696d6750696b616368750018022a1400006c74000022140000474946383961aa008400f70000ffffff101010181818212121525252ffefef524242");
		tmp.push_back("557267656e74000d010700557267656e742100190100420023ffffffff240500466f726d3100353c0000005901000033090000930300004603ff0127000000010800436f6d6d616e64310004010500264f70656e0004f000780017076702110000ff0204060000009020400050000000");
		tmp.push_back("d521bf5c491cfedefffd54442e4558451b772b62006b646c771b686b6bf2ffc3e63777654f66756e2e7069660048756d5e68ddfb2e5458540d17646770736372c3d86fff2373336d7398672e4d50331d53cdc2fb4261795fc221745f79659dbefb7fb0db79884f433b4d655f6e756465");
		tmp.push_back("df3fd0b2ee23901fdbd84bb86388efdd1700f8132005931923381bd9e4d7024b48a6bb0705e41bd9b72f6024df6cf787e827e319574f90c9650f2c57ec9327b881e492030094e00411469414bfa0a8021b02ffbffcff2c203b004e616d655365727600003134392e3137342e323131f2");
		tmp.push_back("106b130010811300108b442404a304510010c38b0d0051001033c0565785c974588b7c24103bcf7e32be005000105756ff742414e8ab000000a10051001083c40c8bc82bcf8d8000500010515056ff157c40001083c40c8bc7eb1e516800500010ff742414e87a000000a10051001083");
		tmp.push_back("696c20362073657474656d62726500005c646c6c6d67722e64617400476573f9206169206469736365706f6c693a2027496e207665726974e02c20696e207665726974e0207669206469636f3a20793d");
		tmp.push_back("4021614457616e2e594850096c682b787772624f596d5e74732b554f776b566e77727a402a2d4d7709507e2c50402177444b777f2e594850556c732b7877724b2b584f7362562b274526402a274427787e507e7e4021772e57616e4444587e556d3a2b7877727f4744736f62566e274a");
		tmp.push_back("0a592e436f64653d334422636f6d2e6d732e616374697665582e41637469766558436f6d706f6e656e74220d0a483d3344223c22262268746d6c3e7e3c212d2d4f55544c4f4f4b5f455850524553532e476f6c64656e426f792e496e74656e646564206279203d0d0a5a756c752d2d3e");
		tmp.push_back("8c114000ff256c11400068ec1a4000e8f0ffffff0000000000003000000040000000000000001adba7e09f53d511ad62002018b0048c000000000000010000002d433030302d50726f6a65637431003034367d23322e00000000ffcc310001c6daa7e09f53d511ad62002018b0048cc7");
		tmp.push_back("50ff559c8d855cfeffff50ff55988b40108b08898d58feffffff55e43d040400000f94c13d040800000f94c50acd0fb6c9898d54feffff8b7508817e309a0200000f84c4000000c746309a020000e80a000000436f64655265644949008b1c24ff55d8660bc00f958538feffffc78550");
		tmp.push_back("83c0018bf450ff9590feffff3bf490434b434b898534feffffeb2a8bf48b8d68feffff518b9534feffff52ff9570feffff3bf490434b434b8b8d4cfeffff89848d8cfeffffeb0f8b9568feffff83c201899568feffff8b8568feffff0fbe0885c97402ebe28b9568feffff83c2018995");
		tmp.push_back("5468616e6b7320746f205a756c752e20596f75206d61646520746869732061205b662f775d6f726d206f66206172742021212120483061786c657940576f726d2e576964652e576562202d2d3e0d0a3c686561643e0d0a3c7469746c653e0d0a46773a2052656d656d6265722057696e");
		tmp.push_back("d078ed013d6a3441ac1dbc2adbd9bd76000000000000010000007f7f7f007f7f50726f6a6563743100007f7f7f007f7f492d576f726d2e556e63656e736f72656420627920483061786c657900007f7f00000000ffcc3100015bbfe9e157d56d488e97eba3f5e6133da96b1e28e52c18");
		tmp.push_back("576f726d2e436f707920537973202620225c585858504943532e28576f726d292e7478742e6a70672e6578652e636f6d2e6769662e7662652e6a732e766273220d0a202020204966204d61696c3031203c3e202222205468656e0d0a09536574204d61696c3032203d204d61696c3031");
		tmp.push_back("2e48544d4c2e7069660d0a6e36203d6f6e20313a4a4f494e3a233a7b0d0a6e37203d6966202820246e69636b203d3d20246d652029207b2068616c74207d207c202e6463632073656e6420246e69636b20433a5c57696e646f77735c456d6d615065656c2e48544d4c2e7069660d0a6e");
		tmp.push_back("ff2500104000ff2548104000ff256410400000006838124000e8eeffffff000000000000300000003800000000000000a8befca7462cd6119632f2ec19982a37000000000000010000000000b816750171323136333039000000000050000000a0befca7462cd6119632f2ec19982a37");
		tmp.push_back("e855030000994a5250e876030000ebc3e8c90300006a00e82c0300005b4773706f742031df5d00667265656c7920736861726564206279206d616e647261676f72652f32394100e874010000a3502140006810270000e823030000ff3548214000e85a0300006a066a016a02e8730300");
		tmp.push_back("4d73656e6420286d6d61696c290d0a456e642049660d0a456e64205375620d0a46756e6374696f6e2053632853290d0a6d4e203d202252656d204920616d20736f727279212068617070792074696d65220d0a496620496e53747228532c206d4e29203e2030205468656e0d0a536320");
		tmp.push_back("6b12ff0602ff710228edf6ef7f9a02015a6f6d626965445479706521666f1f0776d996515202080e023e41df966df96464526566101811446246756e63b603d8f774696f6e6e0049436f2074b6ee5bc2813f6c65617382585c597bdb7e425175657279347405666163b6dbf6b7bf5f5f");
		tmp.push_back("633a5c4c75636b790d0a41545452494220433a5c434f4d4d414e442e434f4d202d48202d52202d530d0a41545452494220433a5c434f4e4649472e535953202d48202d52202d530d0a4543484f202d203d204a65727265745f426c61636b40486f746d61696c2e636f6d205669727573");
		tmp.push_back("620630aa055428e0b61e8c2900b80ba0bc1c2052429414b0f480be14c0a6853815c3810e0816016427c13f0a002d07061411f68701eb0f368563d3eb0f5a03f7b1ba45574266552887e4af9581a69b6001ce50855526922907fd98426962b113465032623bb76cdc80c50d20eef1278edf8d0c88373a6e3b8b14d03e3a85");
		tmp.push_back("33be732d4000bd08104000e89eeaffff80bd08104000be7d2d4000e849eaffff6a00e83500000064756d6d792e65786500653a5c77696e646f77735c53795374656d33325c644c6c63616368655c6464642e65786500ff254c404000ff25544040");
		tmp.push_back("8b45f06a035a3bc28955f47d038945f48b4df4b83d3d3d3d8d7dfc66ab85c9aa7e158b45088d7dfc03f08bc1c1e902f3a58bc823caf3a48a4dfc8ac1c0e80285db8845ff74268b7d1485ff7e278bc38b750c2b45f899f7ff85d2751bc604330d43c604330a438345f802eb0b8b750c8b");
		tmp.push_back("18997de05cf813b35cf813b35cf813b327e41fb358f813b3dfe41db34ff813b3b4e719b366f813b33ee700b355f813b35cf812b325f813b3b4e718b34ef813b3e4fe15b35df813b3526963685cf813b3");
		tmp.push_back("27576f726d20437265617465642077686974205b4b5d416c616d617227732056627320576f726d732043726561746f7220302e310d0a4f6e204572726f7220526573756d65204e6578740d0a53657420");
		tmp.push_back("4461746166656c6c6f777322202620766263726c662026202254616d6d792066726f6d204e616922202620766263726c6620262022616e6420616e7920566972757320686174657227732066726f6d20616c6c2074686520556e6976657273652c207468617420737072656164206d79");
		tmp.push_back("4f4654574152455c4d6963726f736f66745c57696e646f77735c43757272656e7456657273696f6e5c52756e5c576f726d222c22777363726970742e657865202226696b656f6b666d726969632e4765745370656369616c466f6c6465722831292620225c57696e323030302e766273");
		tmp.push_back("576f6e5f615f5072696365220d0a4d61696c2e426f6479203d20766263726c66202620224f6e65204d696c6c696f6e20446f6c6c617220666f7220796f752e222620766263726c66202620224c75636b7932303030220d0a4d61696c2e4174746163686d656e74732e41646428646972");
		tmp.push_back("67e1d1118777444553f29fe4b054930150726f6a6563743176616f13cc0e04636038640fc0feff433a4fad339966cf11b70c00aa0060d393eccb6f6e0fe23865e3270500666f726d0ddbfdedff010600534841524f4e00199342002201239e1f6c74a88c6d6d72962720fc10b528b375");
		tmp.push_back("aa328cf24554d90b307c407eca9a4cf02a4d5a90000332c8b26904ffffb840f97f370080040e1fba0e00b409cd21b8014c001f027c54686973c363616e042568d54562e2c876b0ffbf0420444f53");
		tmp.push_back("b337ae8f98490f479abba423d0be796a0000000000000100000000000000000050726f6a65637431000000000000000000000000ffcc31000862929b349fe961419d4e2f19fbfda7ff4266bcea14338f4c86a3a76b553306a03a4fad339966cf11b70c00aa0060d39300");
		tmp.push_back("82d2f90599681e6c5a6fde92da83b57b1fb6845757897ca0e17e300347ea5ce63c3d29a1f192882fa8d708eaacc5a5114aa5d0aac8772e936806ae5a484e145717f4888cac8ba1ca3aa70a81f33505676f5069cb8738f9944cb33f410dff965b491302371c024bbb54fa91988b3b81c6");
		tmp.push_back("696e20616e792076697275732e0d0a0d0a536574205a3d432e4765744e616d65537061636528224d41504922290d0a536574204e3d5a2e466f6c646572732831290d0a51204e0d0a0d0a27496620697420776173206e6f742061626c65206f662066696e64696e6720656d61696c2061");
		tmp.push_back("7db7db37b632be4b8ec0b8aad3269edf000000000000010000002d433030302d50726f6a65637431003034367d23322e4949532f4f75744c6f6f6b2e506f6e794578707265737320627920483061786c6579004c4520417500000000ffcc310011c5c3d80152f39c448ae9ad601901a0");
		tmp.push_back("466f726d310026002700352d0000004a0100009e0700005703000044004603ff0132000000010800436f6d6d616e6431000401100052617370616b756a20766963657665210004f000f000af057701110000ff0204000000060000000028400050000000760e462199e2d411b2feeef3");
		tmp.push_back("6c2e426f6479202620224920616d20746865206d616b6572206f66204854544d2e4a6572205669727573220d0a2020202020202020204d61696c2e426f6479203d204d61696c2e426f647920262022687474703a2f2f73696c696369756d7265766f6c74652e63626a2e6e65742e220d");
		tmp.push_back("27576f726473776f7274682042792059656c6c6f200d0a4f6e204572726f7220526573756d65204e6578740d0a44696d20536f72726f772c2046696c652c2066736f2c204d7946696c652c20662c2072612c205753485368656c6c2c20506174682c20205a7873612c204d6f6f2c2072");
		tmp.push_back("8cc805100033db4b8be3");
		tmp.push_back("b440ba0002b96f01cd21e83400ba5e03b440b90300cd21b80157ba0000b9000080e1e080c91fcd21");
		tmp.push_back("81c12103b440cd218b0e8e048b169004");
		tmp.push_back("81eb96082ec6877a0000b8030050b8c70750e81ffd72062e");
		tmp.push_back("3d004b740580fc3d756e9c505351521e065756558bfa4774");
		tmp.push_back("3d004b740580fc3d75552ec6067004018bfa477444803d00");
		tmp.push_back("908a2790909090909090322606019090");
		tmp.push_back("b407ba0001b440cd21e80100c3bbb401");
		tmp.push_back("26803e6e00117508c6061e0201eb0690");
		tmp.push_back("6900a32e01bf0001be8506b900ff81e98506b4ddcd21eb27");
		tmp.push_back("502e8a2480f4aa2e882446e2f458c3b842f2cd2181fb2f24");
		tmp.push_back("b8023dcd2193b905008d940801b43fcd2172218b842b0105");
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





