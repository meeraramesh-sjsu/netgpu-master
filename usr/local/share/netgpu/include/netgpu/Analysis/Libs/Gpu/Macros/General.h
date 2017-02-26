/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef General_h
#define General_h

#include "../../../../Util.h"

//Macros used to create CONCATENATED names
#define _COMPOUND_NAME(PREFIX,NAME)\
	PREFIX##_##NAME

#define COMPOUND_NAME(PREFIX,NAME)\
	_COMPOUND_NAME(PREFIX,NAME)

/* GENERAL MACROS */

#define BUFFER_BLOCKS (MAX_BUFFER_PACKETS/ANALYSIS_TPB)
#define ARRAY_SIZE(type) \
	(sizeof(type)*MAX_BUFFER_PACKETS) 

#if HAS_WINDOW == 1
	//Window special case
	#define POS threadIdx.x + (state.blockIterator*blockDim.x) 
	#define RELATIVE_MINING_POS (threadIdx.x + ((state.blockIterator-state.windowState.blocksPreviouslyMined)*blockDim.x)) 
#else
	#define POS threadIdx.x + (blockIdx.x*blockDim.x) //Absolute thread number 
#endif

/* Packet inside buffer */
#if HAS_WINDOW == 1
	#define PACKET (&GPU_buffer[RELATIVE_MINING_POS])
#else
	#define PACKET (&GPU_buffer[POS])
#endif

/*GPU_data element */
#define DATA_ELEMENT GPU_data[POS]  
/*GPU_results element */
#define RESULT_ELEMENT GPU_results[POS]  

/* GETS HEADERS POINTER at level*/
#define GET_HEADER_POINTER(level) \
	(((uint8_t*)&(PACKET->packet))+PACKET->headers.offset[level])
/*#define GET_HEADER_POINTER(level) \
		(((PACKET->packet))+PACKET->headers.offset[level])*/

#define GET_HEADER_POINTERCHAR ((const u_char*) /*(uint8_t* )*/ &(PACKET->packet))

//#define GET_HEADER_POINTERCHAR ((const u_char*) /*(uint8_t* )*/ (PACKET->packet))

#define GET_HEADER_TCP_POINTER(level)\
	PACKET->headers.offset[level]


//Gets field safely, to get disaligned fields 
#define GET_FIELD(field) cudaNetworkToHost(cudaSafeGet(&(field))) //TODO: ENDIANISME ELIMINAR EL CUDANETWORKTOHOST

#define GET_FIELDNETWORK(field) cudaSafeGet(&(field)) //TODO: ENDIANISME ELIMINAR EL CUDANETWORKTOHOST


//* BARRIERS */

//Block barrier

#define SYNCTHREADS() __syncthreads()

//Predefined Analysis Barrier (syncblocks)

#ifndef DONT_EXPAND_SYNCBLOCKS
	#define SYNCBLOCKS_PRECODED() } \
		template<typename T,typename R>\
		__global__ void COMPOUND_NAME(COMPOUND_NAME(ANALYSIS_NAME,PredefinedKernel),_PRECODED_COUNTER)(packet_t* GPU_buffer,T* GPU_data,R* GPU_results, analysisState_t state){\
		do{}while(0)
#endif
//User Grid barrier

#ifndef DONT_EXPAND_SYNCBLOCKS
	#define SYNCBLOCKS() } \
	template<typename T,typename R>\
	__device__ __inline__ void COMPOUND_NAME(COMPOUND_NAME(ANALYSIS_NAME,AnalysisExtraRoutine),__COUNTER__)(packet_t* GPU_buffer, T* GPU_data,R* GPU_results, analysisState_t state){\
	do{}while(0)
#endif

/* HAS_REACHED_WINDOW_LIMIT MACRO */
#define IF_HAS_REACHED_WINDOW_LIMIT()\
		if(state.windowState.hasReachedWindowLimit)

#endif //General_h
