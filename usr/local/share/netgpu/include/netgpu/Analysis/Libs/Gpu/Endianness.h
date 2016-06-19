/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef Endianness_h
#define Endianness_h

#include <pcap.h>
#include <inttypes.h>
#include <iostream>
#include <arpa/inet.h>
#include <cuda.h>

/*Inline functions*/
	
#ifdef __CUDACC__

template<typename T>
__device__ __inline__ T cudaNetworkToHost16(T GPU_toBeFlipped)
{
	//TODO: check architecture

	uint16_t toBeFlipped = ((*((uint16_t*)&GPU_toBeFlipped) & 0xFF) << 8) | ((*((uint16_t*)&GPU_toBeFlipped) & 0xFF00) >> 8);

	return *((T*)&toBeFlipped);	
}


template<typename T>
__device__ __inline__ T cudaNetworkToHost32(T GPU_toBeFlipped)
{
	//TODO: check architecture	
	uint32_t toBeFlipped = ((*((uint32_t*)&GPU_toBeFlipped) & 0xFF) << 24) | ((*((uint32_t*)&GPU_toBeFlipped) & 0xFF00) << 8) | ((*((uint32_t*)&GPU_toBeFlipped) & 0xFF0000) >> 8) | ((*((uint32_t*)&GPU_toBeFlipped) & 0xFF000000) >> 24);

	return *((T*)&toBeFlipped);	
}

template<typename T>
__device__ __inline__ T cudaNetworkToHost64(T GPU_toBeFlipped)
{
	//TODO: check architecture	

	uint64_t toBeFlipped = (
		 ((( *((uint64_t*)&GPU_toBeFlipped))<<56) & 0xFF00000000000000)  | 
                 ((( *((uint64_t*)&GPU_toBeFlipped))<<40) & 0x00FF000000000000)  | 
                 ((( *((uint64_t*)&GPU_toBeFlipped))<<24) & 0x0000FF0000000000)  | 
                 ((( *((uint64_t*)&GPU_toBeFlipped))<< 8) & 0x000000FF00000000)  | 
                 ((( *((uint64_t*)&GPU_toBeFlipped))>> 8) & 0x00000000FF000000)  | 
                 ((( *((uint64_t*)&GPU_toBeFlipped))>>24) & 0x0000000000FF0000)  | 
                 ((( *((uint64_t*)&GPU_toBeFlipped))>>40) & 0x000000000000FF00)  | 
                 ((( *((uint64_t*)&GPU_toBeFlipped))>>56) & 0x00000000000000FF)
				);


	return *((T*)&toBeFlipped);	

}

template<typename T>
__device__ __inline__ T cudaNetworkToHost128(T GPU_toBeFlipped)
{


//	return *((T*)&toBeFlipped);	
	return GPU_toBeFlipped;	
}

template<typename T>
__device__ __inline__ T cudaNetworkToHost(T GPU_toBeFlipped)
{
	switch(sizeof(T)){
		
		case 1: return GPU_toBeFlipped;
			
		case 2: return cudaNetworkToHost16(GPU_toBeFlipped);
		
		case 4: return cudaNetworkToHost32(GPU_toBeFlipped);
		
		case 8: return cudaNetworkToHost64(GPU_toBeFlipped);
		
		case 16: return cudaNetworkToHost128(GPU_toBeFlipped);
		
		default:
			return GPU_toBeFlipped;	
	}
}

#endif //CUADACC

#endif // Endianness_h
