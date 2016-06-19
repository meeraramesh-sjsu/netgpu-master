/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef CudaSafeAccess_h
#define CudaSafeAccess_h

#include <inttypes.h>
#include <stdio.h>

#define GPU_ADDRESS_ALIGNMENT 8

#ifdef __CUDACC__

template<typename T> 
__device__ __inline__  T cudaSafeGet(T* pointer){
	
	switch(sizeof(T)){

		case 1:	return cudaSafeGet8(pointer);

		case 2:	return cudaSafeGet16(pointer);

		case 4:	return cudaSafeGet32(pointer);
		case 8:	return cudaSafeGet64(pointer);

		default: return *pointer;	

	}
}

template<typename T> __device__ __inline__  T cudaSafeGet8(T* pointerT)
{
	uint64_t tmp1;
	uint8_t to_return;
	short int rest = (uint64_t)((uint8_t*)pointerT)%GPU_ADDRESS_ALIGNMENT;
	
	if(rest == 0)
		return *(pointerT);	
	
	tmp1 = *((uint64_t*)(((uint8_t*)pointerT)-rest));
	
	if(rest == 7)
	{
		to_return=(uint8_t)(
					((tmp1&0xFF00000000000000)>>56)
					
				 ); 
	}
	else if(rest == 6)
	{
		to_return=(uint8_t)(
					((tmp1&0x00FF000000000000)>>48)
					
				 ); 
	}
	else if(rest == 5)
	{
		to_return=(uint8_t)(
					((tmp1&0x0000FF0000000000)>>40)
					
				 ); 
	}
	else if(rest == 4)
	{
		to_return=(uint8_t)((tmp1&0x000000FF00000000)>>32);
	}
	else if(rest == 3)
	{
		to_return=(uint8_t)((tmp1&0x00000000FF000000)>>24);
	}
	else if(rest == 2)
	{
		to_return=(uint8_t)((tmp1&0x0000000000FF0000)>>16);
	}
	else //(rest == 1)
	{
		to_return=(uint8_t)((tmp1&0x000000000000FF00)>>8);
	}

	return *((T*)&to_return);
}


template<typename T> __device__ __inline__ T cudaSafeGet16(T* pointerT)
{
	uint64_t tmp1,tmp2;
	short int rest = (uint64_t)((uint8_t*)pointerT)%GPU_ADDRESS_ALIGNMENT;
	uint16_t to_return;

	if(rest == 0)
		return *(pointerT);	
	
	
	tmp1 = *((uint64_t*)(((uint8_t*)pointerT)-rest));
	
		
	if(rest == 7)
	{
		tmp2 = *((uint64_t*)(((uint8_t*)pointerT)-rest+GPU_ADDRESS_ALIGNMENT));
		to_return= (uint16_t)(
					((tmp1&0xFF00000000000000)>>56)
					 |
					((tmp2&0x00000000000000FF)<<8)
					
				 ); 
	}
	else if(rest == 6)
	{
		to_return =  (uint16_t)((tmp1&0xFFFF000000000000)>>48); 
	}
	else if(rest == 5)
	{
		to_return =  (uint16_t)((tmp1&0x00FFFF0000000000)>>40); 
	}
	else if(rest == 4)
	{
		to_return = (uint16_t)((tmp1&0x0000FFFF00000000)>>32);
	}
	else if(rest == 3)
	{
		to_return=(uint16_t)((tmp1&0x000000FFFF000000)>>24);
	}
	else if(rest == 2)
	{
		to_return=(uint16_t)((tmp1&0x00000000FFFF0000)>>16);
	}
	else //(rest == 1)
	{
		to_return = (uint16_t)((tmp1&0x0000000000FFFF00)>>8);
	}
	
	return *((T*)&to_return);
}



template<typename T>
__device__ __inline__  T cudaSafeGet32(T* pointerT)
{
	uint64_t tmp1,tmp2;
	short int rest = (uint64_t)((uint8_t*)pointerT)%GPU_ADDRESS_ALIGNMENT;
	uint32_t to_return;
	
	if(rest == 0)
		return *(pointerT);	
	
	tmp1 = *((uint64_t*)(((uint8_t*)pointerT)-rest));
	
	
	if(rest == 7)
	{
		tmp2 = *((uint64_t*)(((uint8_t*)pointerT)-rest+GPU_ADDRESS_ALIGNMENT));
		to_return=(uint32_t)(
					((tmp1&0xFF00000000000000)>>56)
					 |
					((tmp2&0x0000000000FFFFFF)<<8 )
					
				 ); 
	}
	else if(rest == 6)
	{
		tmp2 = *((uint64_t*)(((uint8_t*)pointerT)-rest+GPU_ADDRESS_ALIGNMENT));
		to_return=(uint32_t)(
					((tmp1&0xFFFF000000000000)>>48)
					 |
					((tmp2&0x000000000000FFFF)<<16)
					
				 ); 
	}
	else if(rest == 5)
	{
		tmp2 = *((uint64_t*)(((uint8_t*)pointerT)-rest+GPU_ADDRESS_ALIGNMENT));
		to_return=(uint32_t)(
					((tmp1&0xFFFFFF0000000000)>>40)
					 |
					((tmp2&0x00000000000000FF)<<24)
					
				 ); 
	}
	else if(rest == 4)
	{
		to_return=(uint32_t)((tmp1&0xFFFFFFFF00000000)>>32);
	}
	else if(rest == 3)
	{
		to_return=(uint32_t)((tmp1&0x00FFFFFFFF000000)>>24);
	}
	else if(rest == 2)
	{
		to_return=(uint32_t)((tmp1&0x0000FFFFFFFF0000)>>16);
	}
	else //(rest == 1)
	{
		to_return=(uint32_t)((tmp1&0x000000FFFFFFFF00)>>8);
	}
	return *((T*)&to_return);
}

template<typename T>
__device__ __inline__  T cudaSafeGet64(T* pointerT)
{
	uint64_t tmp1,tmp2;
	short int rest = (uint64_t)((uint8_t*)pointerT)%GPU_ADDRESS_ALIGNMENT;
	uint64_t to_return;
	
	if(rest == 0)
		return *(pointerT);	
	
	tmp1 = *((uint64_t*)(((uint8_t*)pointerT)-rest));
	tmp2 = *((uint64_t*)(((uint8_t*)pointerT)-rest+GPU_ADDRESS_ALIGNMENT));
	
	
	if(rest == 7)
	{
		to_return=(uint64_t)(
					((tmp1&0xFF00000000000000)>>56)
					 |
					((tmp2&0x00FFFFFFFFFFFFFF)<<8 )
					
				 ); 
	}
	else if(rest == 6)
	{
		to_return=(uint64_t)(
					((tmp1&0xFFFF000000000000)>>48)
					 |
					((tmp2&0x0000FFFFFFFFFFFF)<<16)
					
				 ); 
	
	}
	else if(rest == 5)
	{
		to_return=(uint64_t)(
					((tmp1&0xFFFFFF0000000000)>>40)
					 |
					((tmp2&0x000000FFFFFFFFFF)<<24)
					
				 ); 
	}
	else if(rest == 4)
	{
		to_return=(uint64_t)(
					((tmp1&0xFFFFFFFF00000000)>>32)
					 |
					((tmp2&0x00000000FFFFFFFF)<<32)
					
				 ); 
	
	
	}
	else if(rest == 3)
	{
		to_return=(uint64_t)(
					((tmp1&0xFFFFFFFFFF000000)>>24)
					 |
					((tmp2&0x0000000000FFFFFF)<<40)
					
				 ); 
	
	
	}
	else if(rest == 2)
	{
		to_return=(uint64_t)(
					((tmp1&0xFFFFFFFFFFFF0000)>>16)
					 |
					((tmp2&0x000000000000FFFF)<<48)
					
				 ); 
	
	
	}
	else //(rest == 1)
	{
		to_return=(uint64_t)(
					((tmp1&0xFFFFFFFFFFFFFF00)>>8)
					 |
					((tmp2&0x00000000000000FF)<<56)
					
				 ); 
	
	
	}
	return *((T*)&to_return);
}
/*
typedef struct{
	
	uint64_t element[2];
}type128bits;

typedef struct{
	
	uint64_t element[4];
}type256bits;

template<typename T>
__device__ __inline__  T cudaSafeGet128(T* pointerT)
{
	short int rest = (uint64_t)((uint8_t*)pointerT)%GPU_ADDRESS_ALIGNMENT;
	uint64_t to_return;
	
	if(rest == 0)
		return *(pointerT);	
	
	type256bits tmp1 = *((type256bits*)(((uint8_t*)pointerT)-rest));
	
	if(rest == 7)
	{
		to_return=(uint64_t)(
					((tmp1&0xFF00000000000000)>>56)
					 |
					((tmp2&0x00FFFFFFFFFFFFFF)<<8 )
					
				 ); 
	}
	else if(rest == 6)
	{
		to_return=(uint64_t)(
					((tmp1&0xFFFF000000000000)>>48)
					 |
					((tmp2&0x0000FFFFFFFFFFFF)<<16)
					
				 ); 
	
	}
	else if(rest == 5)
	{
		to_return=(uint64_t)(
					((tmp1&0xFFFFFF0000000000)>>40)
					 |
					((tmp2&0x000000FFFFFFFFFF)<<24)
					
				 ); 
	}
	else if(rest == 4)
	{
		to_return=(uint64_t)(
					((tmp1&0xFFFFFFFF00000000)>>32)
					 |
					((tmp2&0x00000000FFFFFFFF)<<32)
					
				 ); 
	
	
	}
	else if(rest == 3)
	{
		to_return=(uint64_t)(
					((tmp1&0xFFFFFFFFFF000000)>>24)
					 |
					((tmp2&0x0000000000FFFFFF)<<40)
					
				 ); 
	
	
	}
	else if(rest == 2)
	{
		to_return=(uint64_t)(
					((tmp1&0xFFFFFFFFFFFF0000)>>16)
					 |
					((tmp2&0x000000000000FFFF)<<48)
					
				 ); 
	
	
	}
	else //(rest == 1)
	{
		to_return=(uint64_t)(
					((tmp1&0xFFFFFFFFFFFFFF00)>>8)
					 |
					((tmp2&0x00000000000000FF)<<56)
					
				 ); 
	
	
	}
	return *((T*)&to_return);
}
*/
#endif //CUDACC

#endif // CudaSafeAccess_h
