/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef BMMS_H
#define BMMS_H

#include <inttypes.h>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cstring>
#include <list>
#include "/usr/local/cuda/include/cuda.h"
#include "/usr/local/cuda/include/cuda_runtime.h"


#define FREE 0
#define USED 1

#include "../../../../Util.h"

using namespace std;

typedef struct{
	bool isUsed;
	unsigned int blockStart;
	unsigned int numOfPartitionBlocks;
}bitmap_t;

/**
	Basic Memory Management System.
	It's a very (very) simple and basic MMS that tries to reduce number of cudaMallocs/cudaFrees,
	by defining it's own memory management system. Must be initialized by a Fixed (and no resizable)
	Memory size;
	
**/

#define BMMS_USE_CUDA_MMS

class BMMS{

public:
	
	/*BMMS(unsigned int totalSize,unsigned int blockSize);
	~BMMS();
	*/
	static void init(unsigned int totalSize,unsigned int blockSize);
	static void mallocBMMS(void** pointer,unsigned int reqSize);
	static void* _mallocBMMS(unsigned int reqSize);
	static void freeBMMS(void* pointer);
	static void printBitmap(void);


private:
	static bool work;
	
	static bitmap_t* bitmap;	
	static uint8_t* allocatedMemory; 
	
	static unsigned int blockSize;
	static unsigned int numOfBlocks;
	
	static int findAndAssignPortion(unsigned int blocksRequired);
		
		
};	
#endif //BMMS_H
