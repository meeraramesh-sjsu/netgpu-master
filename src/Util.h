/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef Util_h
#define Util_h

#include <errno.h> 
#include <string.h> 

/* Define debug level 0,1,2 */
/* less verbose <---> more verbose */
#define DEBUG_LEVEL 3

/* GENERAL CONSTANTS */
#define APP_NAME NetGpu


/* CUDA CONSTANTS */
#define CUDA_MAX_BLOCKS_PER_DIM 512
#define CUDA_MAX_THREADS 65536 

/* Stringificator */
#define _STR(a) #a
#define STR(a)	_STR(a)



#define cudaAssert(f) \
	do {	\
		cudaError_t err=f;\
		if(err != cudaSuccess) { \
			fprintf(stderr,"cudaError at %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err));\
			exit(-1);\
		}\
	}while(0)

#define ABORT(msg)	\
	do {	\
		fprintf(stderr,"ABORTING: %s:%d: %s\n",__FILE__,__LINE__,msg);\
		exit(-1);	\
	} while(0)

#define WARN(msg)	\
	do {	\
		fprintf(stderr,"WARNING: %s:\n",msg);\
	} while(0)

#define WARN_ERRNO(msg)	\
	do {	\
		fprintf(stderr,"WARNING: %s -> %s\n",msg,strerror(errno));\
	} while(0)



#if DEBUG_LEVEL > 0 
	#define DEBUG(msg,...) do {	\
		fprintf(stderr,"DEBUG: %s:%d:" msg "\n",__FILE__,__LINE__,##__VA_ARGS__);\
	} while(0)

#else
	#define DEBUG(a,...) do{}while(0)
#endif

#if DEBUG_LEVEL == 2
	#define DEBUG2(msg,...) do {	\
		fprintf(stderr,"DEBUG2: %s:%d:" msg "\n",__FILE__,__LINE__,##__VA_ARGS__);\
	} while(0)
//#else
	//#define DEBUG2(a,...)	do{}while(0)
#endif

#endif //Util_h
