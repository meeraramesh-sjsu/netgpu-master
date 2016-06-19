/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef libAnalysisThresholds_h 
#define libAnalysisThresholds_h 
 
typedef struct{
	ANALYSIS_INPUT_TYPE user;
	int32_t counter;
	int64_t flow;
	float rate;

}COMPOUND_NAME(ANALYSIS_NAME,thresholdAnalysis_t);


/*Diff time function */
__device__ __inline__ long cudaTimevaldiff(struct timeval starttime, struct timeval finishtime)
{
                long msec;
                msec=(finishtime.tv_sec-starttime.tv_sec)*1000;
                msec+=(finishtime.tv_usec-starttime.tv_usec)/1000;
                return msec;
}

/* SIZE MACROS */

#define KB 1024
#define KBPS KB
#define MB 1048576
#define MBPS MB
#define GB 1073741824 
#define GBPS GB

#endif 

/* Undefine TYPES */
#undef  ANALYSIS_INPUT_TYPE
#define ANALYSIS_INPUT_TYPE COMPOUND_NAME(ANALYSIS_NAME,thresholdAnalysis_t)


