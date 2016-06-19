/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef Windows_h
#define Windows_h

#include <time.h>
#include "../Common/PacketBuffer.h"

#define PACKET_WINDOW 1
#define TIME_WINDOW 2

typedef struct{
	bool hasReachedWindowLimit;
	uint32_t blocksPreviouslyMined;
	uint32_t totalNumberOfBlocks;

	struct timeval windowStartTime;
	struct timeval windowEndTime;
}windowState_t;

bool inline hasReachedPacketLimitWindow(unsigned int current, unsigned int limit){

	//TODO: when large array access CUDA BUG is solved uncomment this and delete the rest (and uncomment Skeleton line too, BOTH!)
	//return (current>= limit);
	
	if(limit%MAX_BUFFER_PACKETS == 0)
		return (current>= limit);
	else
		return (current>= MAX_BUFFER_PACKETS*(limit/MAX_BUFFER_PACKETS-1));

}

bool inline hasReachedTimeLimitWindow(struct timeval start, struct timeval end,unsigned int limit_seconds){

	long msec;

	msec=(end.tv_sec-start.tv_sec)*1000;
	msec+=(end.tv_usec-start.tv_usec)/1000;

	return (((float)msec/1000) >= limit_seconds);
			
}
#endif //Windows_h
