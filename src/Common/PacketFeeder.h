/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef PacketFeeder_h
#define PacketFeeder_h

#include <pthread.h>

#include "../Util.h"
#include "PacketBuffer.h"
	
class PacketFeeder {

public:
	//Create a pthread and start buffering
	virtual pthread_t* start(int limit)=0;	
	
	//Get a filled PacketBuffer
	virtual PacketBuffer* getSniffedPacketBuffer(void)=0;
	
	//Force to stop feeding and mark last PacketBuffer with flag "flush" to true
	virtual void flushAndExit(void)=0;

private:

};

#endif // PacketFeeder_h
