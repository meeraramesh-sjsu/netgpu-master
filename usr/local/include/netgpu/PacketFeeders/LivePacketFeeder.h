/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef LivePacketFeeder_h
#define LivePacketFeeder_h

#include <pcap.h>
#include <cstring>
#include <cstdlib>
#include <inttypes.h> 
#include <pthread.h> 
#include <semaphore.h> 


#include "../Util.h"
#include "../Common/PacketFeeder.h"
#include "SizeDissector.h"

#define CAPTURING_TIMEms 1000
#define SNIFFER_BUFFER_SIZE 8192
#define SNIFFER_NUM_OF_BUFFERS 2


#define SNIFFER_GO_STATE 0
#define SNIFFER_LASTBUFFER_STATE 1
#define SNIFFER_END_STATE 2

using namespace std;
	
class LivePacketFeeder:public PacketFeeder {

public:

	LivePacketFeeder(const char* device);
	~LivePacketFeeder(void);

	pthread_t* start(int limit);	

	//captured packet callback method	
	static void packetCallback(u_char *sniffer,const struct pcap_pkthdr* pkthdr,const u_char* packet);
	
	//Method for the consumer thread to get the sniffed PacketBuffer
	PacketBuffer* getSniffedPacketBuffer(void);

	void flushAndExit(void);
private:
	//PCAP descriptor
        pcap_t* descr;
	
	//Counter and limit
	int packetCounter;
	int maxPackets;

	//Array of 2 packetBuffers and actualindex
	PacketBuffer* packetBufferArray;
	short int bufferIndex;	

	//Device name
	const char* dev;

	//State
	int state;

	//Mutex pthread semaphore
	pthread_mutex_t mutex;

	//Synchronization pthreads  semaphore
	sem_t* waitForSwap;
	sem_t* waitForLivePacketFeederToEnd;

	static void* startThreadWrapper(void* object); 
	void _start(void);	
	inline void setDeviceDataLinkInfoToBuffers(int deviceDataLink);
};

#endif // LivePacketFeeder_h
