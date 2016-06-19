/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef PacketBuffer_h
#define PacketBuffer_h

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <pcap.h>
#include <inttypes.h>
#include <fstream>
#include "/usr/local/cuda/include/cuda.h"
#include "/usr/local/cuda/include/cuda_runtime.h"
 
#include "../Util.h"
#include "../PacketFeeders/SizeDissector.h"

#define MAX_BUFFER_PACKETS 3840 //Max number of packets
#define MAX_BUFFER_PACKET_SIZE 94 //Packet max size
#define TIMESTAMP_OFFSET sizeof(int)

typedef struct{
	int proto[7];
	int offset[7];		
}headers_t;


typedef struct{
	timeval timestamp;
	headers_t headers;
	uint8_t packet[MAX_BUFFER_PACKET_SIZE];
}packet_t;

using namespace std;

class PacketBuffer {

 public:
	
	PacketBuffer(void);
	~PacketBuffer(void);

	void setDeviceDataLinkInfo(int deviceDataLinkInfo);
	int getDeviceDataLinkInfo(void);
	
	unsigned int getNumOfPackets(void);
	unsigned int getNumOfLostPackets(void);
	packet_t* getBuffer(void);
	
	int pushPacket(uint8_t* packetPointer, const struct pcap_pkthdr* hdr);
	packet_t* getPacket(int index);
	void clearContent(void);
	
	bool inline getFlushFlag(void){return flushWindows;}
	void inline setFlushFlag(bool value){flushWindows = value;}
	packet_t* buffer;
 protected:
	//DataLink info for all packets
	int deviceDataLink;

	unsigned int lastPacketIndex;
	unsigned int lostPackets;
	//packet_t* buffer;

	//Flag to flush all windows (end of data or interrupted)
	bool flushWindows;
};

#endif // PacketBuffer_h
