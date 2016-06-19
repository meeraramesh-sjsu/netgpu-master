/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef UdpHeader_h
#define UdpHeader_h

#include <inttypes.h>
#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "../VirtualHeader.h"

#define UDP_HEADER_SIZE 8
#define HEADER_UDP_HEXVALUE 0x0011 

/*MACROS HEADERS */
#define UDP_HEADER_TYPENAME struct udp_header

#define INSERT_HEADER_UDP(headers, level, offseT) INSERT_HEADER(headers, level, offseT,HEADER_UDP_HEXVALUE)
#define IS_HEADER_TYPE_UDP(headers, level) IS_HEADER_TYPE(headers, level,HEADER_UDP_HEXVALUE)
 
/*END MACROS */
using namespace std;

struct udp_header{
	uint16_t sport;           // source port 
	uint16_t dport;           // destination port
	uint16_t ulen;            // udp length 
	uint16_t sum;             // udp checksum 
};

class UdpHeader : public VirtualHeader {
	
private:
	struct udp_header* udp;	
public:
	inline UdpHeader(const uint8_t* udpPointer){udp=(struct udp_header*)udpPointer;};
	void dump(void);	

};

#endif // UdpHeader_h
