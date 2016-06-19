/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef IcmpHeader_h
#define IcmpHeader_h

#include <inttypes.h>
#include <iostream>
#include <pcap.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>


#include <inttypes.h>

#include "../VirtualHeader.h"

#define ICMP_HEADER_SIZE 8
#define HEADER_ICMP_HEXVALUE 0x0001 

/*MACROS HEADERS */
#define ICMP_HEADER_TYPENAME struct icmp_header

#define INSERT_HEADER_ICMP(headers, level, offseT) INSERT_HEADER(headers, level, offseT,HEADER_ICMP_HEXVALUE)
#define IS_HEADER_TYPE_ICMP(headers, level) IS_HEADER_TYPE(headers, level,HEADER_ICMP_HEXVALUE)

using namespace std;

struct icmp_header{
	uint8_t	 type;
	uint8_t	 code; 
	uint16_t checksum; 
	uint16_t id;
	uint16_t seq; 
};

class IcmpHeader : public VirtualHeader {

private:	
	struct icmp_header* icmp;
public:
	inline IcmpHeader(const uint8_t* icmpPointer){icmp=(struct icmp_header*)icmpPointer;};
	void dump(void);
};

#endif // IcmpHeader_h
