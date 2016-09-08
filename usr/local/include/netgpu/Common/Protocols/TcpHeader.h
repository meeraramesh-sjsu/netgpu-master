/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef TcpHeader_h
#define TcpHeader_h

#include <iostream>
#include <inttypes.h>
#include <pcap.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "../VirtualHeader.h"

#define TCP_NO_OPTIONS_HEADER_SIZE 20
#define HEADER_TCP_HEXVALUE 0x0006 

/*MACROS HEADERS */
#define TCP_HEADER_TYPENAME struct tcp_header

#define INSERT_HEADER_TCP(headers, level, offseT) INSERT_HEADER(headers, level, offseT,HEADER_TCP_HEXVALUE)
#define IS_HEADER_TYPE_TCP(headers, level) IS_HEADER_TYPE(headers, level,HEADER_TCP_HEXVALUE)
 

 
/*END MACROS */


using namespace std;

struct tcp_header{
	uint16_t 	sport;		// source port 
	uint16_t 	dport;		// destination port 
	uint32_t 	seq;			// sequence number 
	uint32_t	ack;			// acknowledgement number 

#if BYTE_ORDER == LITTLE_ENDIAN 
	uint8_t		unused:4,		// (unused) 
			offset:4;		// data offset 
#endif
#if BYTE_ORDER == BIG_ENDIAN 
	uint8_t		offset:4,		// data offset 
			unused:4;		// (unused) 
#endif
	uint8_t		flags;
	uint16_t	window;			// window 
	uint16_t	checksum;			// checksum 
	uint16_t	urp;			// urgent pointer 
	//unsigned char *data;		//User Data
};

class TcpHeader : public VirtualHeader {

private: 
	struct tcp_header* tcp;
public:
	inline TcpHeader(const uint8_t* tcpPointer){tcp=(struct tcp_header*)tcpPointer;};
	void dump(void);	 
	uint8_t getHeaderLength(void);	 
	uint32_t getHeaderLengthInBytes(void);	 
	static uint32_t calcHeaderLengthInBytes(const uint8_t * tcpPointer);

};
#endif // TcpHeader_h
