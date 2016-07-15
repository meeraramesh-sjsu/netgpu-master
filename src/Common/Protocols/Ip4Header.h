/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef Ip4Header_h
#define Ip4Header_h

#include <iostream>
#include <inttypes.h>
#include <pcap.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "../VirtualHeader.h"

#define IP4_NO_OPTIONS_HEADER_SIZE 20
#define HEADER_IP4_HEXVALUE 0x0800 

/*MACROS HEADERS */
#define IP4_HEADER_TYPENAME struct ip4_header
#define IP4_HEADER_TYPENAME_16Byte struct ip4_header16Byte

#define INSERT_HEADER_IP4(headers, level, offseT) INSERT_HEADER(headers, level, offseT,HEADER_IP4_HEXVALUE)
#define IS_HEADER_TYPE_IP4(headers, level) IS_HEADER_TYPE(headers, level,HEADER_IP4_HEXVALUE)

/*END MACROS */

/*Onboard Protocol types */

#define IP_PROT_ICMP 	0x01 	// 	ICMP 	Internet Control Message Protocol
#define IP_PROT_IP4 	0x04 	//	IP(4) 	IP in IP (encapsulation)
#define IP_PROT_TCP	0x06 	// 	TCP 	Transmission Control Protocol
#define IP_PROT_UDP	0x11 	// 	UDP 	User Datagram Protocol
#define IP_PROT_IP6	0x29 	//	IPv6 	IPv6

//add more here

/*End of Protocol types */

using namespace std;

struct ip4_header{ 
#if BYTE_ORDER == LITTLE_ENDIAN
        uint8_t   headerLength:4,    /* header length */
                  version:4;         /* version */
#elif BYTE_ORDER == BIG_ENDIAN
        uint8_t   version:4,         /* version */
                  headerLength:4;    /* header length */
#endif
	uint8_t	 tos;			
	uint16_t totalLength;			
	uint16_t identification;			
	uint16_t flagsAndOffset;
	uint8_t  ttl;			
	uint8_t  protocol;			
	uint16_t checksum;			
	uint32_t ip_src;
	uint32_t ip_dst;
};
/*
 * @author: Meera Ramesh
 */
struct ip4_header16Byte{
    uint16_t headerVertos;
	uint16_t totalLength;
	uint16_t identification;
	uint16_t flagsAndOffset;
	uint16_t ttlprotocol;
	uint16_t checksum;
	uint32_t ip_srcFirstHalf:16,ip_srcSecHalf:16;
	uint32_t ip_dstFirstHalf:16,ip_dstSecHalf:16;
};
/*End*/

class Ip4Header : public VirtualHeader {

private:
	struct ip4_header* ip4;

public: 
	inline Ip4Header(const uint8_t* ip4Pointer){ip4=(struct ip4_header*)ip4Pointer;};
	void dump(void);
	uint8_t getHeaderLength(void);
	uint32_t getHeaderLengthInBytes(void);
	uint8_t getProtocol(void);
	
	//static method to calculate header length in bytes from a ip4_header struct -> dissector
	static uint32_t calcHeaderLengthInBytes(const uint8_t * ipPointer);

};

#endif // Ip4Header_h
