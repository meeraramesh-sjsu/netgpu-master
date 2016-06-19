/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef Ip6Header_h
#define Ip6Header_h

#include <inttypes.h>
#include <pcap.h>

#include "../VirtualHeader.h"
#include "Ip4Header.h" //Protocol numbers

#define HEADER_IP6_HEXVALUE 0x86DD 

/*MACROS HEADERS */

#define IP6_HEADER_TYPENAME struct ip6_header

#define INSERT_HEADER_IP6(headers, level, offseT) INSERT_HEADER(headers, level, offseT,HEADER_IP6_HEXVALUE)
#define IS_HEADER_TYPE_IP6(headers, level) IS_HEADER_TYPE(headers, level,HEADER_IP6_HEXVALUE)
 
 
/*END MACROS */




struct ip6_header{
#if BYTE_ORDER == LITTLE_ENDIAN
	uint8_t	 priority:4,
                 version:4;
#elif BYTE_ORDER == BIG_ENDIAN
        uint8_t  version:4,
                 priority:4;
#endif
        uint8_t  flow_lbl[3];
	uint16_t payload_length;
	uint8_t  next_header;
	uint8_t  hop_limit;
	uint8_t  ip_src[16];
	uint8_t  ip_dst[16];
};

class Ip6Header : public VirtualHeader {

private:
	
	struct ip6_header* ip6;

public:
	inline Ip6Header(const uint8_t* ip6Pointer){ip6=(struct ip6_header*)ip6Pointer;};
	void dump(void);
	uint8_t getNextHeader(void);
//	uint16_t getNextHeader(void);
};

#endif // Ip6Header_h
