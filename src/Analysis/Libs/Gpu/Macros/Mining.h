/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef Mining_h
#define Mining_h

/*Import protocols (should be loaded at this time although)*/
#include "../Protocols.h"

/****** MINING MACROS & functions ******/

/* Protocol MACROS */

// ETHERNET
 
#define ETHERNET_HEADER (*((ETHERNET_HEADER_TYPENAME*) GET_HEADER_POINTER(2) ))
#define ETHERNET_HEADER_LEVEL(level) GET_HEADER_POINTER(level)

#define IS_ETHERNET_LEVEL(level) \
	( IS_HEADER_TYPE_ETHERNET(&(PACKET->headers), level) )

#define IS_ETHERNET() \
	IS_ETHERNET_LEVEL(2)

//End of ETHERNET
	

// IP4
 
#define IP4_HEADER (*((IP4_HEADER_TYPENAME*) GET_HEADER_POINTER(3) ))
//#define IP4_HEADER_LEVEL(level) GET_HEADER_POINTER(level)


#define IS_IP4_LEVEL(level) \
	( IS_HEADER_TYPE_IP4(&(PACKET->headers), level) )

#define IS_IP4() \
	IS_IP4_LEVEL(3)

#define IP4(a,b,c,d)\
		((uint32_t)((a<<24)|(b<<16)|(c<<8)|d))

#define IP4_MASK(a,b,c,d) IP4(a,b,c,d)
#define IP4_NETID(a,b)\
		(a & ((uint32_t)(0xFFFFFFFF<<(32-b))))

//End of IP4

// IP6 
#define IP6_HEADER (*((IP6_HEADER_TYPENAME*) GET_HEADER_POINTER(3) ))
#define IP6_HEADER_LEVEL(level) GET_HEADER_POINTER(level)


#define IS_IP6_LEVEL(level) \
	( IS_HEADER_TYPE_IP6(&(PACKET->headers), level) )

#define IS_IP6() \
	IS_IP6_LEVEL(3)

#define IP6(a,b,c,d)\
		((uint32_t)((a<<24)+(b<<16)+(c<<8)+d))

#define IP6_MASK(a,b,c,d) IP4(a,b,c,d)

//End of IP6


// TCP  
#define TCP_HEADER (*((TCP_HEADER_TYPENAME*) GET_HEADER_POINTER(4) ))
#define TCP_HEADER_LEVEL(level) GET_HEADER_POINTER(level)


#define IS_TCP_LEVEL(level) \
	( IS_HEADER_TYPE_TCP(&(PACKET->headers), level) )

#define IS_TCP() \
	IS_TCP_LEVEL(4)

//End of TCP

// UDP  
#define UDP_HEADER (*((UDP_HEADER_TYPENAME*) GET_HEADER_POINTER(4) ))
#define UDP_HEADER_LEVEL(level) GET_HEADER_POINTER(level)


#define IS_UDP_LEVEL(level) \
	( IS_HEADER_TYPE_UDP(&(PACKET->headers), level) )

#define IS_UDP() \
	IS_UDP_LEVEL(4)

//End of ICMP


// ICMP  
#define ICMP_HEADER (*((ICMP_HEADER_TYPENAME*) GET_HEADER_POINTER(4) ))
#define ICMP_HEADER_LEVEL(level) GET_HEADER_POINTER(level)


#define IS_ICMP_LEVEL(level) \
	( IS_HEADER_TYPE_ICMP(&(PACKET->headers), level) )

#define IS_ICMP() \
	IS_ICMP_LEVEL(4)

//End of ICMP



/*End of file */

#endif //Mining_h

