/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef Ethernet2Header_h
#define Ethernet2Header_h

#include <iostream>
#include <inttypes.h>
#include <arpa/inet.h>

#include "../VirtualHeader.h"

#define ETHERNET_HEADER_SIZE 14
#define HEADER_ETHERNET_HEXVALUE 0x0001

/*MACROS HEADERS */
#define ETHERNET_HEADER_TYPENAME struct ether_header

#define INSERT_HEADER_ETHERNET(headers, level, offseT) INSERT_HEADER(headers, level, offseT,HEADER_ETHERNET_HEXVALUE)

#define IS_HEADER_TYPE_ETHERNET(headers, level) IS_HEADER_TYPE(headers, level,HEADER_ETHERNET_HEXVALUE)






/*TYPES (onBoard) */
#define ETHER2_TYPE_IP4 0x0800 	//Internet Protocol, Version 4 (IPv4)
#define ETHER2_TYPE_ARP	0x0806 	//Address Resolution Protocol (ARP)
#define ETHER2_TYPE_WOL	0x0842 	//Wake-on-LAN
#define ETHER2_TYPE_RARP	0x8035 	//Reverse Address Resolution Protocol (RARP)
/*	0x809B 	AppleTalk (Ethertalk)
	0x80F3 	AppleTalk Address Resolution Protocol (AARP)
	0x8100 	VLAN-tagged frame (IEEE 802.1Q)
	0x8137 	Novell IPX (alt)
	0x8138 	Novell
*/
#define ETHER2_TYPE_IP6 0x86DD 	//Internet Protocol, Version 6 (IPv6)
/*	0x8808 	MAC Control
	0x8819 	CobraNet
	0x8847 	MPLS unicast
	0x8848 	MPLS multicast
	0x8863 	PPPoE Discovery Stage
	0x8864 	PPPoE Session Stage
	0x888E 	EAP over LAN (IEEE 802.1X)
	0x889A 	HyperSCSI (SCSI over Ethernet)
	0x88A2 	ATA over Ethernet
	0x88A4 	EtherCAT Protocol
	0x88A8 	Provider Bridging (IEEE 802.1ad)
	0x88B5 	AVB Transport Protocol (AVBTP)
	0x88CD 	SERCOS-III
	0x88D8 	Circuit Emulation Services over Ethernet (MEF-8)
	0x88E1 	HomePlug
	0x88E5 	MAC security (IEEE 802.1AE)
	0x8906 	Fibre Channel over Ethernet
	0x8914 	FCoE initialization protocol
	0x9100 	Q-in-Q
	0xCAFE 	Veritas Low Latency Transport (LLT)

*/

/*End of TYPES */

using namespace std;


/*Header */

struct ether_header{
	uint8_t src[6];
	uint8_t dst[6];
	uint16_t type;
};

class Ethernet2Header : public VirtualHeader {

private:
	struct ether_header* ether;
public:

	inline Ethernet2Header(const uint8_t* ethPointer){ether=(struct ether_header*)ethPointer;};
	uint16_t getType(void);
	void dump(void);	
};

#endif // Ethernet2Header_h
