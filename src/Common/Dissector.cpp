#include "Dissector.h"

#define _DISSECTOR_CHECK_OVERFLOW(a,b) \
		do{ \
			if(hdr!=NULL){ \
				if(a>b) \
				return; \
			} \
		}while(0)

//TotalPacketLength, used to find the payload
int packetLength;
/*L2 dissectors */
void Dissector::dissectEthernet(const uint8_t* packetPointer,unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,void* user){
	//cout<<"call4:dissectethernet";
	uint8_t* onBoardProtocol;	

	//Setting pointer to on board protocol	
	onBoardProtocol = (uint8_t *)packetPointer+ETHERNET_HEADER_SIZE; //TODO: IMprove to save memory erase this pointer, do it inline

	//ACTIONS::VIRTUAL ACTION
	EthernetVirtualAction(packetPointer,totalHeaderLength,hdr,new Ethernet2Header(packetPointer),user);

	//Adding size of this header	
	*totalHeaderLength += ETHERNET_HEADER_SIZE;

	DEBUG2("Ethernet packet");
	switch(ntohs(((struct ether_header*)packetPointer)->type)){

	case ETHER2_TYPE_IP4:
		_DISSECTOR_CHECK_OVERFLOW(*totalHeaderLength+IP4_NO_OPTIONS_HEADER_SIZE,hdr->caplen);
		dissectIp4(onBoardProtocol,totalHeaderLength,hdr,user);
		break;
	case ETHER2_TYPE_IP6:
		//dissectIp6( onBoardProtocol,bufferPointer,totalHeaderLength);
		//break;

	default://NOT SUPPORTED
		DEBUG2("NOT supported");
		break;

	}	

}

/*end of L2 dissectors*/

/*L3 dissectors*/

void Dissector::dissectIp4(const uint8_t* packetPointer,unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,void* user){
	//cout<<"dissectIpv4";
	uint8_t* onBoardProtocol;	

	//Setting pointer to on board protocol	
	onBoardProtocol =(uint8_t *)packetPointer+Ip4Header::calcHeaderLengthInBytes(packetPointer);

	//ACTIONS::VIRTUAL ACTION
	Ip4VirtualAction(packetPointer,totalHeaderLength,hdr,new Ip4Header(packetPointer),user);
	//Only for checksum
	Ip4VirtualActionnew(packetPointer,totalHeaderLength,hdr,new Ip4Header16(packetPointer),user);

	//Adding size of this header	
	*totalHeaderLength +=Ip4Header::calcHeaderLengthInBytes(packetPointer);

	DEBUG2("IP4 packet");
	switch(((struct ip4_header*)packetPointer)->protocol){

	case IP_PROT_ICMP:
		_DISSECTOR_CHECK_OVERFLOW(*totalHeaderLength+ICMP_HEADER_SIZE,hdr->caplen);

		dissectIcmp(onBoardProtocol,totalHeaderLength,hdr,user);
		break;
		//case IP_PROT_IP4: 
		//Tunnel Ip4
		//break;
	case IP_PROT_TCP:
		_DISSECTOR_CHECK_OVERFLOW(*totalHeaderLength+TCP_NO_OPTIONS_HEADER_SIZE,hdr->caplen);
		dissectTcp(onBoardProtocol,totalHeaderLength,hdr,user);
		break;
	case IP_PROT_UDP:
		_DISSECTOR_CHECK_OVERFLOW(*totalHeaderLength+UDP_HEADER_SIZE,hdr->caplen);
		dissectUdp(onBoardProtocol,totalHeaderLength,hdr,user);
		break;
		//case IP_PROT_IP6: 
		//Tunnel Ip6
		//break;
	default://NOT SUPPORTED
		DEBUG2("NOT supported");
		break;

	}
	*totalHeaderLength = Ip4Header::totalPacketLength(packetPointer);
	packetLength = ntohs(((struct ip4_header*)packetPointer)->totalLength) & 0x0000FFFF;
}

/*end of L3 dissectors*/
/*L4 Dissectors*/

void Dissector::dissectTcp(const uint8_t* packetPointer,unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,void* user){
	//cout<<"Dissect TCP";	
	//uint8_t* onBoardProtocol;	

	//ACTIONS::VIRTUAL ACTION
	TcpVirtualAction(packetPointer,totalHeaderLength,hdr,new TcpHeader(packetPointer),user);

	//Adding size of this header	
	*totalHeaderLength +=TcpHeader::calcHeaderLengthInBytes(packetPointer);
	DEBUG2("TCP packet");

	//Setting pointer to Payload
	uint8_t* onBoardProtocol;

	//Setting pointer to on board protocol
	onBoardProtocol =(uint8_t *)packetPointer+TcpHeader::calcHeaderLengthInBytes(packetPointer);

	payLoadRabinKarp(onBoardProtocol);
}

long hashCal(char* key, int  m, int offset) {
	long h = 0;
	for (int j = 0; j < m; j++)
		h = (256 * h + key[offset + j]) % 997;
	return h;
}

bool compare(string a,string b)
{
	return a.length() < b.length();
}

void Dissector::payLoadRabinKarp(const uint8_t* packetPointer) {
	unordered_map<int,int> mapHash;
	vector<string> tmp;
    ifstream myFile ("Pattern/patterns10.cpp", ios::in);
    std::string line;
    std::vector<std::string> myLines;
    while (std::getline(myFile, line))
    {
       tmp.push_back(line);
    }

    sort(tmp.begin(),tmp.end(),compare);

    //Fill the map with patternhashes
    for(int i=0;i<tmp.size();i++)
    {
    	long patHash = hashCal(tmp[i], tmp[i].size(),0);
    	mapHash[patHash] = i;
    }

	int payLoadLength = packetLength - 40;
/*	int m = 5;
	char* pattern = "Hello";
	cout<<"payLoadLength= "<<payLoadLength<<endl;
	/*while(payLoadLength-- > 0)
	{
		cout<<*(char*) packetPointer;
		packetPointer++;
	}
		int RM = 1;
	for (int i = 1; i <= m-1; i++)
		RM = (256 * RM) % 997;*/
	int minLen = tmp[0].length();

	int q = 997;
	int R = 256;
	int hy = 0;
	if(payLoadLength < minLen) return;
	int hy = hashCal((char*)packetPointer,minLen,0);

	if(mapHash.count(hy)>0 && memcmp((char*)packetPointer,tmp[hy].c_str(),minLen) == 0)
		               cout<<"Pattern "<<tmp[hy]<<" exists!"<<endl;


	   for(int i=0;i<payLoadLength;i++) {
		   for(int j=0;j<tmp.size();j++) {
			   if(tmp[j].size()==minLen) break;
	            if(i+tmp[j].size()>= payLoadLength) break;
	                for(int k=minLen;k<tmp[j].size();k++)
	                hy = (hy * 256 + packetPointer[k+i]) % 997;
	                if(mapHash.count(hy)>0 && memcmp((char*)packetPointer,tmp[hy].c_str(),tmp[j].size()) == 0)
	               cout<<"Pattern "<<tmp[hy]<<" exists!"<<endl;
	                minLen = tmp[j].size();
		   }
		}

/*	if (payLoadLength < m) return;
	long txtHash = hashCal((char*)packetPointer, m,0);
	long patHash = hashCal((char*)pattern, m,0);
	cout<<"textHash= "<<txtHash<<"patHash= "<<patHash<<endl;
	// check for match at offset 0
	if ((patHash == txtHash) && memcmp((char*)packetPointer,pattern,5)==0)
		{ cout<<"Pattern Hello exists"<<endl; return;}

	// check for hash match; if hash match, check for exact match
	for (int i = m; i < payLoadLength; i++) {
		// Remove leading digit, add trailing digit, check for match.
		txtHash = (txtHash + q - RM*packetPointer[i-m] % q) % q;
		txtHash = (txtHash*R + packetPointer[i]) % q;

		// match
		int offset = i - m + 1;
		if ((patHash == txtHash) && memcmp((char*) (packetPointer + offset), pattern,m)==0)
			{ cout<<"Pattern Hello exists"<<endl;return;}
	}
*/
}

void Dissector::dissectUdp(const uint8_t* packetPointer, unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,void* user){
	//cout<<"Dissect UDP";
	//uint8_t* onBoardProtocol;	

	//Setting pointer to on board protocol	for L5 analysis

	//ACTIONS::VIRTUAL ACTION
	UdpVirtualAction(packetPointer,totalHeaderLength,hdr, new UdpHeader(packetPointer),user);

	//Adding size of this header	
	*totalHeaderLength +=UDP_HEADER_SIZE;
	DEBUG2("UDP packet");


	//HERE should come L5 switch
	//TODO: L5 support
}

void Dissector::dissectIcmp(const uint8_t* packetPointer, unsigned int* totalHeaderLength,const struct pcap_pkthdr* hdr,void* user){
	//cout<<"dissectICMP";
	//uint8_t* onBoardProtocol;	

	//ACTIONS::VIRTUAL ACTION
	IcmpVirtualAction(packetPointer,totalHeaderLength,hdr,new IcmpHeader(packetPointer),user);

	//Adding size of this header	
	*totalHeaderLength +=ICMP_HEADER_SIZE;
	DEBUG2("ICMP packet");

	//HERE should come L5 switch
	//TODO: L5 support
}
/*end of L4 dissectors*/




//Main dissector method
unsigned int Dissector::dissect(const uint8_t* packetPointer,const struct pcap_pkthdr* hdr,const int deviceDataLinkInfo,void* user){
	//cout<<"Starting disscetion";
	unsigned int totalHeaderLength = 0;

	//CHECKING LINKLAYER and dissect rest of packet

	DEBUG2("Starting dissection...");
	switch(deviceDataLinkInfo){

	case DLT_EN10MB: //Ethernet
		dissectEthernet(packetPointer,&totalHeaderLength,hdr,user);
		break;
		/*		case DLT_IEEE802: //Token ring
				break;
		case DLT_PPP:	//PPP
				break;
		case DLT_FDDI: //FDDI
				break;
		case DLT_ATM_RFC1483: //ATM_RFC1483
					break;
		case DLT_RAW: //Raw IP, starts with IP
				break;
		case DLT_PPP_ETHER: //PPPoE
				break;
		case DLT_IEEE802_11: //802.11
				break;
		case DLT_FRELAY: //Frame Relay
				break;
		case DLT_IP_OVER_FC: //Ip over Fiber Channel
				break;		
		case DLT_IEEE802_11_RADIO: //802.11 RADIO
				break;
		 */
	default: //Rest of LLtypes, check documentation and add code
		DEBUG2("Not implemented yet");
		return -1;
		break;
	}
	DEBUG2("End of dissection\n");
	//ACTIONS::VIRTUAL ACTION
	EndOfDissectionVirtualAction(&totalHeaderLength,hdr,user);


	return totalHeaderLength;
}

