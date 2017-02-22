#include "Dissector.h"
#include "AhoCorasick.h"
#include "WuManber.h"
#include <omp.h>
#define _DISSECTOR_CHECK_OVERFLOW(a,b) \
		do{ \
			if(hdr!=NULL){ \
				if(a>b) \
				return; \
			} \
		}while(0)

struct length {
	bool operator() ( const string& a, const string& b )
	{
		return a.size() < b.size();
	}
};

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
	char* onBoardProtocol;

	//Setting pointer to on board protocol
	onBoardProtocol =(char *)packetPointer+TcpHeader::calcHeaderLengthInBytes(packetPointer);

	//payLoadRabinKarp(onBoardProtocol);
	string packet(onBoardProtocol);

	vector<string> tmp;
	ifstream myFile ("/home/meera/gpudir/netgpu-master/src/Common/Pattern/patterns10.cpp", ios::in);
	std::string line;

	if (myFile.is_open()) {
	#pragma omp parallel for
		for(;std::getline(myFile, line);)
		{
			tmp.push_back(line);
		}
	}
	else cout << "Unable to open file";
	cout<<"added file contents to vector"<<endl;

	//Aho-Corasick
	//searchWords(tmp,tmp.size(),packet);

	//WuManber
	int m = (*min_element(tmp.begin(),tmp.end(),length())).size();
	int shiftsize = wu_determine_shiftsize(256);
	int p_size = tmp.size();
	int *SHIFT = (int *) malloc(shiftsize * sizeof(int)); //shiftsize = maximum hash value of the B-size suffix of the patterns

	//The hash value of the B'-character prefix of a pattern
	int *PREFIX_value = (int *) malloc(shiftsize * p_size * sizeof(int)); //The possible prefixes for the hash values.

	//The pattern number
	int *PREFIX_index = (int *) malloc(shiftsize * p_size * sizeof(int));

	//How many patterns with the same prefix hash exist
	int *PREFIX_size = (int *) malloc(shiftsize * sizeof(int));

	#pragma omp parallel for
	for (int i = 0; i < shiftsize; i++) {

		//*( *SHIFT + i ) = m - B + 1;
		SHIFT[i] = m - 3 + 1;
		PREFIX_size[i] = 0;
	}

	preproc_wu(tmp,m,3,SHIFT,PREFIX_value,PREFIX_index,PREFIX_size);

//	preproc_wu(tmp,  m, 3,
	//		int *SHIFT, int *PREFIX_value, int *PREFIX_index, int *PREFIX_size);

	search_wu(tmp,m,packet,packet.size(),SHIFT,PREFIX_value,PREFIX_index,PREFIX_size);

}

long hashCal(const char* key, int  m, int offset) {
	long h = 0;
	for (int j = 0; j < m; j++)
	{
		h = (256 * h + key[offset + j]) % 997;
	}
	return h;
}

bool compare(string a,string b)
{
	return a.size() < b.size();
}

// This function finds all occurrences of all array words
// in text.
void Dissector::searchWords(vector<string> arr, int k, string text)
{
	// Preprocess patterns.
	// Build machine with goto, failure and output functions
	buildMatchingMachine(arr, k);
	cout<<"Completed building machine"<<endl;
	// Initialize current state
	int currentState = 0;

	// Traverse the text through the nuilt machine to find
	// all occurrences of words in arr[]
	for (int i = 0; i < text.size(); ++i)
	{
		currentState = findNextState(currentState, text[i]);

		// If match not found, move to next state
		if (out[currentState] == 0)
			continue;

		// Match found, print all matching words of arr[]
		// using output function.
		for (int j = 0; j < k; ++j)
		{
			if (out[currentState] & (1 << j))
			{
				cout << "Word " << arr[j] << " appears from "
						<< i - arr[j].size() + 1 << " to " << i << endl;
			}
		}
	}
}


void Dissector::payLoadRabinKarp(char* packetPointer) {
	vector<int> mapHash(997,-1);
	vector<string> tmp;
	set<int> setlen;
	ifstream myFile ("/home/meera/gpudir/netgpu-master/src/Common/Pattern/patterns10.cpp", ios::in);
	std::string line;

	if (myFile.is_open()) {
		while (std::getline(myFile, line))
		{
			tmp.push_back(line);
		}
	}
	else cout << "Unable to open file";

	for(int i=0;i<tmp.size();i++)
		setlen.insert(tmp[i].length());

	sort(tmp.begin(),tmp.end(),compare);

	//Fill the map with pattern hashes
	for(int i=0;i<tmp.size();i++)
	{
		long patHash = hashCal(tmp[i].c_str(), tmp[i].size(),0);
		mapHash[patHash] = i;
	}

	int payLoadLength = packetLength - 40;
	/*int m = 5;
	char* pattern = "Hello";
	cout<<"payLoadLength= "<<payLoadLength<<endl;
	/*while(payLoadLength-- > 0)
	{
		cout<<*(char*) packetPointer;
		packetPointer++;
	}*/


	int minLen = tmp[0].length();

	int q = 997;
	int R = 256;

	/*long hy = 0;

	if(payLoadLength < minLen) return;

	for (int j = 0; j < minLen; j++)
	{
		hy = (256 * hy + packetPointer[j]) % 997;
	}

	if(mapHash[hy]>0 && memcmp((char*)packetPointer,tmp[mapHash[hy]].c_str(),minLen) == 0)
		cout<<"Pattern "<<tmp[hy]<<" exists!"<<endl;

	for(int i=0;i<payLoadLength;i++) {
		bool found = false;
		for(int j=0;j<tmp.size();j++) {
			if(found) continue;
			if(i+tmp[j].size()> payLoadLength) break;
			for(int k=minLen;k<tmp[j].size();k++)
				hy = (hy * 256 + packetPointer[k+i]) % 997;
			int patIndex = mapHash[hy];
			if(patIndex>=0)
			{
				if(memcmp((packetPointer+i),tmp[patIndex].c_str(),tmp[patIndex].size()) == 0) cout<<"Pattern "<<tmp[patIndex]<<" exists!"<<endl;
				minLen = tmp[patIndex].size();
				found = true;
			}
		}
		hy = 0;
		minLen = 0;
	}*/
	//More optimal Rabin karp algorithm
	//Starting from 0, move for every pattern length, computing the hash values
	//Time complexity O(N* number of pattern lengths)
	for(auto it= setlen.begin();it!=setlen.end();it++)
	{
		int m = *it;
		int RM = 1;
		for (int i = 1; i <= m-1; i++)
			RM = (256 * RM) % 997;

		if (m > payLoadLength) break;
		int txtHash = hashCal((char*)packetPointer, m,0);

		// check for match at offset 0
		if ((mapHash[txtHash]>0) && memcmp((char*)packetPointer,tmp[mapHash[txtHash]].c_str(),m)==0)
		{ cout<<"Virus Pattern " << tmp[mapHash[txtHash]] <<" exists"<<endl; break;}

		// check for hash match; if hash match, check for exact match
		for (int j = m; j < payLoadLength; j++) {
			// Remove leading digit, add trailing digit, check for match.
			txtHash = (txtHash + q - RM*packetPointer[j-m] % q) % q;
			txtHash = (txtHash*R + packetPointer[j]) % q;

			// match
			int offset = j - m + 1;
			if ((mapHash[txtHash]>0) && memcmp((char*) (packetPointer + offset), tmp[mapHash[txtHash]].c_str(),m)==0)
			{ cout<<"Virus Pattern " << tmp[mapHash[txtHash]] <<" exists"<<endl; break;}
		}
	}

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

