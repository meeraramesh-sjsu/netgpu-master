#include "Ip4Header.h"

void Ip4Header::dump(void){

	cout <<"IP4 Header ; ";

	//fprintf(stdout,"Src address: %s ",inet_ntoa(*(struct in_addr*)&ip4->ip_src));
	//fprintf(stdout,"Dst address: %s \n\n",inet_ntoa(*(struct in_addr*)&ip4->ip_dst));

	//Check if IP address is in private address range
	int octet4 = (ip4->ip_src & 0xff000000) >> 24;
	int octet3 = (ip4->ip_src & 0x00ff0000) >> 16;
	int octet2 = (ip4->ip_src & 0x0000ff00) >> 8;
	int octet1 = (ip4->ip_src & 0x000000ff);

	int octet4Dst = (ip4->ip_dst & 0xff000000) >> 24;
	int octet3Dst = (ip4->ip_dst & 0x00ff0000) >> 16;
	int octet2Dst = (ip4->ip_dst & 0x0000ff00) >> 8;
	int octet1Dst = (ip4->ip_dst & 0x000000ff);

	string ipsrc =  inet_ntoa(*(struct in_addr*)&ip4->ip_src);
	string ipdst =  inet_ntoa(*(struct in_addr*)&ip4->ip_dst);
	//cout<<"Src octets"<<octet1<<" "<<octet2<<" "<<octet3<<" "<<octet4<<" "<<endl;
	//cout<<"Dst octets"<<octet1Dst<<" "<<octet2Dst<<" "<<octet3Dst<<" "<<octet4Dst<<" "<<endl;

	if (octet1 == 10 || octet1Dst==10)
	{
		cout<<"Packet src IP " << ipsrc << "or Dst IP address" << ipdst << "is in private address range" <<endl;
	}

	// 172.16.0.0 - 172.31.255.255
	else  if ((octet1 == 172 || octet1Dst==172) && (octet2 >= 16 || octet2Dst >=16 ) && (octet2 <= 31 || octet2Dst<=31))
	{
		cout<<"Packet src IP " << ipsrc << "or Dst IP address" << ipdst << "is in private address range" <<endl;
	}
	// 192.168.0.0 - 192.168.255.255
	else  if ((octet1 == 192 || octet1Dst == 192) && (octet2 == 168 || octet2Dst == 168))
	{
		cout<<"Packet src IP " << ipsrc << "or Dst IP address" << ipdst << "is in private address range" <<endl;
	}

	//Check if IP checksum is malicious
	int headerVerTos = ((ip4->headerLength & 0x0000000F) << 12) | ((ip4->version & 0x0000000F) << 8) | (ip4->tos & 0x000000FF);
	int ttlprotocol = ((ip4->ttl & 0x000000FF)<<8) | (ip4->protocol & 0x000000FF);
	int srcFirstHalf = (ip4->ip_src & 0xFFFF0000)>>16;
	int srcSecHalf = (ip4->ip_src & 0x0000FFFF)>>16;
	int dstFirstHalf = (ip4->ip_src & 0xFFFF0000)>>16;
	int dstSecHalf = (ip4->ip_src & 0x0000FFFF)>>16;
	cout<<"Checksum Calculation "<<endl;
	cout<<headerVerTos<<" "<<ttlprotocol<<" "<<srcFirstHalf<<" "<<srcSecHalf<<" "<<dstFirstHalf<<" "<<dstSecHalf<<" ";
	cout<<" "<<ip4->totalLength<<" "<<ip4->identification<<" "<<ip4->flagsAndOffset<<" "<<ip4->checksum<<endl;
	int result = headerVerTos + ttlprotocol
			+ srcFirstHalf + srcSecHalf
			+ dstFirstHalf + dstSecHalf
			+ ip4->totalLength + ip4->identification
			+ ip4->flagsAndOffset + ip4->checksum;
	unsigned int sum = ~(result>>16 + (result & 0xFFFF));
	if(sum!=-1) cout<<"The checksum is malicious"<<endl;

	//Checking if it is a broadcast packet
	if( (ip4->protocol==6 && octet4Dst == 0) || octet4Dst == 255)
	 cout<<"TCP Broadcast packets are not allowed!"<<endl;
}

uint8_t Ip4Header::getHeaderLength(void){
	return ip4->headerLength;
}
uint32_t Ip4Header::getHeaderLengthInBytes(void){
	return (uint32_t)ip4->headerLength*4;
}
uint8_t Ip4Header::getProtocol(void){

	return ip4->protocol; 
}

uint32_t Ip4Header::calcHeaderLengthInBytes(const uint8_t * ipPointer){
	return (uint32_t)((struct ip4_header *)ipPointer)->headerLength*4;
}
uint32_t Ip4Header::totalPacketLength(const uint8_t * ipPointer)
{
	return (uint32_t) ntohs(((struct ip4_header *)ipPointer)->totalLength) + 14;
}
