#include "Ip4Header.h"
#include <string>
#include <iostream>
#include <arpa/inet.h>
using namespace std;
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

	//Checking if it is a broadcast packet
	if( (ip4->protocol==6 && octet4Dst == 0) || octet4Dst == 255)
	 cout<<"TCP Broadcast packets are not allowed!"<<endl;
}

void Ip4Header16::dump(void){

	cout<<" "<<ntohs(ip4->headerVertos) <<" "<< ntohs(ip4->ttlprotocol)
				<<" "<<  ntohs(ip4->ip_srcFirstHalf) <<" "<<  ntohs(ip4->ip_srcSecHalf)
				<<" "<<  ntohs(ip4->ip_dstFirstHalf) <<" "<<  ntohs(ip4->ip_dstSecHalf)
				<<" "<<  ntohs(ip4->totalLength) <<" "<<  ntohs(ip4->identification)
				<<" "<<  ntohs(ip4->flagsAndOffset) <<" "<<  ntohs(ip4->checksum)<<endl;

	cout <<"IP4 Header checksum computation";
	int result =  ntohs(ip4->headerVertos) + ntohs(ip4->ttlprotocol)
					+ ntohs(ip4->ip_srcFirstHalf)+ ntohs(ip4->ip_srcSecHalf)
						+ ntohs(ip4->ip_dstFirstHalf) + ntohs(ip4->ip_dstSecHalf)
						+  ntohs(ip4->totalLength) + ntohs(ip4->identification)
						+  ntohs(ip4->flagsAndOffset) +  ntohs(ip4->checksum);
	cout<<"Result= "<<hex<<result<<endl;
	cout<<hex<<(result>>16)<<" "<<hex<<(result & 0xFFFF)<<" "<<hex<<(result>>16 + (result & 0xFFFF))<<endl;
	unsigned int sum = ~(result>>16 + (result & 0xFFFF));

	cout<<"checksum= "<<sum<<endl;
	if(sum!=-1) cout<<"The checksum is malicious"<<endl;
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
