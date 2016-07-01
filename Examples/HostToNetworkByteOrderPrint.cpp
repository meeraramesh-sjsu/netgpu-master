#include <stdio.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <iostream>
using namespace std;
int main ()
{
uint32_t network_byte_order;
uint32_t some_long = 3232236034;
network_byte_order = htonl(some_long);
struct in_addr in;
in.s_addr = network_byte_order;
cout<<inet_ntoa(in);
return 0;
}
