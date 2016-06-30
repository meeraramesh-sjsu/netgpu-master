#include <stdlib.h>
#include <stdio.h>
#include <arpa/inet.h>
#include <stdbool.h>

bool IsPrivateAddress(uint32_t ip)
{
    uint8_t b1, b2, b3, b4;
    b1 = (uint8_t)(ip >> 24);
    b2 = (uint8_t)((ip >> 16) & 0x0ff);
    b3 = (uint8_t)((ip >> 8) & 0x0ff);
    b3 = (uint8_t)(ip & 0x0ff);

    // 10.x.y.z
    if (b1 == 10)
        return true;

    // 172.16.0.0 - 172.31.255.255
    if ((b1 == 172) && (b2 >= 16) && (b2 <= 31))
        return true;

    // 192.168.0.0 - 192.168.255.255
    if ((b1 == 192) && (b2 == 168))
        return true;

    return false;
}

int main(int argc, char **argv) {
  struct in_addr pin;

  if (argc != 2) {
    printf("%d\n",argc);
    fprintf(stderr,"Usage: inet_aton dotted-quad\n");
    exit(1);
  }

  int valid = inet_aton(argv[1],&pin);

  if (!valid) {
    fprintf(stderr,"inet_aton could not parse \"%s\"\n",argv[1]);
    exit(1);
  }

  /* pin.s_addr is in network by order, convert to host byte order */
  unsigned int address = ntohl(pin.s_addr);

  unsigned int octet1 = (0xff000000 & address)>>24;
  unsigned int octet2 = (0x00ff0000 & address)>>16;
  unsigned int octet3 = (0x0000ff00 & address)>>8;
  unsigned int octet4 = 0x000000ff & address;

  printf("ping %u.%u.%u.%u\n",octet1,octet2,octet3,octet4);
  printf("ping %u.%u.%u\n",octet1,octet2,(octet3<<8) + octet4);
  printf("ping %u.%u\n",octet1,(octet2<<16) + (octet3<<8) + octet4);
  printf("ping %u\n",address);
}
