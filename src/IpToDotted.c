#include <stdlib.h>
#include <stdio.h>
#include <arpa/inet.h>

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
