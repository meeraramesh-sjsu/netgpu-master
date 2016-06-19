#!/bin/sh

#Set filename
FILE=.syncblocks_counters.ppph

#Erases content (and avoids cpp error on inclusion)
echo -n > $FILE 

#include PPP MACRO
echo \#ifndef __PPP__ >>$FILE 
echo     \#define __PPP__ >>$FILE
echo \#endif >>$FILE

echo -n \#define\ __SYNCBLOCKS_COUNTER\   >> $FILE 
#cpp -I $PWD -D __CUDACC__ -D DONT_EXPAND_SYNCBLOCKS $1 -nostdinc  2>/dev/null | grep SYNCBLOCKS\(\)\; | wc -l >> $FILE
cpp -I $PWD -D __CUDACC__ -D DONT_EXPAND_SYNCBLOCKS $1  2>/dev/null | grep SYNCBLOCKS\(\)\; | wc -l >> $FILE

echo >> $FILE 

echo -n \#define\ __SYNCBLOCKS_PRECODED_COUNTER\   >> $FILE
#cpp -I $PWD -D __CUDACC__ -D DONT_EXPAND_SYNCBLOCKS $1 -nostdinc 2>/dev/null | grep SYNCBLOCKS_PRECODED\(\)\; | wc -l >> $FILE
cpp -I $PWD -D __CUDACC__ -D DONT_EXPAND_SYNCBLOCKS $1 2>/dev/null | grep SYNCBLOCKS_PRECODED\(\)\; | wc -l >> $FILE

