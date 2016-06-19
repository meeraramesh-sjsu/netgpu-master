/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef hostTimeUtils_h
#define hostTimeUtils_h

static inline unsigned long long tv2usec(struct timeval const arg) {
  return (unsigned long long)arg.tv_sec; //* 1000000 + arg.tv_usec; //Loses precision
}



#endif //hostTimeUtils_h
