/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef VirtualHeader_h
#define VirtualHeader_h

/*General header MACROS*/
#define INSERT_HEADER(headers, level, offseT,protocol) do{ \
		(headers)->proto[level] = protocol; \
                (headers)->offset[level] = offseT; \
        }while(0)

#define IS_HEADER_TYPE(headers, level,protocol) (headers)->proto[level] == protocol



class VirtualHeader { //Abstract base class

private:

protected:

public:
	virtual void dump(void)=0; //pure virtual

};

#endif // VirtualHeader_h
