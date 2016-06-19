/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef Modules_Util_h
#define Modules_Util_h

#define SET_EXTENDED_PARAMETER(position,value)\
	if(POS == 0)\
		state.inputs.GPU_extendedParameters[position] = value

#define GET_EXTENDED_PARAMETER(position)\
	state.inputs.GPU_extendedParameters[position]

#endif //Modules_Util.h

