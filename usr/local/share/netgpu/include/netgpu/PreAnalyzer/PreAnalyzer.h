/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef PreAnalyzer_h
#define PreAnalyzer_h

#include <iostream>
#include <arpa/inet.h>

#include "../Util.h"

#include "../Common/PacketBuffer.h"
#include "PreAnalyzerDissector.h"


using namespace std;

class PreAnalyzer {


public:
	void preAnalyze(PacketBuffer* bufferPointer);
	
private:
	PreAnalyzerDissector preAnalyzerDissector;
	void dumpBufferStats(PacketBuffer* bufferPointer);

};
#endif // PreAnalyzer_h
