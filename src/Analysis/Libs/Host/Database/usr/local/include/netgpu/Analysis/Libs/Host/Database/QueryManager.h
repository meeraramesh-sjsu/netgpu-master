/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef QueryManager_h
#define QueryManager_h

#include <sql.h>
#include <sqlext.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <iostream>

#include "DatabaseManager.h"
#include "../../../../Scheduler/Scheduler.h"
#include "../../../../Util.h"

//Query buffer
#define MAX_QUERY_LENGTH 400*1024 //400Kbytes

using namespace std;

class QueryManager {

 public:
	
	QueryManager();
	~QueryManager(void);
	void scheduleQuery(const char* line);
	void executePackedQuerys(void);
 private:
	DatabaseManager* databaseManager;
	char packedQuery[MAX_QUERY_LENGTH];
	unsigned int cursor;

};

#endif // QueryManager_h

