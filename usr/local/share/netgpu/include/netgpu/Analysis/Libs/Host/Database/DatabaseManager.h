/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef DatabaseManager_h
#define DatabaseManager_h

#include <sql.h>
#include <sqlext.h>

/* DataBaseManager abstract class */

class DatabaseManager {

 public:
	virtual void executeQuery(SQLCHAR* query, unsigned int length)=0;
	virtual DatabaseManager* getManager(void)=0;

 private:
	
	virtual void connect(void)=0;
	virtual void disconnect(void)=0;
};

#endif // DatabaseManager_h

