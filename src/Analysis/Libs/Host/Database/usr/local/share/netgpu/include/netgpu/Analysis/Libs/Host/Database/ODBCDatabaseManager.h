/*

Copyright 2009 Marc Suñe Clos, Isaac Gelado

This file is part of the NetGPU framework.

The NetGPU framework is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The NetGPU framework is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

*/

#ifndef ODBCDatabaseManager_h
#define ODBCDatabaseManager_h

#include <sql.h>
#include <sqlext.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <signal.h>
#include <sys/time.h>
#include <unistd.h>

#include "DatabaseManager.h"
#include "../../../../Util.h"

#define MAX_OUTPUT_LENGTH 2048


#if 1
#define ODBC_ASSERT(action)  \
	 do{ \
    SQLRETURN	 ret;\
    ret = action;\
    if(!SQL_SUCCEEDED(ret)){\
	    SQLINTEGER	 i = 0;\
	    SQLINTEGER	 native;\
	    SQLCHAR	 state[ 7 ];\
	    SQLCHAR	 text[256];\
	    SQLSMALLINT	 len;\
	    i=0;\
	    do{\
        	ret = SQLGetDiagRec(SQL_HANDLE_STMT,stmt , ++i, state, &native, text,sizeof(text), &len );\
	        if(SQL_SUCCEEDED(ret))\
			printf("%s:%d:%d:%s\n", state, i, native, text);\
	    }\
	    while( ret == SQL_SUCCESS );\
	}\
    }while(0)
#endif

#define ODBC_SOURCE_NAME Netgpu
#define IDLE_DB_TIMEOUT 15 //10 minuts (seconds)
//#define IDLE_DB_TIMEOUT 600 //10 minuts (seconds)

using namespace std;

class ODBCDatabaseManager: public DatabaseManager{

 public:
	ODBCDatabaseManager();
	~ODBCDatabaseManager();
	void executeQuery(SQLCHAR* query, unsigned int length);
	DatabaseManager* getManager(void);

#ifndef DBM_LOCAL_DB 
	void refreshTimestamp(void);
	void handlerDisconnect(void){ cerr<<"Handler disconnect"<<endl; disconnect();}
#endif

 private:
	void connect(void);
	void disconnect(void);
 	SQLHENV env;
  	SQLHDBC dbc;
  	SQLHSTMT stmt; /* Statement */
  	SQLRETURN ret; /* ODBC API return status */
  	SQLSMALLINT outstrlen;
  	SQLCHAR outstr[MAX_OUTPUT_LENGTH]; /* Ouput String*/

};

#endif // ODBCDatabaseManager_h

