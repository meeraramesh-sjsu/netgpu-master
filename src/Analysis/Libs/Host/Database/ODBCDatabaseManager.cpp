#include "ODBCDatabaseManager.h"

//#define DBM_LOCAL_DB

/*
	SIGALRM handler
	used to disconnect from DataBase if connection is not local (unix socket or loopback)
	If it's a local connection, cpp FLAG DBM_LOCAL_DB should be set to enhance performance.

*/

#ifndef DBM_LOCAL_DB 
	//Global vars
	struct timeval lastOperationTimestamp;	
	pthread_mutex_t mutex;
	ODBCDatabaseManager *odbcPointer; 
	//Handler
	void timeoutHandler(int sigNum){
	
		struct timeval actualTime;
//		cerr<<"Handler "<<getpid()<<endl;
		//Lock
		pthread_mutex_lock(&mutex);

		gettimeofday(&actualTime,NULL);
		if( actualTime.tv_sec >= (lastOperationTimestamp.tv_sec+IDLE_DB_TIMEOUT)){
//			cerr<<"Handler: vaig a desconectar"<<endl;
			odbcPointer->handlerDisconnect();
		}else{
			//SIGALRM from another(user) alarm, sleep etc.. call
			//Set new alarm with right time

//			cerr<<"NO i resetejo a:"<<(IDLE_DB_TIMEOUT-(actualTime.tv_sec-lastOperationTimestamp.tv_sec))<<endl;

			alarm((IDLE_DB_TIMEOUT-(actualTime.tv_sec-lastOperationTimestamp.tv_sec)));
		}				
		pthread_mutex_unlock(&mutex);
	}
#endif

ODBCDatabaseManager::ODBCDatabaseManager(void){

#ifndef DBM_LOCAL_DB 
	//Init lock
	pthread_mutex_init(&mutex,NULL);
	odbcPointer = this;
	//handler programing
	struct sigaction action;
	memset(&action,0,sizeof(struct sigaction));
	action.sa_handler = timeoutHandler;
	sigaction(SIGALRM,&action,NULL);	
#endif

	dbc =NULL;

}

ODBCDatabaseManager::~ODBCDatabaseManager(void){
	
#ifndef DBM_LOCAL_DB 
	pthread_mutex_lock(&mutex);
	//Disable ALARM
	sigset_t set;
	sigemptyset(&set);
	sigaddset(&set,SIGALRM);
	sigprocmask(SIG_BLOCK,&set, NULL);
#endif
//	cerr<<"destructor"<<endl;
	disconnect();

	/* Allocate an environment handle */
  	SQLFreeHandle(SQL_HANDLE_ENV,env);

  	/* Allocate a connection handle */
  	SQLFreeHandle(SQL_HANDLE_DBC,dbc);
	
#ifndef DBM_LOCAL_DB 
	pthread_mutex_unlock(&mutex);
	pthread_mutex_destroy(&mutex);	
#endif
}


#ifndef DBM_LOCAL_DB 
void ODBCDatabaseManager::refreshTimestamp(void){
	
	//Lock
	pthread_mutex_lock(&mutex);
	
	gettimeofday(&lastOperationTimestamp,NULL);

	//Unlock	
	pthread_mutex_unlock(&mutex);
	alarm(IDLE_DB_TIMEOUT);
}
#endif

DatabaseManager* ODBCDatabaseManager::getManager(void){
	return this;	
}


void ODBCDatabaseManager::connect(void){

	char string[256];	

//	cerr<<"connect"<<endl;

	if(dbc == NULL){
		/* Allocate an environment handle */
	  	SQLAllocHandle(SQL_HANDLE_ENV, SQL_NULL_HANDLE, &env);

  		/* We want ODBC 3 support */
	  	SQLSetEnvAttr(env, SQL_ATTR_ODBC_VERSION, (void *) SQL_OV_ODBC3, 0);

  		/* Allocate a connection handle */
	  	SQLAllocHandle(SQL_HANDLE_DBC, env, &dbc);
//		cerr<<"acabo constructor"<<endl;
	
		//Prepare connection string	
  		snprintf(string, 256,"DSN="STR(ODBC_SOURCE_NAME)";");
//		cerr<<string<<endl;
		//Connect
    		ret = SQLDriverConnect(dbc, NULL, (SQLCHAR*)string,strlen(string) ,outstr, sizeof(outstr), &outstrlen,SQL_DRIVER_COMPLETE);

		//Track error
    		if(!SQL_SUCCEEDED(ret)){
    			SQLINTEGER i = 0;
	    		SQLINTEGER native;
    			SQLCHAR	state[ 7 ];
    			SQLCHAR	text[256];
	    		SQLSMALLINT len;
			i=0;
	    		do{
        			ret = SQLGetDiagRec(SQL_HANDLE_DBC,dbc , ++i, state, &native, text,sizeof(text), &len );
		        	if (SQL_SUCCEEDED(ret))
        		    		fprintf(stderr,"%s:%d:%d:%s\n", state, i, native, text);
	    		}while(ret == SQL_SUCCESS);
			WARN("Database connection not available.\nAll Database actions will be ommitted\n");
			dbc = NULL;
       		}else{	
		
			//Allocate stmt handle
			ODBC_ASSERT(SQLAllocHandle(SQL_HANDLE_STMT,dbc,&stmt));

			//SetAlarmHandler and alarm timeout
#ifndef DBM_LOCAL_DB
			refreshTimestamp();	
#endif
		}	
	}
}

void ODBCDatabaseManager::disconnect(void){
//	cerr<<"disconnect"<<endl;
	if(dbc!= NULL){
		//Free stmt handler
		ODBC_ASSERT(SQLFreeHandle(SQL_HANDLE_STMT,stmt));
		SQLDisconnect(dbc);
		dbc = NULL;
		stmt = NULL;
	}
}


//No return parameters query only!
void ODBCDatabaseManager::executeQuery(SQLCHAR* query, unsigned int length){

	//Line MUST include ; at the end of SQL statement
	size_t query_length = strlen((const char*)query);

	//cerr<<"executeQuery"<<endl;

	if(!dbc){
		
		//cerr<<"No tinc connexió"<<endl;
		connect();
	}
	if(dbc){	
		//cerr<<"abans execDirect"<<endl;
		ODBC_ASSERT(SQLExecDirect(stmt,query, query_length));
		//cerr<<"despres execDirect"<<endl;
#ifndef DBM_LOCAL_DB
		refreshTimestamp();	
#endif
	}else{
		WARN("Connection could not be performed.Missed results");
	}

}

