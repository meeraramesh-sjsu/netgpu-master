#include "QueryManager.h"

QueryManager::QueryManager(void){

	databaseManager = NULL;
	cursor = 0;	
}

QueryManager::~QueryManager(void){

}

void QueryManager::scheduleQuery(const char* line){

	//Line MUST include ; at the end of SQL statement
	size_t line_length = strlen(line);

	//Check for available space
	if( (cursor+line_length+1) >= (MAX_QUERY_LENGTH) ){
		//Execute & flush buffer
		executePackedQuerys();
	}
	//Add line to packedQuery
  	snprintf(packedQuery+cursor, line_length+1, line);
	cursor +=line_length;	

}

void QueryManager::executePackedQuerys(void){

	if(databaseManager == NULL){
		//First initialization avoids "static initialization order fiasco"
		databaseManager = Scheduler::dbManager->getManager();	
	}	

	if(cursor > 0){
		//Execute Packed query
		databaseManager->executeQuery((SQLCHAR*)packedQuery, cursor);
		
		//set pointer to 0
		cursor=0;	
	}
}
