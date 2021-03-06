#include "Scheduler.h"
//#include "/home/meera/gpudir/NetGPU/netgpu-master/Examples/Analysis/IpScan/IpScan.h"
/* Static member initialization */
//extern "C"  void kernel_wrapper(int *a,int *b);
//extern void kernel_wrapper(int *a, int *b);
//#include "kernel.h"


DatabaseManager* Scheduler::dbManager = new ODBCDatabaseManager();

void (*Scheduler::analysisFunctions[SCHEDULER_MAX_ANALYSIS_POOL_SIZE])(PacketBuffer* packetBuffer, packet_t* GPU_buffer);

feeders_t Scheduler::feedersPool[SCHEDULER_MAX_FEEDERS_POOL_SIZE];
 
/* Scheduler initializer */
void Scheduler::init(void){
	
	sigset_t set;

	//Memset pools
	memset(&feedersPool,0,sizeof(feedersPool));		
	memset(&analysisFunctions,0,sizeof(analysisFunctions));	
	
	//Ignore SIGTERM signal (to be inherit by feeders threads)
	sigemptyset(&set);
	sigaddset(&set,SIGTERM);
	sigaddset(&set,SIGINT);
       	sigprocmask(SIG_BLOCK, &set, NULL);
}


/* SIGTERM Signal handler*/

void Scheduler_sigterm_handler(int signum){

	Scheduler::term();
}

/* SIGTERM handler programmer */
void Scheduler::programHandler(void){

	struct sigaction action;
	sigset_t set;
	
	sigemptyset(&set);
	sigaddset(&set,SIGTERM);
	
	//sigaction
	memset(&action,0,sizeof(struct sigaction));	
	action.sa_handler = Scheduler_sigterm_handler;
   
	if(sigaction(SIGTERM, &action, NULL)<0)
		ABORT("Error programming signal handler on Scheduler");

	if(sigaction(SIGINT, &action, NULL)<0)
		ABORT("Error programming signal handler on Scheduler");

	//Unblock my SIGTERM
       	sigprocmask(SIG_UNBLOCK, &set, NULL);
}

packet_t* Scheduler::loadBufferToGPU(PacketBuffer* packetBuffer){
	
	/* Loads buffer to the GPU */	
	//cout<<"Loading Buffer To GPU "<<endl;        

        cout<<"\n ---- buffer from CPU to GPU \n" <<endl;

	packet_t* GPU_buffer;
	int size = sizeof(packet_t)*MAX_BUFFER_PACKETS;

	BMMS::mallocBMMS((void**)&GPU_buffer,size);
	cudaAssert(cudaThreadSynchronize());

        cout<<"\n ---- ******************************************************************" <<endl;

	/* Checks if buffer is NULL */
	if(packetBuffer == NULL)
        {
          cout<<"\n ---- packet buffer is NULL \n" <<endl;
	  return NULL;
        } 
	
	if(GPU_buffer == NULL)
          ABORT("cudaMalloc failed at Scheduler");
	if(packetBuffer->getBuffer()==NULL)
    	  ABORT("PacketBuffer is NULL");

        cudaAssert(cudaMemcpy(GPU_buffer,packetBuffer->getBuffer(),size,cudaMemcpyHostToDevice));
	cudaAssert(cudaThreadSynchronize());

/*
// added on June 1st 
//Modified: June 7th Uncomment the below block if needed to print the packets passing from the CPU
   int index=packetBuffer->getNumOfPackets();
   
   for(int i=0;i<index;i++)
   {
	packet_t* curpacket=packetBuffer->getPacket(i);
	cout<<"Header information of packet"<<i<<endl;
	cout<<curpacket->headers.offset<<endl;
	cout<<curpacket->headers.proto<<endl;

	for(int i=0;i<94;i++)
	{
           cout<<ntohs(curpacket->packet[i]);
	//  cout<<endl;
	}
cout<<endl;
   }
*/
  return GPU_buffer;	
}

void Scheduler::unloadBufferFromGPU(packet_t* GPU_buffer){
/*Modified: June 7th Uncomment the below block if needed to print the packets returned from the GPU

PacketBuffer* pacbuf=new PacketBuffer();
int size=sizeof(packet_t)*MAX_BUFFER_PACKETS;
cout<<endl<<"Unloading from GPU called";

	cudaAssert(cudaMemcpy(pacbuf->buffer,GPU_buffer,size,cudaMemcpyDeviceToHost));
		cudaAssert(cudaThreadSynchronize());

   for(int i=0;i<911;i++)
   {		
		packet_t* curpacket=pacbuf->getPacket(i);
		cout<<"Unloaded: Header information of packet"<<i<<endl;
		cout<<"Unloaded"<<curpacket->headers.offset<<endl;
		cout<<"Unloaded"<<curpacket->headers.proto<<endl;

		for(int j=0;j<94;j++)
		{
		cout <<curpacket->packet[j];
		}
cout<<endl; 
}
*/
	/* Unloads buffer from the GPU */	
	BMMS::freeBMMS(GPU_buffer);
}


/* Adds feeder to the pool and stores pthread_t */

void Scheduler::addFeederToPool(PacketFeeder* feeder,int limit){
	int i;	
	cout<<"Add Feeder To Pool";
	for(i=0;i<SCHEDULER_MAX_FEEDERS_POOL_SIZE;i++){
		if(feedersPool[i].feeder == NULL){
			feedersPool[i].feeder = feeder;
			feedersPool[i].thread = feedersPool[i].feeder->start(limit);
			return;
		}		

	}

	ABORT("No more feeders can be placed into the pool");
}

/* Adds an analysis to the pool */

void Scheduler::addAnalysisToPool(void (*func)(PacketBuffer* packetBuffer, packet_t* GPU_buffer)){
	int i;
        
        cout << "\n ---- addAnalysisToPool called \n";

	for(i=0;i<SCHEDULER_MAX_ANALYSIS_POOL_SIZE;i++){
		if(analysisFunctions[i] == NULL){
			analysisFunctions[i] = func;
			return;
		}		

	}

	ABORT("No more analysis can be placed into the pool");
}

/* Buffer analyze routine */
void Scheduler::analyzeBuffer(PacketBuffer* packetBuffer){
	int i;
	static int counter=0;	

	packet_t* GPU_buffer; 

	DEBUG("Entering analyzeBuffer");
	
        cout <<"\n ---- calling loadBufferToGPU \n"; 

	//Load buffer from PacketBuffer to GPU
	GPU_buffer = loadBufferToGPU(packetBuffer);
	
	DEBUG("Loaded: %d",counter);


	/*** Throwing Analysis ***/
	for(i=0;i<SCHEDULER_MAX_ANALYSIS_POOL_SIZE;i++){
		if(analysisFunctions[i] != NULL){
			analysisFunctions[i](packetBuffer,GPU_buffer);
		}else
			break;
	}
	
	cout<<"Unloading Buffer From GPU called";
	//UNload buffer from GPU
	unloadBufferFromGPU(GPU_buffer);

	counter++;
	cerr<<endl<<endl;
}

/* Start routine. Infinite loop that obtains buffer and analyzes it*/

void Scheduler::start(void){
	
	int i;
	bool hasFeedersLeft;
	PacketBuffer* buffer=NULL;

	/* SIGTERM signal handler*/

	programHandler();	

	/* Implements infinite loop */
	for(;;){
		for(i=0,hasFeedersLeft = false;i<SCHEDULER_MAX_FEEDERS_POOL_SIZE;i++){	 
			
			//If slot has valid Feeder pointer
        		if(feedersPool[i].feeder != NULL){
				//Get buffer
				buffer = feedersPool[i].feeder->getSniffedPacketBuffer();
			
				//Analyse it
                  //analyzeBuffer(buffer);

				//Check if(offline) feeder has no more packets to get
				if(buffer == NULL || buffer->getFlushFlag())
					feedersPool[i].feeder = NULL;
				else
				{
					hasFeedersLeft = true;	
				}}
		}

		if(hasFeedersLeft == false)
			break;
	}

}

void Scheduler::term(void){

	int i;
	
	cerr<<"Sending term"<<endl;
	
	//Force all feeders to flush their buffers and to exit

	for(i=0;i<SCHEDULER_MAX_FEEDERS_POOL_SIZE;i++){	 
		if(feedersPool[i].feeder != NULL)	
			feedersPool[i].feeder->flushAndExit();	
	}

		
}
