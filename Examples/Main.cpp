#include <cstring>
#include <iostream>
#include <time.h>
#include <netgpu/netgpu.h>
//#include "Analysis/Rate/Rate.h"
//#include "Analysis/Anomalies/Anomalies.h"
//#include "Analysis/Throughput/Throughput.h"
//#include "Analysis/Histogram/Histogram.h"
//#include "Analysis/PortScan/PortScan.h"
#include "Analysis/IpScan/IpScan.h"
//#include "Analysis/IpScan/kernel.h"
//#include "Analysis/Advanced/Advanced.h"

using namespace std;
 int main(int args,char *argv[]) {

	 if(args < 4) {
	printf("Please provide input parameters as Executable pcap_file noOfPAtterns");
	 return 0;
	 }
	 clock_t start,stop;
	 start = clock();
	//Capture packets from a pcap capture file (argv[1])
	OfflinePacketFeeder* feeder = new OfflinePacketFeeder(argv[1]);
	int noOfPatterns = atoi(argv[2]);
	//Capturing from lo
	//LivePacketFeeder* feeder = new LivePacketFeeder("lo");
	

	//cout <<"Starting analysis in few minutes...\n";

//	std::cout <<"Starting analysis now      ...\n";


	//Adding analysis to pool
//	Scheduler::addAnalysisToPool(IpScan::launchAnalysis);
//	Scheduler::addAnalysisToPool(Kernel::launchAnalysis);
//	Scheduler::addAnalysisToPool(PortScan::launchAnalysis);
//	Scheduler::addAnalysisToPool(Anomalies::launchAnalysis);
//	Scheduler::addAnalysisToPool(Rate::launchAnalysis);
//	Scheduler::addAnalysisToPool(Histogram::launchAnalysis);
//	Scheduler::addAnalysisToPool(Throughput::launchAnalysis);
//	Scheduler::addAnalysisToPool(AdvancedExample::launchAnalysis);

	cout<<"Adding feeder to pool......"<<endl;
	//Adding a single feeder
	Scheduler::addFeederToPool(feeder,noOfPatterns);

	//Starting execution (infinite loop)
	Scheduler::start();
	stop = clock();
	cout<<"Time=   "<<(stop - start)/CLOCKS_PER_SEC<<endl;
	//std::cout<<" \n Ending ......."<<endl;

	delete feeder;
	exit(EXIT_SUCCESS);
        //sample 
}

