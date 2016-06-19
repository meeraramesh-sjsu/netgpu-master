#include <cstring>
#include <iostream>


#include <netgpu/netgpu.h>
//#include "Analysis/Rate/Rate.h"
//#include "Analysis/Anomalies/Anomalies.h"
//#include "Analysis/Throughput/Throughput.h"
//#include "Analysis/Histogram/Histogram.h"
//#include "Analysis/PortScan/PortScan.h"
#include "Analysis/IpScan/IpScan.h"
//#include "Analysis/IpScan/kernel.h"
//#include "Analysis/Advanced/Advanced.h"

 int main(int args,char *argv[]) {

	//Capture packets from a pcap capture file (argv[1])
	OfflinePacketFeeder* feeder = new OfflinePacketFeeder(argv[1]);

	//Capturing from lo
	//LivePacketFeeder* feeder = new LivePacketFeeder("lo");
	

	std::cout <<"Starting analysis now...\n";

//	std::cout <<"Starting analysis now      ...\n";


	//Adding analysis to pool
	Scheduler::addAnalysisToPool(IpScan::launchAnalysis);
//	Scheduler::addAnalysisToPool(Kernel::launchAnalysis);
//	Scheduler::addAnalysisToPool(PortScan::launchAnalysis);
//	Scheduler::addAnalysisToPool(Anomalies::launchAnalysis);
//	Scheduler::addAnalysisToPool(Rate::launchAnalysis);
//	Scheduler::addAnalysisToPool(Histogram::launchAnalysis);
//	Scheduler::addAnalysisToPool(Throughput::launchAnalysis);
//	Scheduler::addAnalysisToPool(AdvancedExample::launchAnalysis);

	//Adding a single feeder
	Scheduler::addFeederToPool(feeder);

	//Starting execution (infinite loop)
	Scheduler::start();

	std::cout<<"Ending."<<endl;

	delete feeder;
	exit(EXIT_SUCCESS);
        //sample 
}

