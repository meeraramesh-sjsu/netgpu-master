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

using namespace std;
int main(int args,char *argv[]) {

	//Capture packets from a pcap capture file (argv[1])
	OfflinePacketFeeder* feeder = new OfflinePacketFeeder(argv[1]);
	string AnoOfPatterns = argv[2];

	ANALYSIS_NAME::noOfPatterns = AnoOfPatterns;
	//Capturing from lo
	//LivePacketFeeder* feeder = new LivePacketFeeder("lo");


	//cout <<"Starting analysis in few minutes...\n";

	//	std::cout <<"Starting analysis now      ...\n";


	//Adding analysis to pool
	Scheduler::addAnalysisToPool(IpScan::launchAnalysis);

	//Adding a single feeder
	Scheduler::addFeederToPool(feeder);

	//Starting execution (infinite loop)
	Scheduler::start();

	//std::cout<<" \n Ending ......."<<endl;

	delete feeder;
	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaAssert(cudaDeviceReset());
	exit(EXIT_SUCCESS);
	//sample
}

