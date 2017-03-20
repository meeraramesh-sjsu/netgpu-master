#!/bin/bash

 nvprof --csv --log-file wumanprintgpu10.csv --print-gpu-trace ./Examples/example pcap/packet256new.pcap 10

 nvprof --csv --log-file wumanprintgpu50.csv --print-gpu-trace ./Examples/example pcap/packet256new.pcap 50

 nvprof --csv --log-file wumanprintgpu100.csv --print-gpu-trace ./Examples/example pcap/packet256new.pcap 100


 nvprof --csv --log-file wumanprintgpu200.csv --print-gpu-trace ./Examples/example pcap/packet256new.pcap 200


 nvprof --csv --log-file wumanprintgpu500.csv --print-gpu-trace ./Examples/example pcap/packet256new.pcap 500


 nvprof --csv --log-file wumanprintgpu1000.csv --print-gpu-trace ./Examples/example pcap/packet256new.pcap 1000


 nvprof --csv --log-file wumanprintgpu1500.csv --print-gpu-trace ./Examples/example pcap/packet256new.pcap 1500


 nvprof --csv --log-file wumanprintgpu1800.csv --print-gpu-trace ./Examples/example pcap/packet256new.pcap 1800

