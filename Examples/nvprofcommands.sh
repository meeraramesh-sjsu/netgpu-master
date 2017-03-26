#!/bin/bash

 
 nvprof --metrics l2_l1_read_hit_rate, stall_inst_fetch, stall_exec_dependency stall_memory_dependency, stall_texture, stall_sync, stall_other, stall_pipe_busy,stall_constant_memory_dependency, stall_memory_throttle, stall_not_selected, ipc, sm_efficiency, warp_execution_efficiency, warp_nonpred_execution_efficiency, inst_per_warp, issue_slot_utilization, shared_efficiency, gld_efficiency, gst_efficiency ./example pcap/packet256new.pcap 1800

