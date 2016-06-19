MAINTAINERCLEANFILES = Makefile.in

SUBDIRS = Common Scheduler PacketFeeders PreAnalyzer Analysis Tools 

export INCLUDES += -I$(CUDA_DIR)/include\
	-I$(abs_srcdir) -I$(abs_builddir)

lib_LTLIBRARIES = libnetgpu.la
libnetgpu_la_SOURCES = Util.h
libnetgpu_la_LIBADD = Common/Protocols/libcommon-protocols.la Common/libcommon.la Scheduler/libscheduler.la PacketFeeders/libpacketfeeders.la PreAnalyzer/libpreanalyzer.la Analysis/Libs/Gpu/libanalysis-libs-gpu.la Analysis/Libs/Host/Database/libanalysis-libs-host-db.la Analysis/Libs/Host/GpuMemoryManagement/libanalysis-libs-host-mm.la

library_includedir=$(includedir)/netgpu

library_include_HEADERS = Util.h netgpu.h

utildir=$(datadir)/netgpu/include/netgpu

util_DATA = Util.h

analysistemplatedir=$(datadir)/netgpu/AnalysisTemplate

analysistemplate_DATA = AnalysisTemplate/AnalysisTemplate.h\
	AnalysisTemplate/AnalysisTemplate.cu\
	AnalysisTemplate/Makefile

