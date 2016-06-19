MAINTAINERCLEANFILES = Makefile.in

noinst_LTLIBRARIES = libanalysis-libs-host-db.la
libanalysis_libs_host_db_la_SOURCES = \
	DatabaseManager.h\
	ODBCDatabaseManager.h ODBCDatabaseManager.cpp\
	QueryManager.h QueryManager.cpp

library_includedir=$(includedir)/netgpu/Analysis/Libs/Host/Database

library_include_HEADERS = DatabaseManager.h\
			ODBCDatabaseManager.h\
			QueryManager.h

databasedir=$(datadir)/netgpu/include/netgpu/Analysis/Libs/Host/Database

database_DATA = DatabaseManager.h\
			ODBCDatabaseManager.h\
			QueryManager.h


