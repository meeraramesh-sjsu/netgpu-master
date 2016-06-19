MAINTAINERCLEANFILES = aclocal.m4 config.guess config.sub configure COPYING\
depcomp INSTALL install-sh ltmain.sh Makefile.in missing

SUBDIRS = src 

install-exec-hook:
		@echo
		@echo "POST INSTALLATION Script (hook)"
		@echo "------------------"
		@echo
	

# Configuring netgpu_ppp
		@echo -n "Configuring netgpu_ppp..."
		@sed -e "3s@_path_to_scripts_@$(datadir)/netgpu/Tools/@" ../src/Tools/netgpu_ppp > $(bindir)/netgpu_ppp
		@echo "Done."
	

# Configuring dmodule.sh	
		@echo -n "Configuring dmodule.sh..."
		@sed -e "3s@_netgpu_path_@$(datadir)/netgpu/include/netgpu/@" ../src/Tools/dmodule.sh > $(datadir)/netgpu/Tools/dmodule.sh
		@echo "Done."


# Configuring netgpu__compile_analysis
		@echo -n "Configuring netgpu__compile_analysis..."
		@sed -e "2s@_library_path_@$(datadir)/netgpu/include/@" ../src/Tools/netgpu__compile_analysis > $(bindir)/netgpu__compile_analysis
		@echo "Done."

# Configuring netgpu_create_analysis
		@echo -n "Configuring netgpu_create_analysis..."
		@sed -e "3s@_library_path_@$(datadir)/netgpu/AnalysisTemplate/@" ../src/Tools/netgpu_create_analysis > $(bindir)/netgpu_create_analysis
		@echo "Done."


# Installing modules
		@echo "Calling Module installer script (datadir)..." 
		@../module-installer.sh $(datadir)/netgpu/include/netgpu/
		@echo "Calling Module installer script (includedir)..." 
		@../module-installer.sh $(includedir)/netgpu/
		
		@echo "Calling ldconfig"
		@/sbin/ldconfig	


uninstall-hook:
		@echo
		@echo "POST UNINSTALLATION Script (hook)"
		@echo "------------------"
		@echo
	
		@echo -n Uninstalling Modules...

		@rm $(datadir)/netgpu/ -Rf
		@echo Done.
		
	
