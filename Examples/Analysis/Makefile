SUBDIRS:=$(shell list=`/bin/ls -1 -t | egrep -v '^(svn|include)$$'`; for i in $$list; do if test -d $$i; then echo $$i; fi; done)

all:
#SUBDIRS:=$(shell list=`/bin/ls -1 -t | grep -v '^(svn|include)$$'`; for i in $$list; do if test -d $$i; then echo $$i; fi; done)	
	@for i in $(SUBDIRS); do \
        echo "make all in $$i..."; \
        (cd $$i && make); done
#	$(OBJS):=$((shell list=`/bin/find -name "*.o" | egrep -v '^(svn|include)$$'`; for i in $$list; do echo $$i; done)
clean: 
	@for i in $(SUBDIRS); do \
        echo "Cleaning $$i/..."; \
        (cd $$i && make clean); done

