#!/bin/sh

DIRS=`cd src/Analysis/Modules/ && find . -type d ` 
MODULES=`cd src/Analysis/Modules/ && find . -name *.module`
HEADERS=`cd src/Analysis/Modules/ && find . -name *.h`

#create folder
mkdir -p $1/Analysis/Modules

echo     Creating Module folder tree...
#dirs
for DIR in $DIRS
do
	DIR=`echo $DIR | grep -v .svn`
	mkdir -p $1/Analysis/Modules/$DIR
done

echo      Installing following Modules:

#Copying modules
for MODULE in $MODULES
do
	echo      $MODULE
	#cp -pR ../src/Analysis/Modules/$MODULE $1/Analysis/Modules/$MODULE
	cp -pR src/Analysis/Modules/$MODULE $1/Analysis/Modules/$MODULE
done

#Copying .h files
echo -n Copying headers within Modules folder...
for HEADER in $HEADERS
do
#	cp -pR ../src/Analysis/Modules/$HEADER $1/Analysis/Modules/$HEADER
	cp -pR src/Analysis/Modules/$HEADER $1/Analysis/Modules/$HEADER
done

echo Done.
echo
