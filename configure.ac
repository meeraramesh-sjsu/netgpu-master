AC_INIT(GMAC, 0.1, marcdevel@gmail.com, netgpu)
AM_INIT_AUTOMAKE([-Wall -Wno-extra-portability -Werror foreign])
AC_CONFIG_MACRO_DIR([m4])
AC_GNU_SOURCE

#AM_PROG_AR
AC_PROG_INSTALL
AC_PROG_CC
AC_PROG_CXX
AC_PROG_LD
AUTOCONF_ENV="SHELL=/bin/sh"
AM_PROG_LIBTOOL

# Get the NVIDIA CUDA compiler
AC_PATH_PROG([NVCC], [nvcc])
if test -z "$NVCC"; then
	AC_MSG_ERROR([CUDA compiler not found])
fi
CUDA_DIR="`dirname $NVCC`/.."
API_DIR="cuda"
AC_DEFINE([HAVE_CUDA], [1])

#Check gcc4.3 compiler (CUDA still does not support gcc4.4)
AC_PATH_PROG([GPLUSPLUS], [g++])
if test -z "$GPLUSPLUS"; then
	AC_MSG_ERROR([g++ GNU compiler not found])
fi

# Check for debug mode - MUST BE THE FIRST CHECK
AC_MSG_CHECKING(whether to enable debug mode)
debug_default="no"
AC_ARG_ENABLE(debug,
	AC_HELP_STRING([--enable-debug], [turn on debug mode [[default=no]]])
		, , enable_debug=$debug_default)
if test "$enable_debug" = "yes"; then
	CFLAGS="-g -O0"
	CXXFLAGS="-g -O0 -fno-inline"
	AC_DEFINE(DEBUG)
	AC_MSG_RESULT(yes)
else
	CXXFLAGS="-O3 -fomit-frame-pointer"
	CFLAGS="-O3 --compiler-options -fno-strict-aliasing --compiler-options -fno-inline --compiler-bindir=/usr/bin/g++"
	AC_MSG_RESULT(no)
fi

# Check for profiling mode
AC_MSG_CHECKING(whether to enable profiling mode)
profile_default="no"
AC_ARG_ENABLE(profile,
	AC_HELP_STRING([--enable-profile], [turn on profile mode [[default=no]]])
		, , enable_profile=$profile_default)
if test "$enable_profile" = "yes"; then
	CFLAGS="$( echo $CFLAGS | sed s/-fomit-frame-pointer//g )"
	CXXFLAGS="$( echo $CXXFLAGS | sed s/-fomit-frame-pointer//g )"
	CXXFLAGS="$CXXFLAGS -pg"
	LDFLAGS="$LDFLAGS -pg"
	AC_MSG_RESULT(yes)
else
	AC_MSG_RESULT(no)
fi

AC_SUBST(NVCC)
AC_SUBST(CUDA_DIR)

AC_OUTPUT(
	Makefile
	src/Makefile
	src/Common/Makefile
	src/Common/Protocols/Makefile
	src/Scheduler/Makefile
	src/PacketFeeders/Makefile
	src/PreAnalyzer/Makefile
	src/Analysis/Makefile
	src/Analysis/Modules/Makefile
	src/Analysis/Libs/Makefile
	src/Analysis/Libs/Gpu/Makefile
	src/Analysis/Libs/Gpu/Macros/Makefile
	src/Analysis/Libs/Host/Makefile
	src/Analysis/Libs/Host/Database/Makefile
	src/Analysis/Libs/Host/GpuMemoryManagement/Makefile
	src/Tools/Makefile
)
