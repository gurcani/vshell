#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ(2.61)

AC_INIT([vshell], [0.1], [ozgur.gurcan@lpp.polytechnique.fr])

AM_INIT_AUTOMAKE([-Wall -Werror foreign])

AC_CONFIG_SRCDIR([])
AC_CONFIG_HEADER([config.h])

# Checks for programs.
AC_PROG_CC

AC_PROG_RANLIB

PKG_CHECK_MODULES([gsl], [gsl])

# Checks for header files.
AC_HEADER_STDC
AC_CHECK_HEADERS([stdlib.h])
AC_CHECK_HEADERS([omp.h],[CFLAGS+=" -fopenmp"],[])

# Checks for typedefs, structures, and compiler characteristics.

AC_ARG_ENABLE(debug, AC_HELP_STRING([--enable-debug],[enables debugging]), ok=$enableval, ok=no)
if test "$ok" = "yes"; then
        AC_DEFINE(VSHELL_DEBUG,1,[define to enable debugging code.])
        CFLAGS="-g"
fi

# Checks for library functions.
AC_FUNC_MALLOC

#AC_CONFIG_FILES([Makefile])
AC_OUTPUT(Makefile)
