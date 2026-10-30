# Dorado COBOL kernels - GnuCOBOL 3.2 (apt gnucobol3) on ubuntu-latest.
#
# cobc -C lowers each COBOL program to C; shim.c owns the one-time
# cob_init() runtime startup and the single-op IEEE helpers the kernels
# use for float arithmetic (see shim.c for why). The result is one
# shared object, lib/dorado_cobol.so, loaded by dorado/cobol_walk.py.

COBC ?= cobc
CC ?= gcc

KERNELS := particle_pick.c particle_trav.c
# the adapter (dorado/cobol_walk.py) loads the shared object from the
# package directory, so build it there
LIB := ../dorado/lib/dorado_cobol.so

all: $(LIB)

particle_pick.c: particle_pick.cob
	$(COBC) -C -O2 particle_pick.cob

particle_trav.c: particle_trav.cob
	$(COBC) -C -O2 particle_trav.cob

$(LIB): $(KERNELS) shim.c
	mkdir -p ../dorado/lib
	$(CC) -shared -fPIC -O2 -o $@ shim.c $(KERNELS) \
	    $$(cob-config --cflags) $$(cob-config --libs)

clean:
	rm -f $(KERNELS) *.h $(LIB)

.PHONY: all clean
