
ifneq ($(TARGET),)
include ../flags_$(TARGET).def
else
TARGET=generic
CFLAGS=-fopenmp
LDFLAGS=-fopenmp
endif

# parameters and settings for execution
RUN_SETTINGS    = OMP_PLACES=cores OMP_PROC_BIND=spread I_MPI_PIN=1 I_MPI_PIN_DOMAIN=auto
NUM_RANKS       ?= 2 
PROG_EXE        ?= ch_intel_fine
MATRIX_SIZE     ?= 5120
BLOCK_SIZE      ?= 512
BOOL_CHECK      ?= 1
# CFLAGS          += -DDEBUG
# LDFLAGS         += -DDEBUG

all : ch_$(TARGET)_fine ch_$(TARGET)_fine_timing

ch_$(TARGET)_fine: ch_ompss.c ../common/ch_common.c Makefile
	$(CC) $(CFLAGS) -o $@ ch_ompss.c -g ../common/ch_common.c $(LDFLAGS)

ch_$(TARGET)_fine_extrae: ch_ompss.c ../common/ch_common.c Makefile
	$(CC) $(CFLAGS) -o $@ ch_ompss.c -g ../common/ch_common.c $(LDFLAGS) -DUSE_EXTRAE $(EXTRAE_FLAGS)

ch_$(TARGET)_fine_timing: ch_ompss.c ../common/ch_common.c Makefile
	$(CC) $(CFLAGS) -o $@ ch_ompss.c -g ../common/ch_common.c $(LDFLAGS) -DUSE_TIMING

run:
	$(RUN_SETTINGS) mpiexec -np $(NUM_RANKS) -genvall ./$(PROG_EXE) $(MATRIX_SIZE) $(BLOCK_SIZE) $(BOOL_CHECK)

clean:
	rm -f *.o ch_$(TARGET)_fine ch_$(TARGET)_fine_extrae ch_$(TARGET)_fine_timing

cleanall: 
	rm -f *.o ch_*_fine ch_*_fine_extrae ch_*_fine_timing
