NVCC := nvcc
NVCC_FLAGS := -std=c++17 -arch=sm_120a

CUTLASS_DIR := $(abspath ../../3rd/cutlass)
INCLUDES := -I$(CUTLASS_DIR)/include \
            -I$(CUTLASS_DIR)/tools/util/include

SRC := main.cu
TARGET := tma_test

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) TMA.cuh
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(SRC) -o $@

clean:
	rm -f $(TARGET)
