# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda-6.0
NVCC := $(CUDA_PATH)/bin/nvcc
INCLUDES := -I./include

.PHONY: clean enableATW enableVW useATWparameters useVWparameters

all: clean SpinAccGPU

clean: 
	rm -rf ./bin
	rm -rf ./tests
cleantests:
	rm -rf ./tests

SpinAccGPU:
	mkdir ./bin
	$(NVCC) ./src/SpinAccGPU.cu -o ./bin/SpinAccGPU $(INCLUDES) -arch=sm_20
debug:
	mkdir ./bin
	$(NVCC) ./src/SpinAccGPU.cu -g ./bin/SpinAccGPU $(INCLUDES) -arch=sm_20

useATWparameters:
	cp ./data/ATW-parameters.h ./include/parameters.h

useVWparameters:
	cp ./data/VW-parameters.h ./include/parameters.h

enableATW:
	mkdir ./tests
	cp ./data/ATWpm-magn-2.5nm.dat ./tests
	cp ./bin/SpinAccGPU ./tests

enableVW:
	mkdir ./tests
	cp ./data/upVW-magn-2.5nm.dat ./tests
	cp ./bin/SpinAccGPU ./tests

testATW: clean useATWparameters SpinAccGPU enableATW

testVW: clean useVWparameters SpinAccGPU enableVW

