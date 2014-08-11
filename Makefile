# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda-6.0
NVCC := $(CUDA_PATH)/bin/nvcc
INCLUDES := -I./include

all: SpinAccGPU

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

testATW:
	mkdir ./tests
	cp ./data/ATWpm-magn-2.5nm.dat.gz ./tests
	gunzip ./tests/ATWpm-magn-2.5nm.dat.gz
	cp ./bin/SpinAccGPU ./tests

testVW:
	mkdir ./tests
	cp ./data/upVW-magn-2.5nm.dat.gz ./tests
	gunzip ./tests/upVW-magn-2.5nm.dat.gz
	cp ./bin/SpinAccGPU ./tests
