CXX  ?= g++
NVCC ?= nvcc
CPP   = $(wildcard *.cpp)
.PHONY: clean

default: $(CPP:.cpp=.bin)
	
PB11INC  := $(shell python3 -m pybind11 --includes)
PB11SUFF := $(shell python3-config --extension-suffix)

clean:
	rm -f *.so

%: %.cu
	$(NVCC) -std=c++14 -O3 \
	        --expt-relaxed-constexpr \
		    --use_fast_math \
	        --maxrregcount 64 \
	        -lineinfo -Xptxas -v \
	        --generate-code arch=compute_30,code=sm_30 \
	        --generate-code arch=compute_61,code=sm_61 \
	        --generate-code arch=compute_70,code=sm_70 \
	        -I../cpp $(PB11INC) \
	        -Xcompiler -fPIC -shared \
	        $^ -lcudart -lrt -o $@$(PB11SUFF)
