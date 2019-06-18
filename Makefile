export SHELL = /bin/bash
export OS = $(shell uname)
export DIR = $(shell pwd)

export MKDIR = mkdir -p
export RM = rm -rf

.PHONY: all
all: build/bin/siphon

build/bin/siphon:
	$(RM) build
	$(MKDIR) build
	. scl_source enable devtoolset-8; \
	set -e; \
	cd build; \
	. /opt/intel/mkl/bin/mklvars.sh intel64; \
	time cmake -DCMAKE_VERBOSE_MAKEFILE=ON -GNinja ..; \
	time cmake --build .;

.PHONY: run
run: build/bin/siphon
	. scl_source enable devtoolset-8; \
	set -e; \
	cd build; \
	. /opt/intel/mkl/bin/mklvars.sh intel64; \
	$(RM) models; \
	$(MKDIR) models; \
	time bin/siphon --caffe2_log_level=0 --load ../test/resnet50 --save models/c2_native --save_onnx models/onnx_from_c2; \
	time bin/siphon --caffe2_log_level=0 --load models/onnx_from_c2 --save models/c2_from_onnx --save_onnx models/onnx_from_onnx;

.PHONY: convert
convert: build/bin/siphon
	. scl_source enable devtoolset-8; \
	set -e; \
	cd build; \
	. /opt/intel/mkl/bin/mklvars.sh intel64; \
	$(RM) models; \
	$(MKDIR) models; \
	find "../test/contrib/" -mindepth 1 -type d \
	| parallel --bar -j0 'bash -c '"'"' \
	    set -e; \
	    time if grep "_onnx$$" <<< {} > /dev/null; then \
	        bin/siphon --caffe2_log_level=0 --load {} --save      "models/$$(basename {} | sed "s/_onnx$$//")"; \
	    else \
	        bin/siphon --caffe2_log_level=0 --load {} --save_onnx "models/$$(basename {})_onnx"; \
	    fi; \
	'"'";

.PHONY: debug
debug: build/bin/siphon
	. scl_source enable devtoolset-8; \
	set -e; \
	cd build; \
	. /opt/intel/mkl/bin/mklvars.sh intel64; \
	$(RM) models; \
	$(MKDIR) models; \
	LD_DEBUG=files bin/siphon --caffe2_log_level=0 --load ../test/resnet50 --save models/c2_native --save_onnx models/onnx_from_c2; \
	LD_DEBUG= gdb --args bin/siphon --caffe2_log_level=0 --load models/onnx_from_c2 --save models/c2_from_onnx --save_onnx models/onnx_from_onnx;

.PHONY: clean
clean:
	$(RM) build
