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
	cd build \
	&& . /opt/intel/mkl/bin/mklvars.sh intel64 \
	&& . scl_source enable devtoolset-7 rh-python36 \
	&& cmake -DCMAKE_VERBOSE_MAKEFILE=ON -GNinja .. \
	&& cmake --build . \
	&& export PYTHONPATH="/usr/local/lib/python3.6/site-packages:$$PYTHONPATH" \
	&& bin/siphon --caffe2_log_level=0 --load ../test/resnet50 --save c2model --save_onnx model.onnx

.PHONY: run
run: build/bin/siphon
	cd build \
	&& . /opt/intel/mkl/bin/mklvars.sh intel64 \
	&& . scl_source enable devtoolset-7 rh-python36 \
	&& export PYTHONPATH="/usr/local/lib/python3.6/site-packages:$$PYTHONPATH" \
	&& rm -rf c2model model.onnx \
	&& bin/siphon --caffe2_log_level=0 --load ../test/resnet50 --save c2model --save_onnx model.onnx

.PHONY: debug
debug: build/bin/siphon
	cd build \
	&& . /opt/intel/mkl/bin/mklvars.sh intel64 \
	&& . scl_source enable devtoolset-7 rh-python36 \
	&& export PYTHONPATH="/usr/local/lib/python3.6/site-packages:$$PYTHONPATH" \
	&& rm -rf c2model model.onnx \
	&& LD_DEBUG=files bin/siphon --caffe2_log_level=0 --load ../test/resnet50 --save c2model --save_onnx model.onnx
