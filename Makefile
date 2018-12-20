export SHELL = /bin/bash
export OS = $(shell uname)
export DIR = $(shell pwd)

export MKDIR = mkdir -p
export RM = rm -rf

.PHONY: all
all:
	$(RM) build
	$(MKDIR) build
	cd build \
	&& . /opt/intel/mkl/bin/mklvars.sh intel64 \
	&& . scl_source enable devtoolset-7 rh-python36 \
	&& cmake -GNinja .. \
	&& cmake --build . \
	&& export PYTHONPATH="/usr/local/lib/python3.6/site-packages:$$PYTHONPATH" \
	&& bin/siphon --load ../test/resnet50 --save c2model --save_onnx model.onnx
