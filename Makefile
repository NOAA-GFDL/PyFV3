SHELL=/bin/bash
CWD=$(shell pwd)
CMD ?= bash
DEV ?=y
TEST_ARGS ?=
PORT ?=8888
RUN_FLAGS ?=--rm
RUN_FLAGS += -e FV3_DACEMODE=$(FV3_DACEMODE)
IMAGE_NAME ?= noaa-gfdl/pyfv3
APP_NAME ?=pyFV3_dev

BUILD_FLAGS ?= 
VOLUMES ?=

ifeq ($(DEV), y)
	VOLUMES += -v $(CWD):/pyFV3
endif

CONTAINER_CMD?=docker run $(RUN_FLAGS) $(VOLUMES) $(IMAGE_NAME)

.PHONY: build enter dev

build:
	DOCKER_BUILDKIT=1 docker build \
		$(BUILD_FLAGS) \
		-f $(CWD)/Dockerfile \
		-t $(IMAGE_NAME) \
		.

enter:
	docker run --rm -it \
		$(VOLUMES) \
		-p=$(PORT):$(PORT) \
		--name="$(APP_NAME)" \
		$(IMAGE_NAME) $(CMD)

dev:
	DEV=y $(MAKE) enter

run:
	CMD="jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --notebook-dir=pyFV3/examples/notebook" \
	DEV=y \
	$(MAKE) enter

# test_sequential:

# test_mpi:

# test_all:
