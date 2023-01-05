
include .env

export DOCKER_BUILDKIT ?= 1

IMAGE_TAG ?= ds-docker

BUILD_ARGS += --shm-size=1g
GPU ?= none
ifeq ($(GPU),none)
    RUN_GPU_ARGS = $(RUN_ARGS)
else
    RUN_GPU_ARGS = $(RUN_ARGS) --gpus='"device=$(GPU)"'
endif

help:
	@cat Makefile

.ssh_host_keys:
	mkdir .ssh_host_keys
	ssh-keygen -t ecdsa -f .ssh_host_keys/ssh_host_ecdsa_key -N ''
	ssh-keygen -t ed25519 -f .ssh_host_keys/ssh_host_ed25519_key -N ''
	ssh-keygen -t rsa -f .ssh_host_keys/ssh_host_rsa_key -N ''

rebuild:
	$(MAKE) build BUILD_ARGS="$(BUILD_ARGS) --no-cache"

build: .ssh_host_keys
	docker build --pull $(BUILD_ARGS) --tag=$(IMAGE_TAG) .
	$(MAKE) test
	docker images $(IMAGE_TAG)

format:
	docker run --rm --interactive $(RUN_ARGS) \
		--volume="$(CURDIR):/work:rw" \
		--workdir="/work" \
		--user=$(shell id -u) \
		$(IMAGE_TAG) bash -cx "pyfltr --commands=pyupgrade,isort,black,pflake8 tests"

test:
	docker run --rm --interactive $(RUN_GPU_ARGS) \
		--volume="$(CURDIR):/work:ro" \
		--workdir="/work" \
		--env="GPU=$(GPU)" \
		$(IMAGE_TAG) bash -cx "pyfltr"

shell:
	docker run --rm --interactive --tty $(RUN_GPU_ARGS) \
		--volume="$(CURDIR):/work:ro" \
		--workdir="/work" \
		$(IMAGE_TAG) bash

base-shell:
	docker run --rm --interactive --tty $(RUN_GPU_ARGS) $(shell grep '^FROM nvidia' Dockerfile | head -n1 | awk '{print$$2}') bash

lint:
	docker pull hadolint/hadolint
	docker run --rm -i hadolint/hadolint < Dockerfile
