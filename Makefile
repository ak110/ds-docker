
include .env

ifdef http_proxy
export http_proxy
BUILD_ARGS += --build-arg=http_proxy
RUN_ARGS += --env=http_proxy
endif
ifdef https_proxy
export https_proxy
BUILD_ARGS += --build-arg=https_proxy
RUN_ARGS += --env=https_proxy
endif
ifdef no_proxy
export no_proxy
BUILD_ARGS += --build-arg=no_proxy
RUN_ARGS += --env=no_proxy
endif
ifdef APT_PROXY
export APT_PROXY
BUILD_ARGS += --build-arg=APT_PROXY
endif
ifdef PIP_TRUSTED_HOST
export PIP_TRUSTED_HOST
BUILD_ARGS += --build-arg=PIP_TRUSTED_HOST
endif
ifdef PIP_INDEX_URL
export PIP_INDEX_URL
BUILD_ARGS += --build-arg=PIP_INDEX_URL
endif

BUILD_ARGS += --shm-size=1g

GPU ?= none
ifeq ($(GPU),none)
    RUN_GPU_ARGS = $(RUN_ARGS)
else
    RUN_GPU_ARGS = $(RUN_ARGS) --gpus='"device=$(GPU)"'
endif

export DOCKER_BUILDKIT=1

IMAGE_TAG ?= ds-docker

help:
	@cat Makefile

.ssh_host_keys:
	mkdir .ssh_host_keys
	ssh-keygen -t ecdsa -f .ssh_host_keys/ssh_host_ecdsa_key -N ''
	ssh-keygen -t ed25519 -f .ssh_host_keys/ssh_host_ed25519_key -N ''
	ssh-keygen -t rsa -f .ssh_host_keys/ssh_host_rsa_key -N ''

update:
	uv sync --no-group=compile
	uv sync --upgrade
	uv export --format=requirements-txt --no-hashes > requirements.txt
	# https://github.com/Dao-AILab/flash-attention
	MAX_JOBS=4 uv sync --upgrade --group=compile
	uv run pyfltr --exit-zero-even-if-formatted tests

rebuild:
	$(MAKE) build BUILD_ARGS="$(BUILD_ARGS) --no-cache"

build: .ssh_host_keys
	docker build --pull --progress=plain $(BUILD_ARGS) --tag=$(IMAGE_TAG) .
	$(MAKE) test
	docker images $(IMAGE_TAG)

format:
	docker run --rm --interactive $(RUN_ARGS) \
		--volume="$(CURDIR):/work:rw" \
		--workdir="/work" \
		--user=$(shell id -u) \
		$(IMAGE_TAG) bash -cx "pyfltr --exit-zero-even-if-formatted --commands=fast tests"

test:
	docker run --rm --interactive $(RUN_GPU_ARGS) \
		--volume="$(CURDIR):/work:ro" \
		--workdir="/work" \
		--env="GPU=$(GPU)" \
		$(IMAGE_TAG) bash -cx "nvidia-smi && pip freeze && pytest"

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
