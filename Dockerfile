# syntax=docker/dockerfile:1

# https://hub.docker.com/r/nvidia/cuda/tags
# https://www.tensorflow.org/install/source?hl=ja#gpu
# https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 AS base-stage

# apt用プロキシ(apt-cacher-ng用)
ARG APT_PROXY=$http_proxy

# APTのキャッシュ https://github.com/moby/buildkit/blob/master/frontend/dockerfile/docs/reference.md#example-cache-apt-packages
RUN rm -f /etc/apt/apt.conf.d/docker-clean; \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

ARG DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
RUN set -x \
    && rm /etc/apt/apt.conf.d/docker-gzip-indexes \
    && rm /etc/apt/apt.conf.d/docker-no-languages \
    # libcuda.so.1を参照できるようにする
    && echo '/usr/local/cuda/compat' > /etc/ld.so.conf.d/nvidia-compat.conf \
    && ldconfig

# aptその1
# pyenv用: https://github.com/pyenv/pyenv/wiki#suggested-build-environment
# その他？: libpng-dev, libjpeg-dev
RUN --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    --mount=type=cache,target=/var/cache/apt/archives,sharing=locked \
    set -x \
    && sed -ie 's@http://archive.ubuntu.com/ubuntu/@http://ftp.riken.go.jp/Linux/ubuntu/@g' /etc/apt/sources.list \
    && sed -ie 's@^deb-src@# deb-src@g' /etc/apt/sources.list \
    && apt-get update \
    && apt-get install --yes --no-install-recommends \
    apt-transport-https \
    apt-utils \
    locales \
    software-properties-common \
    && apt-get upgrade --yes \
    && apt-get install --yes --no-install-recommends \
    build-essential \
    curl \
    libbluetooth-dev \
    libbz2-dev \
    libdb-dev \
    libexpat-dev \
    libffi-dev \
    libffi8 \
    libgdbm-dev \
    libjpeg-dev \
    liblzma-dev \
    libncurses5-dev \
    libnss3-dev \
    libpng-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    libtool \
    libwebp-dev \
    libxml2-dev \
    libxmlsec1-dev \
    llvm \
    make \
    tk-dev \
    unminimize \
    uuid-dev \
    wget \
    xz-utils \
    zlib1g-dev \
    zstd \
    && locale-gen ja_JP.UTF-8 \
    && update-locale LANG=ja_JP.UTF-8 LANGUAGE='ja_JP:ja'

# 本体ここから。
FROM base-stage AS main-stage
ARG DEBIAN_FRONTEND=noninteractive

# unminimize
RUN --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    --mount=type=cache,target=/var/cache/apt/archives,sharing=locked \
    set -x \
    && yes | unminimize

# aptその2
# scipy用: gfortran
# horovodなど用: g++ <https://github.com/horovod/horovod/blob/master/Dockerfile.test.cpu>
# lightgbm用: libboost-devなど <https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html>
# opencv用: libgl1
# soundfile用: libsndfile1
# その他？: liblapack3 libatlas3-base libgfortran5
# torch? : libcusparselt-dev
# <https://github.com/scipy/scipy/issues/9005>: gfortran libopenblas-dev liblapack-dev
RUN --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    --mount=type=cache,target=/var/cache/apt/archives,sharing=locked \
    set -x \
    && apt-get update \
    && apt-get install --yes --no-install-recommends \
    accountsservice \
    ack-grep \
    apparmor \
    apt-file \
    build-essential \
    automake \
    bash-completion \
    bc \
    clang \
    bind9-host \
    bsdmainutils \
    busybox-initramfs \
    busybox-static \
    cifs-utils \
    cloc \
    cmake \
    command-not-found \
    connect-proxy \
    console-setup \
    console-setup-linux \
    corkscrew \
    cpio \
    cron \
    dbus \
    debconf-i18n \
    dialog \
    distro-info-data \
    dmidecode \
    dmsetup \
    dnsutils \
    dosfstools \
    ed \
    eject \
    emacs \
    entr \
    file \
    fonts-ipafont \
    fonts-liberation \
    friendly-recovery \
    ftp \
    g++ \
    gdb \
    geoip-database \
    gettext-base \
    gfortran \
    gh \
    git \
    git-filter-repo \
    git-lfs \
    golang-cfssl \
    google-perftools \
    graphviz \
    groff-base \
    hdf5-tools \
    hdparm \
    htop \
    iftop \
    imagemagick \
    inetutils-traceroute \
    info \
    init \
    initramfs-tools \
    initramfs-tools-bin \
    initramfs-tools-core \
    install-info \
    iotop \
    iproute2 \
    iptables \
    iputils-ping \
    iputils-tracepath \
    irqbalance \
    isc-dhcp-client \
    isc-dhcp-common \
    jq \
    kbd \
    keyboard-configuration \
    klibc-utils \
    kmod \
    krb5-locales \
    language-pack-ja \
    language-selector-common \
    less \
    libatlas3-base \
    libboost-dev \
    libboost-filesystem-dev \
    libboost-system-dev \
    libcusparselt-dev \
    libgfortran5 \
    libgl1 \
    liblapack-dev \
    liblapack3 \
    libopenblas-dev \
    libsndfile1 \
    libstdc++-14-dev \
    linux-base \
    locate \
    logrotate \
    lshw \
    lsof \
    ltrace \
    man-db \
    manpages \
    mecab \
    mecab-ipadic-utf8 \
    mecab-jumandic-utf8 \
    mtr-tiny \
    nano \
    net-tools \
    netbase \
    netcat-openbsd \
    netplan.io \
    networkd-dispatcher \
    ntfs-3g \
    nvidia-opencl-dev \
    openssh-client \
    openssh-server \
    openssl \
    p7zip-full \
    pandoc \
    parted \
    pciutils \
    plymouth \
    plymouth-theme-ubuntu-text \
    poppler-data \
    poppler-utils \
    popularity-contest \
    powermgmt-base \
    psmisc \
    publicsuffix \
    ripgrep \
    rsync \
    rsyslog \
    samba \
    screen \
    shared-mime-info \
    sl \
    smbclient \
    sshpass \
    strace \
    subversion \
    sudo \
    swig \
    systemd \
    systemd-sysv \
    tcl-dev \
    tcpdump \
    telnet \
    tesseract-ocr \
    tesseract-ocr-jpn \
    tesseract-ocr-jpn-vert \
    tesseract-ocr-script-jpan \
    tesseract-ocr-script-jpan-vert \
    texlive-fonts-recommended \
    texlive-plain-generic \
    texlive-xetex \
    time \
    tmux \
    tmuxinator \
    tzdata \
    ubuntu-advantage-tools \
    ubuntu-minimal \
    ubuntu-release-upgrader-core \
    ubuntu-standard \
    ucf \
    udev \
    ufw \
    unzip \
    update-manager-core \
    usbutils \
    uuid-dev \
    uuid-runtime \
    valgrind \
    vim \
    whiptail \
    wrk \
    xauth \
    xclip \
    xdg-user-dirs \
    xkb-data \
    xxd \
    zip \
    zsh \
    # MeCabの標準はIPA辞書にしておく
    && update-alternatives --set mecab-dictionary /var/lib/mecab/dic/ipadic-utf8

# python
# https://gregoryszorc.com/docs/python-build-standalone/main/running.html
# https://github.com/astral-sh/python-build-standalone/releases
ARG PYTHON_VERSION=3.12.10
ARG PYTHON_TAG=20250521
ARG PYTHON_ARCH=x86_64_v2-unknown-linux-gnu
RUN set -x \
    && curl -sSL "https://github.com/astral-sh/python-build-standalone/releases/download/${PYTHON_TAG}/cpython-${PYTHON_VERSION}+${PYTHON_TAG}-${PYTHON_ARCH}-pgo+lto-full.tar.zst" -o python.tar.zst \
    && tar -axvf python.tar.zst \
    && cp -avn python/install/. /usr/local/ \
    && rm -rf python.tar.zst python \
    && ldconfig \
    && export PYTHONDONTWRITEBYTECODE=1 \
    && python3 --version | grep -q "${PYTHON_VERSION}"

# nodejs
COPY --from=node:lts /usr/local/bin/node /usr/local/bin/
COPY --from=node:lts /usr/local/lib/node_modules/ /usr/local/lib/node_modules/
COPY --from=node:lts /usr/local/include/node/ /usr/local/include/node/
COPY --from=node:lts /usr/local/share/doc/node/ /usr/local/share/doc/node/
COPY --from=node:lts /usr/local/share/man/man1/node.1 /usr/local/share/man/man1/
RUN --mount=type=cache,target=/root/.npm set -x \
    && ln -s /usr/local/bin/node /usr/local/bin/nodejs \
    && ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && ln -s /usr/local/lib/node_modules/corepack/dist/corepack.js /usr/local/bin/corepack \
    && npm -g config set proxy $http_proxy \
    && npm -g config set https-proxy $https_proxy \
    && npm -g config set cafile /etc/ssl/certs/ca-certificates.crt \
    && npm -g update \
    && npm -g install \
        @anthropic-ai/claude-code \
        @biomejs/biome \
        @google/gemini-cli \
        @openai/codex \
        aicommits \
        eslint \
        npm-check-updates \
        opencommit \
        prettier \
        pyright \
        xo \
    && yarn config set proxy $http_proxy -g \
    && yarn config set https-proxy $https_proxy -g \
    && yarn config set strict-ssl false -g

# Docker <https://docs.docker.com/engine/install/debian/>
RUN --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    --mount=type=cache,target=/var/cache/apt/archives,sharing=locked \
    set -x \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update \
    && apt-get install --yes --no-install-recommends docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Azure CLI
RUN --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    --mount=type=cache,target=/var/cache/apt/archives,sharing=locked \
    set -x \
    && curl -sL https://aka.ms/InstallAzureCLIDeb | bash

# devpi-server用
ARG PIP_TRUSTED_HOST=""
ARG PIP_INDEX_URL=""
ARG PIP_RETRIES=10
ARG PIP_TIMEOUT=180
ARG PIP_DEFAULT_TIMEOUT=180
ARG PIP_ROOT_USER_ACTION=ignore

# pip
RUN --mount=type=cache,target=/root/.cache \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    set -x \
    && pip install --upgrade pip pip_system_certs uv \
    && UV_PROJECT_ENVIRONMENT=/usr/local uv sync --frozen --no-group=step2 \
    && UV_PROJECT_ENVIRONMENT=/usr/local uv sync --frozen --group=step2 \
    && pip install --upgrade "tensorflow[and-cuda]>=2.19,<2.20" \
    && poetry self add poetry-plugin-export poetry-plugin-shell \
    ;

# TFがエラーにならないことの確認
RUN set -x \
    && python3 -c "import tensorflow as tf;print(tf.config.list_physical_devices('GPU'))" 2>&1 | tee /tmp/check.log \
    && grep -q 'cuInit: CUDA_ERROR_NO_DEVICE' /tmp/check.log \
    && rm -f /tmp/check.log

# PyTorchがエラーにならないことの確認
RUN set -x \
    && python3 -c "import torch;print(torch.cuda.get_device_name())" 2>&1 | tee /tmp/check.log \
    && grep -q "RuntimeError: No CUDA GPUs are available" /tmp/check.log \
    && rm -f /tmp/check.log

# 辞書など
# https://github.com/nltk/nltk/issues/1825
RUN set -x \
    && python3 -m nltk.downloader --exit-on-error --dir=/usr/local/share/nltk_data popular punkt_tab
RUN set -x \
    && ldconfig /usr/local/cuda/lib64/stubs \
    && python3 -m spacy download en_core_web_sm --no-cache \
    && ldconfig
RUN set -x \
    && python3 -m unidic download

# jupyter関連
# RUN --mount=type=cache,target=/root/.cache \
#     --mount=type=cache,target=/root/.npm \
#     set -x \
#     && (jupyter lab build --dev-build=False --minimize=False --debug-log-path=/tmp/jupyterlab-build.log || (cat /tmp/jupyterlab-build.log && false))

# # LightGBM
# # 参考: https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html
# # 参考: https://github.com/microsoft/LightGBM/issues/586
# # https://github.com/microsoft/LightGBM/releases
# RUN --mount=type=cache,target=/root/.cache set -x \
#     && mkdir -p /etc/OpenCL/vendors \
#     && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd \
#     && git clone --recursive --depth=1 --branch=v4.5.0 https://github.com/microsoft/LightGBM /usr/local/src/LightGBM \
#     && cd /usr/local/src/LightGBM \
#     && cmake -DUSE_GPU=1 \
#     && make -j$(nproc) \
#     && sh ./build-python.sh install --precompile

# horovod
# 参考: https://github.com/horovod/horovod/blob/master/docker/horovod/Dockerfile
#RUN --mount=type=cache,target=/root/.cache set -x \
#    && HOROVOD_WITH_MPI=1 HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MXNET=0 pip install horovod \
#    && ldconfig

# 最後にPillow-SIMD
# → 2021/08/13現在メンテされてなさそう: <https://github.com/uploadcare/pillow-simd/issues/93>
#RUN --mount=type=cache,target=/root/.cache set -x && \
#    CC="cc -mavx2" pip install --force-reinstall Pillow-SIMD

# サイズは増えるけどやっておくと使うときに便利かもしれない諸々
RUN set -x \
    && apt-get update \
    && updatedb

# ユーザー作成
# ubuntu24のDockerイメージはUID=1000でubuntuがあるようなので、重複する場合は削除してから作成
ARG RUN_USER=user
ARG RUN_UID=1000
RUN set -x \
    && if getent passwd 1000 ; then userdel --remove `getent passwd 1000 | cut -d: -f1` ; fi \
    && useradd --create-home --shell=/bin/bash --uid=$RUN_UID --groups=sudo $RUN_USER

RUN set -x \
    # sshd用ディレクトリ作成
    && mkdir --mode=744 /var/run/sshd \
    # sshd用設定(~/.ssh/environmentを読む、KeepAliveする)
    && echo 'PermitUserEnvironment yes' > /etc/ssh/sshd_config.d/docker.conf \
    && echo 'ClientAliveInterval 15' >> /etc/ssh/sshd_config.d/docker.conf \
    && echo 'ClientAliveCountMax 10' >> /etc/ssh/sshd_config.d/docker.conf \
    && /usr/sbin/sshd -t \
    # NCCL設定
    && echo 'NCCL_DEBUG=INFO' >> /etc/nccl.conf \
    # 環境変数設定
    && echo "export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin" > /etc/environment \
    && echo "export PYTHONIOENCODING=utf-8" >> /etc/environment \
    && echo "export PYTHONDONTWRITEBYTECODE=1" >> /etc/environment \
    && echo "export BETTER_EXCEPTIONS=1" >> /etc/environment \
    && echo "export TF_FORCE_GPU_ALLOW_GROWTH=true" >> /etc/environment \
    && echo "export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python" >> /etc/environment \
    # sudoでhttp_proxyなどが引き継がれるようにしておく
    && echo 'Defaults env_keep += "http_proxy https_proxy ftp_proxy no_proxy PIP_TRUSTED_HOST PIP_INDEX_URL SSL_CERT_FILE PIP_CERT REQUESTS_CA_BUNDLE"' > /etc/sudoers.d/docker \
    && echo 'Defaults always_set_home' >> /etc/sudoers.d/docker \
    # $RUN_USERをパスワード無しでsudoできるようにしておく
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers.d/docker \
    && chmod 0440 /etc/sudoers.d/* \
    && visudo --check \
    # completion
    && poetry completions bash > /etc/bash_completion.d/poetry.bash-completion \
    && uv generate-shell-completion bash > /etc/bash_completion.d/uv.bash-completion \
    && uvx --generate-shell-completion bash > /etc/bash_completion.d/uvx.bash-completion \
    # 念のため最後にldconfig
    && ldconfig

# sshd以外の使い方をするとき用環境変数色々
ENV TZ='Asia/Tokyo' \
    LANG='ja_JP.UTF-8' \
    PYTHONIOENCODING='utf-8' \
    PYTHONDONTWRITEBYTECODE=1 \
    BETTER_EXCEPTIONS=1 \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# SSHホストキーを固定で用意
COPY --chown=root:root .ssh_host_keys/ssh_host_* /etc/ssh/
RUN set -x \
    && chmod 600 /etc/ssh/ssh_host_* \
    && chmod 644 /etc/ssh/ssh_host_*.pub
# 作った日時を記録しておく (一応)
RUN date '+%Y/%m/%d %H:%M:%S' > /image.version
# sshd
# -D: デタッチしない
# -e: ログを標準エラーへ
CMD ["/usr/sbin/sshd", "-D", "-e"]
