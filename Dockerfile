# syntax=docker/dockerfile:1

FROM node:lts as node

# https://hub.docker.com/r/nvidia/cuda/tags
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# APTのキャッシュ https://github.com/moby/buildkit/blob/master/frontend/dockerfile/docs/reference.md#example-cache-apt-packages
RUN rm -f /etc/apt/apt.conf.d/docker-clean; \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

ARG DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8
RUN set -x \
    && rm /etc/apt/apt.conf.d/docker-gzip-indexes \
    && rm /etc/apt/apt.conf.d/docker-no-languages \
    # # libcuda.so.1を参照できるようにする
    && echo '/usr/local/cuda/compat' > /etc/ld.so.conf.d/nvidia-compat.conf \
    && ldconfig

RUN --mount=type=cache,target=/var/lib/apt/lists,sharing=private \
    --mount=type=cache,target=/var/cache/apt/archives,sharing=private \
    set -x \
    && sed -ie 's@http://archive.ubuntu.com/ubuntu/@http://ftp.riken.go.jp/Linux/ubuntu/@g' /etc/apt/sources.list \
    && sed -ie 's@^deb-src@# deb-src@g' /etc/apt/sources.list \
    && apt-get update \
    && apt-get install --yes --no-install-recommends \
    apt-transport-https \
    apt-utils \
    ca-certificates \
    curl \
    locales \
    software-properties-common \
    wget \
    && locale-gen ja_JP.UTF-8 \
    && update-locale LANG=ja_JP.UTF-8 LANGUAGE='ja_JP:ja'

RUN --mount=type=cache,target=/var/lib/apt/lists,sharing=private \
    --mount=type=cache,target=/var/cache/apt/archives,sharing=private \
    set -x \
    && yes | unminimize

ARG PYTHON_VERSION=3.11

# libgl1 libglib2.0-0 libsm6 libxrender1 libxext6: opencv用
# libgomp1: LightGBM用
# puppeteer用: libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libpango-1.0-0 libcairo2 libasound2 libxcursor1 libgtk-3-0
# pyenv用: https://github.com/pyenv/pyenv/wiki#suggested-build-environment
RUN --mount=type=cache,target=/var/lib/apt/lists,sharing=private \
    --mount=type=cache,target=/var/cache/apt/archives,sharing=private \
    set -x \
    && apt-get update \
    && apt-get install --yes --no-install-recommends \
    bash-completion \
    build-essential \
    cargo \
    connect-proxy \
    curl \
    dialog \
    git \
    git-lfs \
    google-perftools \
    graphviz \
    language-pack-ja \
    less \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libbz2-dev \
    libcairo2 \
    libcups2 \
    libffi-dev \
    libgbm1 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libgtk-3-0 \
    liblzma-dev\
    libncursesw5-dev \
    libnss3 \
    libopencv-core-dev \
    libpango-1.0-0 \
    libreadline-dev \
    libsm6 \
    libsqlite3-dev \
    libssl-dev \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxkbcommon0 \
    libxml2-dev \
    libxmlsec1-dev \
    libxrandr2 \
    libxrender1 \
    locate \
    netcat-openbsd \
    openssh-client \
    openssh-server \
    p7zip-full \
    pandoc \
    poppler-data \
    poppler-utils \
    psmisc \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-full \
    python3-pip \
    rsync \
    sudo \
    tesseract-ocr \
    tesseract-ocr-jpn \
    texlive-fonts-recommended \
    texlive-plain-generic \
    texlive-xetex \
    tk-dev \
    tmux \
    tmuxinator \
    vim \
    xz-utils \
    zip \
    zlib1g-dev \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

# Docker (DooD用。--volume="/var/run/docker.sock:/var/run/docker.sock" をつけて実行する。)
RUN --mount=type=cache,target=/var/lib/apt/lists,sharing=private \
    --mount=type=cache,target=/var/cache/apt/archives,sharing=private \
    set -x \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null  \
    && apt-get update \
    && apt-get install --yes --no-install-recommends docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# devpi-server用
ARG PIP_TRUSTED_HOST=""
ARG PIP_INDEX_URL=""
ARG PIP_RETRIES=10
ARG PIP_TIMEOUT=180
ARG PIP_DEFAULT_TIMEOUT=180

RUN --mount=type=cache,target=/root/.cache/pip set -x \
    && pip install --upgrade pip \
    && pip install wheel cython
RUN --mount=type=cache,target=/root/.cache/pip set -x \
    && pip install --upgrade pip \
    && pip install \
    albumentations \
    ansible \
    ansible-lint \
    av \
    awscli \
    azure-identity \
    bashplotlib \
    better_exceptions \
    boto3 \
    catboost \
    category_encoders \
    cookiecutter \
    diskcache \
    eli5 \
    ensemble-boxes \
    feather-format \
    imageio \
    imbalanced-learn \
    imgaug \
    imgdup \
    iterative-stratification \
    jaconv \
    janome \
    japanize-matplotlib \
    joblib \
    jpholiday \
    lightgbm \
    matplotlib \
    mecab-python3 \
    mojimoji \
    motpy \
    natsort \
    nltk \
    numba \
    nvitop \
    opencv-python-headless \
    openpyxl \
    optuna \
    pandas \
    pandas-profiling \
    passlib \
    pillow \
    plotly \
    polars[all] \
    pycryptodome \
    pydot \
    pykalman \
    pyod \
    pypandoc \
    python-datauri \
    python-dotenv \
    python-utils \
    pyyaml \
    requests \
    schedule \
    scikit-image \
    scikit-learn \
    seaborn \
    solrpy \
    sympy \
    tabulate \
    tqdm \
    ulid-py\>=1.1 \
    xgboost \
    xlrd \
    xlwt \
    ;
RUN --mount=type=cache,target=/root/.cache/pip set -x \
    && pip install \
    keras-cv \
    onnxmltools \
    segmentation-models \
    tensorboard \
    tensorboard-plugin-profile \
    tensorflow~=2.14.0 \
    tensorflow-addons[tensorflow] \
    tensorflow-datasets \
    tensorflow-hub \
    tf2onnx \
    ;

# TFがエラーにならないことの確認
RUN set -x \
    && python3 -c "import tensorflow as tf;print(tf.config.list_physical_devices('GPU'))" 2>&1 | tee /tmp/check.log \
    && grep -q 'failed call to cuInit: CUDA_ERROR_NO_DEVICE' /tmp/check.log \
    && rm -f /tmp/check.log

RUN --mount=type=cache,target=/root/.cache/pip set -x \
    && pip install \
    pandas-stubs \
    pre-commit \
    pyfltr \
    sqlalchemy-stubs \
    types-Flask \
    types-Pillow \
    types-PyYAML \
    types-Werkzeug \
    types-click \
    types-requests \
    ;
RUN --mount=type=cache,target=/root/.cache/pip set -x \
    && pip install \
    pip-tools \
    pipdeptree \
    pipenv \
    poetry \
    && poetry self add poetry-plugin-export
RUN --mount=type=cache,target=/root/.cache/pip set -x \
    && pip install \
    flask\<2.3 \
    flask-login \
    flask-migrate \
    flask-sqlalchemy \
    asyncio \
    fastapi[all] \
    sse-starlette \
    ;
RUN --mount=type=cache,target=/root/.cache/pip set -x \
    && pip install \
    sphinx \
    sphinx-autobuild \
    sphinx-autodoc-typehints \
    sphinx_rtd_theme \
    ;
RUN --mount=type=cache,target=/root/.cache/pip set -x \
    && pip install \
    kaggle \
    signate \
    stickytape \
    ;

# PyTorch関連: https://pytorch.org/get-started/locally/
RUN --mount=type=cache,target=/root/.cache/pip set -x \
    # PyTorchが既にインストールされてしまっていないことの確認
    && test $(pip freeze | grep ^torch== | wc -l) -eq 0 \
    # PyTorchとそれに依存するものたちのインストール
    && pip install torch torchtext torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 \
    && pip install \
    accelerate \
    anthropic \
    auto-gptq \
    bitsandbytes \
    datasets \
    diffusers \
    einops \
    ginza \
    google-cloud-aiplatform \
    google-generativeai \
    jmespath \
    langchain \
    lightning \
    litellm \
    llama-index\>=0.8.8 \
    openai \
    pretrainedmodels \
    safetensors \
    sentence-transformers \
    spacy \
    tiktoken \
    tokenizers \
    torch \
    torchaudio \
    torchtext \
    torchvision \
    transformers-stream-generator \
    transformers[ja,sentencepiece]\>=4.34.0 \
    triton \
    unstructured[all-docs] \
    --extra-index-url https://download.pytorch.org/whl/cu118
    # faiss-gpu\>=1.7.2

# PyTorchがエラーにならないことの確認
RUN set -x \
    && python3 -c "import torch;print(torch.cuda.get_device_name())" 2>&1 | tee /tmp/check.log \
    #&& grep -q "Error 34: CUDA driver is a stub library" /tmp/check.log \
    && grep -q "RuntimeError: No CUDA GPUs are available" /tmp/check.log \
    && rm -f /tmp/check.log

# 辞書など
# https://github.com/nltk/nltk/issues/1825
RUN set -x \
    && python3 -m nltk.downloader --dir=/usr/local/share/nltk_data popular --quiet --exit-on-error
RUN set -x \
    && python3 -m spacy download en_core_web_sm --no-cache
RUN set -x \
    && python3 -m unidic download

# nodejs
COPY --from=node /usr/local/bin/node /usr/local/bin/
COPY --from=node /usr/local/lib/node_modules/ /usr/local/lib/node_modules/
COPY --from=node /usr/local/include/node/ /usr/local/include/node/
COPY --from=node /usr/local/share/doc/node/ /usr/local/share/doc/node/
COPY --from=node /usr/local/share/man/man1/node.1 /usr/local/share/man/man1/
RUN --mount=type=cache,target=/root/.npm set -x \
    && ln -s /usr/local/bin/node /usr/local/bin/nodejs \
    && ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && ln -s /usr/local/lib/node_modules/corepack/dist/corepack.js /usr/local/bin/corepack \
    && npm update -g \
    && npm install -g pyright npm-check-updates prettier eslint

# jupyter関連
RUN --mount=type=cache,target=/root/.cache \
    --mount=type=cache,target=/root/.npm \
    set -x \
    && pip install \
    ipywidgets \
    jupyter-dash \
    jupyterlab-code-formatter \
    jupyterlab-git \
    jupyterlab-language-pack-ja-JP \
    jupyterlab-widgets \
    jupyterlab\>=3.4 \
    && (jupyter lab build --dev-build=False --minimize=False --debug-log-path=/tmp/jupyterlab-build.log || (cat /tmp/jupyterlab-build.log && false))

# # LightGBM
# # 参考: https://github.com/microsoft/LightGBM/issues/586
# RUN set -x \
#     && mkdir -p /etc/OpenCL/vendors \
#     && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd \
#     && pip install --no-binary :all: --install-option=--gpu lightgbm || (cat /root/LightGBM_compilation.log && false)

# # horovod
# # 参考: https://github.com/horovod/horovod/blob/master/docker/horovod/Dockerfile
# RUN set -x \
#     && ldconfig /usr/local/cuda/lib64/stubs \
#     && HOROVOD_WITH_MPI=1 HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MXNET=0 \
#     pip install horovod \
#     && ldconfig

# # 最後にPillow-SIMD
# # → 2021/08/13現在メンテされてなさそう: <https://github.com/uploadcare/pillow-simd/issues/93>
# #RUN set -x && \
# #    CC="cc -mavx2" pip install --force-reinstall Pillow-SIMD

# # npm
# RUN --mount=type=cache,target=/root/.npm set -x \
#     && npm install -g pyright npm-check-updates

# サイズは増えるけどやっておくと使うときに便利かもしれない諸々
RUN set -x \
    && apt-get update \
    && updatedb

# ユーザー作成
ARG RUN_USER=user
ARG RUN_UID=1000
RUN set -x \
    && useradd --create-home --shell=/bin/bash --uid=$RUN_UID --groups=sudo $RUN_USER

RUN set -x \
    # sshd用ディレクトリ作成
    && mkdir --mode=744 /var/run/sshd \
    # sshd用設定(~/.ssh/environmentを読む、KeepAliveする)
    && echo 'PermitUserEnvironment yes' > /etc/ssh/sshd_config.d/docker.conf \
    && echo 'ClientAliveInterval 15' >> /etc/ssh/sshd_config.d/docker.conf \
    && echo 'ClientAliveCountMax 10' >> /etc/ssh/sshd_config.d/docker.conf \
    && /usr/sbin/sshd -t \
    # 環境変数設定
    && echo 'export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH' > /etc/profile.d/docker.sh \
    && echo 'export BETTER_EXCEPTIONS=1' >> /etc/profile.d/docker.sh \
    && echo 'export TF_FORCE_GPU_ALLOW_GROWTH=true' >> /etc/profile.d/docker.sh \
    # sudoでhttp_proxyなどが引き継がれるようにしておく
    && echo 'Defaults env_keep += "http_proxy https_proxy ftp_proxy no_proxy PIP_TRUSTED_HOST PIP_INDEX_URL"' > /etc/sudoers.d/docker \
    && echo 'Defaults always_set_home' >> /etc/sudoers.d/docker \
    # $RUN_USERをパスワード無しでsudoできるようにしておく
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers.d/docker \
    && chmod 0440 /etc/sudoers.d/* \
    && visudo --check \
    # # completion
    # && poetry completions bash > /etc/bash_completion.d/poetry.bash-completion \
    # 最後に念のため
    && ldconfig

# sshd以外の使い方をするとき用環境変数色々
ENV TZ='Asia/Tokyo' \
    LANG='ja_JP.UTF-8' \
    PYTHONIOENCODING='utf-8' \
    PYTHONDONTWRITEBYTECODE=1 \
    BETTER_EXCEPTIONS=1 \
    TF_FORCE_GPU_ALLOW_GROWTH='true'

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
