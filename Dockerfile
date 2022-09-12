# https://hub.docker.com/r/nvidia/cuda/tags
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8
RUN set -x \
    && rm /etc/apt/apt.conf.d/docker-gzip-indexes \
    && rm /etc/apt/apt.conf.d/docker-no-languages

RUN set -x \
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
    && update-locale LANG=ja_JP.UTF-8 LANGUAGE='ja_JP:ja' \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN yes | unminimize

ARG PYTHON_VERSION=3.10

# libgl1 libglib2.0-0 libsm6 libxrender1 libxext6: opencv用
# libgomp1: LightGBM用
RUN set -ex \
    && apt-get update \
    && apt-get install --yes --no-install-recommends \
    git \
    git-lfs \
    graphviz \
    language-pack-ja \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libopencv-core-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    locate \
    openssh-client \
    openssh-server \
    p7zip-full \
    pandoc \
    python3-pip \
    python${PYTHON_VERSION}-full \
    sudo \
    texlive-fonts-recommended \
    texlive-plain-generic \
    texlive-xetex \
    tmux \
    tmuxinator \
    vim \
    zip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# python
RUN set -ex \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# # devpi-server用
# ARG PIP_TRUSTED_HOST=""
# ARG PIP_INDEX_URL=""
# ARG PIP_RETRIES=10
# ARG PIP_TIMEOUT=180
# ARG PIP_DEFAULT_TIMEOUT=180

RUN set -ex \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir wheel cython
RUN set -x \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
    albumentations \
    av \
    bashplotlib \
    better_exceptions \
    catboost \
    category_encoders \
    cookiecutter \
    eli5 \
    ensemble-boxes \
    feather-format \
    imageio \
    imbalanced-learn \
    imgaug \
    imgdup \
    iterative-stratification \
    janome \
    japanize-matplotlib \
    joblib \
    jpholiday \
    lightgbm \
    matplotlib \
    mecab-python3 \
    motpy \
    natsort \
    numba \
    opencv-python-headless \
    openpyxl \
    optuna \
    pandas \
    pandas-profiling \
    pillow \
    plotly \
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
    scikit-image \
    scikit-learn \
    seaborn \
    sympy \
    tabulate \
    tqdm \
    xgboost \
    xlrd \
    xlwt \
    ;
RUN set -x \
    && pip install --no-cache-dir \
    onnxmltools \
    segmentation-models \
    tensorboard \
    tensorboard-plugin-profile \
    tensorflow-addons[tensorflow] \
    tensorflow-datasets \
    tensorflow-hub \
    tensorflow \
    tf2onnx \
    ;

ENV LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH

# TFがエラーにならないことの確認
RUN set -x \
    && python3 -c "import tensorflow as tf;print(tf.config.list_physical_devices('GPU'))" 2>&1 | tee /tmp/check.log \
    && (if grep -q 'Could not load dynamic library' /tmp/check.log  ; then false ; else echo OK! ; fi) \
    && rm -f /tmp/check.log

RUN set -x \
    && pip install --no-cache-dir \
    pre-commit \
    pyfltr \
    types-Pillow \
    types-PyYAML \
    types-requests \
    ;
RUN set -x \
    && pip install --no-cache-dir \
    pip-tools \
    pipdeptree \
    pipenv \
    poetry \
    ;
RUN set -x \
    && pip install --no-cache-dir \
    Flask \
    Flask-Login \
    Flask-Migrate \
    Flask-Restless \
    Flask-SQLAlchemy \
    ;
RUN set -x \
    && pip install --no-cache-dir \
    sphinx \
    sphinx-autobuild \
    sphinx-autodoc-typehints \
    sphinx_rtd_theme \
    ;
RUN set -x \
    && pip install --no-cache-dir \
    kaggle \
    signate \
    stickytape \
    ;

# PyTorch関連: https://pytorch.org/get-started/locally/
RUN set -x \
    # PyTorchが既にインストールされてしまっていないことの確認
    && test $(pip freeze | grep ^torch== | wc -l) -eq 0 \
    # PyTorchとそれに依存するものたちのインストール
    && pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 \
    && pip install --no-cache-dir \
    datasets \
    diffusers \
    faiss-gpu \
    ginza \
    pretrainedmodels \
    pytorch-lightning \
    spacy \
    tokenizers \
    torchtext \
    transformers[ja] \
    torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# PyTorchがエラーにならないことの確認
RUN set -x \
    && python3 -c "import torch;print(torch.cuda.get_device_name())" 2>&1 | tee /tmp/check.log \
    #&& grep -q "Error 34: CUDA driver is a stub library" /tmp/check.log \
    && grep -q "RuntimeError: No CUDA GPUs are available" /tmp/check.log \
    && rm -f /tmp/check.log

# # 辞書など
# # https://github.com/nltk/nltk/issues/1825
# RUN set -x \
#     && python3 -m nltk.downloader --dir=/usr/local/share/nltk_data popular --quiet --exit-on-error
# RUN set -x \
#     && python3 -m spacy download en_core_web_sm --no-cache
# RUN set -x \
#     && python3 -m unidic download

# # nodejs
# ARG NODEJS_VERSION=v12.18.3
# RUN set -x \
#     && wget -q -O- https://nodejs.org/dist/$NODEJS_VERSION/node-$NODEJS_VERSION-linux-x64.tar.xz | tar xJ -C /tmp/ \
#     && mv /tmp/node-$NODEJS_VERSION-linux-x64/bin/* /usr/local/bin/ \
#     && mv /tmp/node-$NODEJS_VERSION-linux-x64/lib/* /usr/local/lib/ \
#     && mv /tmp/node-$NODEJS_VERSION-linux-x64/include/* /usr/local/include/ \
#     && mv /tmp/node-$NODEJS_VERSION-linux-x64/share/doc/* /usr/local/share/doc/ \
#     && mv /tmp/node-$NODEJS_VERSION-linux-x64/share/man/man1/* /usr/local/share/man/man1 \
#     && rm -rf /tmp/node-$NODEJS_VERSION-linux-x64

# jupyter関連
RUN set -x \
    && pip install --no-cache-dir \
    jupyterlab \
    ;

# # LightGBM
# # 参考: https://github.com/microsoft/LightGBM/issues/586
# RUN set -x \
#     && mkdir -p /etc/OpenCL/vendors \
#     && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd \
#     && pip install --no-cache-dir --no-binary :all: --install-option=--gpu lightgbm || (cat /root/LightGBM_compilation.log && false)

# # horovod
# # 参考: https://github.com/horovod/horovod/blob/master/docker/horovod/Dockerfile
# RUN set -x \
#     && ldconfig /usr/local/cuda/lib64/stubs \
#     && HOROVOD_WITH_MPI=1 HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MXNET=0 \
#     pip install --no-cache-dir horovod \
#     && ldconfig

# # 最後にPillow-SIMD
# # → 2021/08/13現在メンテされてなさそう: <https://github.com/uploadcare/pillow-simd/issues/93>
# #RUN set -x && \
# #    CC="cc -mavx2" pip install --no-cache-dir --force-reinstall Pillow-SIMD

# # npm
# RUN set -x \
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
    && echo 'PermitUserEnvironment yes' > /etc/ssh/sshd_config.d/aiserver.conf \
    && echo 'ClientAliveInterval 15' >> /etc/ssh/sshd_config.d/aiserver.conf \
    && echo 'ClientAliveCountMax 10' >> /etc/ssh/sshd_config.d/aiserver.conf \
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
    # # libcuda.so.1を参照できるようにする
    # && echo '/usr/local/cuda/compat' > /etc/ld.so.conf.d/nvidia-compat.conf \
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
