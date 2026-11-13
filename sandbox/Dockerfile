# Based on Fedora
FROM fedora:42

# Accept TARGET_ARCH build argument
ARG TARGET_ARCH=x86_64

# Set up the environment variables
ENV SANDBOX_DIR=/home/agent \
    SHELL=/bin/bash \
    TZ=Etc/UTC \
    DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/miniconda3/bin:$PATH

# Setup agent user with sudo access
RUN useradd -m -d /home/agent -s ${SHELL} agent && \
    echo "agent ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/agent && \
    echo "Defaults    env_keep += \"PATH\"" >> /etc/sudoers.d/agent && \
    chmod 0440 /etc/sudoers.d/agent

# Install common dev tools
RUN dnf -y install dnf-plugins-core && \
    dnf -y remove selinux-policy* && \
    dnf -y update && \
    dnf -y install \
    gcc gcc-c++ make git git-lfs llvm llvm-devel clang clang-devel \
    nodejs python3.12 python3.12-devel cmake openssh-server \
    tmux lsof strace gdb ltrace valgrind inotify-tools jq pv bzip2 unzip \
    p7zip wget curl sudo file tree which gettext-envsubst patch openssl \
    rsync zip nmap-ncat ripgrep perf poppler-utils lapack-devel blas-devel \
    openssl-devel libffi-devel procps-ng sysstat htop \
    libtiff-devel golang awk \
    # System and networking utilities
    hostname net-tools iproute iputils bind-utils tcpdump traceroute mtr \
    psmisc lsof netcat telnet whois tar gzip less findutils

# Install Miniconda in /opt and set permissions
USER root
RUN mkdir -p /opt/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${TARGET_ARCH}.sh -O /opt/miniconda3/miniconda.sh && \
    bash /opt/miniconda3/miniconda.sh -b -u -p /opt/miniconda3 && \
    rm /opt/miniconda3/miniconda.sh && \
    chown -R agent:agent /opt/miniconda3 && \
    chmod -R u+w /opt/miniconda3


# Configure conda
RUN /opt/miniconda3/bin/conda init --all && \
    /opt/miniconda3/bin/conda config --append channels conda-forge

# Create system-wide conda initialization
RUN echo '. /opt/miniconda3/etc/profile.d/conda.sh' >> /etc/bashrc && \
    echo 'source /opt/miniconda3/bin/activate' >> /etc/bashrc && \
    mkdir -p /etc/profile.d && \
    echo '. /opt/miniconda3/etc/profile.d/conda.sh' >> /etc/profile.d/conda.sh && \
    echo 'source /opt/miniconda3/bin/activate' >> /etc/profile.d/conda.sh && \
    chmod +x /etc/profile.d/conda.sh

# Switch back to root for system configurations
USER root

# Setup Python 3.12 as default python
RUN alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    alternatives --set python3 /usr/bin/python3.12 && \
    alternatives --install /usr/bin/python python /usr/bin/python3 1

# Create necessary directories for pnpm
RUN mkdir -p ${SANDBOX_DIR}/.local/share/pnpm && \
    touch ${SANDBOX_DIR}/.bashrc && \
    chown -R agent:agent ${SANDBOX_DIR}/.local && \
    chown agent:agent ${SANDBOX_DIR}/.bashrc && \
    chown agent:agent ${SANDBOX_DIR}

COPY configs/gitignore ${SANDBOX_DIR}/.gitignore
RUN chown agent:agent ${SANDBOX_DIR}/.gitignore && \
    chmod +w ${SANDBOX_DIR}/.gitignore

# Switch back to agent user for remaining setup
USER agent
WORKDIR ${SANDBOX_DIR}

# Set directory permissions
RUN mkdir -p ${SANDBOX_DIR}/.ssh && \
    chmod 700 ${SANDBOX_DIR}/.ssh && \
    touch ${SANDBOX_DIR}/.ssh/authorized_keys && \
    chmod 600 ${SANDBOX_DIR}/.ssh/authorized_keys

RUN curl https://raw.githubusercontent.com/github/gitignore/main/Python.gitignore >> ${SANDBOX_DIR}/.gitignore && \
    curl https://raw.githubusercontent.com/github/gitignore/main/Node.gitignore >> ${SANDBOX_DIR}/.gitignore

# Install and setup pnpm
ENV PNPM_HOME=${SANDBOX_DIR}/.local/share/pnpm
ENV PATH=$PNPM_HOME:$PATH
ENV NODE_OPTIONS=--max_old_space_size=4096
RUN curl -fsSL https://get.pnpm.io/install.sh | ENV="${SANDBOX_DIR}/.bashrc" SHELL="/bin/bash" bash - && \
    . ${SANDBOX_DIR}/.bashrc && \
    echo "export PNPM_HOME=$PNPM_HOME" >> ${SANDBOX_DIR}/.bashrc && \
    echo "export PATH=$PNPM_HOME:\$PATH" >> ${SANDBOX_DIR}/.bashrc && \
    . ${SANDBOX_DIR}/.bashrc && \
    pnpm install -g typescript ts-node @types/node prettier eslint tsx

# Install and configure Rust using rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    . ${SANDBOX_DIR}/.cargo/env && \
    rustup component add rust-src && \
    echo '. ${SANDBOX_DIR}/.cargo/env' >> ${SANDBOX_DIR}/.bashrc

# Install LSP Servers for common languages with architecture awareness
RUN . ${SANDBOX_DIR}/.bashrc && \
    # Python - Pyright
    sudo dnf install -y npm && \
    sudo npm install -g pyright && \
    # JavaScript/TypeScript
    pnpm install -g typescript-language-server typescript && \
    # Rust - Install and configure Rust using rustup
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    . ${SANDBOX_DIR}/.cargo/env && \
    rustup component add rust-src rust-analyzer && \
    echo '. ${SANDBOX_DIR}/.cargo/env' >> ${SANDBOX_DIR}/.bashrc \
    # Go - Install gopls
    go install golang.org/x/tools/gopls@latest

# Configure environment
ENV HOME=${SANDBOX_DIR}

# Copy and install some base requirements
COPY base_requirements.txt /tmp/base_requirements.txt
RUN pip install -r /tmp/base_requirements.txt && \
    sudo rm /tmp/base_requirements.txt

# Copy and install agent dependencies (maintaining current approach)
COPY --from=base_agent --chown=agent:agent . /tmp/base_agent
RUN cd /tmp/base_agent && pip install -r requirements.txt

WORKDIR ${SANDBOX_DIR}

# Expose necessary ports (maintaining current approach)
EXPOSE 5000 80 22 443 8080 8000

# No credentials are built into the image: API keys and the Google
# credentials file are handed to containers when they are started (see
# sandbox_secret_args in runner.py, and SANDBOX_SECRETS in the Makefile).

# Set the entrypoint (maintaining current approach)
CMD ["/bin/bash", "--login"]
