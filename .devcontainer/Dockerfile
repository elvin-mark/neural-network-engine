FROM mcr.microsoft.com/devcontainers/rust:latest

# Install build dependencies, debugging tools, and performance profiling tools
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        build-essential \
        cmake \
        pkg-config \
        libssl-dev \
        lldb \
        gdb \
        valgrind \
        htop \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Add standard Rust components
RUN rustup component add clippy rustfmt llvm-tools-preview
