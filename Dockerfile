# ==============================================================================
# STAGE 1: Builder (Shared compilation environment)
# ==============================================================================
FROM rust:1.76-bookworm AS builder
WORKDIR /usr/src/ocrrs

# Copy configuration files
COPY Cargo.toml Cargo.lock* ./

# Create dummy source files to pre-compile and cache dependencies
RUN mkdir -p src/nets src/math src/data src/bin && \
    echo "fn main() {}" > src/bin/train.rs && \
    echo "fn main() {}" > src/bin/server.rs && \
    echo "" > src/lib.rs && \
    echo "<h1>Stub</h1>" > src/bin/index.html

# Pre-compile dependencies using cache mounts
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/src/ocrrs/target \
    cargo build --release

# Copy actual source code
COPY src ./src/

# Force cargo to rebuild with the actual source code instead of the cached dummy files
RUN touch src/bin/train.rs src/bin/server.rs src/lib.rs

# Build release binaries
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/src/ocrrs/target \
    cargo build --release && \
    cp target/release/train . && \
    cp target/release/server .

# ==============================================================================
# STAGE 2: Train Target Runtime Image
# ==============================================================================
FROM debian:bookworm-slim AS train
WORKDIR /app

# Install dynamic library support
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy the train binary
COPY --from=builder /usr/src/ocrrs/train /app/train

# Default entrypoint to run training CLI
ENTRYPOINT ["/app/train"]

# ==============================================================================
# STAGE 3: Server Target Runtime Image
# ==============================================================================
FROM debian:bookworm-slim AS server
WORKDIR /app

# Install dynamic library support
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy the server binary
COPY --from=builder /usr/src/ocrrs/server /app/server

# Expose server port
EXPOSE 3000

# Default entrypoint/command to run the inference server
ENTRYPOINT ["/app/server"]
CMD ["--weights-dir", "/app/weights", "--port", "3000"]
