# Reproducible environment: the official vLLM 0.31.0 image + this plugin.
# Pin by digest for publication, e.g. vllm/vllm-openai@sha256:<digest>
FROM vllm/vllm-openai:v0.31.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/vllm-fastdecode
COPY . .
RUN pip install --no-cache-dir ninja datasets scipy pytest \
    && pip install --no-cache-dir -e . --no-deps

# The CPU kernel is JIT-compiled on first use with -march=native, i.e. for the
# machine that runs the container, not the one that built it.
ENTRYPOINT []
CMD ["bash"]
