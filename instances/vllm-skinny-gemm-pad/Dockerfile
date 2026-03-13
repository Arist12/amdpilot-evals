ARG BASE_IMAGE=rocm/vllm-dev:nightly
FROM ${BASE_IMAGE}
WORKDIR /workspace

# Clone vllm source at pre-fix commit using treeless clone (much faster than full)
RUN git clone --filter=blob:none https://github.com/vllm-project/vllm.git /workspace/vllm && \
    cd /workspace/vllm && git checkout 9251ed5c4fc6c954a5cdc5399d9d4f25ea5a8dd3

# Copy pre-compiled .so extensions from the nightly image into the source tree
RUN cp /usr/local/lib/python3.12/dist-packages/vllm/_rocm_C.abi3.so /workspace/vllm/vllm/ && \
    cp /usr/local/lib/python3.12/dist-packages/vllm/_C.abi3.so /workspace/vllm/vllm/ && \
    cp /usr/local/lib/python3.12/dist-packages/vllm/_moe_C.abi3.so /workspace/vllm/vllm/ 2>/dev/null || true && \
    cp /usr/local/lib/python3.12/dist-packages/vllm/cumem_allocator.abi3.so /workspace/vllm/vllm/ 2>/dev/null || true

# Make the source tree importable by prepending to PYTHONPATH.
# The agent edits files in /workspace/vllm/ and they take effect immediately.
ENV PYTHONPATH=/workspace/vllm:${PYTHONPATH}

# Generate a stub _version.py to suppress import warnings
RUN echo '__version__ = "0.0.dev0"; __version_tuple__ = (0, 0, "dev0")' \
    > /workspace/vllm/vllm/_version.py

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
CMD ["sleep", "infinity"]
