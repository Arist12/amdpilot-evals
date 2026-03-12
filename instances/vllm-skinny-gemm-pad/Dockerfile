ARG BASE_IMAGE=rocm/vllm-dev:nightly
FROM ${BASE_IMAGE}
WORKDIR /workspace

RUN git clone https://github.com/vllm-project/vllm.git /workspace/vllm && \
    cd /workspace/vllm && git checkout 9251ed5c4fc6c954a5cdc5399d9d4f25ea5a8dd3

RUN cd /workspace/vllm && \
    pip install -e . --no-build-isolation 2>&1 | tail -30 || \
    (echo "Full build failed, trying extension-only build..." && \
     python setup.py build_ext --inplace 2>&1 | tail -30) || true

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
CMD ["sleep", "infinity"]
