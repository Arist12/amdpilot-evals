FROM rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260226
WORKDIR /workspace
RUN git clone https://github.com/vllm-project/vllm.git /workspace/vllm && \
    cd /workspace/vllm && git checkout 2ce6f3cf67934ebe199188c9a1f83ff1c2d8ba96
RUN pip install cbor2 lm-format-enforcer 2>/dev/null || true
COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
CMD ["sleep", "infinity"]
