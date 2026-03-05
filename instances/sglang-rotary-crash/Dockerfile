FROM rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260226
WORKDIR /workspace
RUN git clone https://github.com/sgl-project/sglang.git /workspace/sglang && \
    cd /workspace/sglang && git checkout 90a0d66e1e46f67ceba1c1c5061180e6af1cc748
COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
CMD ["sleep", "infinity"]
