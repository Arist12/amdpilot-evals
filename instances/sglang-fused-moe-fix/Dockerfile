FROM rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260226
WORKDIR /workspace
RUN git clone https://github.com/sgl-project/sglang.git /workspace/sglang && \
    cd /workspace/sglang && git checkout ee5ccde0ad9eefe96c2c29c9d7d9087d78a4500c
COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
CMD ["sleep", "infinity"]
