ARG BASE_IMAGE=rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260226
FROM ${BASE_IMAGE}
WORKDIR /workspace
RUN git clone https://github.com/sgl-project/sglang.git /workspace/sglang && \
    cd /workspace/sglang && git checkout 91230dcca890252018e04f838db7f19995a62aa1
COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
CMD ["sleep", "infinity"]
