ARG BASE_IMAGE=rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260226
FROM ${BASE_IMAGE}
WORKDIR /workspace
RUN git clone https://github.com/sgl-project/sglang.git /workspace/sglang && \
    cd /workspace/sglang && git checkout f1088beb6ab0ead4406b87b9e46d7cac69f40500
COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
CMD ["sleep", "infinity"]
