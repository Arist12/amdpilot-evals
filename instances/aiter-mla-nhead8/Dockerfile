# Dockerfile for aiter-mla-nhead8 eval instance
#
# Uses a ROCm base image with aiter pre-built. The agent's task is to add
# nhead=8 support for MLA decode via padding.
#
# Build: docker build -t amdpilot-eval-aiter-mla-nhead8 .
# Note: Must be built on an AMD GPU machine (aiter requires HIP compilation).

ARG BASE_IMAGE=rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260226
FROM ${BASE_IMAGE}

WORKDIR /workspace

# Clone aiter at the commit before PR #2138
# Use main branch which doesn't have nhead=8 support yet
RUN git clone --recursive https://github.com/ROCm/aiter.git /workspace/aiter && \
    cd /workspace/aiter && \
    git checkout main

# Build aiter (requires GPU at build time for HIP compilation)
RUN cd /workspace/aiter && \
    /opt/venv/bin/python3 setup.py develop

# Copy test harness and task description
COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md

CMD ["sleep", "infinity"]
