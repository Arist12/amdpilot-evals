# Dockerfile for sglang-fused-moe-fix eval instance
#
# Builds on a ROCm base image with sglang cloned at the buggy commit.
# The agent's task is to find and fix the NameError in fused_moe.py.
#
# Build: docker build -t amdpilot-eval-sglang-fused-moe-fix .
# Note: The base image should be a ROCm image with PyTorch pre-installed.
#       Adjust BASE_IMAGE to match your environment.

ARG BASE_IMAGE=rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260226
FROM ${BASE_IMAGE}

WORKDIR /workspace

# Clone sglang at the buggy commit (ee5ccde introduced the bug)
RUN git clone --depth 50 https://github.com/sgl-project/sglang.git /workspace/sglang && \
    cd /workspace/sglang && \
    git checkout ee5ccde0ad9eefe96c2c29c9d7d9087d78a4500c

# Copy the test harness
COPY test_harness.py /workspace/test_harness.py

# Copy the task description (agent reads this)
COPY task_description.md /workspace/task_description.md

CMD ["sleep", "infinity"]
