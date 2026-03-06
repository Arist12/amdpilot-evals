FROM rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260226
WORKDIR /workspace

# Revert the sigmoid kernel to the pre-optimization version
RUN cd /sgl-workspace/aiter && \
    git checkout cbf2df77632d4f3389178b6528d40af9f34be7e6 -- csrc/kernels/unary_operator.cu && \
    /opt/venv/bin/python3 setup.py develop 2>&1 | tail -5

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
CMD ["sleep", "infinity"]
