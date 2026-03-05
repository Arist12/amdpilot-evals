FROM rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260226
WORKDIR /workspace

# Revert the moe_align_block_size kernel to pre-optimization version
RUN cd /sgl-workspace/aiter && \
    git checkout 2d76e1812385b7688f938120ea86f8a70778dfdb -- csrc/kernels/moe_align_block_size_kernels.cu && \
    /opt/venv/bin/python3 setup.py develop 2>&1 | tail -5

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
CMD ["sleep", "infinity"]
