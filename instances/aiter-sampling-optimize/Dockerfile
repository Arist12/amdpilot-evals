FROM rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260226
WORKDIR /workspace

# Revert the sampling kernel to the pre-optimization version
RUN cd /sgl-workspace/aiter && \
    git checkout 7411c99753f0661a3eecdbdb1b36feb58539f62b -- csrc/cpp_itfs/sampling/sampling.cuh && \
    /opt/venv/bin/python3 setup.py develop 2>&1 | tail -5

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
CMD ["sleep", "infinity"]
