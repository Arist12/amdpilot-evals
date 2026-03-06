FROM rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260226
WORKDIR /workspace

# Revert the MLA reduce kernel to the pre-optimization version,
# force JIT rebuild, and prevent solution leak via git history
RUN cd /sgl-workspace/aiter && \
    git checkout 9f8ef5a5dd429ef24227cc8c97306a5ff912d604 -- csrc/kernels/mla/reduce.cu && \
    rm -rf aiter/jit/module_mla_reduce.so aiter/jit/build/module_mla_reduce && \
    /opt/venv/bin/python3 setup.py develop 2>&1 | tail -5 && \
    git add -A && \
    git checkout --orphan _baseline && \
    git commit -m "baseline" --allow-empty && \
    git branch -D main 2>/dev/null; \
    git branch -M _baseline main && \
    git reflog expire --expire=now --all && \
    git gc --prune=now -q

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
CMD ["sleep", "infinity"]
