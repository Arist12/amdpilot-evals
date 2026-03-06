FROM rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260226
WORKDIR /workspace

# Revert the moe_align_block_size kernel to pre-optimization version,
# force JIT rebuild, and prevent solution leak via git history
RUN cd /sgl-workspace/aiter && \
    git checkout 2d76e1812385b7688f938120ea86f8a70778dfdb -- csrc/kernels/moe_align_block_size_kernels.cu && \
    rm -rf aiter/jit/module_moe_asm.so aiter/jit/build/module_moe_asm && \
    /opt/venv/bin/python3 setup.py develop 2>&1 | tail -5 && \
    git config user.email "build@eval" && git config user.name "eval" && \
    git add -A && \
    git checkout --orphan _baseline && \
    git commit -m "baseline" && \
    git branch -D main 2>/dev/null; \
    git branch -M _baseline main && \
    git reflog expire --expire=now --all && \
    git remote remove origin 2>/dev/null; \
    git gc --prune=now -q

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
CMD ["sleep", "infinity"]
