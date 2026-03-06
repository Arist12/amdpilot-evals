FROM rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260226
WORKDIR /workspace

# Revert the sigmoid kernel to the pre-optimization version,
# force JIT rebuild, and prevent solution leak via git history
RUN cd /sgl-workspace/aiter && \
    git checkout cbf2df77632d4f3389178b6528d40af9f34be7e6 -- csrc/kernels/unary_operator.cu && \
    rm -rf aiter/jit/module_aiter_unary.so aiter/jit/build/module_aiter_unary && \
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
