FROM rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260226
WORKDIR /workspace

# Revert the sampling kernel to the pre-optimization version,
# clear JIT cache, and prevent solution leak via git history
RUN cd /sgl-workspace/aiter && \
    git checkout 7411c99753f0661a3eecdbdb1b36feb58539f62b -- csrc/cpp_itfs/sampling/sampling.cuh && \
    rm -rf /root/.aiter/build/top_k_top_p_sampling_from_probs* && \
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
