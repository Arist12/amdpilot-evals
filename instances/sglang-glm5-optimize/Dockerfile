FROM rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260226

WORKDIR /workspace

RUN git clone --depth 1 https://github.com/sgl-project/sglang.git /workspace/sglang

COPY patch_glm5.py /tmp/patch_glm5.py
RUN python3 /tmp/patch_glm5.py && rm /tmp/patch_glm5.py

COPY bench_glm5.sh /workspace/bench_glm5.sh
RUN chmod +x /workspace/bench_glm5.sh

COPY task_description.md /workspace/task_description.md

CMD ["sleep", "infinity"]
