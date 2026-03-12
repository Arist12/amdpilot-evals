#!/usr/bin/env python3
"""Patch sglang's get_config() to handle glm_moe_dsa model type.

transformers 4.57.1 doesn't recognize glm_moe_dsa, causing AutoConfig to
raise ValueError.  sglang already catches deepseek_v32 — extend the same
pattern for glm_moe_dsa.
"""

import sys

target = "/sgl-workspace/sglang/python/sglang/srt/utils/hf_transformers_utils.py"

with open(target) as f:
    content = f.read()

OLD = '''        except ValueError as e:
            if not "deepseek_v32" in str(e):
                raise e
            config = _load_deepseek_v32_model('''

NEW = '''        except ValueError as e:
            if "glm_moe_dsa" in str(e):
                from transformers import PretrainedConfig
                config = PretrainedConfig.from_pretrained(
                    model, trust_remote_code=trust_remote_code,
                    revision=revision, **kwargs
                )
            elif "deepseek_v32" in str(e):
                config = _load_deepseek_v32_model('''

if OLD not in content:
    print("WARNING: patch target not found — file may already be patched or changed", file=sys.stderr)
    sys.exit(0)

content = content.replace(OLD, NEW)

with open(target, "w") as f:
    f.write(content)

print("Patched get_config() for glm_moe_dsa support")
