import yaml
from copy import deepcopy
import re
import numpy as np


class PromptLibrary:
    def __init__(self, yaml_path):
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.raw_prompts = cfg["prompts"]
        self.prompts = self.raw_prompts
        self.layouts = cfg["layouts"]

    def render(self, name, context, layout_id=0):
        spec = self.prompts[name]
        if "layout" in spec:
            layout = spec["layout"]
        else:
            layout_id = layout_id or np.random.randint(0, len(self.layouts) - 1)
            layout = self.layouts[layout_id]
        comp = spec["components"]
        # ex layout: '{intro}\n{skill_token}\n{output_instruct}\n{examples}\n{input_fmt}\n'
        layout_fields = re.findall(r"{(.*?)}", layout)
        # check that all components in layout are in spec
        assert all(
            k in comp for k in layout_fields if k not in ["examples"]
        ), f"Layout components not in prompt spec, missing: {[k for k in layout_fields if k not in comp]}"

        # Generate example block using input_fmt and output_fmt
        def render_examples(input_fmt, output_fmt, ex_list):
            lines = []
            for ex in ex_list:
                line = input_fmt.format(**ex) + output_fmt.format(**ex)
                lines.append(line)
            return "\n".join(lines) if lines else ""

        filled = {}
        for key, val in comp.items():
            if isinstance(val, list):
                val = np.random.choice(val)
            filled[key] = val.format(**context)

        # Add examples if any
        if "examples" in layout_fields:
            ex = context["examples"]
            input_fmt = comp["input_fmt"]
            output_fmt = comp["output_fmt"]
            filled["examples"] = render_examples(input_fmt, output_fmt, ex)

        return layout.format(**filled)
