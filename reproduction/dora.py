import math

import torch
from transformers import PreTrainedModel

# TODO: (to consider) the reference code has `Wdecompose` as a param, to toggle between LoRA and DoRA
class DoraLayer(torch.nn.Module):
    def __init__(
            self,
            base_layer: torch.nn.Linear,
            r: int = 16,
            lora_alpha: int = 32,
        ):
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad = False # Freeze V

        out_features, in_features = base_layer.weight.shape

        # LoRA adapters
        self.lora_A = torch.nn.Linear(in_features, r, bias=False)
        self.lora_B = torch.nn.Linear(r, out_features, bias=False)
        self.scaling = lora_alpha / r

        # Init LoRA weights
        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B.weight)

        # DoRA magnitude m: Initialize with ||W_0||_c
        with torch.no_grad():
            m = torch.linalg.norm(self.base_layer.weight, dim=1, keepdim=True)
        self.m = torch.nn.Parameter(m)

    def forward(self, x):
        previous_dtype = self.base_layer.weight.dtype

        with torch.no_grad():
            # delta_V = B @ A * scaling
            delta_v = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            v_new = self.base_layer.weight + delta_v

        # From reference code: they don't calculate the full m * (V_new / ||V_new||), because this generates a new full tensor in memory
        # norm_scale = m / ||V_new|| (shape: [out_features, 1])
        # detach(): this is the "Reduction of Training Overhead" logic from the paper
        norm_scale = self.m / torch.linalg.norm(v_new, dim=1, keepdim=True).detach()

        # y = norm_scale * (V @ x + delta_v @ x)
        base_output = self.base_layer(x) # V @ x
        lora_output = self.lora_B(self.lora_A(x)) * self.scaling  # delta_v @ x

        result = norm_scale.view(1, 1, -1) * (base_output + lora_output)

        if self.base_layer.bias is not None:
            result += self.base_layer.bias

        if result.dtype != previous_dtype:
            result = result.to(previous_dtype)
        return result

def inject_dora(model: PreTrainedModel, r: int, lora_alpha: int, target_modules: list[str]):
    print(f"Injecting DoRA into pre-trained model")
    module_names = list(model.named_modules())

    for name, module in module_names:
        # e.g. "model.layers.0.self_attn.q_proj"
        leaf_name = name.split(".")[-1]
        if leaf_name in target_modules or any(target in name for target in target_modules):
            if isinstance(module, torch.nn.Linear):
                # e.g "model.layers.0.self_attn", ".", "q_proj"
                parent_name, _, layer_name = name.rpartition(".")
                parent = model.get_submodule(parent_name)
                print(f"Injecting DoRA into: {name}")

                dora_layer = DoraLayer(
                    base_layer=module,
                    r=r,
                    lora_alpha=lora_alpha
                )

                setattr(parent, layer_name, dora_layer)

    return model

@torch.no_grad()
def merge_and_unload_dora(model: torch.nn.Module):
    module_names = list(model.named_modules())

    for name, module in module_names:
        if isinstance(module, DoraLayer):
            # Calculate merged weights
            base = module.base_layer
            lora_A = module.lora_A.weight
            lora_B = module.lora_B.weight
            m = module.m
            scaling = module.scaling

            delta_v = (lora_B @ lora_A) * scaling
            v_new = base.weight + delta_v

            v_norm = torch.linalg.norm(v_new, dim=1, keepdim=True)

            w_merged = m * (v_new / v_norm)

            # Create new layer
            new_linear = torch.nn.Linear(
                base.in_features,
                base.out_features,
                bias=(base.bias is not None)
            )

            new_linear.weight.copy_(w_merged.to(base.weight.dtype))
            if base.bias is not None:
                new_linear.bias.copy_(base.bias)

            new_linear.to(device=base.weight.device, dtype=base.weight.dtype)

            # Replace with new layer
            parent_name, _, layer_name = name.rpartition(".")
            parent = model.get_submodule(parent_name)
            setattr(parent, layer_name, new_linear)

            print(f"Merged and unloaded DoRA for module: {name}")

    return model