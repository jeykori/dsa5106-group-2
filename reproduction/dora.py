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
            lora_dropout: float = 0.0,
        ):
        super().__init__()
        self.base_layer = base_layer

        out_features, in_features = base_layer.weight.shape

        # LoRA adapters
        self.lora_A = torch.nn.Linear(in_features, r, bias=False)
        self.lora_B = torch.nn.Linear(r, out_features, bias=False)
        self.scaling = lora_alpha / r

        if lora_dropout > 0.0:
            self.lora_dropout = torch.nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = torch.nn.Identity()

        # Init LoRA weights
        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B.weight)

        # DoRA magnitude m: Initialize with ||W_0||_c
        with torch.no_grad():
            m = torch.linalg.norm(self.base_layer.weight, dim=1, keepdim=True)
        self.m = torch.nn.Parameter(m)

    def forward(self, x):
        previous_dtype = self.base_layer.weight.dtype

        # delta_V = B @ A * scaling
        delta_v = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        v_new = self.base_layer.weight + delta_v

        # From reference code: they don't calculate the full m * (V_new / ||V_new||), because this generates a new full tensor in memory
        # norm_scale = m / ||V_new|| (shape: [out_features, 1])
        # detach(): this is the "Reduction of Training Overhead" logic from the paper
        norm_scale = self.m / torch.linalg.norm(v_new, dim=1, keepdim=True).detach()

        norm_scale_view = norm_scale.view(1, 1, -1)
        dropout_x = self.lora_dropout(x)

        # y = norm_scale * (V @ x + delta_v @ x)
        base_output = torch.nn.functional.linear(x, self.base_layer.weight) # V @ x
        base_output_dropout = torch.nn.functional.linear(dropout_x, self.base_layer.weight) # V @ dropout(x)
        lora_output = self.lora_B(self.lora_A(dropout_x)) * self.scaling  # delta_v @ dropout(x)

        result = base_output + (norm_scale_view - 1) * base_output_dropout + norm_scale_view * lora_output

        if self.base_layer.bias is not None:
            result += self.base_layer.bias

        if result.dtype != previous_dtype:
            result = result.to(previous_dtype)
        return result

def inject_dora(
        model: PreTrainedModel,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        target_modules: list[str],
        modules_to_save: list[str] = None
    ):
    print(f"Injecting DoRA into pre-trained model")

    # Freeze all original weights
    for param in model.parameters():
            param.requires_grad = False

    # Unfreeze modules specified in modules_to_save
    # Needed for "classifier" layer in ViT
    if modules_to_save is not None:
        for name, param in model.named_parameters():
            if any(target in name for target in modules_to_save):
                param.requires_grad = True
                print(f"Unfrozen for training: {name}")

    module_names = list(model.named_modules())

    for name, module in module_names:
        # e.g. "model.layers.0.self_attn.q_proj"
        is_target = any(name.endswith("." + target) for target in target_modules)

        # patch for scenario where target is `output.dense` and we don't want to match `attention.output.dense`
        if is_target and "attention.output.dense" in name and "attention.output.dense" not in target_modules:
             is_target = False

        if is_target:
            if isinstance(module, torch.nn.Linear):
                # e.g "model.layers.0.self_attn", ".", "q_proj"
                parent_name, _, layer_name = name.rpartition(".")
                parent = model.get_submodule(parent_name)
                print(f"Injecting DoRA into: {name}")

                dora_layer = DoraLayer(
                    base_layer=module,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
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