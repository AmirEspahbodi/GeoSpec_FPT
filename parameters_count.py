import torch
import json
from typing import Optional, Literal, Dict, Any, List, Set

def _bytes_to_readable(nbytes: int) -> str:
    kb = nbytes / 1024
    mb = kb / 1024
    gb = mb / 1024
    if gb >= 1:
        return f"{gb:.3f} GB"
    if mb >= 1:
        return f"{mb:.3f} MB"
    if kb >= 1:
        return f"{kb:.3f} KB"
    return f"{nbytes} bytes"

def model_params_summary(
    model: torch.nn.Module,
    *,
    include_param_table: bool = True,
    include_module_summary: bool = True,
    max_param_items: Optional[int] = None,
    return_type: Literal["json", "dict"] = "dict"
) -> Dict[str, Any] | str:
    """
    Produce a summary of a PyTorch model and return it as JSON (string) by default.
    If return_type == "dict" a Python dictionary is returned.

    Args:
        model: torch.nn.Module instance
        include_param_table: include per-parameter list (can be long)
        include_module_summary: include per-module param counts (direct params only)
        max_param_items: if provided, limits how many parameters to include in the param table
        return_type: "json" or "dict"

    Returns:
        JSON string or Python dict describing the model parameters and sizes.
    """
    # Collect unique parameters to avoid double-counting (shared params)
    unique_params: Dict[int, Dict[str, Any]] = {}
    ordered_param_names: List[str] = []

    for name, p in model.named_parameters(recurse=True):
        pid = id(p)
        if pid not in unique_params:
            unique_params[pid] = {
                "param_obj": p,
                "names": [name],  # keep names that refer to the same param (rare)
            }
        else:
            unique_params[pid]["names"].append(name)
        ordered_param_names.append((name, pid))

    # Totals
    total_params = 0
    trainable_params = 0
    total_bytes = 0
    devices: Set[str] = set()
    dtypes: Set[str] = set()

    for pid, info in unique_params.items():
        p = info["param_obj"]
        nelems = int(p.numel())
        total_params += nelems
        if p.requires_grad:
            trainable_params += nelems
        # element_size returns bytes per element
        total_bytes += int(nelems * p.element_size())
        devices.add(str(p.device))
        dtypes.add(str(p.dtype))

    non_trainable_params = total_params - trainable_params

    # Build per-parameter table (ordered by named_parameters order)
    param_list: List[Dict[str, Any]] = []
    seen_pids: Set[int] = set()
    for idx, (name, pid) in enumerate(ordered_param_names):
        if "cnn_backbone" in name or "side_vit" in name:
            continue
        if pid in seen_pids:
            # skip duplicates in the ordered list — we already report param once
            continue
        seen_pids.add(pid)
        info = unique_params[pid]
        p = info["param_obj"]
        entry = {
            "name": name,
            "all_aliases": info["names"],  # other names that point to same param (if any)
            "shape": tuple(p.shape),
            "numel": int(p.numel()),
            "dtype": str(p.dtype),
            "device": str(p.device),
            "requires_grad": bool(p.requires_grad),
            "bytes": int(p.numel() * p.element_size())
        }
        param_list.append(entry)
        if (max_param_items is not None) and (len(param_list) >= max_param_items):
            break

    # Module-level direct param counts (recurse=False)
    module_list: List[Dict[str, Any]] = []
    if include_module_summary:
        for mod_name, mod in model.named_modules():
            if "cnn_backbone" in mod_name or "side_vit" in mod_name:
                continue
            # parameters that are direct attributes of this module
            direct_params_iter = list(mod.parameters(recurse=False))
            if not direct_params_iter:
                continue
            direct_numel = sum(int(p.numel()) for p in direct_params_iter)
            direct_trainable = any(p.requires_grad for p in direct_params_iter)
            module_list.append({
                "module_name": mod_name if mod_name != "" else "<root>",
                "direct_param_count": int(direct_numel),
                "has_trainable_direct_params": bool(direct_trainable)
            })

    result: Dict[str, Any] = {
        "model_class": model.__class__.__name__,
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "non_trainable_parameters": int(non_trainable_params),
        "total_bytes": int(total_bytes),
        "total_bytes_human_readable": _bytes_to_readable(int(total_bytes)),
        "devices": sorted(list(devices)),
        "dtypes": sorted(list(dtypes)),
        "param_table_included": bool(include_param_table),
        "module_summary_included": bool(include_module_summary),
    }

    if include_param_table:
        result["parameters"] = param_list
        if max_param_items is not None and len(param_list) < len(unique_params):
            result["parameters_truncated"] = True
            result["parameters_total_count"] = len(unique_params)
        else:
            result["parameters_truncated"] = False

    if include_module_summary:
        result["module_list"] = module_list

    if return_type == "dict":
        return result
    else:
        # JSON dump - ensure all items are JSON serializable (we used only primitives)
        return json.dumps(result, ensure_ascii=False, indent=2)

