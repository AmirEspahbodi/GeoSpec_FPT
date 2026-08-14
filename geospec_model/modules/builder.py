import os
import torch

from transformers import ViTConfig
from geospec_train import CoreConfig
from .bridge import FineGrainedPromptTuning, FusionModule
from .side_vit import ViTForImageClassification as SideViT
from .frozen_vit import ViTForImageClassification as FrozenViT

def get_terminal_col():
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80

def print_msg(msg, appendixs=[], warning=False):
    color = "\033[93m"
    end = "\033[0m"
    print_fn = (lambda x: print(color + x + end)) if warning else print

    max_len = len(max([msg, *appendixs], key=len))
    max_len = min(max_len, get_terminal_col())
    print_fn("=" * max_len)
    print_fn(msg)
    for appendix in appendixs:
        print_fn(appendix)
    print_fn("=" * max_len)


regression_loss = ["mean_square_error", "mean_absolute_error", "smooth_L1"]

def select_out_features(num_classes, criterion):
    out_features = num_classes
    if criterion in regression_loss:
        out_features = 1
    return out_features

def generate_model(cfg: CoreConfig):
    model = build_model(cfg)
    model = model.to(cfg.base.device)

    # the computation of the number of learnable parameters only works when the preloading is disabled
    if cfg.dataset.preload_path:
        frozen_encoder = None
    else:
        frozen_encoder = build_frozen_encoder(cfg).to(cfg.base.device)

        num_learnable_params = 0
        total_params = 0
        for _, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                num_learnable_params += param.numel()
        if frozen_encoder is not None:
            for _, param in frozen_encoder.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    num_learnable_params += param.numel()

        print("Total params: {}".format(total_params))
        print("Learnable params: {}".format(num_learnable_params))
        print(
            "Learnable params ratio: {:.4f}%".format(
                num_learnable_params / total_params * 100
            )
        )

    return frozen_encoder, model


def load_weights(model, checkpoint):
    weights = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(weights, strict=True)
    print_msg("Load weights form {}".format(checkpoint))



def build_model(cfg: CoreConfig):
    out_features = select_out_features(cfg.dataset.num_classes, cfg.train.criterion)
    num_layers = len(parse_layers(cfg.network.layers_to_extract))
    vit_config = ViTConfig.from_pretrained(cfg.network.pretrained_path)
    side_dimension = vit_config.hidden_size // cfg.network.side_reduction_ratio
    prompts_dim = vit_config.hidden_size // cfg.network.prompt_reduction_ratio
    fusion_module = FusionModule(
        num_layers=num_layers,
        in_dim=vit_config.hidden_size,
        out_dim=side_dimension,
        num_heads=vit_config.num_attention_heads,
        num_prompts=cfg.network.num_prompts,
        prompt_dim=prompts_dim,
        prompt_norm=cfg.network.prompt_norm,
        prompt_proj=cfg.network.prompt_proj,
    )

    side_config = ViTConfig.from_pretrained(
        cfg.network.pretrained_path,
        num_hidden_layers=num_layers,
        hidden_size=side_dimension,
        intermediate_size=side_dimension * 4,
        image_size=cfg.network.side_input_size,
        num_labels=out_features,
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
    )
    side_encoder = SideViT(side_config)

    model = FineGrainedPromptTuning(side_encoder, fusion_module)
    return model


def build_frozen_encoder(cfg: CoreConfig):
    frozen_config = ViTConfig.from_pretrained(cfg.network.pretrained_path)
    frozen_config.token_imp = cfg.network.token_imp
    frozen_config.token_ratio = cfg.network.token_ratio
    frozen_config.layers_to_extract = parse_layers(cfg.network.layers_to_extract)

    frozen_encoder = FrozenViT.from_pretrained(
        cfg.network.pretrained_path, config=frozen_config
    )

    frozen_encoder.eval()
    for p in frozen_encoder.parameters():
        p.requires_grad = False

    return frozen_encoder


def parse_layers(layers_to_extract):
    if "-" in layers_to_extract:
        start, end = layers_to_extract.split("-")
        return list(range(int(start), int(end) + 1))
    elif "," in layers_to_extract:
        return list(map(int, layers_to_extract.split(",")))
    else:
        return [int(layers_to_extract)]
