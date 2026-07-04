import torch
from torchmetrics import (
    AUROC,
    Accuracy,
    ConfusionMatrix,
    F1Score,
    Precision,
    Recall,
)

from src.utils.func import select_target_type


def prepare_batch(batch, cfg, frozen_encoder, device):
    """Prepares a single batch of data, moving it to the correct device."""
    if cfg.dataset.get("preload_path"):
        X_side, key_states, value_states, y = batch
        key_states, value_states = key_states.to(device), value_states.to(device)
        key_states, value_states = (
            key_states.transpose(0, 1),
            value_states.transpose(0, 1),
        )
    else:
        X_lpm, X_side, y = batch
        X_lpm = X_lpm.to(device)
        with torch.no_grad():
            _, key_states, value_states = frozen_encoder(
                X_lpm, interpolate_pos_encoding=True
            )

    X_side, y = X_side.to(device), y.to(device)
    y_true = select_target_type(y, cfg.train.get("criterion", "cross_entropy"))
    return X_side, key_states, value_states, y_true, y


def evaluate_model(
    cfg, frozen_encoder, model, dataloader, loss_function, device, just_loss_acc=False
):
    model.eval()

    # 1. “Dry run” first batch to infer binary vs. multiclass
    sample_batch = next(iter(dataloader))
    X_side, key_states, value_states, y_true, _ = prepare_batch(
        sample_batch, cfg, frozen_encoder, device
    )
    with torch.no_grad():
        sample_out = model(X_side, key_states, value_states)

    if sample_out.ndim == 1 or sample_out.shape[1] == 1:
        task = "binary-single-logit"
        num_classes = 2
    else:
        task = "multiclass"
        num_classes = sample_out.shape[1]

    # 2. Setup metrics on device
    if task.startswith("binary"):
        # binary classification metrics
        acc_metric = Accuracy(task="binary", average="weighted").to(device)
        prec_metric = Precision(task="binary", average="weighted", zero_division=0).to(
            device
        )
        rec_metric = Recall(task="binary", average="weighted", zero_division=0).to(
            device
        )
        f1_metric = F1Score(task="binary", average="weighted", zero_division=0).to(
            device
        )
        cm_metric = ConfusionMatrix(task="binary").to(device)
        # AUROC for raw logits
        auroc_metric = AUROC(task="binary", input="logits").to(device)

    else:
        # multiclass classification metrics
        acc_metric = Accuracy(
            task="multiclass", num_classes=num_classes, average="weighted"
        ).to(device)
        prec_metric = Precision(
            task="multiclass",
            num_classes=num_classes,
            average="weighted",
            zero_division=0,
        ).to(device)
        rec_metric = Recall(
            task="multiclass",
            num_classes=num_classes,
            average="weighted",
            zero_division=0,
        ).to(device)
        f1_metric = F1Score(
            task="multiclass",
            num_classes=num_classes,
            average="weighted",
            zero_division=0,
        ).to(device)
        cm_metric = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(
            device
        )
        # AUROC for probability vectors
        auroc_metric = AUROC(
            task="multiclass",
            num_classes=num_classes,
            average="weighted",
            # multi_class="ovo"
        ).to(device)

    total_loss = 0.0
    total_samples = 0

    # 3. Main loop
    with torch.no_grad():
        for batch in dataloader:
            X_side, key_states, value_states, y_true, _ = prepare_batch(
                batch, cfg, frozen_encoder, device
            )
            logits = model(X_side, key_states, value_states)

            # accumulate loss
            bs = y_true.size(0)
            total_loss += loss_function(logits, y_true).item() * bs
            total_samples += bs

            # get probabilities & preds
            if task == "binary-single-logit":
                scores = logits.view(-1)
                probs = torch.sigmoid(scores)
                preds = (probs >= 0.5).long()
                auroc_metric.update(scores, y_true)
            else:
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                auroc_metric.update(probs, y_true)

            # update all other metrics
            acc_metric.update(preds, y_true)
            prec_metric.update(preds, y_true)
            rec_metric.update(preds, y_true)
            f1_metric.update(preds, y_true)
            cm_metric.update(preds, y_true)

    # 4. Compute final values
    avg_loss = total_loss / total_samples
    acc = acc_metric.compute().item()
    precision = prec_metric.compute().item()
    recall = rec_metric.compute().item()
    f1 = f1_metric.compute().item()
    cm_tensor = cm_metric.compute().to(torch.int64)

    # AUROC may throw if only one class seen
    try:
        auc = auroc_metric.compute().item()
    except Exception:
        auc = float("nan")
    if just_loss_acc:
        return avg_loss, acc
    else:
        return acc, f1, auc, precision, recall, cm_tensor.cpu().numpy()
