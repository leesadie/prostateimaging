import numpy as np
import torch

def train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        metrics_fn,
        device,
        epochs=10,
        writer=None,
        task_name="detection",
        path="best_model.pt"
):
    """
    Base training loop for classification models with TensorBoard logging
    """
    best_val_auc = -np.inf
    
    for epoch in range(epochs):
        model = model.to(device)

        # Train
        model.train()
        train_losses = []

        for batch in train_loader:
            inputs = batch["image"].to(device)
            labels = batch["cls_label"].to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs)
            if outputs.shape != labels.shape:
                outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)

        # Val
        model.eval()
        val_losses = []
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["image"].to(device)
                labels = batch["cls_label"].to(device).float()

                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())

                probs = torch.sigmoid(outputs).cpu().numpy()
                all_preds.append(probs)
                all_labels.append(labels.cpu().numpy())

        val_loss = np.mean(val_losses)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        metrics = metrics_fn(all_labels, all_preds)
        
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        )

        if writer:
            writer.add_scalar(f"{task_name}/Loss/train", train_loss, epoch)
            writer.add_scalar(f"{task_name}/Loss/val", val_loss, epoch)
            for k, v in metrics.items():
                writer.add_scalar(f"{task_name}/{k}/val", v, epoch)

        if metrics["AUC"] > best_val_auc:
            best_val_auc = metrics["AUC"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_auc": best_val_auc
            }, path)

    print(f"Best val AUC: {best_val_auc:.4f}")  
    return model

def train_seg_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    dice_metric,
    device,
    epochs=50,
    writer=None,
    task_name="segmentation",
    path="best_model.pt"
):
    best_val_dice = -1.0

    for epoch in range(epochs):
        model = model.to(device)

        # Train
        model.train()
        train_losses = []

        for batch in train_loader:
            images = batch["image"].to(device).float()
            masks = batch["mask"].to(device).float()

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)

        # Val
        model.eval()
        dice_metric.reset()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device).float()
                masks = batch["mask"].to(device).float()

                logits = model(images)
                loss = criterion(logits, masks)
                val_losses.append(loss.item())

                preds = torch.sigmoid(logits)
                preds = (preds > 0.5).float()
                dice_metric(preds, masks)

        val_loss = np.mean(val_losses)
        val_dice = dice_metric.aggregate().item()

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f}"
        )

        if writer:
            writer.add_scalar(f"{task_name}/Loss/train", train_loss, epoch)
            writer.add_scalar(f"{task_name}/Loss/val", val_loss, epoch)
            writer.add_scalar(f"{task_name}/Dice/val", val_dice, epoch)

        # Save model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), path)

    print(f"Best val Dice: {best_val_dice:.4f}")
    return model
                