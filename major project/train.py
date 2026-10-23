import os
import sys
import time
import signal
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import math

from src.config import config
from src.dataset import DonutInvoiceDataset, DonutDataCollator
from src.model import build_donut_model_and_processor, save_checkpoint, load_checkpoint


# Global state for graceful interrupt handling
training_state = {
    "model": None,
    "processor": None,
    "optimizer": None,
    "scheduler": None,
    "scaler": None,
    "epoch": 0,
    "step": 0,
    "val_loss": float("inf"),
    "interrupted": False
}


def handle_interrupt(sig, frame):
    """
    Catches Ctrl+C / SIGINT and saves the current model progress immediately.
    """
    print("\n" + "=" * 70)
    print(" [SIGNAL DETECTED] Interrupting training gracefully...")
    print(" Saving latest progress and model weights to prevent any data loss...")
    print("=" * 70)
    
    training_state["interrupted"] = True
    
    if training_state["model"] is not None:
        save_path = os.path.join(config.CHECKPOINT_DIR, "interrupt_checkpoint.pt")
        latest_path = os.path.join(config.CHECKPOINT_DIR, "checkpoint_latest.pt")
        
        save_checkpoint(
            save_path=save_path,
            model=training_state["model"],
            processor=training_state["processor"],
            optimizer=training_state["optimizer"],
            scheduler=training_state["scheduler"],
            scaler=training_state["scaler"],
            epoch=training_state["epoch"],
            step=training_state["step"],
            val_loss=training_state["val_loss"],
            save_hf_dir=os.path.join(config.CHECKPOINT_DIR, "interrupted_model")
        )
        
        save_checkpoint(
            save_path=latest_path,
            model=training_state["model"],
            processor=training_state["processor"],
            optimizer=training_state["optimizer"],
            scheduler=training_state["scheduler"],
            scaler=training_state["scaler"],
            epoch=training_state["epoch"],
            step=training_state["step"],
            val_loss=training_state["val_loss"]
        )
        print("\nProgress safely stored! You can resume anytime using:")
        print(f"python train.py --resume {latest_path}")
        
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, handle_interrupt)
if hasattr(signal, "SIGTERM"):
    signal.signal(signal.SIGTERM, handle_interrupt)


def evaluate(model, val_loader, device, fp16=True):
    """
    Runs evaluation on the validation split and returns average loss.
    """
    model.eval()
    total_loss = 0.0
    steps = 0
    
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=fp16):
                outputs = model(
                    pixel_values=pixel_values,
                    labels=labels
                )
                loss = outputs.loss
                
            total_loss += loss.item()
            steps += 1
            
    model.train()
    return total_loss / max(1, steps)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("Starting Donut Invoice Model Training Pipeline")
    print(f"Device:            {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Batch Size:        {args.batch_size} (Grad Accum: {args.grad_accum}) -> Effective Batch: {args.batch_size * args.grad_accum}")
    print(f"Learning Rate:     {args.lr}")
    print(f"Mixed Precision:   {args.fp16 and torch.cuda.is_available()}")
    print(f"Checkpoint Dir:    {config.CHECKPOINT_DIR}")
    print("=" * 70)
    
    # 1. Datasets & Loaders
    print("Loading Train and Validation datasets...")
    train_dataset = DonutInvoiceDataset(split="train")
    val_dataset = DonutInvoiceDataset(split="val")
    
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    if len(train_dataset) == 0:
        raise ValueError("No training samples found in metadata.jsonl. Please run 'python -m src.data_prep' first.")
        
    # 2. Model & Processor
    model, processor = build_donut_model_and_processor(
        pretrained_model_name=args.pretrained_model,
        device=device
    )
    
    collator = DonutDataCollator(processor=processor, max_length=config.MAX_LENGTH)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 3. Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY
    )
    
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and torch.cuda.is_available())
    
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    
    # 4. Resume from Checkpoint if specified
    if args.resume:
        if os.path.exists(args.resume):
            start_epoch, global_step, best_val_loss = load_checkpoint(
                checkpoint_path=args.resume,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device
            )
        else:
            print(f"[Warning] Resume checkpoint '{args.resume}' not found. Starting from scratch.")
            
    # Update global state for interrupt handler
    training_state["model"] = model
    training_state["processor"] = processor
    training_state["optimizer"] = optimizer
    training_state["scheduler"] = scheduler
    training_state["scaler"] = scaler
    training_state["epoch"] = start_epoch
    training_state["step"] = global_step
    training_state["val_loss"] = best_val_loss
    
    # 5. Training Loop
    model.train()
    print("\nStarting Training Loop...")
    
    for epoch in range(start_epoch, args.epochs):
        training_state["epoch"] = epoch
        running_loss = 0.0
        step_in_epoch = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        
        for step, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=args.fp16):
                outputs = model(
                    pixel_values=pixel_values,
                    labels=labels
                )
                loss = outputs.loss / args.grad_accum
                
            scaler.scale(loss).backward()
            
            running_loss += loss.item() * args.grad_accum
            step_in_epoch += 1
            
            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                training_state["step"] = global_step
                
            # Progress bar display
            avg_loss = running_loss / step_in_epoch
            current_lr = scheduler.get_last_lr()[0] if scheduler else args.lr
            gpu_mem = f"{torch.cuda.memory_allocated(0) / 1024**2:.0f}MB" if torch.cuda.is_available() else "N/A"
            
            pbar.set_postfix({
                "loss": f"{loss.item() * args.grad_accum:.4f}",
                "avg_loss": f"{avg_loss:.4f}",
                "lr": f"{current_lr:.2e}",
                "gpu": gpu_mem
            })
            
            # Periodic Step Checkpointing
            if global_step > 0 and global_step % config.SAVE_STEP_INTERVAL == 0 and (step + 1) % args.grad_accum == 0:
                latest_ckpt_path = os.path.join(config.CHECKPOINT_DIR, "checkpoint_latest.pt")
                save_checkpoint(
                    save_path=latest_ckpt_path,
                    model=model,
                    processor=processor,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    step=global_step,
                    val_loss=best_val_loss
                )
                
        # 6. End of Epoch Validation & Checkpoint
        print(f"\nEvaluating Epoch {epoch + 1}...")
        val_loss = evaluate(model, val_loader, device, fp16=args.fp16)
        print(f"[Epoch {epoch + 1} Result] Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        training_state["val_loss"] = val_loss
        
        # Save Latest Checkpoint
        latest_ckpt_path = os.path.join(config.CHECKPOINT_DIR, "checkpoint_latest.pt")
        save_checkpoint(
            save_path=latest_ckpt_path,
            model=model,
            processor=processor,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch + 1,
            step=global_step,
            val_loss=val_loss
        )
        
        # Save Epoch Checkpoint
        epoch_ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch + 1}.pt")
        save_checkpoint(
            save_path=epoch_ckpt_path,
            model=model,
            processor=processor,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch + 1,
            step=global_step,
            val_loss=val_loss
        )
        
        # Save Best Model if validation improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f" [*] New best validation loss: {best_val_loss:.4f}! Saving best model...")
            save_checkpoint(
                save_path=os.path.join(config.CHECKPOINT_DIR, "checkpoint_best.pt"),
                model=model,
                processor=processor,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
                step=global_step,
                val_loss=best_val_loss,
                save_hf_dir=config.BEST_MODEL_DIR
            )
            
    print("=" * 70)
    print("Training Completed Successfully!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Model Directory: {config.BEST_MODEL_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Donut Model for Invoice Feature Extraction")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size per step")
    parser.add_argument("--grad_accum", type=int, default=config.GRAD_ACCUM_STEPS, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--fp16", action="store_true", default=config.FP16, help="Use mixed precision")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint file to resume")
    parser.add_argument("--pretrained_model", type=str, default=config.PRETRAINED_MODEL_NAME, help="Base Donut model name")
    
    args = parser.parse_args()
    train(args)
