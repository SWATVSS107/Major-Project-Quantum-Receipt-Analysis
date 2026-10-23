import os
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional
from transformers import DonutProcessor, VisionEncoderDecoderModel

from src.config import config


def _load_hf_component(cls, name_or_path: str, **kwargs):
    """
    Attempts to load a HuggingFace component locally first, then falls back to online.
    """
    try:
        return cls.from_pretrained(name_or_path, local_files_only=True, **kwargs)
    except Exception:
        return cls.from_pretrained(name_or_path, **kwargs)


def build_donut_model_and_processor(
    pretrained_model_name: str = config.PRETRAINED_MODEL_NAME,
    image_size: list = config.IMAGE_SIZE,
    max_length: int = config.MAX_LENGTH,
    device: Optional[torch.device] = None
) -> Tuple[VisionEncoderDecoderModel, DonutProcessor]:
    """
    Initializes the Donut model and processor with custom invoice tokens.
    """
    print(f"Loading pretrained Donut processor and model from '{pretrained_model_name}'...")
    
    # Load processor
    processor = _load_hf_component(DonutProcessor, pretrained_model_name)
    
    # Update image size in feature extractor if specified
    if hasattr(processor, "image_processor"):
        processor.image_processor.size = {"height": image_size[0], "width": image_size[1]}
    elif hasattr(processor, "feature_extractor"):
        processor.feature_extractor.size = {"height": image_size[0], "width": image_size[1]}
        
    # Add custom special tokens for invoice extraction
    new_tokens = [t for t in config.SPECIAL_TOKENS if t not in processor.tokenizer.get_vocab()]
    if new_tokens:
        num_added = processor.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        print(f"Added {num_added} custom special tokens to Donut tokenizer.")
        
    # Load model
    model = _load_hf_component(VisionEncoderDecoderModel, pretrained_model_name)
    
    # Resize decoder token embeddings to match tokenizer with new tokens
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    
    # Configure generation and decoder start tokens
    task_token_id = processor.tokenizer.convert_tokens_to_ids(config.TASK_START_TOKEN)
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = task_token_id
    model.config.max_length = max_length
    
    if device is not None:
        model.to(device)
        
    print(f"Donut model configured successfully. Vocab size: {len(processor.tokenizer)}")
    return model, processor


def save_checkpoint(
    save_path: str,
    model: VisionEncoderDecoderModel,
    processor: DonutProcessor,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[torch.amp.GradScaler],
    epoch: int,
    step: int,
    val_loss: float = float("inf"),
    save_hf_dir: Optional[str] = None
):
    """
    Saves complete training checkpoint with optimizer, scheduler, scaler, and model states.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Checkpoint dictionary
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "val_loss": val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
    }
    
    # Save torch state checkpoint
    torch.save(checkpoint, save_path)
    print(f"[Checkpoint Saved] -> {save_path} (Epoch: {epoch}, Step: {step}, Val Loss: {val_loss:.4f})")
    
    # Save HF model and processor directory if specified (e.g. for best model)
    if save_hf_dir:
        os.makedirs(save_hf_dir, exist_ok=True)
        model.save_pretrained(save_hf_dir)
        processor.save_pretrained(save_hf_dir)
        print(f"[HuggingFace Model Exported] -> {save_hf_dir}")


def load_checkpoint(
    checkpoint_path: str,
    model: VisionEncoderDecoderModel,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    device: Optional[torch.device] = None
) -> Tuple[int, int, float]:
    """
    Loads model and training states from checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'")
        
    print(f"Loading checkpoint from '{checkpoint_path}'...")
    map_location = device if device is not None else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    
    model.load_state_dict(ckpt["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"]:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        
    if scheduler and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"]:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        
    if scaler and "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"]:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        
    start_epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("step", 0)
    val_loss = ckpt.get("val_loss", float("inf"))
    
    print(f"Successfully restored checkpoint (Resuming from Epoch: {start_epoch}, Step: {global_step}, Val Loss: {val_loss:.4f})")
    return start_epoch, global_step, val_loss
