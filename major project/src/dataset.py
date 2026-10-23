import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Dict, Any, Optional
from transformers import DonutProcessor

from src.config import config


class DonutInvoiceDataset(Dataset):
    """
    PyTorch Dataset for Donut Document Understanding on Invoice Bills.
    """
    def __init__(
        self,
        metadata_path: str = config.METADATA_FILE,
        split: str = "train",
        max_samples: Optional[int] = None
    ):
        self.split = split
        self.records: List[Dict[str, Any]] = []
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Metadata file not found at '{metadata_path}'. Please run 'python -m src.data_prep' first."
            )
            
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if item.get("split") == split:
                        self.records.append(item)
                        
        if max_samples is not None:
            self.records = self.records[:max_samples]
            
    def __len__(self) -> int:
        return len(self.records)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.records[idx]
        img_path = item["image_path"]
        
        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback to blank white image if corrupted
            image = Image.new("RGB", (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]), (255, 255, 255))
            
        gt_text = item.get("ground_truth_text", "")
        
        return {
            "image": image,
            "ground_truth_text": gt_text,
            "raw_record": item
        }


class DonutDataCollator:
    """
    Custom collator for Donut Processor to batch images and tokenized target texts.
    """
    def __init__(
        self,
        processor: DonutProcessor,
        max_length: int = config.MAX_LENGTH,
        ignore_id: int = -100
    ):
        self.processor = processor
        self.max_length = max_length
        self.ignore_id = ignore_id
        
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [item["image"] for item in batch]
        texts = [item["ground_truth_text"] for item in batch]
        raw_records = [item["raw_record"] for item in batch]
        
        # Process visual image inputs
        pixel_values = self.processor(
            images=images,
            random_padding=False,
            return_tensors="pt"
        ).pixel_values
        
        # Tokenize target text strings
        labels = self.processor.tokenizer(
            texts,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        # Mask padding tokens with -100 so they are ignored in cross-entropy loss
        labels[labels == self.processor.tokenizer.pad_token_id] = self.ignore_id
        
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "raw_records": raw_records
        }
