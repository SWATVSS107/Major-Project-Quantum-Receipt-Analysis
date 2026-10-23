import os
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Config:
    # Directories
    WORKSPACE_DIR: str = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    DATASET_PDF_DIR: str = os.path.join(WORKSPACE_DIR, "Dataset")
    ANNOTATION_DIR: str = os.path.join(WORKSPACE_DIR, "annotated_results")
    ANNOTATION_JSON_DIR: str = os.path.join(ANNOTATION_DIR, "json")
    
    PROCESSED_DIR: str = os.path.join(WORKSPACE_DIR, "processed_data")
    PROCESSED_IMG_DIR: str = os.path.join(PROCESSED_DIR, "images")
    PROCESSED_JSON_DIR: str = os.path.join(PROCESSED_DIR, "ground_truth")
    METADATA_FILE: str = os.path.join(PROCESSED_DIR, "metadata.jsonl")
    
    CHECKPOINT_DIR: str = os.path.join(WORKSPACE_DIR, "checkpoints")
    BEST_MODEL_DIR: str = os.path.join(CHECKPOINT_DIR, "best_model")
    OUTPUT_DIR: str = os.path.join(WORKSPACE_DIR, "output")
    
    # Model parameters
    PRETRAINED_MODEL_NAME: str = "naver-clova-ix/donut-base"
    MAX_LENGTH: int = 768
    IMAGE_SIZE: List[int] = field(default_factory=lambda: [1280, 960])  # [height, width]
    
    # Training parameters (Optimized for RTX 4060 8GB GPU)
    BATCH_SIZE: int = 2
    GRAD_ACCUM_STEPS: int = 4
    LEARNING_RATE: float = 3e-5
    NUM_EPOCHS: int = 3
    WEIGHT_DECAY: float = 0.01
    WARMUP_RATIO: float = 0.05
    FP16: bool = True
    NUM_WORKERS: int = 0  # set 0 for Windows multiprocessing safety
    
    # Checkpointing & Validation
    VAL_CHECK_INTERVAL: int = 1  # evaluate every epoch
    SAVE_EPOCH_INTERVAL: int = 1
    SAVE_STEP_INTERVAL: int = 100
    
    # Target Schema Fields
    TASK_START_TOKEN: str = "<s_invoice>"
    SPECIAL_TOKENS: List[str] = field(default_factory=lambda: [
        "<s_invoice>", "</s_invoice>",
        "<s_vendor_name>", "</s_vendor_name>",
        "<s_gstin>", "</s_gstin>",
        "<s_invoice_no>", "</s_invoice_no>",
        "<s_date>", "</s_date>",
        "<s_items>", "</s_items>",
        "<s_item>", "</s_item>",
        "<s_product_description>", "</s_product_description>",
        "<s_qty>", "</s_qty>",
        "<s_unit>", "</s_unit>",
        "<s_rate>", "</s_rate>",
        "<s_subtotal>", "</s_subtotal>",
        "<s_discount>", "</s_discount>",
        "<s_freight>", "</s_freight>",
        "<s_cgst>", "</s_cgst>",
        "<s_sgst>", "</s_sgst>",
        "<s_igst>", "</s_igst>",
        "<s_grand_total>", "</s_grand_total>",
    ])
    
    # Excel Output Columns
    EXCEL_COLUMNS: List[str] = field(default_factory=lambda: [
        "Source File",
        "Vendor Name",
        "GSTIN",
        "Invoice No",
        "Date",
        "Product Description",
        "Qty",
        "Unit",
        "Rate",
        "Subtotal",
        "Discount",
        "Freight",
        "CGST",
        "SGST",
        "IGST",
        "Grand Total",
    ])

config = Config()
