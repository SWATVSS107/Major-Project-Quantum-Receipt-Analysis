import os
import sys
import re
import glob
import json
import argparse
from typing import List, Dict, Any, Union
import torch
from PIL import Image
from tqdm import tqdm
from transformers import DonutProcessor, VisionEncoderDecoderModel

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from src.config import config
from src.model import _load_hf_component
from export_excel import export_predictions_to_excel


def parse_donut_output_xml(xml_text: str, source_file: str = "") -> Dict[str, Any]:
    """
    Parses Donut output token string into a structured dictionary matching the 16 target schema fields.
    """
    def extract_tag(text: str, tag: str) -> str:
        m = re.search(rf"<s_{tag}>(.*?)</s_{tag}>", text, re.DOTALL)
        return m.group(1).strip() if m else ""
        
    vendor_name = extract_tag(xml_text, "vendor_name")
    gstin = extract_tag(xml_text, "gstin")
    invoice_no = extract_tag(xml_text, "invoice_no")
    date = extract_tag(xml_text, "date")
    grand_total = extract_tag(xml_text, "grand_total")
    
    # Parse line items
    items = []
    items_block = extract_tag(xml_text, "items")
    if items_block:
        item_matches = re.findall(r"<s_item>(.*?)</s_item>", items_block, re.DOTALL)
        for im in item_matches:
            item_dict = {
                "product_description": extract_tag(im, "product_description"),
                "qty": extract_tag(im, "qty"),
                "unit": extract_tag(im, "unit"),
                "rate": extract_tag(im, "rate"),
                "subtotal": extract_tag(im, "subtotal"),
                "discount": extract_tag(im, "discount"),
                "freight": extract_tag(im, "freight"),
                "cgst": extract_tag(im, "cgst"),
                "sgst": extract_tag(im, "sgst"),
                "igst": extract_tag(im, "igst")
            }
            items.append(item_dict)
            
    if not items:
        items.append({
            "product_description": "",
            "qty": "",
            "unit": "",
            "rate": "",
            "subtotal": "",
            "discount": "",
            "freight": "",
            "cgst": "",
            "sgst": "",
            "igst": ""
        })
        
    return {
        "source_file": source_file,
        "vendor_name": vendor_name,
        "gstin": gstin,
        "invoice_no": invoice_no,
        "date": date,
        "items": items,
        "grand_total": grand_total
    }


class DonutInvoiceExtractor:
    """
    Inference Engine for Donut Invoice Feature Extraction.
    """
    def __init__(self, model_path: str = None, device: str = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Initializing Donut Inference Engine on: {self.device}")
        
        if model_path is None or not os.path.exists(model_path):
            if os.path.exists(config.BEST_MODEL_DIR):
                model_path = config.BEST_MODEL_DIR
            elif os.path.exists(os.path.join(config.CHECKPOINT_DIR, "checkpoint_latest.pt")):
                model_path = os.path.join(config.CHECKPOINT_DIR, "checkpoint_latest.pt")
            else:
                model_path = config.PRETRAINED_MODEL_NAME
                
        print(f"Loading model from: {model_path}")
        
        if os.path.isdir(model_path):
            self.processor = _load_hf_component(DonutProcessor, model_path)
            self.model = _load_hf_component(VisionEncoderDecoderModel, model_path)
        elif model_path.endswith(".pt"):
            self.processor = _load_hf_component(DonutProcessor, config.PRETRAINED_MODEL_NAME)
            self.processor.tokenizer.add_special_tokens({"additional_special_tokens": config.SPECIAL_TOKENS})
            self.model = _load_hf_component(VisionEncoderDecoderModel, config.PRETRAINED_MODEL_NAME)
            self.model.decoder.resize_token_embeddings(len(self.processor.tokenizer))
            
            ckpt = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(ckpt["model_state_dict"])
        else:
            self.processor = _load_hf_component(DonutProcessor, model_path)
            self.processor.tokenizer.add_special_tokens({"additional_special_tokens": config.SPECIAL_TOKENS})
            self.model = _load_hf_component(VisionEncoderDecoderModel, model_path)
            self.model.decoder.resize_token_embeddings(len(self.processor.tokenizer))
            
        self.model.to(self.device)
        self.model.eval()
        
        self.task_prompt = config.TASK_START_TOKEN
        self.decoder_input_ids = self.processor.tokenizer(
            self.task_prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids.to(self.device)
        
    def predict_image(self, image: Image.Image, source_name: str = "") -> Dict[str, Any]:
        """
        Runs Donut extraction on a PIL Image.
        """
        pixel_values = self.processor(
            images=[image],
            return_tensors="pt"
        ).pixel_values.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=self.decoder_input_ids,
                max_length=config.MAX_LENGTH,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True
            )
            
        seq = self.processor.batch_decode(outputs.sequences)[0]
        seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
        seq = re.sub(r"<.*?(?:vocab|pad).*?>", "", seq).strip()
        
        parsed = parse_donut_output_xml(seq, source_file=source_name)
        return parsed
        
    def predict_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Renders PDF pages and extracts features for each page.
        """
        results = []
        if fitz is None:
            print("[Warning] PyMuPDF not available. Please install 'fitz'.")
            return results
            
        doc = fitz.open(pdf_path)
        base_name = os.path.basename(pdf_path)
        
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            source_name = f"{base_name} (Page {page_idx + 1})" if len(doc) > 1 else base_name
            res = self.predict_image(img, source_name=source_name)
            results.append(res)
            
        doc.close()
        return results
        
    def predict_batch(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Batch prediction over a list of images or PDFs.
        """
        all_results = []
        for fp in tqdm(file_paths, desc="Extracting Invoices"):
            if fp.lower().endswith(".pdf"):
                res = self.predict_pdf(fp)
                all_results.extend(res)
            elif fp.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                img = Image.open(fp).convert("RGB")
                res = self.predict_image(img, source_name=os.path.basename(fp))
                all_results.append(res)
        return all_results


def run_inference(args):
    extractor = DonutInvoiceExtractor(model_path=args.model_path)
    
    # Collect input files
    input_files = []
    if os.path.isdir(args.input):
        for ext in ["*.pdf", "*.jpg", "*.jpeg", "*.png"]:
            input_files.extend(glob.glob(os.path.join(args.input, ext)))
    elif os.path.isfile(args.input):
        input_files = [args.input]
    else:
        raise FileNotFoundError(f"Input path '{args.input}' not found.")
        
    if args.max_files > 0:
        input_files = input_files[:args.max_files]
        
    print(f"Found {len(input_files)} document(s) to process.")
    predictions = extractor.predict_batch(input_files)
    
    # Export to Excel
    out_excel = args.output_excel if args.output_excel else os.path.join(config.OUTPUT_DIR, "extracted_invoices.xlsx")
    export_predictions_to_excel(predictions, output_path=out_excel)
    print(f"\nCompleted! Extracted data saved to: {out_excel}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Donut Invoice Feature Extractor & Excel Generator")
    parser.add_argument("--input", type=str, default=config.DATASET_PDF_DIR, help="Path to input invoice PDF/image or directory")
    parser.add_argument("--model_path", type=str, default="", help="Path to trained Donut model checkpoint/dir")
    parser.add_argument("--output_excel", type=str, default="", help="Output Excel file path (.xlsx)")
    parser.add_argument("--max_files", type=int, default=10, help="Max files to process (0 for all)")
    
    args = parser.parse_args()
    run_inference(args)
