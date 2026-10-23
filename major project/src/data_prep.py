import os
import sys
import json
import re
import random
import glob
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from PIL import Image

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from src.config import config


def json_to_donut_xml(gt_dict: Dict[str, Any]) -> str:
    """
    Converts a ground truth dictionary into Donut XML token format.
    """
    xml = f"{config.TASK_START_TOKEN}"
    
    # Metadata fields
    for field in ["vendor_name", "gstin", "invoice_no", "date"]:
        val = str(gt_dict.get(field, "")).strip()
        xml += f"<s_{field}>{val}</s_{field}>"
        
    # Line items
    xml += "<s_items>"
    items = gt_dict.get("items", [])
    if not items:
        items = [{
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
        }]
        
    for item in items:
        xml += "<s_item>"
        for field in [
            "product_description", "qty", "unit", "rate", 
            "subtotal", "discount", "freight", "cgst", "sgst", "igst"
        ]:
            val = str(item.get(field, "")).strip()
            xml += f"<s_{field}>{val}</s_{field}>"
        xml += "</s_item>"
    xml += "</s_items>"
    
    # Grand Total
    grand_total = str(gt_dict.get("grand_total", "")).strip()
    xml += f"<s_grand_total>{grand_total}</s_grand_total>"
    xml += "</s_invoice>"
    
    return xml


def extract_fields_from_ocr(lines: List[Dict[str, Any]], filename: str) -> Dict[str, Any]:
    """
    Extracts structured invoice fields from PaddleOCR lines.
    """
    if not lines:
        return {
            "source_file": filename,
            "vendor_name": "",
            "gstin": "",
            "invoice_no": "",
            "date": "",
            "items": [],
            "grand_total": ""
        }
        
    full_text = " \n ".join([l.get("text", "").strip() for l in lines if l.get("text")])
    
    # 1. GSTIN (15-character Indian GST format)
    gstin_match = re.search(r'\b([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}[Zz][0-9A-Z]{1})\b', full_text)
    gstin = gstin_match.group(1).upper() if gstin_match else ""
    
    # 2. Invoice No
    inv_patterns = [
        r'(?:Invoice\s*(?:No|Number|#)|Bill\s*(?:No|Number|#)|Inv\s*No)[.:\s]*([A-Za-z0-9\-_/]+)',
        r'(?:Inv\.|Bill\.)[.:\s]*([A-Za-z0-9\-_/]+)',
        r'([A-Za-z0-9\-_/]+\s*[-/]\s*[0-9]{2,4})'
    ]
    invoice_no = ""
    for pat in inv_patterns:
        m = re.search(pat, full_text, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip(" :.,-")
            if candidate and len(candidate) > 2 and not candidate.lower() in ["date", "gstin", "total"]:
                invoice_no = candidate
                break
                
    # 3. Date
    date_patterns = [
        r'(?:Date|Dated|Dt|Invoice Date)[.:\s]*([0-9]{1,2}[\-/\.][0-9]{1,2}[\-/\.][0-9]{2,4}|[0-9]{1,2}[\-/\.][A-Za-z]{3}[\-/\.][0-9]{2,4})',
        r'([0-9]{1,2}[\-/\.][0-9]{1,2}[\-/\.][0-9]{2,4}|[0-9]{1,2}[\-/\.][A-Za-z]{3}[\-/\.][0-9]{2,4})'
    ]
    invoice_date = ""
    for pat in date_patterns:
        m = re.search(pat, full_text, re.IGNORECASE)
        if m:
            invoice_date = m.group(1).strip()
            break
            
    # 4. Vendor Name
    vendor_name = ""
    ignore_keywords = [
        "INVOICE", "TAX INVOICE", "BILL OF SUPPLY", "ORIGINAL", "DUPLICATE", 
        "BUYER", "CONSIGNEE", "IRN", "ACK", "GSTIN", "STATE CODE", "MSME", "FSSAI",
        "E-WAY", "DELIVERY", "TRANSPORT", "VEHICLE"
    ]
    for l in lines[:12]:
        t = l.get("text", "").strip()
        t_upper = t.upper()
        if len(t) > 3 and not any(k in t_upper for k in ignore_keywords) and not re.match(r'^[0-9\-_/.: ]+$', t):
            vendor_name = t
            break
            
    # 5. Grand Total
    total_patterns = [
        r'(?:Grand\s*Total|Total\s*Amount|Net\s*Amount|Invoice\s*Total|Total\s*Payable|Total)[.:\s]*[₹Rs\.]*\s*([0-9,]+\.?[0-9]*)',
        r'(?:₹|Rs\.?)\s*([0-9,]+\.?[0-9]{2})'
    ]
    grand_total = ""
    for pat in total_patterns:
        m = re.search(pat, full_text, re.IGNORECASE)
        if m:
            cand = m.group(1).replace(",", "").strip()
            try:
                val = float(cand)
                if val > 0:
                    grand_total = f"{val:.2f}"
                    break
            except ValueError:
                pass
                
    # 6. Table Line Items Extraction
    unit_regex = re.compile(r'\b(KGS?|BAGS?|MT|M\.T\.|NOS?|PCS?|PKT|QTL|BOX|LTR|UNT)\b', re.IGNORECASE)
    items = []
    
    for i, l in enumerate(lines):
        text = l.get("text", "").strip()
        unit_m = unit_regex.search(text)
        
        # Check if line contains numbers/amounts typical of item rows
        num_matches = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        if unit_m or (len(num_matches) >= 2 and any(float(n) > 0 for n in num_matches if n.replace('.', '').isdigit())):
            # Heuristic item row parsing
            u = unit_m.group(1).upper() if unit_m else "NOS"
            qty = num_matches[0] if len(num_matches) > 0 else "1"
            rate = num_matches[1] if len(num_matches) > 1 else ""
            subtotal = num_matches[2] if len(num_matches) > 2 else ""
            
            # Clean description
            desc = unit_regex.sub("", text)
            desc = re.sub(r'[\d,.]+', '', desc).strip(" -:")
            if not desc and i > 0:
                prev_text = lines[i-1].get("text", "").strip()
                if len(prev_text) > 3 and not any(k in prev_text.upper() for k in ignore_keywords):
                    desc = prev_text
            if not desc:
                desc = "Goods / Services"
                
            items.append({
                "product_description": desc[:60],
                "qty": qty,
                "unit": u,
                "rate": rate,
                "subtotal": subtotal,
                "discount": "0.00",
                "freight": "0.00",
                "cgst": "0.00",
                "sgst": "0.00",
                "igst": "0.00"
            })
            
    if not items:
        items.append({
            "product_description": "Goods / Services",
            "qty": "1",
            "unit": "NOS",
            "rate": grand_total if grand_total else "0.00",
            "subtotal": grand_total if grand_total else "0.00",
            "discount": "0.00",
            "freight": "0.00",
            "cgst": "0.00",
            "sgst": "0.00",
            "igst": "0.00"
        })
        
    return {
        "source_file": filename,
        "vendor_name": vendor_name,
        "gstin": gstin,
        "invoice_no": invoice_no,
        "date": invoice_date,
        "items": items[:10],  # cap at 10 items for Donut token length
        "grand_total": grand_total
    }


def find_matching_pdf_or_image(stem: str) -> str:
    """
    Finds the source PDF or image corresponding to a JSON file stem.
    """
    # 1. Check existing annotated image
    ann_img_name = f"{stem}_annotated.jpg"
    ann_img_path = os.path.join(config.ANNOTATION_DIR, "images", ann_img_name)
    
    # 2. Check if we can render clean image from PDF in Dataset
    # Parse stem e.g., '01-1_page_0' -> base '01-1.pdf', page 0
    page_match = re.search(r'_page_(\d+)$', stem)
    page_num = int(page_match.group(1)) if page_match else 0
    base_name = re.sub(r'_page_\d+$', '', stem)
    
    # Try finding candidate PDF in Dataset
    candidate_pdfs = [
        os.path.join(config.DATASET_PDF_DIR, f"{base_name}.pdf"),
        os.path.join(config.DATASET_PDF_DIR, f"{base_name.replace('_', ' ')}.pdf"),
        os.path.join(config.DATASET_PDF_DIR, f"{base_name}.PDF")
    ]
    
    for pdf_path in candidate_pdfs:
        if os.path.exists(pdf_path) and fitz is not None:
            try:
                doc = fitz.open(pdf_path)
                if page_num < len(doc):
                    target_img_path = os.path.join(config.PROCESSED_IMG_DIR, f"{stem}.png")
                    if not os.path.exists(target_img_path):
                        page = doc[page_num]
                        pix = page.get_pixmap(dpi=150)
                        pix.save(target_img_path)
                    doc.close()
                    return target_img_path
                doc.close()
            except Exception:
                pass
                
    # Fallback to annotated image if PDF rendering not available
    if os.path.exists(ann_img_path):
        target_img_path = os.path.join(config.PROCESSED_IMG_DIR, f"{stem}.jpg")
        if not os.path.exists(target_img_path):
            img = Image.open(ann_img_path).convert("RGB")
            img.save(target_img_path)
        return target_img_path
        
    return ""


def prepare_dataset(val_ratio: float = 0.1, test_ratio: float = 0.1):
    """
    Processes all PaddleOCR JSONs and PDF/images into clean Donut training format.
    """
    print("=" * 70)
    print("Starting Donut Dataset Preprocessing & Ground Truth Builder...")
    print(f"Source JSONs: {config.ANNOTATION_JSON_DIR}")
    print(f"Source PDFs:  {config.DATASET_PDF_DIR}")
    print(f"Output Imgs:  {config.PROCESSED_IMG_DIR}")
    print("=" * 70)
    
    os.makedirs(config.PROCESSED_IMG_DIR, exist_ok=True)
    os.makedirs(config.PROCESSED_JSON_DIR, exist_ok=True)
    
    json_files = glob.glob(os.path.join(config.ANNOTATION_JSON_DIR, "*.json"))
    print(f"Found {len(json_files)} annotation files to process.")
    
    records = []
    skipped = 0
    
    for json_path in tqdm(json_files, desc="Building Donut Dataset"):
        stem = os.path.splitext(os.path.basename(json_path))[0]
        
        try:
            with open(json_path, "r", encoding="utf-8", errors="ignore") as f:
                ocr_data = json.load(f)
        except Exception as e:
            skipped += 1
            continue
            
        lines = ocr_data.get("lines", [])
        if not lines:
            # Skip empty OCR results
            skipped += 1
            continue
            
        img_path = find_matching_pdf_or_image(stem)
        if not img_path or not os.path.exists(img_path):
            skipped += 1
            continue
            
        # Extract ground truth schema
        gt_dict = extract_fields_from_ocr(lines, filename=f"{stem}.pdf")
        
        # Save structured ground truth JSON
        gt_json_path = os.path.join(config.PROCESSED_JSON_DIR, f"{stem}.json")
        with open(gt_json_path, "w", encoding="utf-8") as f:
            json.dump(gt_dict, f, indent=2)
            
        # Convert to Donut target XML string
        gt_xml = json_to_donut_xml(gt_dict)
        
        records.append({
            "stem": stem,
            "image_path": img_path,
            "ground_truth_json": gt_dict,
            "ground_truth_text": gt_xml
        })
        
    print(f"\nSuccessfully processed {len(records)} samples ({skipped} skipped/empty).")
    
    # Shuffle and split into Train / Val / Test
    random.seed(42)
    random.shuffle(records)
    
    total = len(records)
    n_val = int(total * val_ratio)
    n_test = int(total * test_ratio)
    n_train = total - n_val - n_test
    
    for i, rec in enumerate(records):
        if i < n_train:
            rec["split"] = "train"
        elif i < n_train + n_val:
            rec["split"] = "val"
        else:
            rec["split"] = "test"
            
    # Write metadata.jsonl
    with open(config.METADATA_FILE, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
            
    print(f"Split Summary: Train={n_train}, Val={n_val}, Test={n_test}")
    print(f"Metadata saved to: {config.METADATA_FILE}")
    print("=" * 70)
    return len(records)


if __name__ == "__main__":
    prepare_dataset()
