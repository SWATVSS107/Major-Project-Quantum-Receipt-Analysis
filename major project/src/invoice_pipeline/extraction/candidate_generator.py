import re
from src.invoice_pipeline.knowledge.synonyms import (LABEL_SYNONYMS, GSTIN_REGEX, DATE_REGEX,
                                                      PAN_REGEX, SELLER_INDICATORS, BUYER_INDICATORS)


def generate_field_candidates(nodes, full_text=""):
    """
    Scans all text tokens and full text to extract multiple candidate matches for each field.
    Uses regex, keyword proximity, spatial graph, and header heuristics.
    """
    candidates = {
        "vendor_name": [],
        "seller_gstin": [],
        "buyer_gstin": [],
        "seller_pan": [],
        "invoice_number": [],
        "invoice_date": [],
        "subtotal": [],
        "discount": [],
        "freight": [],
        "cgst": [],
        "sgst": [],
        "igst": [],
        "grand_total": [],
        "all_numbers": []
    }

    all_text = full_text + " " + " ".join([n["text"] for n in nodes])

    # --- 1. Find ALL GSTINs with their spatial position & text index ---
    gstin_with_positions = []
    for n in nodes:
        matches = re.findall(GSTIN_REGEX, n["text"])
        for g in matches:
            gstin_with_positions.append({"value": g, "cx": n["cx"], "cy": n["cy"], "conf": 0.98})

    all_gstin_values = {g["value"] for g in gstin_with_positions}
    for m in re.finditer(GSTIN_REGEX, full_text):
        g = m.group()
        if g not in all_gstin_values:
            gstin_with_positions.append({"value": g, "cx": 0, "cy": 0, "conf": 0.90, "pos": m.start()})
        else:
            for item in gstin_with_positions:
                if item["value"] == g and "pos" not in item:
                    item["pos"] = m.start()

    # --- 2. Split GSTINs into Seller vs Buyer using full_text context & spatial cues ---
    buyer_positions = [m.start() for m in re.finditer(r'(?:bill\s*to|buyer|consignee|ship\s*to|name\s*&\s*address\s*of\s*buyer|details\s*of\s*receiver)', full_text, re.IGNORECASE)]
    first_buyer_pos = min(buyer_positions) if buyer_positions else len(full_text)

    known_buyer_pans = ["AACCA8432H"]
    is_avenue_buyer = "avenue supermarts" in full_text.lower()

    seller_candidates = []
    buyer_candidates = []

    unique_gstins = list(dict.fromkeys([g["value"] for g in gstin_with_positions] + re.findall(GSTIN_REGEX, full_text)))

    for g in unique_gstins:
        pos = full_text.find(g)
        is_known_buyer_pan = any(pan in g for pan in known_buyer_pans) and is_avenue_buyer

        if is_known_buyer_pan:
            buyer_candidates.append({"value": g, "conf": 0.99, "source": "known_buyer_pan"})
        elif pos >= 0 and pos > first_buyer_pos:
            buyer_candidates.append({"value": g, "conf": 0.95, "source": "after_buyer_header"})
        elif pos >= 0 and pos < first_buyer_pos:
            seller_candidates.append({"value": g, "conf": 0.95, "source": "before_buyer_header"})
        else:
            seller_candidates.append({"value": g, "conf": 0.80, "source": "position_fallback"})

    if seller_candidates:
        candidates["seller_gstin"] = seller_candidates
    if buyer_candidates:
        candidates["buyer_gstin"] = buyer_candidates

    if len(unique_gstins) == 1 and not candidates["seller_gstin"] and candidates["buyer_gstin"]:
        if unique_gstins[0] not in [b["value"] for b in buyer_candidates if b["source"] == "known_buyer_pan"]:
            candidates["seller_gstin"] = [{"value": unique_gstins[0], "conf": 0.85, "source": "single_gstin"}]

    # --- 2b. PAN number fallback --- 
    def _clean_pan(p_str):
        p = p_str.upper().strip()
        if len(p) == 10 and p[:5].isalpha() and p[9].isalpha():
            digits = p[5:9].replace('O', '0').replace('I', '1').replace('Z', '2').replace('B', '8')
            if digits.isdigit():
                return p[:5] + digits + p[9]
        return p if len(p) == 10 else None

    if not candidates["seller_gstin"]:
        for n in nodes:
            if "pan" in n["text"].lower():
                for m in nodes:
                    if m["cy"] >= n["cy"] - 30 and m["cy"] <= n["cy"] + 40:
                        pan_matches = re.findall(r'[A-Z]{5}[0-9OIZB]{4}[A-Z]', m["text"].upper())
                        for p in pan_matches:
                            clean_p = _clean_pan(p)
                            if clean_p:
                                candidates["seller_pan"].append({"value": clean_p, "conf": 0.90, "source": "label_pan"})
        pan_in_text = re.findall(r'PAN\s*(?:No\.?)?\s*:?\s*([A-Z]{5}[0-9OIZB]{4}[A-Z])', full_text, re.IGNORECASE)
        for p in pan_in_text:
            clean_p = _clean_pan(p)
            if clean_p and not any(cp["value"] == clean_p for cp in candidates["seller_pan"]):
                candidates["seller_pan"].append({"value": clean_p, "conf": 0.85, "source": "regex_pan"})

    # --- 3. Date candidates via regex ---
    date_matches = re.findall(DATE_REGEX, all_text, re.IGNORECASE)
    for d in set(date_matches):
        d_clean = d.strip()
        if len(d_clean) >= 6:
            candidates["invoice_date"].append({"value": d_clean, "conf": 0.95, "source": "regex"})

    inv_date_matches = re.findall(r'(?:invoice\s*date|dated|bill\s*date|doc\s*date)\s*[:\s\-]*(\d{1,2}[\-/\.][\w]+[\-/\.]\d{2,4})', full_text, re.IGNORECASE)
    for d in inv_date_matches:
        candidates["invoice_date"].insert(0, {"value": d.strip(), "conf": 0.99, "source": "label_proximity"})

    inv_date_before = re.findall(r'(\d{1,2}[\-/\.][\w]+[\-/\.]\d{2,4})\s*\n\s*(?:invoice\s*date|bill\s*date)', full_text, re.IGNORECASE)
    for d in inv_date_before:
        candidates["invoice_date"].insert(0, {"value": d.strip(), "conf": 0.99, "source": "label_before"})

    # --- 4. Numeric candidates (exclude phone, pincodes, and HSN codes) ---
    hsn_codes = set(re.findall(r'\b(?:\d{8}|\d{6})\b', full_text))
    hsn_numbers = {float(h) for h in hsn_codes if h.isdigit()}
    for h in hsn_codes:
        if h.startswith('0') and h[1:].isdigit():
            hsn_numbers.add(float(h[1:]))

    for n in nodes:
        clean_text = n["text"].replace(",", "").replace("ī", "").replace("₹", "").strip()
        if re.match(r'^(?:\+91[\s\-]?)?[6-9]\d{9}$', clean_text):
            continue
        if re.match(r'^\d{6}$', clean_text):
            continue
        if len(clean_text) > 10 and clean_text.isdigit():
            continue
        if clean_text.isdigit() and (len(clean_text) == 8 or (len(clean_text) == 7 and clean_text.startswith(('7', '8', '9')))):
            continue
        if re.match(GSTIN_REGEX, clean_text):
            continue

        try:
            val = float(clean_text)
            if 0 < val < 100_000_000.0 and val not in hsn_numbers:
                candidates["all_numbers"].append({
                    "value": val,
                    "text": n["text"],
                    "node_id": n["id"],
                    "cx": n["cx"],
                    "cy": n["cy"],
                    "conf": n["confidence"]
                })
        except ValueError:
            pass

    for m in re.finditer(r'\b\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?\b|\b\d+(?:\.\d{1,2})?\b', full_text):
        clean_s = m.group().replace(",", "")
        try:
            val = float(clean_s)
            if 0 < val < 100_000_000.0 and val not in hsn_numbers:
                if not any(abs(an["value"] - val) < 0.01 for an in candidates["all_numbers"]):
                    candidates["all_numbers"].append({
                        "value": val,
                        "text": m.group(),
                        "node_id": -1,
                        "cx": 0,
                        "cy": 0,
                        "conf": 0.85
                    })
        except ValueError:
            pass

    # --- 5. Vendor Name from header heuristics ---
    lines = [line.strip() for line in full_text.splitlines() if line.strip()]
    skip_keywords = [
        "bill of supply", "tax invoice", "original", "duplicate", "triplicate",
        "subject to", "jurisdiction", "gstin", "invoice no", "date", "phone",
        "mobile", "fax", "email", "address", "pan:", "cin:", "fssai",
        "tan:", "tan ", "cin ", "cin no", "state code",
        "transporter", "transporter name", "transport", "transp", "broker",
        "broker name", "destination", "dispatched", "dispatch", "driver", "vehicle",
        "e-way", "eway", "bill to", "billed to", "consignee", "ship to",
        "shipped to", "buyer", "sold to", "party name", "customer", "recipient",
        "details of receiver", "deliver to", "bank details", "declaration",
        "terms & conditions", "authorised signatory", "authorized signatory",
        "messers", "messrs", "name & address of buyer", "vehicle no", "irn",
        "ack no", "ack dt"
    ]

    seller_name_candidates = []
    for line in lines[:30]:
        line_clean = re.sub(r'[^a-zA-Z0-9\s&\.\-\(\)]', '', line).strip()
        line_lower = line.lower()
        letters_only = re.sub(r'[^a-zA-Z]', '', line)
        if len(letters_only) < 4:
            continue
        if re.match(r'^[a-f0-9]{25,}', line_clean.lower()):
            continue
        if line_lower.startswith(("irn", "ack", "tax invoice", "bill of supply", "original", "duplicate", "tan:", "tan ", "cin:", "cin ")):
            continue
        if any(bkw in line_lower for bkw in ["buyer (bill to)", "name & address of buyer", "billed to", "consignee", "receiver details"]):
            break
        if (len(line_clean) > 4
                and not any(k in line_lower for k in skip_keywords)
                and not re.match(r'^[\d\s]+$', line_clean)
                and not re.match(GSTIN_REGEX, line_clean)
                and len(line_clean) < 80):
            score = 0.85
            if any(si in line_lower for si in SELLER_INDICATORS):
                score = 0.98
            seller_name_candidates.append({
                "value": line_clean,
                "conf": score,
                "source": "header_heuristic"
            })

    if not any(sc["conf"] >= 0.90 for sc in seller_name_candidates):
        for line in lines:
            m = re.search(r'\b(?:FOR|For)\s*,?\s+([A-Z0-9\s&\.\-]{4,60})\b', line)
            if m:
                val = m.group(1).strip()
                if not any(k in val.lower() for k in skip_keywords):
                    seller_name_candidates.append({
                        "value": val,
                        "conf": 0.92,
                        "source": "for_company"
                    })

    # Sort descending by confidence so strong seller indicator lines come first
    seller_name_candidates.sort(key=lambda x: x["conf"], reverse=True)
    candidates["vendor_name"] = seller_name_candidates

    # --- 6. Invoice Number ---
    for i, n in enumerate(nodes):
        txt_lower = n["text"].lower()
        for synonym in LABEL_SYNONYMS["invoice_number"]:
            if synonym in txt_lower and not any(k in txt_lower for k in ["supply", "e-way", "eway", "e -way", "lading", "lr", "lr-rr", "note", "vehicle", "driver", "licence", "license", "mobile", "contact", "a/c", "account"]):
                for neighbor in nodes[i + 1:i + 6]:
                    val = neighbor["text"].strip(":# ")
                    is_date = bool(re.search(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b', val, re.IGNORECASE) or re.match(r'^\d{1,2}[\-/\.]\d{1,2}[\-/\.]\d{2,4}$', val))
                    if val and len(val) >= 2 and not is_date and not any(k in val.lower() for k in ["date", "gstin", "bill", "supply", "order", "state", "vehicle", "driver", "mobile", "licence", "license"]):
                        candidates["invoice_number"].append({
                            "value": val,
                            "conf": 0.90,
                            "source": f"keyword:{synonym}"
                        })
                        break
                break

    # Clean text to prevent "Bill of Supply" matching "bill no" and strip e-way bill / lading
    clean_inv_text = re.sub(r'bill\s+of\s+supply', '', full_text, flags=re.IGNORECASE)
    clean_inv_text = re.sub(r'e\s*[-]?\s*way\s*bill\s*(?:no\.?)?\s*[:\s]*\d+', '', clean_inv_text, flags=re.IGNORECASE)
    clean_inv_text = re.sub(r'bill\s+of\s+lading[^\n]*', '', clean_inv_text, flags=re.IGNORECASE)
    clean_inv_text = re.sub(r'lr[-/ ]rr\s*no\.?[^\n]*', '', clean_inv_text, flags=re.IGNORECASE)

    inv_no_matches = re.findall(r'(?:invoice\s*no\.?|inv\s*no\.?)\s*[:\s\n]*([A-Za-z0-9\-_/]{2,30})', clean_inv_text, re.IGNORECASE)
    for inm in inv_no_matches:
        is_date = bool(re.search(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b', inm, re.IGNORECASE) or re.match(r'^\d{1,2}[\-/\.]\d{1,2}[\-/\.]\d{2,4}$', inm))
        if not is_date and not any(k in inm.lower() for k in ["date", "dated", "delivery", "note", "gstin", "supply", "order"]):
            if len(inm.strip()) == 12 and inm.strip().isdigit():
                continue
            candidates["invoice_number"].append({"value": inm.strip(), "conf": 0.99, "source": "regex_inv_no"})

    inv_no_before = re.findall(r'([A-Za-z0-9\-_/]{2,30})\s*\n\s*(?:invoice\s*no\.?|bill\s*no\.?)', clean_inv_text, re.IGNORECASE)
    for inm in inv_no_before:
        is_date = bool(re.search(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b', inm, re.IGNORECASE) or re.match(r'^\d{1,2}[\-/\.]\d{1,2}[\-/\.]\d{2,4}$', inm))
        if not is_date and not any(k in inm.lower() for k in ["date", "dated", "delivery", "note", "gstin", "buyer", "ack", "supply", "order", "vehicle"]):
            if len(inm.strip()) == 12 and inm.strip().isdigit():
                continue
            candidates["invoice_number"].append({"value": inm.strip(), "conf": 0.95, "source": "regex_inv_before"})

    candidates["invoice_number"].sort(key=lambda x: x["conf"], reverse=True)

    # --- 7. Spatial Amount Fields ---
    for field_name in ["subtotal", "discount", "freight", "cgst", "sgst", "igst", "grand_total"]:
        for synonym in LABEL_SYNONYMS[field_name]:
            for n in nodes:
                if synonym in n["text"].lower():
                    # For discount, skip if line mentions payment terms or conditions
                    if field_name == "discount" and any(pk in n["text"].lower() for pk in ["within", "condition", "payment", "month"]):
                        continue
                    for num in candidates["all_numbers"]:
                        if num["cy"] >= n["cy"] - 20 and num["cy"] <= n["cy"] + 40:
                            if field_name == "discount" and num["value"] >= 100000.0:
                                continue
                            candidates[field_name].append({
                                "value": num["value"],
                                "conf": num["conf"],
                                "source": f"spatial:{synonym}"
                            })

    # --- 8. Enhanced Regex for Discount, Freight, Taxes & Grand Total ---
    # Discount: avoid capturing payment terms e.g. "Within 5 days 2% discount"
    clean_disc_text = re.sub(r'(?:within\s*\d+\s*days|payment\s*condition)[^\n]*', '', full_text, flags=re.IGNORECASE)

    disc_patterns = [
        r'(?:\([^\d\n]*\))?[-]?\s*([\d,]+\.\d{2})\)?\s*\n?\s*(?:cash\s*disc|cd\s*amt|trade\s*disc)',
        r'Discount[^\n]*\n\s*\(?[-]?\)?\s*([\d,]+\.\d{2})',
        r'(?:cash\s*disc(?:ount)?|cd\s*amt|trade\s*disc(?:ount)?|less\s*:\s*discount)[^\d\n]*?\(?[-]?\)?\s*([\d,]+\.\d{2})',
        r'Less\s*:\s*Discount[^\n]*\n\s*\(?[-]?\s*([\d,]+\.\d{2})',
    ]
    for dp in disc_patterns:
        matches = re.findall(dp, clean_disc_text, re.IGNORECASE)
        for dm in matches:
            try:
                v = float(dm.replace(",", ""))
                if 0 < v < 500000.0 and v not in hsn_numbers and not any(abs(cd["value"] - v) < 0.1 for cd in candidates["discount"]):
                    candidates["discount"].append({"value": v, "conf": 0.98, "source": "regex_disc"})
            except ValueError:
                pass

    # Freight
    frt_patterns = [
        r'Freight\s*Advance[^\n]*\n\s*([\d,]+\.\d{2})',
        r'(?:freight\s*advance|total\s*freight|lorry\s*freight)[^\d\n]*?([\d,]+\.\d{2})',
        r'([\d,]+\.\d{2})\s*\n?\s*(?:freight\s*advance|total\s*freight|lorry\s*freight)',
        r'(?:freight|fright)[^\d\n]*?([\d,]+\.\d{2})',
    ]
    for fp in frt_patterns:
        matches = re.findall(fp, full_text, re.IGNORECASE)
        for fm in matches:
            try:
                v = float(fm.replace(",", ""))
                if 0 < v < 1000000.0 and v not in hsn_numbers and not any(abs(cf["value"] - v) < 0.1 for cf in candidates["freight"]):
                    candidates["freight"].append({"value": v, "conf": 0.98, "source": "regex_freight"})
            except ValueError:
                pass

    # CGST & SGST
    cgst_matches = re.findall(r'([\d,]+\.\d{2})\s*\n?\s*(?:out\s*put\s*)?cgst', full_text, re.IGNORECASE) + \
                   re.findall(r'(?:out\s*put\s*)?cgst[^\d\n]*?([\d,]+\.\d{2})', full_text, re.IGNORECASE)
    for cm in cgst_matches:
        try:
            v = float(cm.replace(",", ""))
            if 0 < v < 1000000.0 and v not in hsn_numbers and not any(abs(c["value"] - v) < 0.1 for c in candidates["cgst"]):
                candidates["cgst"].append({"value": v, "conf": 0.98, "source": "regex_cgst"})
        except ValueError:
            pass

    sgst_matches = re.findall(r'([\d,]+\.\d{2})\s*\n?\s*(?:out\s*put\s*)?sgst', full_text, re.IGNORECASE) + \
                   re.findall(r'(?:out\s*put\s*)?sgst[^\d\n]*?([\d,]+\.\d{2})', full_text, re.IGNORECASE)
    for sm in sgst_matches:
        try:
            v = float(sm.replace(",", ""))
            if 0 < v < 1000000.0 and v not in hsn_numbers and not any(abs(c["value"] - v) < 0.1 for c in candidates["sgst"]):
                candidates["sgst"].append({"value": v, "conf": 0.98, "source": "regex_sgst"})
        except ValueError:
            pass

    # Grand Total
    gt_matches = (
        re.findall(r'TOTAL[^\d\n\(\-]*?[ī₹\s]*([\d,]+\.\d{2})', full_text, re.IGNORECASE) +
        re.findall(r'Total[^\n]*\n\s*[ī₹]?\s*([\d,]+\.\d{2})', full_text, re.IGNORECASE) +
        re.findall(r'(?:Invoice\s*amount|Net\s*payable|Bill\s*amount)[^\d\n]*?[:\s]*\n?\s*([\d,]+\.\d{2})', full_text, re.IGNORECASE)
    )
    for gm in gt_matches:
        try:
            v = float(gm.replace(",", ""))
            if v > 100.0 and v not in hsn_numbers:
                if not any(abs(cg["value"] - v) < 0.1 for cg in candidates["grand_total"]):
                    candidates["grand_total"].append({"value": v, "conf": 0.98, "source": "regex_grand_total"})
        except ValueError:
            pass

    return candidates
