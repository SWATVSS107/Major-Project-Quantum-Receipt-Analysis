import re


# Keywords that indicate a line is NOT a product
_NON_PRODUCT_KEYWORDS = [
    "rupees in words", "amount in words", "in words", "total bags",
    "total weight", "total qty", "total quantity", "net weight",
    "bank details", "bank name", "account no", "ifsc", "branch",
    "terms & conditions", "terms and conditions", "e & o.e",
    "subject to", "jurisdiction", "declaration", "certified",
    "vehicle no", "truck no", "lorry no", "transporter name",
    "payment terms", "due date", "credit period",
    "authorized signatory", "signature", "proprietor",
    "gut no", "shed no", "mouza", "plot no", "survey no",
    "round off", "rounded off", "rounding",
    "state code", "pin code", "pincode",
    "address", "phone", "mobile", "email",
    "fssai", "msme", "irn", "ack no", "ack dt"
]

_QTY_HEADERS = ["qty", "quantity", "bags", "no of bags", "no. of bags", "pcs", "nos"]
_RATE_HEADERS = ["rate", "rate/qtl", "rate 100 kg", "rate per", "price", "rate/bag"]
_AMOUNT_HEADERS = ["amount", "total value", "value", "total amount", "line total"]
_WEIGHT_HEADERS = ["net wt", "net weight", "total weight", "wt", "weight", "kg per bag",
                   "per bag", "wt/bag", "(kg)", "qtl"]

_COMMODITY_KEYWORDS = [
    'dal', 'chana', 'besan', 'atta', 'rice', 'oil', 'sugar', 'wheat', 'bag', 'kg',
    'pulses', 'toor', 'urad', 'moong', 'masoor', 'sooji', 'maida', 'ragi', 'millet',
    'mustard', 'soyabean', 'groundnut', 'spices', 'chilly', 'turmeric', 'coriander',
    'nagli', 'udid', 'katta', 'sona', 'masoori', 'basmati', 'kolam', 'lucky', 'angur',
    'rawa', 'rava', 'suji', 'flour', 'gram', 'whole', 'dall'
]


def _is_non_product_line(line_lower):
    """Check if a line is clearly not a product line item."""
    for kw in _NON_PRODUCT_KEYWORDS:
        if kw in line_lower:
            return True
    stripped = re.sub(r'[\d,\.\s:]+', '', line_lower).strip()
    if stripped in ('total', 'grand total', 'net total', 'sub total', 'subtotal',
                    'net payable', 'total payable', 'bill total', 'amount payable', 'continued'):
        return True
    return False


def _detect_column_order(header_line):
    """
    Detect the positions of qty, rate, amount, and weight columns from a header line.
    Returns a dict mapping column_type -> approximate character position.
    """
    header_lower = header_line.lower()
    positions = {}

    for header_group, col_type in [
        (_QTY_HEADERS, "qty"),
        (_RATE_HEADERS, "rate"),
        (_AMOUNT_HEADERS, "amount"),
        (_WEIGHT_HEADERS, "weight"),
    ]:
        best_pos = -1
        for kw in header_group:
            idx = header_lower.find(kw)
            if idx >= 0 and (best_pos < 0 or idx < best_pos):
                best_pos = idx
        if best_pos >= 0:
            positions[col_type] = best_pos

    return positions


def _parse_apmc_commodity_table(lines):
    """
    Parses specialized APMC & Flour Mill invoice table where columns are:
    Sr No -> Description -> HSN -> Net Kg -> Quantity -> [Batch] -> Total Qtl -> Rate/Qtl -> Amount -> Disc% -> Disc Amt -> After Disc -> Freight/Qtl -> Freight Amt -> Bill Amt
    Supports zero quantity lines (informational items) and captures line-item discounts/freights.
    """
    items = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        sr_match = re.match(r'^([1-9]\d?)$', line)
        if sr_match and i + 1 < len(lines):
            sr_num = int(sr_match.group(1))
            # Find HSN code (6-8 digits)
            hsn_idx = -1
            for k in range(i + 1, min(len(lines), i + 6)):
                if re.match(r'^\d{6,8}$', lines[k].strip()):
                    hsn_idx = k
                    break

            if hsn_idx != -1:
                desc = ' '.join(lines[i + 1:hsn_idx]).strip()
                hsn = lines[hsn_idx].strip()

                # Collect row tokens
                row_tokens = []
                j = hsn_idx + 1
                while j < len(lines):
                    tok = lines[j].strip()
                    # Check if next line is another serial number with HSN
                    if re.match(r'^[1-9]\d?$', tok) and j + 1 < len(lines):
                        is_next_sr = False
                        for k in range(j + 1, min(len(lines), j + 6)):
                            if re.match(r'^\d{6,8}$', lines[k].strip()):
                                is_next_sr = True
                                break
                        if is_next_sr:
                            break
                    if any(w in tok.lower() for w in ['total', 'payment', 'declaration', 'amount chargeable', 'rounding', 'fssai', 'for ']):
                        break
                    row_tokens.append(tok)
                    j += 1

                # Parse tokens into numbers and optional batch
                nums = []
                batch = ''
                for t in row_tokens:
                    t_clean = t.replace(',', '').replace('ī', '').replace('₹', '').strip()
                    try:
                        nums.append(float(t_clean))
                    except ValueError:
                        batch = t

                # Need at least: net_kg, qty, tot_qtl, rate_qtl, amt
                if len(nums) >= 5:
                    net_kg = nums[0]
                    qty = nums[1]
                    tot_qtl = nums[2]
                    rate_qtl = nums[3]
                    amt = nums[4]
                    disc_pct = nums[5] if len(nums) > 5 else 0.0
                    disc_amt = nums[6] if len(nums) > 6 else 0.0
                    after_disc = nums[7] if len(nums) > 7 else amt - disc_amt
                    frt_qtl = nums[8] if len(nums) > 8 else 0.0
                    frt_amt = nums[9] if len(nums) > 9 else 0.0
                    line_total = nums[10] if len(nums) > 10 else after_disc + frt_amt

                    # Total weight in kg = tot_qtl * 100 or qty * net_kg
                    total_weight_kg = tot_qtl * 100.0 if tot_qtl > 0 else (qty * net_kg if qty > 0 else 0.0)

                    items.append({
                        'description': desc,
                        'hsn': hsn,
                        'quantity': qty,
                        'unit': 'Bag',
                        'rate': round(rate_qtl, 2),
                        'amount': round(amt, 2),
                        'total_weight': round(total_weight_kg, 2),
                        'weight_per_unit': net_kg,
                        'total_qtl': tot_qtl,
                        'discount_pct': disc_pct,
                        'discount_amount': round(disc_amt, 2),
                        'freight_rate': frt_qtl,
                        'freight_amount': round(frt_amt, 2),
                        'line_total': round(line_total, 2)
                    })
                i = j - 1
        i += 1
    return items


def _parse_multi_line_table(lines):
    """
    Parses table where each column/cell is on its own line (e.g. PyMuPDF native text).
    Identifies rows starting with Sr No (1, 2, 3...) or commodity names,
    gathers the values in the row block, and solves for Qty * Rate == Amount
    or (Weight_qtl * Rate_qtl == Amount).
    """
    items = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        sr_match = re.match(r'^(\d{1,2})\.?\s*(.*)$', line)
        desc = None
        next_idx = i + 1

        if sr_match and int(sr_match.group(1)) <= 40:
            desc_part = sr_match.group(2).strip()
            if not desc_part and i + 1 < len(lines):
                candidate_desc = lines[i + 1].strip()
                if any(w in candidate_desc.lower() for w in _COMMODITY_KEYWORDS):
                    desc = candidate_desc
                    next_idx = i + 2
            elif desc_part and any(w in desc_part.lower() for w in _COMMODITY_KEYWORDS):
                desc = desc_part
                next_idx = i + 1

        if desc:
            clean_desc_key = desc.lower().strip()
            item_nums = []
            j = next_idx
            while j < min(len(lines), next_idx + 18):
                cur_line = lines[j].strip()
                if re.match(r'^(?:sub\s*total|total|\d{1,2}\.?\s+[A-Za-z]|less\s*:|freight|continued)', cur_line, re.IGNORECASE):
                    break
                if cur_line in ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10') and j > next_idx:
                    break

                cl = cur_line.replace(',', '').replace('ī', '').replace('₹', '').strip()
                # Handle formatted quantities like "60- 0 BAGS" -> 60.0
                m_dash = re.match(r'^(\d+)[\-\s]+(\d+)\s*(?:bags|bag|qtl|kg)?', cl, re.IGNORECASE)
                if m_dash:
                    item_nums.append(float(f"{m_dash.group(1)}.{m_dash.group(2)}"))
                    j += 1
                    continue

                m_num = re.search(r'[-]?\d+(?:\.\d+)?', cl)
                if m_num:
                    try:
                        v = float(m_num.group())
                        # Skip 8 digit HSN codes
                        if not (len(m_num.group()) == 8 and v > 1000000) and 0 <= v < 10000000:
                            item_nums.append(v)
                    except ValueError:
                        pass
                j += 1

            # Solve for Qty, Rate, Amount using mathematical relations:
            # 1. Qty * Rate == Amount
            found_solution = None
            for q_idx in range(len(item_nums)):
                for r_idx in range(len(item_nums)):
                    if q_idx == r_idx:
                        continue
                    q = item_nums[q_idx]
                    r = item_nums[r_idx]
                    for a_idx in range(len(item_nums)):
                        if a_idx in (q_idx, r_idx):
                            continue
                        a = item_nums[a_idx]
                        if abs(q * r - a) < 2.0 and a > 100.0:
                            qty_cand = min(q, r) if (min(q, r) < 2000 and max(q, r) > 500) else q
                            rate_cand = max(q, r) if (min(q, r) < 2000 and max(q, r) > 500) else r
                            found_solution = (qty_cand, rate_cand, a)
                            break
                    if found_solution:
                        break
                if found_solution:
                    break

            if found_solution:
                qty_val, rate_val, amt_val = found_solution

                # Determine pack weight: check parenthesized outer pack first e.g. "(30Kg)", else "50 KG"
                paren_match = re.search(r'\(\s*(\d+)\s*k(?:g|gs)\s*\)', desc, re.IGNORECASE)
                if paren_match:
                    kg_per_bag = float(paren_match.group(1))
                else:
                    w_match = re.search(r'(\d+)\s*k(?:g|gs)\b', desc, re.IGNORECASE)
                    kg_per_bag = float(w_match.group(1)) if w_match else 0.0

                # Check if any number in item_nums represents QTL (where v * 100 / qty is in standard bag weight range)
                qtl_val = 0.0
                for v in item_nums:
                    if 0.5 <= v <= 2000.0 and v not in (qty_val, rate_val, amt_val):
                        if qty_val > 0 and 15.0 <= (v * 100.0) / qty_val <= 100.0:
                            qtl_val = v
                            kg_per_bag = round((v * 100.0) / qty_val, 1)
                            break

                if qtl_val > 0:
                    tot_weight = round(qtl_val * 100.0, 2)
                elif kg_per_bag > 0 and qty_val > 0:
                    tot_weight = round(qty_val * kg_per_bag, 2)
                else:
                    tot_weight = 0.0

                items.append({
                    "description": desc,
                    "hsn": "",
                    "quantity": qty_val,
                    "unit": "Bag",
                    "rate": round(rate_val, 2),
                    "amount": round(amt_val, 2),
                    "total_weight": tot_weight,
                    "weight_per_unit": kg_per_bag
                })

            i = j - 1
        i += 1

    return items


def _smart_assign_numbers(nums, col_positions, num_positions=None):
    """
    Assign qty, rate, and amount from a list of numbers using detected column positions.
    Falls back to heuristic: amount = last number, qty and rate from remaining.
    """
    if not nums:
        return 1.0, 0.0, 0.0

    amount = nums[-1]
    qty = 1.0
    rate = 0.0

    if len(nums) == 1:
        return 1.0, nums[0], nums[0]

    # Pattern check: does any pair multiply to amount?
    for i in range(len(nums) - 1):
        for j in range(i + 1, len(nums)):
            n1, n2 = nums[i], nums[j]
            for k in range(len(nums)):
                if k not in (i, j):
                    cand_amt = nums[k]
                    if abs(n1 * n2 - cand_amt) < 2.0 and cand_amt > 100:
                        qty = min(n1, n2) if (min(n1, n2) < 2000 and max(n1, n2) > 500) else n1
                        rate = max(n1, n2) if (min(n1, n2) < 2000 and max(n1, n2) > 500) else n2
                        return qty, rate, cand_amt

    if col_positions and num_positions and len(nums) >= 3:
        qty_col_pos = col_positions.get("qty", -1)
        rate_col_pos = col_positions.get("rate", -1)
        interior = list(zip(nums[:-1], num_positions[:-1])) if num_positions else []

        if qty_col_pos >= 0 and interior:
            best_qty = min(interior, key=lambda x: abs(x[1] - qty_col_pos))
            qty = best_qty[0]
            interior = [x for x in interior if x is not best_qty]

        if rate_col_pos >= 0 and interior:
            best_rate = min(interior, key=lambda x: abs(x[1] - rate_col_pos))
            rate = best_rate[0]
        elif qty > 0 and amount > 0:
            rate = round(amount / qty, 2)
    elif len(nums) >= 3:
        if nums[0] <= 20 and len(nums) >= 4:
            qty = nums[1]
            rate = nums[-2] if len(nums) >= 4 else (amount / qty if qty > 0 else 0)
        else:
            qty = nums[0]
            rate = nums[1] if len(nums) >= 3 else (amount / qty if qty > 0 else 0)
    elif len(nums) == 2:
        qty = nums[0]
        rate = round(amount / qty, 2) if qty > 0 else amount

    return qty, rate, amount


def extract_product_table(full_text, nodes):
    """
    Parses product line items from invoice text or OCR tokens.
    Supports APMC/Flour Mill tabular layout, multi-line table layout,
    and single-line row layouts.
    Returns list of dicts with:
      description, hsn, quantity, unit, rate, amount, total_weight, weight_per_unit
    """
    lines = [l.strip() for l in full_text.splitlines() if l.strip()]

    # 1. Specialized APMC / Flour Mill commodity table parser
    apmc_items = _parse_apmc_commodity_table(lines)
    if apmc_items:
        seen = {}
        for it in apmc_items:
            k = (it["description"].lower().strip(), it["amount"])
            if k not in seen:
                seen[k] = it
        return list(seen.values())

    # 2. General multi-line cell parser
    multi_items = _parse_multi_line_table(lines)
    if multi_items:
        seen = {}
        for it in multi_items:
            k = (it["description"].lower().strip(), it["amount"])
            if k not in seen:
                seen[k] = it
        return list(seen.values())

    # 3. Single-line row parsing fallback (standard OCR row lines)
    items = []
    table_started = False
    col_positions = {}

    table_start_kws = [
        "description", "product", "particulars", "item name",
        "goods", "commodity", "item", "sr.no", "s.no", "article"
    ]
    table_end_kws = [
        "subtotal", "sub total", "taxable", "amount in words", "bank details",
        "terms & conditions", "jurisdiction", "total amount", "grand total",
        "net payable", "authorized signatory", "for "
    ]

    for line in lines:
        line_lower = line.lower()

        if not table_started and any(h in line_lower for h in table_start_kws):
            table_started = True
            col_positions = _detect_column_order(line)
            continue

        if any(f in line_lower for f in table_end_kws):
            if table_started and items:
                break

        if _is_non_product_line(line_lower):
            continue

        raw_tokens = line.split()
        nums = []
        num_char_positions = []
        for tok in raw_tokens:
            clean_tok = tok.replace(",", "").strip()
            if re.match(r'^(?:\+91[\s\-]?)?[6-9]\d{9}$', clean_tok):
                continue
            if len(clean_tok) > 12 and clean_tok.replace('.', '').isdigit():
                continue
            if re.match(r'^[0-9]{2}[A-Z]{5}', clean_tok):
                continue
            if clean_tok.isdigit() and (len(clean_tok) == 8 or (len(clean_tok) == 7 and clean_tok.startswith(('7', '8', '9')))):
                continue
            try:
                v = float(clean_tok)
                if 0 <= v < 100_000_000.0:
                    nums.append(v)
                    pos = line.find(tok)
                    num_char_positions.append(pos if pos >= 0 else 0)
            except ValueError:
                pass

        words = [w for w in raw_tokens if re.match(r'^[A-Za-z][A-Za-z0-9\-\.]*$', w)
                 or re.match(r'^[0-9]+[A-Za-z]', w)]

        if len(nums) >= 2 and len(words) >= 1:
            qty, rate, amount = _smart_assign_numbers(nums, col_positions, num_char_positions)

            unit_words = {"kg", "bag", "bags", "kts", "qtl", "nos", "pcs", "mt",
                          "kgs", "ltr", "lts", "ton", "tons", "unit", "units",
                          "quintal", "quintals", "mts", "each", "d", "c"}
            prod_words = [w for w in words if w.lower() not in unit_words]
            prod_desc = " ".join(prod_words) if prod_words else "Commodity Item"

            if not prod_desc or len(prod_desc) < 2 or prod_desc.replace('-', '').strip() == '':
                continue
            if prod_desc.lower() in ('total', 'grand', 'sub', 'net', 'round', 'rupees', 'words'):
                continue
            if amount <= 0:
                continue

            if "bag" in line_lower or "bags" in line_lower:
                unit = "Bag"
            elif "qtl" in line_lower or "quintal" in line_lower:
                unit = "Qtl"
            elif "kg" in line_lower:
                unit = "Kg"
            elif "ltr" in line_lower:
                unit = "Ltr"
            elif "ton" in line_lower or "mt" in line_lower:
                unit = "Ton"
            else:
                unit = "Unit"

            # Compute total weight
            w_match = re.search(r'\(?(\d+)\s*k(?:g|gs)\)?\b', prod_desc, re.IGNORECASE)
            kg_per_bag = float(w_match.group(1)) if w_match else 0.0
            tot_weight = round(qty * kg_per_bag, 2) if kg_per_bag > 0 else (round(qty * 100.0, 2) if unit == "Qtl" else 0.0)

            items.append({
                "description": prod_desc,
                "hsn": "",
                "quantity": qty,
                "unit": unit,
                "rate": round(rate, 2),
                "amount": round(amount, 2),
                "total_weight": tot_weight,
                "weight_per_unit": kg_per_bag
            })

    if not items:
        items.append({
            "description": "Agri / FMCG Purchase Commodity",
            "hsn": "",
            "quantity": 1.0,
            "unit": "Unit",
            "rate": 0.0,
            "amount": 0.0,
            "total_weight": 0.0,
            "weight_per_unit": 0.0
        })

    return items
