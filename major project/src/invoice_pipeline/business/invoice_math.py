"""
Invoice Mathematics, Constraint Verification, and Validation Layer
for APMC / FMCG Purchase Bills.

Equations:
  1. Product Total - Discounts = Net Product Value
  2. Net Product Value + Freight = Taxable Amount
  3. Taxable Amount + CGST + SGST (or IGST) = Grand Total

Validation Formula (per user specification):
  ((rate × total_weight_kg / 100) - discount + freight) × (1 + gst_rate)
  Adapted dynamically based on whether rates are per-bag or per-100kg,
  and whether discount/GST are in absolute values or percentages.
"""


def verify_invoice_equations(subtotal, discount, freight, cgst, sgst, igst, grand_total, tolerance=2.0):
    """
    Evaluates mathematical consistency of candidate amounts.
    Supports both standard (subtotal - discount + freight) and
    post-discount subtotal (subtotal + freight).
    Returns (is_valid, error_margin, calculated_total)
    """
    total_tax = cgst + sgst + igst
    calc_1 = (subtotal - discount + freight) + total_tax
    calc_2 = (subtotal + freight) + total_tax

    diff1 = abs(calc_1 - grand_total)
    diff2 = abs(calc_2 - grand_total)

    if diff2 <= tolerance or (diff2 < diff1 and diff2 < 5.0):
        return True, diff2, calc_2

    diff = diff1
    is_valid = diff <= tolerance
    return is_valid, diff, calc_1


def validate_invoice_formula(result):
    """
    Validation layer using the formula:
      ((((((rate * total_weight) / 100) - discount) + freight) * (1 + gst_rate)))

    Adapts dynamically:
      1. Product Subtotal:
         - When rate is per-bag: computed_subtotal = rate × qty
         - When rate is per-100kg: computed_subtotal = (rate × total_weight_kg) / 100
      2. Net Taxable:
         - If discount is value: taxable = sum(subtotals) - discount + freight
         - If discount is %: taxable = sum(subtotals) × (1 - discount%) + freight
      3. Grand Total:
         - If GST is value (CGST + SGST + IGST): grand_total = taxable + total_tax
         - If GST is %: grand_total = taxable × (1 + gst%)

    Returns dict with status ('VERIFIED', 'MISMATCH', 'SKIPPED'), computed values, and difference.
    """
    products = result.get("products", [])
    discount = result.get("discount", 0.0)
    freight = result.get("freight", 0.0)
    cgst = result.get("cgst", 0.0)
    sgst = result.get("sgst", 0.0)
    igst = result.get("igst", 0.0)
    extracted_gt = result.get("grand_total", 0.0)
    extracted_subtotal = result.get("subtotal", 0.0)

    if not products or extracted_gt <= 0:
        return {
            "status": "SKIPPED",
            "reason": "Missing products or grand total",
            "computed_grand_total": 0.0,
            "extracted_grand_total": extracted_gt,
            "difference": 0.0,
        }

    # --- Step 1: Compute product subtotals ---
    computed_product_total = 0.0
    product_validations = []

    for p in products:
        rate = p.get("rate", 0.0)
        qty = p.get("qty") if p.get("qty") is not None else p.get("quantity", 0.0)
        total_weight = p.get("total_weight", 0.0)
        product_amount = p.get("amount", 0.0)

        computed_amount = 0.0

        # Check if qty is 0 (informational item)
        if qty == 0 and product_amount == 0:
            computed_amount = 0.0
        # Check if rate × qty == amount (rate is per unit/bag)
        elif rate > 0 and qty > 0 and (product_amount <= 0 or abs(rate * qty - product_amount) < max(product_amount * 0.05, 5.0)):
            computed_amount = round(rate * qty, 2)
        # Check if (rate × total_weight) / 100 == amount (rate is per 100 kg / quintal)
        elif rate > 0 and total_weight > 0 and (product_amount <= 0 or abs((rate * total_weight / 100.0) - product_amount) < max(product_amount * 0.05, 5.0)):
            computed_amount = round(rate * total_weight / 100.0, 2)
        elif rate > 0 and qty > 0:
            computed_amount = round(rate * qty, 2)
        elif product_amount > 0:
            computed_amount = product_amount

        if computed_amount > 0 or (qty == 0 and product_amount == 0):
            computed_product_total += computed_amount
            pv = {
                "description": p.get("description", ""),
                "computed_amount": computed_amount,
                "extracted_amount": product_amount,
                "match": abs(computed_amount - product_amount) < max(product_amount * 0.05, 5.0) if product_amount > 0 else True,
            }
            product_validations.append(pv)

    if computed_product_total <= 0:
        return {
            "status": "SKIPPED",
            "reason": "Cannot compute product subtotals (missing rate/qty)",
            "computed_grand_total": 0.0,
            "extracted_grand_total": extracted_gt,
            "difference": 0.0,
        }

    # --- Step 2: Compute taxable amount ---
    # taxable = product_total - discount + freight
    taxable = computed_product_total - discount + freight

    # --- Step 3: Compute grand total ---
    total_tax = cgst + sgst + igst

    if total_tax > 0:
        computed_gt = round(taxable + total_tax, 2)
    elif extracted_subtotal > 0 and extracted_gt > extracted_subtotal:
        implied_tax = extracted_gt - (extracted_subtotal - discount + freight)
        if implied_tax > 0 and (extracted_subtotal - discount + freight) > 0:
            gst_rate = implied_tax / (extracted_subtotal - discount + freight)
            computed_gt = round(taxable * (1 + gst_rate), 2)
        else:
            computed_gt = round(taxable, 2)
    else:
        computed_gt = round(taxable, 2)

    difference = abs(computed_gt - extracted_gt)
    # Tolerance: up to 2.0 (covers round off differences e.g. -0.38) or 0.1% of invoice
    tolerance = max(extracted_gt * 0.001, 2.5)

    status = "VERIFIED" if difference <= tolerance else "MISMATCH"

    return {
        "status": status,
        "computed_grand_total": computed_gt,
        "extracted_grand_total": extracted_gt,
        "difference": round(difference, 2),
        "computed_product_total": round(computed_product_total, 2),
        "extracted_subtotal": round(extracted_subtotal, 2),
        "taxable_amount": round(taxable, 2),
        "total_tax": round(total_tax, 2),
        "product_validations": product_validations,
    }


def solve_field_constraints(candidates, product_items):
    """
    Evaluates candidate numbers to find the mathematically sound set of invoice totals.
    Uses constraint satisfaction: tries combinations until subtotal+tax=grand_total holds.
    """
    numbers = [c["value"] for c in candidates.get("all_numbers", [])]
    if not numbers:
        return {
            "subtotal": 0.0, "discount": 0.0, "freight": 0.0,
            "cgst": 0.0, "sgst": 0.0, "igst": 0.0,
            "grand_total": 0.0, "confidence": 0.0
        }

    numbers = sorted(list(set(numbers)), reverse=True)

    def best_candidate(field):
        field_cands = candidates.get(field, [])
        if field_cands:
            field_cands_sorted = sorted(field_cands, key=lambda x: x["conf"], reverse=True)
            return field_cands_sorted[0]["value"]
        return None

    product_sum = sum(item["amount"] for item in product_items if item.get("amount", 0) > 0)

    gt_val = best_candidate("grand_total")
    sub_val = best_candidate("subtotal")
    disc_val = best_candidate("discount") or 0.0
    frt_val = best_candidate("freight") or 0.0
    cgst_val = best_candidate("cgst") or 0.0
    sgst_val = best_candidate("sgst") or 0.0
    igst_val = best_candidate("igst") or 0.0

    grand_total_cand = gt_val if gt_val else (product_sum if product_sum > 0 else (numbers[0] if numbers else 0.0))

    best_solution = {
        "subtotal": sub_val if sub_val else (product_sum if product_sum > 0 else grand_total_cand),
        "discount": disc_val, "freight": frt_val,
        "cgst": cgst_val, "sgst": sgst_val, "igst": igst_val,
        "grand_total": grand_total_cand,
        "confidence": 0.50
    }

    if gt_val and sub_val:
        is_valid, diff, _ = verify_invoice_equations(
            sub_val, disc_val, frt_val, cgst_val, sgst_val, igst_val, gt_val, tolerance=5.0
        )
        if is_valid:
            return {
                "subtotal": sub_val,
                "discount": disc_val,
                "freight": frt_val,
                "cgst": cgst_val,
                "sgst": sgst_val,
                "igst": igst_val,
                "grand_total": gt_val,
                "confidence": 0.99 if diff < 1.0 else 0.90
            }

    # Fallback: combination search on top candidates
    top_nums = numbers[:8]
    for sub in top_nums:
        for disc in [0.0] + [n for n in top_nums if n < sub]:
            for frt in [0.0] + [n for n in top_nums if n < sub]:
                for tax in [0.0] + [n for n in top_nums if n < sub]:
                    for gt in top_nums:
                        is_valid, diff, _ = verify_invoice_equations(
                            sub, disc, frt, 0.0, 0.0, tax, gt, tolerance=3.0
                        )
                        if is_valid and gt > 0:
                            return {
                                "subtotal": sub,
                                "discount": disc,
                                "freight": frt,
                                "cgst": round(tax / 2.0, 2) if tax > 0 else 0.0,
                                "sgst": round(tax / 2.0, 2) if tax > 0 else 0.0,
                                "igst": tax if tax > 0 else 0.0,
                                "grand_total": gt,
                                "confidence": 0.99 if diff < 0.5 else 0.85
                            }

    return best_solution
