import os
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side


# 20-column schema with Total Weight and Review Flags
HEADERS = [
    "S.No",
    "Source File",
    "Vendor Name",
    "Seller GSTIN",
    "Buyer GSTIN",
    "Invoice No",
    "Date",
    "Product Description",
    "Qty",
    "Unit",
    "Total Weight (Kg)",
    "Rate",
    "Subtotal",
    "Discount",
    "Freight",
    "CGST",
    "SGST",
    "IGST",
    "Grand Total",
    "Review Flags",
]

# Column indices (1-based) that contain numeric values → apply number format
NUMERIC_COLS = {9, 11, 12, 13, 14, 15, 16, 17, 18, 19}


def export_to_excel(records, output_filepath):
    """
    Exports structured invoice extractions to a styled Purchase Register Excel.
    One row per product line item. Multiple products from one invoice span multiple rows.
    """
    os.makedirs(
        os.path.dirname(output_filepath) if os.path.dirname(output_filepath) else ".",
        exist_ok=True
    )

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Purchase Register"
    ws.append(HEADERS)

    # ── Header styling ──────────────────────────────────────────────────────────
    header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    center_wrap = Alignment(horizontal="center", vertical="center", wrap_text=True)

    for col_idx in range(1, len(HEADERS) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = center_wrap

    # ── Borders & alternating row fill ─────────────────────────────────────────
    thin_border = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )
    alt_fill = PatternFill(start_color="EBF3FB", end_color="EBF3FB", fill_type="solid")
    num_fmt = "#,##0.00"

    row_idx = 2

    for item_idx, record in enumerate(records, start=1):
        file_name    = record.get("file_name", "")
        vendor       = record.get("vendor_name", "") or ""
        seller_gstin = record.get("seller_gstin", "") or ""
        buyer_gstin  = record.get("buyer_gstin", "") or ""
        seller_pan   = record.get("seller_pan", "") or ""
        if not seller_gstin and seller_pan:
            seller_gstin = f"PAN: {seller_pan}"
        inv_no       = record.get("invoice_number", "") or ""
        inv_date     = record.get("invoice_date", "") or ""
        subtotal     = record.get("subtotal", 0.0) or 0.0
        discount     = record.get("discount", 0.0) or 0.0
        freight      = record.get("freight", 0.0) or 0.0
        cgst         = record.get("cgst", 0.0) or 0.0
        sgst         = record.get("sgst", 0.0) or 0.0
        igst         = record.get("igst", 0.0) or 0.0
        grand_total  = record.get("grand_total", 0.0) or 0.0
        flags        = "; ".join(record.get("flags", [])) if record.get("flags") else "Verified"

        products = record.get("products", [])
        if not products:
            products = [{"description": "", "qty": 0.0, "unit": "", "rate": 0.0, "amount": 0.0, "total_weight": 0.0}]

        use_alt = (item_idx % 2 == 0)

        for p in products:
            p_desc   = str(p.get("description", "") or "").strip()
            qty      = p.get("qty") or p.get("quantity") or 0.0
            unit     = str(p.get("unit", "") or "").strip()
            tot_wt   = p.get("total_weight", 0.0) or 0.0
            rate     = p.get("rate") or 0.0

            try:
                qty    = float(qty)
                rate   = float(rate)
                tot_wt = float(tot_wt)
            except (TypeError, ValueError):
                qty    = 0.0
                rate   = 0.0
                tot_wt = 0.0

            row_data = [
                item_idx,     # S.No
                file_name,    # Source File
                vendor,       # Vendor Name
                seller_gstin, # Seller GSTIN
                buyer_gstin,  # Buyer GSTIN
                inv_no,       # Invoice No
                inv_date,     # Date
                p_desc,       # Product Description
                qty,          # Qty
                unit,         # Unit
                tot_wt,       # Total Weight (Kg)
                rate,         # Rate
                subtotal,     # Subtotal
                discount,     # Discount
                freight,      # Freight
                cgst,         # CGST
                sgst,         # SGST
                igst,         # IGST
                grand_total,  # Grand Total
                flags,        # Review Flags
            ]
            ws.append(row_data)

            for c_idx in range(1, len(row_data) + 1):
                cell = ws.cell(row=row_idx, column=c_idx)
                cell.border = thin_border
                cell.alignment = Alignment(vertical="center", wrap_text=(c_idx in (8, 20)))
                if use_alt:
                    cell.fill = alt_fill
                if c_idx in NUMERIC_COLS:
                    cell.number_format = num_fmt

            row_idx += 1

    # ── Auto-fit column widths ──────────────────────────────────────────────────
    for col in ws.columns:
        max_len = max((len(str(cell.value or "")) for cell in col), default=0)
        col_letter = openpyxl.utils.get_column_letter(col[0].column)
        ws.column_dimensions[col_letter].width = min(max(max_len + 3, 12), 55)

    ws.freeze_panes = "A2"

    wb.save(output_filepath)
    print(f"[EXPORT] Purchase Register saved → {output_filepath}")
