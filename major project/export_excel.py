import os
import pandas as pd
from typing import List, Dict, Any
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from src.config import config


def format_predictions_to_dataframe(predictions: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Transforms structured invoice prediction dictionaries into a tabular DataFrame
    matching the 16 target invoice columns.
    """
    rows = []
    
    for pred in predictions:
        source_file = pred.get("source_file", "")
        vendor_name = pred.get("vendor_name", "")
        gstin = pred.get("gstin", "")
        invoice_no = pred.get("invoice_no", "")
        date = pred.get("date", "")
        grand_total = pred.get("grand_total", "")
        
        items = pred.get("items", [])
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
            rows.append({
                "Source File": source_file,
                "Vendor Name": vendor_name,
                "GSTIN": gstin,
                "Invoice No": invoice_no,
                "Date": date,
                "Product Description": item.get("product_description", ""),
                "Qty": item.get("qty", ""),
                "Unit": item.get("unit", ""),
                "Rate": item.get("rate", ""),
                "Subtotal": item.get("subtotal", ""),
                "Discount": item.get("discount", ""),
                "Freight": item.get("freight", ""),
                "CGST": item.get("cgst", ""),
                "SGST": item.get("sgst", ""),
                "IGST": item.get("igst", ""),
                "Grand Total": grand_total
            })
            
    df = pd.DataFrame(rows, columns=config.EXCEL_COLUMNS)
    return df


def save_to_styled_excel(df: pd.DataFrame, output_path: str):
    """
    Saves DataFrame to an Excel sheet with clean styling and auto-fitted columns.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Invoice_Extraction")
        
        workbook = writer.book
        worksheet = writer.sheets["Invoice_Extraction"]
        
        # Styles
        header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
        alignment_center = Alignment(horizontal="center", vertical="center")
        alignment_left = Alignment(horizontal="left", vertical="center")
        thin_border = Border(
            left=Side(style="thin", color="D3D3D3"),
            right=Side(style="thin", color="D3D3D3"),
            top=Side(style="thin", color="D3D3D3"),
            bottom=Side(style="thin", color="D3D3D3")
        )
        
        # Style Header Row
        for col_num in range(1, len(df.columns) + 1):
            cell = worksheet.cell(row=1, column=col_num)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = alignment_center
            
        # Style Data Rows and auto-fit widths
        for col_idx, col_name in enumerate(df.columns, start=1):
            max_len = max(
                len(str(col_name)),
                max([len(str(val)) for val in df[col_name].fillna("")], default=0)
            )
            col_letter = get_column_letter(col_idx)
            worksheet.column_dimensions[col_letter].width = min(max(max_len + 4, 12), 45)
            
            for row_num in range(2, len(df) + 2):
                cell = worksheet.cell(row=row_num, column=col_idx)
                cell.border = thin_border
                if col_name in ["Qty", "Rate", "Subtotal", "Discount", "Freight", "CGST", "SGST", "IGST", "Grand Total", "Date", "Invoice No", "GSTIN"]:
                    cell.alignment = alignment_center
                else:
                    cell.alignment = alignment_left
                    
    print(f"Excel report exported successfully to: {output_path}")


def export_predictions_to_excel(predictions: List[Dict[str, Any]], output_path: str = None) -> str:
    if output_path is None:
        output_path = os.path.join(config.OUTPUT_DIR, "extracted_invoices.xlsx")
    df = format_predictions_to_dataframe(predictions)
    save_to_styled_excel(df, output_path)
    return output_path
