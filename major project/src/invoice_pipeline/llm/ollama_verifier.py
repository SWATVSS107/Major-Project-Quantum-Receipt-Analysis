import json
import re
import urllib.request
import urllib.error

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"


def parse_resilient_json(text):
    """
    Robustly extracts JSON from LLM response, handling markdown wrappers and minor syntax errors.
    """
    clean_text = text.strip()
    for prefix in ["```json", "```"]:
        if clean_text.startswith(prefix):
            clean_text = clean_text[len(prefix):]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    clean_text = clean_text.strip()

    try:
        return json.loads(clean_text)
    except Exception:
        pass

    # Regex fallback for individual string fields
    result = {}
    for key in ["vendor_name", "seller_gstin", "buyer_gstin", "seller_pan", "invoice_number", "invoice_date"]:
        m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', clean_text)
        if m:
            result[key] = m.group(1)

    # Regex fallback for numeric fields — handle Indian comma format (e.g. 12,72,000.00)
    for key in ["subtotal", "discount", "freight", "cgst", "sgst", "igst", "grand_total"]:
        m = re.search(rf'"{key}"\s*:\s*"?([\d,]+\.?\d*)"?', clean_text)
        if m:
            try:
                result[key] = float(m.group(1).replace(",", ""))
            except ValueError:
                pass

    # Regex fallback for products array
    products_m = re.search(r'"products"\s*:\s*(\[.*?\])', clean_text, re.DOTALL)
    if products_m:
        try:
            result["products"] = json.loads(products_m.group(1))
        except Exception:
            pass

    return result


def verify_invoice_with_ollama(full_text, current_extraction):
    """
    Sends invoice OCR text to Ollama Llama 3.2 for high-accuracy structured extraction.
    Covers multi-page Indian APMC, Agri-commodity, and FMCG bills.
    """
    prompt = f"""You are an expert at extracting structured data from Indian APMC, Agri-commodity and FMCG purchase invoices.

IMPORTANT RULES FOR VENDOR NAME AND BUYER:
- vendor_name = the company that ISSUED / SOLD the goods at the VERY TOP header of the invoice.
- NEVER pick the buyer/consignee as the vendor:
  * "AVENUE SUPERMARTS LIMITED" is the BUYER when goods are billed to them. The VENDOR is the seller company at the top!
  * Look for "Bill To", "Buyer", "Consignee", "Name & Address Of Buyer", "Receiver Details" — those indicate the BUYER, not the vendor!
- seller_gstin = 15-character GSTIN of the company at the TOP header of the bill (null if not printed).
- buyer_gstin = 15-character GSTIN of the customer / buyer company (null if not printed).
- Note: AVENUE SUPERMARTS GSTINs always contain PAN "AACCA8432H" (e.g. 29AACCA8432H1ZM, 24AACCA8432H1ZW). That is ALWAYS the buyer_gstin!

INVOICE NUMBER RULES:
- invoice_number = actual bill/invoice number (e.g. "BOS/394", "1107", "25-26/162/11", "U-2/25-26/05152").
- NEVER pick E-Way Bill Number (12 digits like 182254846613) as the invoice number!

PRODUCT TABLE RULES:
- Extract each distinct commodity line item accurately.
- Description: Full commodity name (e.g. "Chana Besan 50 KG", "TOOR DAL (P) ANGUR OILY 5Kg Bag (30Kg)", "UDID DAL", "RAWA", "ATTA").
- Qty: Number of bags or units (e.g. 60, 167, 34, 250, 200, 580; items with 0 qty have qty 0.0).
- Unit: "Bag" or "Qtl" or "Kg".
- Rate: Unit price from Rate column (e.g. 3150.00, 3510.00, 3540.00, 8900.00, 3685.00, 3030.00). Never mistake 3150 as 30150.
- Amount: Line item total amount (e.g. 189000.00, 586170.00, 120360.00, 1112500.00, 368500.00, 456924.00).
- total_weight: Total weight of this item in kg (e.g. 125 quintal = 12500 kg; 100 quintal = 10000 kg; 150.80 quintal = 15080 kg; 60 bags * 50kg = 3000 kg).

FINANCIAL BREAKDOWN RULES:
- subtotal = sum of product line totals before discount, freight and taxes (e.g. 1112500.00, 825424.00, 189000.00, 706530.00).
- discount = cash discount, trade discount, rebate (e.g. "CASH DISC. (-)14,130.60" means 14130.60; 2.5% discount means subtotal * 0.025). Enter as positive number. 0 if none.
- freight = freight advance, advance freight, transportation (e.g. "Freight Advance 5,100.00" means 5100.00; 290/qtl * 125 qtl = 36250.00). 0 if none.
- cgst = Central GST amount. 0 if not present / exempted.
- sgst = State GST amount. 0 if not present / exempted.
- igst = Integrated GST amount. 0 if not present.
- grand_total = final payable amount on bill (e.g. 1120937.50, 804788.40, 189375.00, 727019.00).
  Note: When you see APMC, discount, GST, round off, the final value at the bottom is the grand total.
- Note that the Rupee symbol ₹ is sometimes encoded as 'ī' (e.g. 'TOTAL ī 7,27,019.00' is 727019.00).

Raw Invoice Text:
{full_text[:12000]}

Extract this EXACT JSON (no markdown, no explanation):
{{
  "vendor_name": "SELLER company name at TOP of bill (NOT buyer/consignee)",
  "seller_gstin": "seller 15-char GSTIN or null",
  "buyer_gstin": "buyer 15-char GSTIN or null",
  "seller_pan": "seller 10-char PAN if no GSTIN, or null",
  "invoice_number": "invoice/bill number",
  "invoice_date": "date as shown",
  "subtotal": product line total before discount/freight/tax (number),
  "discount": discount amount as positive number (number, 0 if none),
  "freight": freight amount as positive number (number, 0 if none),
  "cgst": CGST amount (number, 0 if not present),
  "sgst": SGST amount (number, 0 if not present),
  "igst": IGST amount (number, 0 if not present),
  "grand_total": final Total payable on the bill (number),
  "products": [
    {{
      "description": "actual commodity name",
      "qty": number of bags/units (number),
      "unit": "Bag" or "Qtl" or "Kg",
      "rate": price per unit from Rate column (number),
      "amount": line item total amount (number),
      "total_weight": total weight in kg (number)
    }}
  ]
}}"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": 4096
        }
    }

    try:
        req = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            response_text = res_data.get("response", "").strip()
            print(f"[LLM Raw] {response_text[:300]}")

            refined = parse_resilient_json(response_text)
            if refined:
                refined["ollama_verified"] = True
                return refined
            else:
                current_extraction["ollama_verified"] = False
                return current_extraction

    except Exception as e:
        print(f"[LLM] Ollama error/fallback: {e}")
        current_extraction["ollama_verified"] = False
        return current_extraction
