import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.invoice_pipeline.ingestion.pdf_loader import load_document
from src.invoice_pipeline.preprocessing.enhance import enhance_image
from src.invoice_pipeline.ocr.paddle import extract_ocr_tokens

bills = [
    'Dataset/01.11.23 NAVI MUMBAI.pdf',
    'Dataset/_Avenue Supermarts ( INDORE ) - 3157.pdf',
    'Dataset/_Dattawadi.xlsx - PUNE.pdf-11450.pdf',
]

for bill in bills:
    print('=' * 70)
    print('FILE:', bill)
    doc = load_document(bill)
    page = doc['pages'][0]
    full_text = page['text'] if page['has_native_text'] else ''
    print(f'Native text: {page["has_native_text"]} | len={len(full_text)}')
    print()
    if full_text:
        print('--- NATIVE TEXT (first 2000 chars) ---')
        print(full_text[:2000])
    print()
