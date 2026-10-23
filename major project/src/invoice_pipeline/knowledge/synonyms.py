"""
Synonym and regex pattern definitions for Indian APMC, FMCG, and Agri-Commodity Invoices.
Covers a wide range of label variations seen in the dataset.
"""

LABEL_SYNONYMS = {
    "invoice_number": [
        "invoice no", "invoice number", "inv no", "inv #", "bill no", "bill number",
        "tax invoice no", "document no", "doc no", "invoice #", "bill of supply no",
        "b.o.s no", "bos no", "ref no", "reference no"
    ],
    "invoice_date": [
        "date", "invoice date", "bill date", "dated", "doc date", "billing date",
        "date of invoice", "invoice dt", "bill dt"
    ],
    "gstin": [
        "gstin", "gst no", "gstin no", "gstin/uin", "supplier gstin",
        "gst number", "gstin:", "gst reg no", "gst registration"
    ],
    "vendor_name": [
        "supplier", "seller", "vendor", "from:", "biller", "company name",
        "sold by", "issued by", "manufacturer"
    ],
    "buyer_name": [
        "buyer", "bill to", "consignee", "billed to:", "party name", "customer",
        "ship to", "deliver to", "recipient"
    ],
    "subtotal": [
        "subtotal", "sub total", "taxable amount", "taxable value", "net amount",
        "total amount before tax", "total taxable", "product total", "goods total",
        "basic amount", "assessable value", "net taxable value"
    ],
    "discount": [
        "discount", "less discount", "trade discount", "scheme discount",
        "disc", "disc.", "less:", "deduction", "rebate", "cash discount", "cd",
        "qty disc", "rate diff", "rate difference", "less", "allowance",
        "cash disc", "cd amt", "disc amt", "disc amount"
    ],
    "freight": [
        "freight", "transport charges", "loading charges", "cartage",
        "freight charges", "packing & forwarding", "coolie", "hamali",
        "delivery charges", "logistics", "handling charges", "bardana", "kanta",
        "tol", "mandi cess", "market fee", "labour charges", "weighment", "tcs",
        "lorry freight", "freight and cartage", "total freight", "fright",
        "total fright", "transportation", "transport", "transp.", "lorry fright"
    ],
    "cgst": [
        "cgst", "central gst", "central tax", "c.g.s.t"
    ],
    "sgst": [
        "sgst", "state gst", "state tax", "utgst", "s.g.s.t", "u.t.g.s.t"
    ],
    "igst": [
        "igst", "integrated gst", "integrated tax", "i.g.s.t"
    ],
    "grand_total": [
        "grand total", "net payable", "total amount", "invoice total",
        "total payable", "amount due", "final amount", "total invoice value",
        "bill total", "net amount payable", "amount payable", "total due",
        "net total", "total (rs)", "total rs.", "net bill", "net bi11", "net biil",
        "apmc", "net payable amount", "bill amount", "total bill"
    ]
}

WEIGHT_SYNONYMS = {
    "weight": [
        "total wt", "gross wt", "gross weight", "wt per bag", "weight per bag",
        "kg/bag", "kg per bag"
    ]
}

# Standard 15-character Indian GSTIN pattern
GSTIN_REGEX = r'[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}'

# Permanent Account Number (PAN) pattern
PAN_REGEX = r'[A-Z]{5}[0-9]{4}[A-Z]'

# Date patterns: DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD, DD Mon YYYY, etc.
DATE_REGEX = (
    r'\b(?:\d{1,2}[\\/\.\-]\d{1,2}[\\/\.\-]\d{2,4}'
    r'|\d{4}[\\/\.\-]\d{1,2}[\\/\.\-]\d{1,2}'
    r'|\d{1,2}[\s\-](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s\-]\d{2,4})\b'
)

# Amount patterns: 1,00,000.00 or 100000.00 or 1000
AMOUNT_REGEX = r'\b\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?\b|\b\d+(?:\.\d{1,2})?\b'

SELLER_INDICATORS = [
    "industries", "enterprises", "enterprise", "private", "pvt", "ltd", "limited",
    "corporation", "cooperations", "manufacturing", "traders", "trading",
    "products", "foods", "starch", "mills", "mill", "agro", "dal", "oil",
    "flour", "pulses", "agencies", "agency", "dall",
    "m/s", "messrs", "supplier", "seller", "manufacturer"
]

BUYER_INDICATORS = [
    "bill to", "billed to", "consignee", "ship to", "shipped to",
    "deliver to", "buyer", "sold to", "party name", "customer",
    "recipient", "details of receiver"
]

NON_PRODUCT_LINE_KEYWORDS = [
    "rupees in words", "amount in words", "in words", "total bags",
    "total weight", "total qty", "total quantity", "net weight",
    "bank details", "bank name", "account no", "ifsc", "branch",
    "terms & conditions", "terms and conditions", "e & o.e",
    "subject to", "jurisdiction", "declaration", "certified",
    "vehicle no", "truck no", "lorry no", "transport", "transporter",
    "payment terms", "due date", "credit period",
    "authorized signatory", "for ", "signature", "proprietor",
    "gut no", "shed no", "mouza", "plot no", "survey no",
    "address", "pin code", "pincode", "state code",
    "round off", "rounded off", "rounding"
]
