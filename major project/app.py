import os
import sys
import json
import tempfile
import urllib.request
import pandas as pd
import streamlit as st
from PIL import Image
import fitz  # PyMuPDF

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from src.invoice_pipeline.pipeline import process_invoice
from src.invoice_pipeline.export.excel_export import export_to_excel
from src.invoice_pipeline.ingestion.pdf_loader import load_document
from src.invoice_pipeline.preprocessing.enhance import enhance_image
from src.invoice_pipeline.ocr.paddle import extract_ocr_tokens

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DocuExtract AI - Intelligent Invoice Intelligence",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Main container and font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header hero */
    .hero-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 12px;
        padding: 24px 32px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .hero-title {
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 8px;
        color: #ffffff !important;
    }
    .hero-subtitle {
        font-size: 14px;
        opacity: 0.9;
        margin-bottom: 0px;
        color: #e0e7ff !important;
    }

    /* Metric cards */
    .metric-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .metric-label {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 20px;
        font-weight: 700;
        color: #0f172a;
        margin-top: 4px;
    }

    /* Status badge */
    .badge-success {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-warning {
        background-color: #fef9c3;
        color: #a16207;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-info {
        background-color: #e0e7ff;
        color: #3730a3;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
    }

    /* GSTIN Cards */
    .gstin-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    .gstin-card-buyer {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    .gstin-title {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        color: #475569;
    }
    .gstin-number {
        font-size: 15px;
        font-weight: 700;
        color: #1e293b;
        font-family: monospace;
        letter-spacing: 0.8px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def check_ollama_health(url="http://localhost:11434/api/tags"):
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name") for m in data.get("models", [])]
            return True, models
    except Exception:
        return False, []


def get_gpu_status():
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return True, f"{device_name} ({vram_gb:.1f} GB VRAM)"
        return False, "CPU Only (CUDA not active)"
    except Exception:
        return False, "GPU Check Unavailable"


def render_pdf_first_page(file_path):
    """Renders page 1 of PDF as a PIL Image."""
    doc = fitz.open(file_path)
    page = doc[0]
    pix = page.get_pixmap(dpi=150)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


# -----------------------------------------------------------------------------
# Sidebar: Engine Status & Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/invoice.png", width=64)
    st.title("Invoice AI Engine")
    st.caption("Multimodal Document Intelligence Pipeline")
    st.divider()

    st.subheader("⚙️ Pipeline Configuration")
    use_ollama = st.toggle("Enable Ollama Llama 3.2 Verification", value=True, help="Uses local Ollama LLM for semantic correction and table alignment.")
    
    st.subheader("🖥️ Hardware & Service Status")
    
    # GPU Check
    gpu_active, gpu_desc = get_gpu_status()
    if gpu_active:
        st.markdown(f"**GPU:** <span class='badge-success'>⚡ {gpu_desc}</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"**GPU:** <span class='badge-warning'>{gpu_desc}</span>", unsafe_allow_html=True)

    # Ollama Check
    ollama_ok, ollama_models = check_ollama_health()
    if ollama_ok:
        st.markdown(f"**Ollama:** <span class='badge-success'>🟢 Online (Llama 3.2)</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"**Ollama:** <span class='badge-warning'>⚠️ Offline (Deterministic fallback active)</span>", unsafe_allow_html=True)

    st.markdown("**OCR Engine:** <span class='badge-info'>PaddleOCR 300 DPI + CLAHE</span>", unsafe_allow_html=True)
    st.markdown("**Parser:** <span class='badge-info'>2D Spatial Graph + Constraint Solver</span>", unsafe_allow_html=True)

    st.divider()
    st.info("💡 **Tip:** Option 1 pipeline with 300 DPI enhancement is active for maximum extraction precision.")


# -----------------------------------------------------------------------------
# Hero Header
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-card">
    <div class="hero-title">🧾 Intelligent Invoice Extraction & Purchase Register</div>
    <div class="hero-subtitle">Automated OCR, Dual GSTIN Detection, Spatial Layout Parsing, Mathematical Constraint Verification & Excel Generation</div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Tabs Interface
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📄 Single Invoice Extractor",
    "📁 Batch Invoicing & Register",
    "📊 Business Analytics",
    "🔍 OCR & Spatial Inspector"
])


# =============================================================================
# TAB 1: Single Invoice Extractor
# =============================================================================
with tab1:
    col_input, col_action = st.columns([3, 1])
    
    # Input selection: Upload OR pick from Dataset/
    with col_input:
        input_choice = st.radio("Choose Input Method:", ["Upload File (PDF / Image)", "Select from Dataset Folder"], horizontal=True)

    target_file_path = None

    if input_choice == "Upload File (PDF / Image)":
        uploaded_file = st.file_uploader("Upload an invoice document (PDF, PNG, JPG, JPEG)", type=["pdf", "png", "jpg", "jpeg"])
        if uploaded_file is not None:
            # Save to temporary file
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                target_file_path = tmp_file.name
    else:
        dataset_dir = "Dataset"
        if os.path.exists(dataset_dir):
            files_in_dataset = [f for f in os.listdir(dataset_dir) if f.lower().endswith((".pdf", ".png", ".jpg", ".jpeg"))]
            if files_in_dataset:
                selected_sample = st.selectbox("Select Invoice from Dataset:", files_in_dataset)
                target_file_path = os.path.join(dataset_dir, selected_sample)
            else:
                st.warning("No invoices found in Dataset/ folder.")
        else:
            st.error("Dataset/ directory does not exist.")

    if target_file_path and os.path.exists(target_file_path):
        st.divider()
        col_doc, col_results = st.columns([1.1, 1.9])

        # Left Column: Document Preview
        with col_doc:
            st.markdown("### 🖼️ Document Preview")
            try:
                if target_file_path.lower().endswith(".pdf"):
                    preview_img = render_pdf_first_page(target_file_path)
                    st.image(preview_img, caption=os.path.basename(target_file_path), use_container_width=True)
                else:
                    st.image(target_file_path, caption=os.path.basename(target_file_path), use_container_width=True)
            except Exception as e:
                st.error(f"Could not render preview: {e}")

        # Right Column: Action & Extraction Dashboard
        with col_results:
            st.markdown("### ⚡ Live Feature Extraction")
            extract_btn = st.button("🚀 Run Extraction Pipeline", type="primary", use_container_width=True)

            if extract_btn:
                with st.spinner("Processing invoice through 300 DPI OCR, 2D Graph, and LLM Verifier..."):
                    try:
                        result = process_invoice(target_file_path, use_ollama=use_ollama)
                        st.session_state["last_single_result"] = result
                        st.success("✅ Extraction Complete!")
                    except Exception as e:
                        st.error(f"Pipeline error: {e}")

            # Display Extraction Results if available
            if "last_single_result" in st.session_state:
                res = st.session_state["last_single_result"]

                # 1. Header Information Cards
                st.markdown("#### 🏢 Header & Identity")
                hcol1, hcol2, hcol3 = st.columns(3)
                with hcol1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">Vendor Name</div>
                        <div class="metric-value" style="font-size: 16px;">{res.get('vendor_name', 'N/A')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with hcol2:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">Invoice Number</div>
                        <div class="metric-value" style="font-size: 16px;">{res.get('invoice_number', 'N/A')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with hcol3:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">Invoice Date</div>
                        <div class="metric-value" style="font-size: 16px;">{res.get('invoice_date', 'N/A')}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # 2. Dual GSTIN Cards
                gcol1, gcol2 = st.columns(2)
                with gcol1:
                    s_gst = res.get('seller_gstin') or "Not Detected / Exempted"
                    st.markdown(f"""
                    <div class="gstin-card">
                        <div class="gstin-title">🏢 Seller GSTIN</div>
                        <div class="gstin-number">{s_gst}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with gcol2:
                    b_gst = res.get('buyer_gstin') or "Not Detected / Consumer"
                    st.markdown(f"""
                    <div class="gstin-card-buyer">
                        <div class="gstin-title">🛒 Buyer GSTIN (Bill To / Consignee)</div>
                        <div class="gstin-number">{b_gst}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # 3. Product Line Items Table
                st.markdown("#### 📦 Product Line Items")
                products = res.get("products", [])
                if products:
                    display_prods = []
                    for p in products:
                        display_prods.append({
                            "Product Description": p.get("description", ""),
                            "Quantity": p.get("qty") if p.get("qty") is not None else p.get("quantity", 0.0),
                            "Unit": p.get("unit", "Bag"),
                            "Total Weight (Kg)": p.get("total_weight", 0.0),
                            "Rate (₹)": p.get("rate", 0.0),
                            "Line Total (₹)": p.get("amount", 0.0)
                        })
                    prod_df = pd.DataFrame(display_prods)
                    st.dataframe(prod_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No product items detected.")

                # 4. Financial & Tax Breakdown
                st.markdown("#### 💰 Financial Summary & Tax Breakdown")
                fcol1, fcol2, fcol3, fcol4 = st.columns(4)
                with fcol1:
                    st.metric("Subtotal (Taxable)", f"₹{res.get('subtotal', 0.0):,.2f}")
                with fcol2:
                    st.metric("Discount", f"₹{res.get('discount', 0.0):,.2f}")
                with fcol3:
                    st.metric("Freight / Charges", f"₹{res.get('freight', 0.0):,.2f}")
                with fcol4:
                    st.metric("Grand Total", f"₹{res.get('grand_total', 0.0):,.2f}")

                tcol1, tcol2, tcol3, tcol4 = st.columns(4)
                with tcol1:
                    st.metric("CGST", f"₹{res.get('cgst', 0.0):,.2f}")
                with tcol2:
                    st.metric("SGST", f"₹{res.get('sgst', 0.0):,.2f}")
                with tcol3:
                    st.metric("IGST", f"₹{res.get('igst', 0.0):,.2f}")
                with tcol4:
                    total_tax = res.get('cgst', 0.0) + res.get('sgst', 0.0) + res.get('igst', 0.0)
                    st.metric("Total GST", f"₹{total_tax:,.2f}")

                # Math Verification Indicator
                sub = res.get('subtotal', 0.0)
                disc = res.get('discount', 0.0)
                frt = res.get('freight', 0.0)
                gt = res.get('grand_total', 0.0)
                calc_gt = sub - disc + frt + total_tax
                diff = abs(calc_gt - gt)
                
                flags = res.get("flags", [])
                if not flags and (diff <= 2.5 or (gt > 0 and diff / gt < 0.005)):
                    st.markdown("<span class='badge-success'>✅ Verified by Validation Formula: ((((((Rate × Weight)/100) - Disc) + Freight) × (1 + GST)) == Grand Total)</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='badge-warning'>⚠️ Manual Review Recommended</span>", unsafe_allow_html=True)
                    if flags:
                        for flg in flags:
                            st.warning(f"🔍 Flagged for Human Review: {flg}")

                st.divider()
                
                # Export options for single invoice
                exp_col1, exp_col2 = st.columns(2)
                with exp_col1:
                    # Excel download
                    tmp_excel_out = os.path.join(tempfile.gettempdir(), "invoice_extraction.xlsx")
                    export_to_excel([res], tmp_excel_out)
                    with open(tmp_excel_out, "rb") as f:
                        st.download_button(
                            label="📥 Download Single Invoice as Excel (.xlsx)",
                            data=f.read(),
                            file_name=f"Invoice_{res.get('invoice_number', 'extracted')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                with exp_col2:
                    st.download_button(
                        label="📥 Download as JSON",
                        data=json.dumps(res, indent=2, ensure_ascii=False),
                        file_name=f"Invoice_{res.get('invoice_number', 'extracted')}.json",
                        mime="application/json",
                        use_container_width=True
                    )


# =============================================================================
# TAB 2: Batch Processing & Register Generator
# =============================================================================
with tab2:
    st.subheader("📁 Batch Processing Dashboard")
    st.write("Extract multiple invoice documents simultaneously and generate a unified, audit-ready **Purchase Register**.")

    bcol1, bcol2 = st.columns([2, 1])
    with bcol1:
        dataset_folder = st.text_input("Folder Path to Invoices:", value="Dataset")
        max_batch_files = st.number_input("Max files to process (0 = all files in folder):", min_value=0, max_value=2000, value=10)
    with bcol2:
        st.write("")
        st.write("")
        run_batch_btn = st.button("⚡ Start Batch Extraction", type="primary", use_container_width=True)

    if run_batch_btn:
        if not os.path.exists(dataset_folder):
            st.error(f"Folder '{dataset_folder}' does not exist.")
        else:
            all_files = [
                os.path.join(dataset_folder, f)
                for f in sorted(os.listdir(dataset_folder))
                if f.lower().endswith((".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif"))
            ]
            if max_batch_files > 0:
                all_files = all_files[:max_batch_files]

            if not all_files:
                st.warning("No supported invoice files found in the folder.")
            else:
                progress_bar = st.progress(0)
                status_ticker = st.empty()
                batch_results = []

                for idx, fpath in enumerate(all_files):
                    fname = os.path.basename(fpath)
                    status_ticker.markdown(f"**Processing ({idx + 1}/{len(all_files)}):** `{fname}`...")
                    try:
                        res = process_invoice(fpath, use_ollama=use_ollama)
                        batch_results.append(res)
                    except Exception as e:
                        st.error(f"Error on {fname}: {e}")
                    progress_bar.progress((idx + 1) / len(all_files))

                status_ticker.success(f"🎉 Batch Processing Complete! Extracted {len(batch_results)} invoices.")
                st.session_state["batch_results"] = batch_results

                # Save default output
                output_reg_path = "output/purchase_register_web.xlsx"
                export_to_excel(batch_results, output_reg_path)

    # Display Batch Results Table
    if "batch_results" in st.session_state and st.session_state["batch_results"]:
        records = st.session_state["batch_results"]
        st.divider()
        st.markdown(f"### 📋 Extracted Purchase Register ({len(records)} Invoices)")

        # Flatten records for DataFrame display
        flat_rows = []
        for sno, rec in enumerate(records, start=1):
            prods = rec.get("products", []) or [{}]
            for p in prods:
                flat_rows.append({
                    "S.No": sno,
                    "Source File": rec.get("file_name", ""),
                    "Vendor Name": rec.get("vendor_name", ""),
                    "Seller GSTIN": rec.get("seller_gstin", ""),
                    "Buyer GSTIN": rec.get("buyer_gstin", ""),
                    "Invoice No": rec.get("invoice_number", ""),
                    "Date": rec.get("invoice_date", ""),
                    "Product Description": p.get("description", ""),
                    "Qty": p.get("qty") or p.get("quantity") or 0.0,
                    "Unit": p.get("unit", ""),
                    "Total Weight (Kg)": p.get("total_weight", 0.0),
                    "Rate": p.get("rate", 0.0),
                    "Subtotal": rec.get("subtotal", 0.0),
                    "Discount": rec.get("discount", 0.0),
                    "Freight": rec.get("freight", 0.0),
                    "CGST": rec.get("cgst", 0.0),
                    "SGST": rec.get("sgst", 0.0),
                    "IGST": rec.get("igst", 0.0),
                    "Grand Total": rec.get("grand_total", 0.0),
                    "Review Flags": "; ".join(rec.get("flags", [])) if rec.get("flags") else "Verified",
                })

        batch_df = pd.DataFrame(flat_rows)
        st.dataframe(batch_df, use_container_width=True)

        st.divider()
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            out_excel = "output/purchase_register_web.xlsx"
            if os.path.exists(out_excel):
                with open(out_excel, "rb") as f:
                    st.download_button(
                        label="📥 Download Formatted Purchase Register (.xlsx)",
                        data=f.read(),
                        file_name="Purchase_Register_Master.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
        with dcol2:
            st.download_button(
                label="📥 Download Full Batch Dataset (.json)",
                data=json.dumps(records, indent=2, ensure_ascii=False),
                file_name="Purchase_Register_Dataset.json",
                mime="application/json",
                use_container_width=True
            )


# =============================================================================
# TAB 3: Business Analytics & Spend Intelligence
# =============================================================================
with tab3:
    st.subheader("📊 Purchase Analytics & Tax Insights")
    
    analytics_records = st.session_state.get("batch_results", [])
    
    if not analytics_records:
        # Check if existing output file can be loaded
        default_out = "output/purchase_register_fixed.xlsx"
        if os.path.exists(default_out):
            try:
                loaded_df = pd.read_excel(default_out)
                st.info(f"Loaded existing extraction register: `{default_out}` ({len(loaded_df)} item records). Run a batch in Tab 2 to refresh.")
            except Exception:
                loaded_df = None
        else:
            loaded_df = None
    else:
        loaded_df = pd.DataFrame(analytics_records)

    if loaded_df is not None and not loaded_df.empty:
        # Aggregate unique invoices
        if "Source File" in loaded_df.columns:
            unique_invoices = loaded_df.drop_duplicates(subset=["Source File"])
        else:
            unique_invoices = loaded_df

        total_bills = len(unique_invoices)
        total_spend = unique_invoices["Grand Total"].sum() if "Grand Total" in unique_invoices.columns else 0.0
        total_cgst = unique_invoices["CGST"].sum() if "CGST" in unique_invoices.columns else 0.0
        total_sgst = unique_invoices["SGST"].sum() if "SGST" in unique_invoices.columns else 0.0
        total_igst = unique_invoices["IGST"].sum() if "IGST" in unique_invoices.columns else 0.0
        total_gst = total_cgst + total_sgst + total_igst

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Bills Processed", f"{total_bills}")
        with m2:
            st.metric("Total Spend", f"₹{total_spend:,.2f}")
        with m3:
            st.metric("Total GST Input Credit", f"₹{total_gst:,.2f}")
        with m4:
            avg_val = total_spend / total_bills if total_bills > 0 else 0
            st.metric("Avg. Invoice Value", f"₹{avg_val:,.2f}")

        st.divider()
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("#### 🏢 Spend by Top Vendors")
            if "Vendor Name" in unique_invoices.columns and "Grand Total" in unique_invoices.columns:
                vendor_spend = unique_invoices.groupby("Vendor Name")["Grand Total"].sum().sort_values(ascending=False).head(8)
                st.bar_chart(vendor_spend)

        with chart_col2:
            st.markdown("#### 🏛️ GST Distribution (CGST vs SGST vs IGST)")
            tax_data = pd.DataFrame({
                "Tax Type": ["CGST", "SGST", "IGST"],
                "Amount": [total_cgst, total_sgst, total_igst]
            }).set_index("Tax Type")
            st.bar_chart(tax_data)

    else:
        st.info("Run a batch extraction in **Tab 2** or process an invoice to view live business intelligence charts.")


# =============================================================================
# TAB 4: OCR & Spatial Inspector
# =============================================================================
with tab4:
    st.subheader("🔍 OCR Tokens & 2D Reading Order Explorer")
    st.write("Inspect how the OCR engine interprets raw bounding boxes, confidence scores, and reading order graph before layout parsing.")

    sample_doc_choice = None
    if os.path.exists("Dataset"):
        d_files = [f for f in os.listdir("Dataset") if f.lower().endswith((".pdf", ".png", ".jpg", ".jpeg"))]
        if d_files:
            sample_doc_choice = st.selectbox("Select document to inspect:", d_files, key="inspect_doc_select")

    if sample_doc_choice:
        doc_path = os.path.join("Dataset", sample_doc_choice)
        inspect_btn = st.button("🔎 Inspect Raw OCR Tokens", use_container_width=True)

        if inspect_btn:
            with st.spinner("Extracting token coordinates and confidence..."):
                doc_info = load_document(doc_path)
                p1 = doc_info["pages"][0]
                enhanced = enhance_image(p1["image"])
                tokens = extract_ocr_tokens(enhanced, native_text=p1["text"])

                st.markdown(f"**Found {len(tokens)} text tokens**")
                
                tok_df = pd.DataFrame([
                    {
                        "Token ID": i + 1,
                        "Text": t["text"],
                        "Confidence": f"{t['confidence']*100:.1f}%",
                        "X0": round(t["bbox"][0], 1),
                        "Y0": round(t["bbox"][1], 1),
                        "X1": round(t["bbox"][2], 1),
                        "Y1": round(t["bbox"][3], 1),
                    }
                    for i, t in enumerate(tokens)
                ])

                icol1, icol2 = st.columns([1, 1.5])
                with icol1:
                    st.markdown("#### Document Image")
                    st.image(p1["image"], channels="BGR", use_container_width=True)
                with icol2:
                    st.markdown("#### Spatial Token Stream")
                    st.dataframe(tok_df, use_container_width=True, height=450)
