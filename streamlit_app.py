# streamlit_app.py
"""
Streamlit front-end for pdf_to_excel_production.py

Place this file alongside your existing pdf_to_excel_production.py in the same folder.

Run:
    streamlit run streamlit_app.py

Notes:
- External dependencies (Tesseract, Poppler, Ghostscript) must be installed on the host.
- Python packages required: streamlit, pandas, openpyxl, pdfplumber, camelot-py[cv] (optional), pdf2image, pytesseract, opencv-python, numpy

This app uploads a PDF, runs the production pipeline, shows a preview of the first sheet and provides a download button for the generated .xlsx file.
"""

import streamlit as st
import tempfile
import os
import io
import shutil
from pathlib import Path
import pandas as pd
import logging

# Import conversion function and helper
try:
    from pdf_to_excel_production import pdf_to_excel_production, ensure_external_tools, logger as pdf_logger
except Exception as e:
    raise SystemExit("Could not import pdf_to_excel_production module. Make sure it's in the same folder. Error: %s" % e)

st.set_page_config(page_title="PDF → Excel Converter", layout="wide")

st.title("PDF → Excel Converter — Bank statements & Reports")
st.markdown(
    "Upload a PDF (bank statement / report / invoice) and convert it to Excel. ``Camelot`` and ``pdfplumber`` are used for table extraction; ``Tesseract`` OCR is used as fallback for scanned pages."
)

# Sidebar options
st.sidebar.header("Conversion options")
dpi = st.sidebar.slider("OCR DPI (pdf2image)", min_value=150, max_value=600, value=300, step=50)
force_ocr = st.sidebar.checkbox("Force OCR on all pages (use for scanned PDFs)", value=False)
camelot_flavor = st.sidebar.selectbox("Camelot preferred flavor", options=["lattice", "stream"], index=0)
show_logs = st.sidebar.checkbox("Show conversion log", value=True)
keep_temp = st.sidebar.checkbox("Keep temporary files (for debugging)", value=False)

# Show external tool notices
notes = ensure_external_tools()
if notes:
    st.sidebar.warning("External tool notices:\n" + "\n".join(f"- {n}" for n in notes))

uploaded = st.file_uploader("Upload PDF file", type=["pdf"])

if uploaded is not None:
    st.info(f"Uploaded: {uploaded.name}  — size: {uploaded.size/1024:.1f} KB")
    col1, col2 = st.columns([3, 1])

    with col1:
        st.write("Preview / options")
        try:
            raw = uploaded.getvalue()
            st.text_area("PDF file bytes preview (first 1024 bytes)", raw[:1024], height=120)
        except Exception:
            pass

    with col2:
        out_name = st.text_input("Output Excel filename", value=Path(uploaded.name).stem + "_output.xlsx")
        convert_btn = st.button("Convert to Excel")

    if convert_btn:
        tmp_dir = tempfile.mkdtemp(prefix="pdf2xlsx_")
        in_path = os.path.join(tmp_dir, uploaded.name)
        out_path = os.path.join(tmp_dir, out_name if out_name.lower().endswith('.xlsx') else out_name + '.xlsx')
        with open(in_path, "wb") as f:
            f.write(uploaded.getvalue())

        # capture logs in a string buffer to show in UI
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
        pdf_logger.addHandler(handler)

        st.info("Starting conversion — this can take a few seconds to a few minutes depending on PDF size and OCR.")
        with st.spinner("Converting..."):
            try:
                pdf_to_excel_production(in_path, out_path, dpi=dpi, force_ocr=force_ocr, camelot_flavor=(camelot_flavor,))
                st.success("Conversion finished.")

                # offer download
                with open(out_path, "rb") as f:
                    data = f.read()
                    st.download_button("Download Excel file", data=data, file_name=os.path.basename(out_path),
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                # show preview of first sheet (if possible)
                try:
                    df_preview = pd.read_excel(out_path, sheet_name=0)
                    st.subheader("Preview: first sheet")
                    st.dataframe(df_preview.head(200))
                except Exception as e:
                    st.warning(f"Could not preview first sheet: {e}")

            except Exception as e:
                st.error(f"Conversion failed: {e}")
            finally:
                pdf_logger.removeHandler(handler)
                log_text = log_stream.getvalue()
                if show_logs and log_text:
                    st.subheader("Conversion log")
                    st.text_area("Log output", log_text, height=300)
                if not keep_temp:
                    try:
                        shutil.rmtree(tmp_dir)
                    except Exception:
                        pass
                else:
                    st.info(f"Temporary folder kept at: {tmp_dir}")

# Footer notes
st.markdown("---")
st.caption("Note: This app runs locally and requires external tools (Tesseract OCR, Poppler, Ghostscript) for best results.")

# Optional: Dockerfile snippet for containerizing the app (copy into a Dockerfile if you want to build)
# ----- Dockerfile (example) -----
# FROM python:3.10-slim
# RUN apt-get update && apt-get install -y poppler-utils tesseract-ocr ghostscript build-essential \
#     && rm -rf /var/lib/apt/lists/*
# WORKDIR /app
# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt
# COPY . /app
# EXPOSE 8501
# CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
# --------------------------------
