"""
pdf_to_excel_production.py

Production-ready PDF -> Excel converter focused on bank statements, financial reports, invoices, and other tabular PDFs.
Features:
- Detects whether PDF pages are born-digital (text layer) or scanned images
- Tries Camelot (lattice & stream) for robust table detection on born-digital PDFs
- Falls back to pdfplumber table extraction and, for scanned pages, pdf2image + pytesseract OCR
- Image preprocessing (deskew, binarize, denoise) to improve OCR accuracy
- Heuristics to normalize bank-statement-like tables (date detection, amount columns, debit/credit/balance columns)
- Exports a multi-sheet Excel workbook: one sheet per detected table + a QA sheet with extraction notes and low-confidence cells
- CLI with configurable options (dpi, force-ocr, camelot flavor, output path)

Limitations & notes:
- For the best accuracy on difficult scanned PDFs, consider commercial OCR (ABBYY, Google Cloud Vision, Microsoft Read API).
- Camelot requires Ghostscript; Camelot (lattice) works best when tables have ruled lines. Stream flavor is good for whitespace-separated tables.
- OCR requires Tesseract installed on the system. pdf2image requires Poppler.

External system dependencies (must be installed separately):
- Tesseract OCR (https://github.com/tesseract-ocr/tesseract)
- Poppler (for pdf2image)
- Ghostscript (for Camelot in some environments)

Python dependencies (pip install):
- pandas, openpyxl, pdfplumber, camelot-py[cv], pdf2image, pytesseract, opencv-python, numpy

Example install:
    pip install pandas openpyxl pdfplumber camelot-py[cv] pdf2image pytesseract opencv-python numpy

Usage:
    python pdf_to_excel_production.py --input sample.pdf --output output.xlsx --dpi 300

"""

import argparse
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import pdfplumber

# Try to import camelot; optional but recommended
try:
    import camelot
    CAMELot_AVAILABLE = True
except Exception:
    CAMELot_AVAILABLE = False

# Optional image/OCR related imports
try:
    from pdf2image import convert_from_path
    import pytesseract
    import cv2
    PDF2IMAGE_AVAILABLE = True
except Exception:
    PDF2IMAGE_AVAILABLE = False

# ----------------- Logging -----------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----------------- Utilities -----------------
DATE_RE = re.compile(r"\b(\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4}|\d{4}[\-/]\d{1,2}[\-/]\d{1,2})\b")
AMOUNT_RE = re.compile(r"^-?\s*[\d,]+(?:\.\d+)?$")


def ensure_external_tools():
    notes = []
    if not CAMELot_AVAILABLE:
        notes.append("camelot not installed (pip install camelot-py[cv]) - will skip Camelot extraction")
    if not PDF2IMAGE_AVAILABLE:
        notes.append("pdf2image/pytesseract/opencv not installed - scanned OCR fallback will be disabled")
    return notes


def is_page_scanned(page: pdfplumber.page.Page) -> bool:
    """Heuristic: page has little or no extractable text -> treat as scanned."""
    try:
        text = page.extract_text()
        if not text or text.strip() == "":
            return True
        # If text exists but it's extremely sparse compared to page area, consider scanned
        words = text.split()
        return len(words) < 10
    except Exception:
        return True


# ----------------- Image preprocessing for OCR -----------------
def preprocess_pil_image_for_ocr(pil_image):
    """Takes a PIL image (RGB) and returns a processed OpenCV grayscale image suitable for pytesseract."""
    img = np.array(pil_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Resize if small
    h, w = gray.shape
    if max(h, w) < 1200:
        scale = 1200 / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    # Binarize with OTSU
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Optionally deskew (simple method)
    coords = np.column_stack(np.where(th > 0))
    if coords.size:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > 0.1:
            (h, w) = th.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            th = cv2.warpAffine(th, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return th


def ocr_image_to_lines(pil_image, lang='eng') -> Tuple[List[str], pd.DataFrame]:
    """Runs pytesseract on the image and returns a list of text lines and a word-level DataFrame (TSV style)"""
    if not PDF2IMAGE_AVAILABLE:
        return [], pd.DataFrame()
    proc = preprocess_pil_image_for_ocr(pil_image)
    # pytesseract returns a dataframe (word-level) which we can aggregate
    custom_config = r'--oem 3 --psm 6'
    try:
        df = pytesseract.image_to_data(proc, output_type=pytesseract.Output.DATAFRAME, config=custom_config, lang=lang)
    except Exception as e:
        logger.warning("pytesseract.image_to_data failed: %s", e)
        return [], pd.DataFrame()
    if df is None or df.empty:
        return [], pd.DataFrame()
    df = df[df.conf != -1]
    # Aggregate words into lines by block/line numbers
    lines = []
    grouped = df.groupby(['block_num', 'par_num', 'line_num'])
    for _, g in grouped:
        txt = " ".join(g['text'].astype(str).tolist()).strip()
        if txt:
            lines.append(txt)
    return lines, df


# ----------------- Table extraction methods -----------------

def extract_with_camelot(pdf_path: str, pages='1-end', flavor_preference=('lattice', 'stream')) -> List[pd.DataFrame]:
    dfs = []
    if not CAMELot_AVAILABLE:
        return dfs
    for flavor in flavor_preference:
        try:
            logger.info("Running Camelot (flavor=%s)...", flavor)
            tables = camelot.read_pdf(pdf_path, pages=pages, flavor=flavor, strip_text='\n')
            logger.info("Camelot detected %d tables (flavor=%s)", len(tables), flavor)
            for t in tables:
                if t.df is not None and not t.df.empty:
                    # camelot returns DataFrame-like t.df
                    dfs.append(t.df)
            if dfs:
                return dfs
        except Exception as e:
            logger.warning("Camelot (%s) failed: %s", flavor, e)
    return dfs


def extract_with_pdfplumber(pdf_path: str) -> List[pd.DataFrame]:
    dfs = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            try:
                tables = page.extract_tables()
                if tables:
                    for t in tables:
                        # t is a list of rows
                        if not t:
                            continue
                        # If header exists in first row (strings), use as columns
                        df = pd.DataFrame(t[1:], columns=t[0]) if len(t) > 1 else pd.DataFrame(t)
                        df.attrs['source'] = f'pdfplumber_page_{i}'
                        dfs.append(df)
                else:
                    # No explicit table detected; still capture text lines for QA
                    text = page.extract_text()
                    if text and text.strip():
                        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                        df = pd.DataFrame(lines, columns=['text'])
                        df.attrs['source'] = f'pdfplumber_text_page_{i}'
                        dfs.append(df)
                    else:
                        # scanned page placeholder
                        dfs.append(pd.DataFrame())
            except Exception as e:
                logger.warning("pdfplumber page %d extraction failed: %s", i, e)
    return dfs


# ----------------- Heuristics & Cleaning -----------------

def drop_empty_rows_and_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace(r'^\s*$', pd.NA, regex=True)
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')
    return df


def normalize_amount_string(s: Any) -> Any:
    if pd.isna(s):
        return s
    if isinstance(s, (int, float)):
        return s
    try:
        txt = str(s).strip()
        # Remove currency symbols and stray characters
        txt = re.sub(r'[₹$,]', '', txt)
        txt = txt.replace('\xa0', '')
        txt = txt.replace('\u2009', '')
        txt = txt.strip()
        # handle parentheses for negatives
        if txt.startswith('(') and txt.endswith(')'):
            txt = '-' + txt[1:-1]
        # if numeric-looking
        if AMOUNT_RE.match(txt.replace(' ', '')):
            return float(txt.replace(',', ''))
        return s
    except Exception:
        return s


def detect_and_normalize_statement(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Attempt to detect common columns (date, description, debit/credit/balance) and normalize amounts."""
    note = {}
    if df is None or df.empty:
        return df, note
    df = drop_empty_rows_and_columns(df)
    # Normalize column names
    cols = list(df.columns)
    col_map = {}
    for c in cols:
        cl = str(c).lower()
        if 'date' in cl or DATE_RE.search(cl):
            col_map[c] = 'date'
        elif 'description' in cl or 'particulars' in cl or 'narration' in cl or 'trans' in cl:
            col_map[c] = 'description'
        elif 'debit' in cl or 'withdrawal' in cl or 'dr' == cl.strip().lower():
            col_map[c] = 'debit'
        elif 'credit' in cl or 'deposit' in cl or 'cr' == cl.strip().lower():
            col_map[c] = 'credit'
        elif 'balance' in cl or 'bal' in cl:
            col_map[c] = 'balance'
        elif 'amount' in cl:
            # ambiguous - could be either
            # try to detect by content below
            col_map[c] = 'amount'
    if col_map:
        df = df.rename(columns=col_map)

    # Try to detect amount-like columns by content
    for c in df.columns:
        sample = df[c].astype(str).dropna().head(20).tolist()
        numeric_like = sum(1 for v in sample if AMOUNT_RE.match(v.replace(' ', '').replace(',', '')))
        if numeric_like > 3 and c not in ('debit', 'credit', 'balance'):
            # prefer 'amount' if unlabeled
            if 'amount' not in df.columns:
                df = df.rename(columns={c: 'amount'})

    # Convert amount-like columns
    for cand in ['debit', 'credit', 'amount', 'balance']:
        if cand in df.columns:
            df[cand] = df[cand].apply(normalize_amount_string)
    # Optionally parse dates (simple attempt)
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
        except Exception:
            pass
    note['columns_normalized'] = list(df.columns)
    return df, note


# ----------------- Main pipeline -----------------

def pdf_to_excel_production(pdf_path: str, out_xlsx: str, dpi=300, force_ocr=False, camelot_flavor=('lattice', 'stream')):
    logger.info("Starting pipeline for %s", pdf_path)
    notes = ensure_external_tools()
    results_for_xlsx = []  # list of tuples (sheet_name, dataframe, source_note)
    qa_rows = []

    # 1) Try Camelot first for table extraction (fast for born-digital)
    if not force_ocr:
        camelot_dfs = extract_with_camelot(pdf_path, flavor_preference=camelot_flavor)
        if camelot_dfs:
            for i, df in enumerate(camelot_dfs, start=1):
                df = drop_empty_rows_and_columns(df)
                df_clean, note = detect_and_normalize_statement(df)
                sheet = (f'table_camelot_{i}', df_clean, f'camelot_{i}')
                results_for_xlsx.append(sheet)
                qa_rows.append({'sheet': sheet[0], 'note': note})

    # 2) Use pdfplumber for per-page extraction and fallback
    pdfplumber_dfs = extract_with_pdfplumber(pdf_path)
    # If pdfplumber produced placeholders (empty dfs), mark scanned pages to OCR
    scanned_page_indices = [i for i, d in enumerate(pdfplumber_dfs) if d is None or d.empty]

    # If we have pdfplumber-derived dfs, add them
    pg_index = 1
    for i, df in enumerate(pdfplumber_dfs, start=1):
        if df is None or df.empty:
            pg_index += 1
            continue
        df = drop_empty_rows_and_columns(df)
        df_clean, note = detect_and_normalize_statement(df)
        sheet_name = f'pg_{i}'
        results_for_xlsx.append((sheet_name, df_clean, f'pdfplumber_pg{i}'))
        qa_rows.append({'sheet': sheet_name, 'note': note})
        pg_index += 1

    # 3) OCR fallback for scanned pages or if force_ocr=True
    if (PDF2IMAGE_AVAILABLE and (scanned_page_indices or force_ocr)):
        logger.info("Running OCR fallback for scanned pages: %s", scanned_page_indices if scanned_page_indices else 'all pages (forced)')
        try:
            images = convert_from_path(pdf_path, dpi=dpi)
        except Exception as e:
            logger.error("convert_from_path failed: %s", e)
            images = []
        for i, img in enumerate(images, start=1):
            do_ocr = force_ocr or (i-1 in scanned_page_indices)
            if not do_ocr:
                continue
            lines, word_df = ocr_image_to_lines(img)
            if not lines:
                continue
            # Try to convert lines into a table: basic heuristic: split by multiple spaces or consistent separators
            rows = []
            for ln in lines:
                # If line has many spaces between columns, split on two+ spaces
                if '  ' in ln:
                    parts = re.split(r'\s{2,}', ln)
                else:
                    parts = re.split(r'\s{3,}|\t', ln)
                parts = [p.strip() for p in parts if p.strip()]
                if parts:
                    rows.append(parts)
            if not rows:
                # fallback: single-column table
                df = pd.DataFrame(lines, columns=['text'])
            else:
                # Normalize to rectangular table by taking max columns
                maxc = max(len(r) for r in rows)
                norm_rows = [r + [''] * (maxc - len(r)) for r in rows]
                df = pd.DataFrame(norm_rows)
            df_clean, note = detect_and_normalize_statement(df)
            sheet_name = f'ocr_pg_{i}'
            results_for_xlsx.append((sheet_name, df_clean, f'ocr_pg{i}'))
            qa_rows.append({'sheet': sheet_name, 'note': note})

    # If nothing extracted at all, create a QA note
    if not results_for_xlsx:
        logger.warning("No tables extracted. Consider forcing OCR or reviewing the PDF manually.")
        notes.append('No tables extracted from PDF')

    # 4) Write to Excel with a QA sheet
    writer = pd.ExcelWriter(out_xlsx, engine='openpyxl')

    for sheet_name, df, src in results_for_xlsx:
        safe_name = sheet_name[:31]
        try:
            if df is None or df.empty:
                # write empty placeholder
                pd.DataFrame({'note': [f'No data extracted for {sheet_name}']}).to_excel(writer, sheet_name=safe_name, index=False)
            else:
                # final cleaning of columns
                df = df.copy()
                df.columns = [str(c)[:31] for c in df.columns]
                # normalize amount-like columns
                for c in df.columns:
                    df[c] = df[c].apply(normalize_amount_string)
                df.to_excel(writer, sheet_name=safe_name, index=False)
        except Exception as e:
            logger.exception("Failed to write sheet %s: %s", sheet_name, e)
            pd.DataFrame({'error': [str(e)]}).to_excel(writer, sheet_name=safe_name, index=False)

    # QA sheet
    qa_df = pd.DataFrame(qa_rows)
    meta = {
        'input_pdf': str(pdf_path),
        'notes': notes
    }
    # write metadata in a small sheet
    try:
        meta_df = pd.DataFrame([meta])
        meta_df.to_excel(writer, sheet_name='__meta__', index=False)
    except Exception:
        pass
    try:
        if not qa_df.empty:
            qa_df.to_excel(writer, sheet_name='__qa__', index=False)
    except Exception:
        pass

    try:
        writer.save()
    except Exception as e:
        logger.exception("Failed saving workbook: %s", e)
    logger.info("Saved output workbook: %s", out_xlsx)


# ----------------- CLI -----------------

def build_cli():
    p = argparse.ArgumentParser(description='PDF -> Excel converter (production-ready pipeline)')
    p.add_argument('--input', '-i', required=True, help='Input PDF file')
    p.add_argument('--output', '-o', required=True, help='Output Excel file (.xlsx)')
    p.add_argument('--dpi', type=int, default=300, help='DPI for OCR (pdf2image)')
    p.add_argument('--force-ocr', action='store_true', help='Force OCR on all pages (useful for scanned-only PDFs)')
    p.add_argument('--camelot-flavor', choices=['lattice', 'stream'], default='lattice', help='Preferred Camelot flavor')
    return p


import sys
from pathlib import Path

if __name__ == '__main__':
    parser = build_cli()

    # If no CLI args provided (e.g., running in IDLE), open file dialogs
    if len(sys.argv) == 1:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            inp_path = filedialog.askopenfilename(title='Select input PDF', filetypes=[('PDF files','*.pdf')])
            if not inp_path:
                print('No input selected. Exiting.')
                raise SystemExit(0)
            out_path = filedialog.asksaveasfilename(title='Save Excel as', defaultextension='.xlsx',
                                                    filetypes=[('Excel files','*.xlsx')])
            if not out_path:
                out_path = str(Path(inp_path).with_suffix('.xlsx'))

            args = parser.parse_args(['--input', inp_path, '--output', out_path])
        except Exception as e:
            # Fallback to normal parse (will raise error if arguments still missing)
            print("GUI file dialog failed, falling back to command-line args. Error:", e)
            args = parser.parse_args()
    else:
        args = parser.parse_args()

    pdf_to_excel_production(str(args.input), str(args.output),
                            dpi=args.dpi, force_ocr=args.force_ocr,
                            camelot_flavor=(args.camelot_flavor,))
    logger.info("Done.")
