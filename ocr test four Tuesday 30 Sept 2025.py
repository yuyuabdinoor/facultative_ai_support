import time
import datetime
from typing import Dict, Any
import ollama
import extract_msg
import json
import numpy as np
from pathlib import Path
import fitz  # PyMuPDF for PDF processing
import cv2
from paddleocr import PaddleOCR
import traceback
import pandas as pd
from docx import Document
from pptx import Presentation
import openpyxl
import tempfile
import os
from PIL import Image
import re
from config import OLLAMA_CONFIG, OCR_CONFIG, PROCESSING_CONFIG

#test ='mychen76/qwen3_cline_roocode:4b'
#embedding = 'qwen3-embedding:0.6b'

class OfficeDocumentProcessor:
    """Process Office documents (docx, pptx, xlsx, csv) with table extraction"""
    @staticmethod
    def extract_docx(file_path):
        """Extract text and tables from Word documents with table detection"""
        doc = Document(file_path)
        content = {
            'paragraphs': [],
            'tables': [],
            'metadata': {
                'total_paragraphs': 0,
                'total_tables': 0,
                'table_validation': []
            }
        }

        # Extract paragraphs
        for para in doc.paragraphs:
            if para.text.strip():
                content['paragraphs'].append(para.text)

        # Enhanced table extraction
        for table_idx, table in enumerate(doc.tables):
            table_data = {
                'table_number': table_idx + 1,
                'rows': len(table.rows),
                'columns': len(table.columns),
                'data': [],
                'dataframe': None,
                'validation': {
                    'has_header': False,
                    'empty_cells': 0,
                    'merged_cells': 0,
                    'data_quality_score': 0.0
                }
            }

            # Extract table data with validation
            total_cells = 0
            empty_cells = 0
            for row_idx, row in enumerate(table.rows):
                row_data = []
                for cell in row.cells:
                    cell_text = cell.text.strip()
                    total_cells += 1
                    if not cell_text:
                        empty_cells += 1
                    row_data.append(cell_text)
                table_data['data'].append(row_data)

            # Calculate data quality metrics
            table_data['validation']['empty_cells'] = empty_cells
            table_data['validation']['data_quality_score'] = (
                                                                         total_cells - empty_cells) / total_cells if total_cells > 0 else 0

            try:
                if len(table_data['data']) > 1:
                    # Check if first row looks like headers (contains mostly non-numeric text)
                    first_row = table_data['data'][0]
                    header_score = sum(
                        1 for cell in first_row if cell and not cell.replace('.', '').replace(',', '').isdigit()) / len(
                        first_row)

                    if header_score > 0.6:  # More than 60% non-numeric
                        table_data['validation']['has_header'] = True
                        df = pd.DataFrame(table_data['data'][1:], columns=first_row)
                    else:
                        df = pd.DataFrame(table_data['data'])
                else:
                    df = pd.DataFrame(table_data['data'])

                # Clean DataFrame
                df = df.dropna(how='all').dropna(axis=1, how='all')

                if not df.empty:
                    table_data['dataframe'] = df.to_dict('records')
                    table_data['dataframe_shape'] = df.shape
                    table_data['columns'] = df.columns.tolist()
                else:
                    table_data['dataframe'] = []
                    table_data['dataframe_shape'] = (0, 0)
                    table_data['columns'] = []

            except Exception as e:
                print(f"Could not convert table {table_idx + 1} to DataFrame: {e}")
                table_data['dataframe'] = []
                table_data['dataframe_shape'] = (0, 0)
                table_data['columns'] = []

            content['tables'].append(table_data)
            content['metadata']['table_validation'].append(table_data['validation'])

        content['metadata']['total_paragraphs'] = len(content['paragraphs'])
        content['metadata']['total_tables'] = len(content['tables'])

        return content

    @staticmethod
    def extract_pptx(file_path):
        """Extract text and tables from PowerPoint presentations"""
        prs = Presentation(file_path)
        content = {
            'slides': [],
            'metadata': {
                'total_slides': len(prs.slides),
                'total_tables': 0
            }
        }

        for slide_idx, slide in enumerate(prs.slides):
            slide_content = {
                'slide_number': slide_idx + 1,
                'title': '',
                'text': [],
                'tables': []
            }

            # Extract title
            if slide.shapes.title:
                slide_content['title'] = slide.shapes.title.text

            # Extract text and tables
            for shape in slide.shapes:
                # Text extraction
                if hasattr(shape, "text") and shape.text.strip():
                    slide_content['text'].append(shape.text)

                # Table extraction
                if shape.has_table:
                    table = shape.table
                    table_data = {
                        'table_number': len(slide_content['tables']) + 1,
                        'rows': len(table.rows),
                        'columns': len(table.columns),
                        'data': [],
                        'dataframe': None
                    }

                    # Extract table data
                    for row in table.rows:
                        row_data = [cell.text.strip() for cell in row.cells]
                        table_data['data'].append(row_data)

                    # Convert to DataFrame
                    try:
                        if len(table_data['data']) > 1:
                            df = pd.DataFrame(table_data['data'][1:], columns=table_data['data'][0])
                            table_data['dataframe'] = df.to_dict('records')
                            table_data['dataframe_shape'] = df.shape
                        else:
                            df = pd.DataFrame(table_data['data'])
                            table_data['dataframe'] = df.to_dict('records')
                            table_data['dataframe_shape'] = df.shape
                    except Exception as e:
                        print(f"Could not convert table to DataFrame: {e}")

                    slide_content['tables'].append(table_data)
                    content['metadata']['total_tables'] += 1

            content['slides'].append(slide_content)

        return content

    @staticmethod
    def extract_xlsx(file_path):
        """Excel extraction with sheet analysis"""
        wb = openpyxl.load_workbook(file_path, data_only=True)
        content = {
            'sheets': [],
            'metadata': {
                'total_sheets': len(wb.sheetnames),
                'sheet_names': wb.sheetnames,
                'sheet_analysis': []
            }
        }

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            sheet_data = {
                'sheet_name': sheet_name,
                'used_range': f"{ws.dimensions}",
                'data': [],
                'dataframe': None,
                'analysis': {
                    'has_data': False,
                    'data_density': 0.0,
                    'likely_headers': False
                }
            }

            # Extract all rows
            all_rows = []
            for row in ws.iter_rows(values_only=True):
                all_rows.append(list(row))

            sheet_data['data'] = all_rows

            # Analyze sheet content
            if all_rows:
                # Check if sheet has meaningful data
                non_empty_cells = sum(1 for row in all_rows for cell in row if cell is not None and str(cell).strip())
                total_cells = sum(len(row) for row in all_rows)
                sheet_data['analysis']['data_density'] = non_empty_cells / total_cells if total_cells > 0 else 0
                sheet_data['analysis']['has_data'] = sheet_data['analysis']['data_density'] > 0.1

                try:
                    if len(all_rows) > 1:
                        # Better header detection
                        first_row = all_rows[0]
                        header_indicators = sum(1 for cell in first_row
                                                if cell and isinstance(cell, str) and
                                                not str(cell).replace('.', '').replace(',', '').replace('-',
                                                                                                        '').isdigit())

                        if header_indicators > len(first_row) * 0.5:
                            sheet_data['analysis']['likely_headers'] = True
                            df = pd.DataFrame(all_rows[1:], columns=first_row)
                        else:
                            df = pd.DataFrame(all_rows)
                    else:
                        df = pd.DataFrame(all_rows)

                    # Clean DataFrame
                    df = df.dropna(how='all').dropna(axis=1, how='all')

                    if not df.empty:
                        sheet_data['dataframe'] = df.to_dict('records')
                        sheet_data['dataframe_shape'] = df.shape
                        sheet_data['columns'] = df.columns.tolist()
                    else:
                        sheet_data['dataframe'] = []
                        sheet_data['dataframe_shape'] = (0, 0)
                        sheet_data['columns'] = []

                except Exception as e:
                    print(f"Could not convert sheet '{sheet_name}' to DataFrame: {e}")
                    sheet_data['dataframe'] = []
                    sheet_data['dataframe_shape'] = (0, 0)
                    sheet_data['columns'] = []

            content['sheets'].append(sheet_data)
            content['metadata']['sheet_analysis'].append(sheet_data['analysis'])

        return content

    @staticmethod
    def extract_csv(file_path):
        content = {
            'filename': Path(file_path).name,
            'data': [],
            'dataframe': None,
            'metadata': {
                'encoding': 'unknown',
                'delimiter': 'unknown',
                'data_quality': 0.0
            }
        }

        # Try multiple encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip')
                content['metadata']['encoding'] = encoding

                # Detect delimiter
                with open(file_path, 'r', encoding=encoding) as f:
                    sample = f.read(1024)
                    delimiters = [',', ';', '\t', '|']
                    delimiter_counts = {delim: sample.count(delim) for delim in delimiters}
                    content['metadata']['delimiter'] = max(delimiter_counts, key=delimiter_counts.get)

                # Calculate data quality
                total_cells = df.size
                non_null_cells = df.count().sum()
                content['metadata']['data_quality'] = non_null_cells / total_cells if total_cells > 0 else 0

                content['dataframe'] = df.to_dict('records')
                content['dataframe_shape'] = df.shape
                content['columns'] = df.columns.tolist()
                content['metadata']['rows'] = len(df)
                content['metadata']['columns'] = len(df.columns)

                # Store raw data
                content['data'] = [df.columns.tolist()] + df.values.tolist()
                break

            except Exception as e:
                continue

        return content

class EmailProcessor:
    """Process .msg files with existing attachments in folder structure"""

    def __init__(self, subfolder_path):
        self.subfolder_path = Path(subfolder_path)
        self.msg_file = None
        self.msg_data = None

    def find_msg_file(self):
        """Find the .msg file in the subfolder"""
        msg_files = list(self.subfolder_path.glob('*.msg'))
        if msg_files:
            self.msg_file = msg_files[0]
            return self.msg_file
        return None

    def _safe_str(self, value):
        """Safely convert any value to string, handling bytes"""
        if value is None:
            return None
        if isinstance(value, bytes):
            try:
                return value.decode('utf-8', errors='ignore')
            except:
                return str(value)
        return str(value)

    def extract_email_data(self):
        """Extract structured data from .msg file"""
        if not self.msg_file:
            self.find_msg_file()

        if not self.msg_file:
            return None

        try:
            msg = extract_msg.Message(str(self.msg_file))

            self.msg_data = {
                'metadata': {
                    'subject': self._safe_str(msg.subject) if msg.subject else '',
                    'sender': self._safe_str(msg.sender) if msg.sender else '',
                    'sender_email': self._safe_str(getattr(msg, 'senderEmail', None)),
                    'date': self._safe_str(msg.date) if msg.date else None,
                    'received_time': self._safe_str(getattr(msg, 'receivedTime', None)) if hasattr(msg,
                                                                                                   'receivedTime') else None,
                    'message_id': self._safe_str(msg.messageId) if msg.messageId else '',
                    'importance': self._safe_str(msg.importance) if msg.importance else '',
                },
                'recipients': {
                    'to': self._format_recipients(msg.to),
                    'cc': self._format_recipients(msg.cc),
                    'bcc': self._format_recipients(msg.bcc)
                },
                'body': {
                    'plain_text': self._safe_str(msg.body) if msg.body else '',
                    'html': self._safe_str(msg.htmlBody) if msg.htmlBody else '',
                },
                'attachments_in_msg': self._get_msg_attachments(msg),
                'folder_path': str(self.subfolder_path),
                'msg_filename': self.msg_file.name
            }

            msg.close()
            return self.msg_data

        except Exception as e:
            print(f"Error extracting {self.msg_file}: {e}")
            traceback.print_exc()
            return None

    def _format_recipients(self, recipients):
        """Format recipient list - handles both string and object formats"""
        if not recipients:
            return []

        formatted = []

        if isinstance(recipients, str):
            return [{'name': self._safe_str(recipients), 'email': self._safe_str(recipients)}]

        if isinstance(recipients, list):
            for r in recipients:
                if isinstance(r, str):
                    formatted.append({'name': self._safe_str(r), 'email': self._safe_str(r)})
                elif hasattr(r, 'name') and hasattr(r, 'email'):
                    formatted.append({
                        'name': self._safe_str(r.name) if r.name else '',
                        'email': self._safe_str(r.email) if r.email else ''
                    })
                else:
                    formatted.append({'name': self._safe_str(r), 'email': self._safe_str(r)})

        return formatted

    def _get_msg_attachments(self, msg):
        """Get attachment info from .msg file - WITHOUT binary data"""
        attachments = []
        try:
            for att in msg.attachments:
                filename = att.longFilename or att.shortFilename or 'unknown'
                att_info = {
                    'filename': self._safe_str(filename),
                    'size': len(att.data) if att.data else 0,
                    # Don't include att.data itself - just metadata
                }
                attachments.append(att_info)
        except Exception as e:
            print(f"  Warning: Could not read attachments from msg: {e}")

        return attachments

    def list_folder_files(self):
        """List all files in the subfolder (saved attachments)"""
        all_files = []
        for file in self.subfolder_path.iterdir():
            if file.is_file() and not file.name.endswith('.msg'):
                all_files.append({
                    'filename': file.name,
                    'path': str(file),
                    'size': file.stat().st_size
                })
        return all_files

    def get_complete_data(self):
        """Get email data + list of existing attachment files"""
        email_data = self.extract_email_data()
        if not email_data:
            return None

        email_data['saved_attachments'] = self.list_folder_files()
        return email_data

    def process_attachments_with_ocr(self, email_data, ocr_engine, save_visuals=True):
        """Process attachments with OCR, office documents, and direct text extraction.
        - If save_visuals=True, saves annotated images and JSON results to:
          <email_folder>/text_detected_and_recognized/<TIMESTAMP>/<attachment>_pageN_annotated.png
          and corresponding <attachment>_pageN_ocr.json
        """
        attachment_texts = []
        office_processor = OfficeDocumentProcessor()

        # If saving visuals, create timestamped output root in the email folder
        out_root = None
        if save_visuals:
            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out_root = Path(self.subfolder_path) / "text_detected_and_recognized" / ts
            out_root.mkdir(parents=True, exist_ok=True)
            print(f"[OCR] Saving visuals to: {out_root}")

        for attachment in email_data.get('saved_attachments', []):
            file_path = attachment['path']
            ext = Path(file_path).suffix.lower()
            base_name = Path(file_path).stem

            if ext in ['.json']:
                continue

            start_time = time.time()

            # OFFICE DOCUMENTS - Process first before OCR
            if ext in ['.docx']:
                try:
                    content = office_processor.extract_docx(file_path)

                    # Extract all text for LLM context
                    text_parts = content['paragraphs'].copy()
                    for table in content['tables']:
                        table_text = f"\n[TABLE {table['table_number']} - {table['rows']}x{table['columns']}]\n"
                        for row in table['data']:
                            table_text += " | ".join(str(cell) for cell in row) + "\n"
                        text_parts.append(table_text)

                    full_text = "\n".join(text_parts)
                    attachment_texts.append({
                        'file': attachment['filename'],
                        'text': full_text,
                        'method': 'docx_extraction',
                        'structured_content': content,
                        'time': time.time() - start_time
                    })

                    # Save structured data
                    if save_visuals and out_root is not None:
                        json_path = out_root / f"{base_name}_docx_extracted.json"
                        with open(json_path, 'w', encoding='utf-8') as jf:
                            json.dump(content, jf, indent=2, ensure_ascii=False)

                except Exception as e:
                    print(f"[DOCX] Error processing {file_path}: {e}")

            elif ext in ['.pptx']:
                try:
                    content = office_processor.extract_pptx(file_path)

                    # Extract all text for LLM context
                    text_parts = []
                    for slide in content['slides']:
                        slide_text = f"\n[SLIDE {slide['slide_number']}: {slide['title']}]\n"
                        slide_text += "\n".join(slide['text'])

                        for table in slide['tables']:
                            table_text = f"\n[TABLE {table['table_number']} - {table['rows']}x{table['columns']}]\n"
                            for row in table['data']:
                                table_text += " | ".join(str(cell) for cell in row) + "\n"
                            slide_text += table_text

                        text_parts.append(slide_text)

                    full_text = "\n".join(text_parts)
                    attachment_texts.append({
                        'file': attachment['filename'],
                        'text': full_text,
                        'method': 'pptx_extraction',
                        'structured_content': content,
                        'time': time.time() - start_time
                    })

                    # Save structured data
                    if save_visuals and out_root is not None:
                        json_path = out_root / f"{base_name}_pptx_extracted.json"
                        with open(json_path, 'w', encoding='utf-8') as jf:
                            json.dump(content, jf, indent=2, ensure_ascii=False)

                except Exception as e:
                    print(f"[PPTX] Error processing {file_path}: {e}")

            elif ext in ['.xlsx']:
                try:
                    content = office_processor.extract_xlsx(file_path)

                    # Extract all sheets for LLM context
                    text_parts = []
                    for sheet in content['sheets']:
                        sheet_text = f"\n[SHEET: {sheet['sheet_name']} - {sheet['used_range']}]\n"
                        if sheet['dataframe']:
                            # Create table representation
                            for row in sheet['data'][:100]:  # Limit to first 100 rows
                                sheet_text += " | ".join(str(cell) if cell is not None else "" for cell in row) + "\n"
                        text_parts.append(sheet_text)

                    full_text = "\n".join(text_parts)
                    attachment_texts.append({
                        'file': attachment['filename'],
                        'text': full_text,
                        'method': 'xlsx_extraction',
                        'structured_content': content,
                        'time': time.time() - start_time
                    })

                    # Save structured data
                    if save_visuals and out_root is not None:
                        json_path = out_root / f"{base_name}_xlsx_extracted.json"
                        with open(json_path, 'w', encoding='utf-8') as jf:
                            json.dump(content, jf, indent=2, ensure_ascii=False)

                except Exception as e:
                    print(f"[XLSX] Error processing {file_path}: {e}")

            elif ext in ['.csv']:
                try:
                    content = office_processor.extract_csv(file_path)

                    # Create text representation
                    text_parts = [
                        f"[CSV FILE - {content['metadata'].get('rows', 0)} rows x {content['metadata'].get('columns', 0)} columns]\n"]
                    for row in content['data'][:100]:  # Limit to first 100 rows
                        text_parts.append(" | ".join(str(cell) if cell is not None else "" for cell in row))

                    full_text = "\n".join(text_parts)
                    attachment_texts.append({
                        'file': attachment['filename'],
                        'text': full_text,
                        'method': 'csv_extraction',
                        'structured_content': content,
                        'time': time.time() - start_time
                    })

                    # Save structured data
                    if save_visuals and out_root is not None:
                        json_path = out_root / f"{base_name}_csv_extracted.json"
                        with open(json_path, 'w', encoding='utf-8') as jf:
                            json.dump(content, jf, indent=2, ensure_ascii=False)

                except Exception as e:
                    print(f"[CSV] Error processing {file_path}: {e}")

            # PDF - OCR Processing
            elif ext in ['.pdf']:
                try:
                    pdf_start = time.time()
                    images, originals = ocr_engine.pdf_to_images(file_path)

                    for page_num, (processed, original) in enumerate(zip(images, originals)):
                        page_start = time.time()

                        # Save temp image for PaddleOCR
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                            img_pil = Image.fromarray(original)
                            img_pil.save(tmp.name)
                            tmp_path = tmp.name

                        # OCR with timing
                        ocr_start = time.time()
                        results = ocr_engine.ocr_with_detection_and_recognition(tmp_path)
                        ocr_time = time.time() - ocr_start

                        # SAVE FIRST - before any filtering
                        if save_visuals and out_root is not None and results:
                            save_start = time.time()
                            try:
                                # PaddleOCR returns a list - get first result
                                if isinstance(results, list) and len(results) > 0:
                                    result_obj = results[0]

                                    # Create page-specific subdirectory for this page's outputs
                                    page_dir = out_root / f"{base_name}_page{page_num + 1}"
                                    page_dir.mkdir(exist_ok=True)

                                    # save_to_img expects a DIRECTORY, not a filename
                                    result_obj.save_to_img(str(page_dir))

                                    # Save JSON with specific filename
                                    json_path = page_dir / "ocr_result.json"
                                    result_obj.save_to_json(str(json_path))

                                    save_time = time.time() - save_start
                                    print(f"  [PDF] Page {page_num + 1} - OCR: {ocr_time:.2f}s, Save: {save_time:.2f}s")
                                    print(f"        Saved to: {page_dir}")
                            except Exception as e:
                                print(f"  [PDF] Warning: Could not save page {page_num + 1}: {e}")
                                traceback.print_exc()

                        # NOW extract text for context - FIXED KEY NAMES
                        text_parts = []
                        for res in results:
                            # Access the JSON dict directly from the result object
                            if isinstance(res, dict):
                                result_data = res
                            else:
                                result_data = getattr(res, "json", None) or res

                            # CORRECT KEY: 'rec_texts' not 'rec_text'
                            rec_texts = result_data.get('rec_texts', []) or []
                            rec_scores = result_data.get('rec_scores', []) or []

                            if rec_scores and len(rec_scores) == len(rec_texts):
                                for t, s in zip(rec_texts, rec_scores):
                                    try:
                                        if float(s) >= 0.7 and str(t).strip():
                                            text_parts.append(str(t).strip())
                                    except Exception:
                                        continue
                            else:
                                for t in rec_texts:
                                    if isinstance(t, str) and len(t.strip()) >= 2:
                                        text_parts.append(t.strip())

                        full_text = ' '.join(text_parts).strip()

                        # Only skip ADDING to attachment_texts if no text
                        # But we already saved the visual outputs above
                        if full_text:
                            attachment_texts.append({
                                'file': attachment['filename'],
                                'page': page_num + 1,
                                'text': full_text,
                                'method': 'ocr',
                                'time': time.time() - page_start,
                                'ocr_time': ocr_time
                            })

                        # Cleanup temp file
                        os.unlink(tmp_path)

                    print(f"  [PDF] Total: {time.time() - pdf_start:.2f}s")

                except Exception as e:
                    print(f"[OCR] Error processing PDF {file_path}: {e}")
                    traceback.print_exc()

            # IMAGE FILES - FIXED (matches your current code structure)
            elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
                try:
                    img_start = time.time()

                    # OCR with timing
                    ocr_start = time.time()
                    results = ocr_engine.ocr_with_detection_and_recognition(file_path)
                    ocr_time = time.time() - ocr_start

                    # SAVE FIRST - before any text filtering
                    if save_visuals and out_root is not None and results:
                        save_start = time.time()
                        try:
                            # Handle results structure - should be a list
                            if isinstance(results, list) and len(results) > 0:
                                result_obj = results[0]
                            else:
                                result_obj = results

                            # Create page-specific subdirectory for this page's outputs
                            page_dir = out_root / f"{base_name}_page{page_num + 1}"
                            page_dir.mkdir(exist_ok=True)

                            # save_to_img expects a DIRECTORY, not a filename
                            result_obj.save_to_img(str(page_dir))

                            # Save JSON with specific filename
                            json_path = page_dir / "ocr_result.json"
                            result_obj.save_to_json(str(json_path))

                            save_time = time.time() - save_start
                            print(f"  [PDF] Page {page_num + 1} - OCR: {ocr_time:.2f}s, Save: {save_time:.2f}s")
                            print(f"        Saved to: {page_dir}")
                        except Exception as e:
                            print(f"  [IMG] Warning: Could not save visuals: {e}")
                            traceback.print_exc()

                    # NOW extract text from result objects - FIXED KEY NAMES
                    text_parts = []
                    for res in results:
                        # Access the JSON dict directly from the result object
                        if isinstance(res, dict):
                            result_data = res
                        else:
                            result_data = getattr(res, "json", None) or res


                        rec_texts = result_data.get('rec_texts', []) or []
                        rec_scores = result_data.get('rec_scores', []) or []

                        if rec_scores and len(rec_scores) == len(rec_texts):
                            for t, s in zip(rec_texts, rec_scores):
                                try:
                                    if float(s) >= 0.7 and str(t).strip():
                                        text_parts.append(str(t).strip())
                                except Exception:
                                    continue
                        else:
                            for t in rec_texts:
                                if isinstance(t, str) and len(t.strip()) >= 2:
                                    text_parts.append(t.strip())

                    full_text = ' '.join(text_parts).strip()

                    # Only add to attachment_texts if there's filtered text
                    # But we already saved visuals above regardless
                    if full_text:
                        attachment_texts.append({
                            'file': attachment['filename'],
                            'text': full_text,
                            'method': 'ocr',
                            'time': time.time() - img_start,
                            'ocr_time': ocr_time
                        })

                except Exception as e:
                    print(f"[OCR] Error processing image {file_path}: {e}")
                    traceback.print_exc()

            # TEXT FILES - Direct extraction
            elif ext in ['.txt', '.log']:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                attachment_texts.append({
                    'file': attachment['filename'],
                    'text': text,
                    'method': 'direct',
                    'time': time.time() - start_time
                })

                # optionally save a JSON copy of the raw text to visuals folder
                if save_visuals and out_root is not None:
                    json_name = f"{base_name}_raw_text.json"
                    json_path = out_root / json_name
                    try:
                        with open(json_path, 'w', encoding='utf-8') as jf:
                            json.dump({"file": attachment['filename'], "text": text}, jf, indent=2, ensure_ascii=False)
                    except Exception as e:
                        print(f"[OCR] Warning: failed to save raw text json {json_path}: {e}")

            # UNKNOWN - Try OCR as fallback
            else:
                try:
                    ocr_result = ocr_engine.ocr_with_detection_and_recognition(file_path)
                    text = ' '.join([t for _, (t, _) in ocr_result.results])
                    attachment_texts.append({
                        'file': attachment['filename'],
                        'text': text,
                        'method': 'ocr',
                        'time': time.time() - start_time
                    })

                    if save_visuals and out_root is not None:
                        png_name = f"{base_name}_annotated.png"
                        json_name = f"{base_name}_ocr.json"
                        png_path = out_root / png_name
                        json_path = out_root / json_name
                        try:
                            ocr_result.save_to_img(str(png_path))
                        except Exception as e:
                            print(f"[OCR] Warning: failed to save image {png_path}: {e}")
                        try:
                            ocr_result.save_to_json(str(json_path))
                        except Exception as e:
                            print(f"[OCR] Warning: failed to save json {json_path}: {e}")
                except Exception as e:
                    print(f"[OCR] Skipping unknown file type {file_path}: {e}")

        return attachment_texts


class LocalLLMProcessor:
    """Process LLM requests using local Ollama with Qwen 3 4B"""

    def __init__(self, model_name=None):
        self.model_name = model_name or OLLAMA_CONFIG["model_name"]

    def test_connection(self):
        """Test if local Ollama is running and model is available"""
        try:
            models = ollama.list()

            # ListResponse object has a 'models' attribute
            model_list = models.models if hasattr(models, 'models') else []

            # Extract model names
            model_names = []
            for m in model_list:
                if hasattr(m, 'model'):
                    model_names.append(m.model)
                elif hasattr(m, 'name'):
                    model_names.append(m.name)

            # Check if target model exists
            if any(self.model_name in name for name in model_names):
                print(f"Local model {self.model_name} is available")
                return True
            else:
                print(f"Model {self.model_name} not found.")
                print(f"   Available models: {model_names}")

                # Try to actually use the model anyway - maybe it's there but list failed
                try:
                    print(f"   Attempting to use model anyway...")
                    result = ollama.generate(
                        model=self.model_name,
                        prompt="test",
                        options={'num_predict': 1}
                    )
                    print(f"Model {self.model_name} is actually available!")
                    return True
                except:
                    print(f"   To pull the model, run: ollama pull {self.model_name}")
                    return False

        except Exception as e:
            print(f"Cannot connect to local Ollama: {e}")
            traceback.print_exc()
            return False

    def generate_response(self, prompt: str, max_tokens: int = 4000, temperature: float = 0.1) -> Dict[str, Any]:
        """
        Generate response from local Ollama model using ollama library

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)

        Returns:
            Dictionary with response data and metadata
        """
        try:
            start_time = time.time()

            result = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_p": 0.9,
                    "top_k": 40
                }
            )

            generation_time = time.time() - start_time

            return {
                "success": True,
                "response": result.get("response", ""),
                "model": result.get("model", self.model_name),
                "generation_time": generation_time,
                "tokens_generated": len(result.get("response", "").split()),
                "metadata": {
                    "prompt_length": len(prompt),
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "generation_time": time.time() - start_time if 'start_time' in locals() else 0
            }

    def extract_structured_data(self, llm_prompt: str) -> Dict[str, Any]:
        """
        Extract structured reinsurance data using the local LLM

        Args:
            llm_prompt: The formatted prompt with email and attachment data

        Returns:
            Dictionary with extracted data and processing metadata
        """
        print("🤖 Sending request to local Ollama...")

        # Generate response
        result = self.generate_response(
            llm_prompt,
            max_tokens=OLLAMA_CONFIG["max_tokens"],
            temperature=OLLAMA_CONFIG["temperature"]
        )

        if not result["success"]:
            return {
                "success": False,
                "error": result["error"],
                "raw_response": None,
                "extracted_data": None
            }

        # Try to parse JSON response
        raw_response = result["response"].strip()

        # Clean up response - remove any markdown formatting
        if raw_response.startswith("```json"):
            raw_response = raw_response[7:]
        if raw_response.endswith("```"):
            raw_response = raw_response[:-3]

        try:
            extracted_data = json.loads(raw_response)
            return {
                "success": True,
                "raw_response": raw_response,
                "extracted_data": extracted_data,
                "generation_time": result["generation_time"],
                "tokens_generated": result["tokens_generated"],
                "model_used": result["model"]
            }
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parsing failed: {e}")
            print(f"Raw response: {raw_response[:500]}...")

            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                try:
                    extracted_data = json.loads(json_match.group())
                    return {
                        "success": True,
                        "raw_response": raw_response,
                        "extracted_data": extracted_data,
                        "generation_time": result["generation_time"],
                        "tokens_generated": result["tokens_generated"],
                        "model_used": result["model"],
                        "warning": "JSON extracted from response with regex"
                    }
                except json.JSONDecodeError:
                    pass

            return {
                "success": False,
                "error": f"JSON parsing failed: {e}",
                "raw_response": raw_response,
                "extracted_data": None,
                "generation_time": result["generation_time"]
            }


def extract_candidates_from_text(text):
    """Return a small dict of candidate values useful as LLM hints."""
    cand = {}

    if not text:
        return cand

    # INSURED - look for common prefixes
    m = re.search(r'(?:Name of the Original Insured|Insured|Assured|Named Insured)[:\s\-]*([A-Z0-9&\-\.,\(\) ]{3,120})', text, re.I)
    if m:
        cand['insured'] = m.group(1).strip()

    # CEDANT
    m = re.search(r'(?:Cedant|Cedant Name|Ceding Company|Insurer)[:\s\-]*([A-Z0-9&\-\.,\(\) ]{3,120})', text, re.I)
    if m:
        cand['cedant'] = m.group(1).strip()

    # TSI / Total Sum Insured
    m = re.search(r'(?:Total Sum Insured|Total \(QAR\)|SUM INSURED|TSI)[^\n\r]{0,120}([\$\€\£A-Z]{0,4}\s*[\d,\.]+(?:\s*\(approx\))?)', text, re.I)
    if m:
        raw = m.group(1).strip()
        cand['total_sum_insured'] = raw
        # numeric parse
        num = re.sub(r'[^\d\.\-]', '', raw)
        try:
            cand['total_sum_insured_float'] = float(num) if num else "TBD"
        except Exception:
            cand['total_sum_insured_float'] = "TBD"

    # Currency attempt from near TSI or standalone currency code
    m = re.search(r'\b(USD|EUR|GBP|KES|QAR|AED|USD\$|US\$)\b', text, re.I)
    if m:
        cand['currency'] = m.group(1).upper().replace('US$', 'USD').replace('USD$', 'USD')

    # PERIOD
    m = re.search(r'(?:Period of Insurance|Period)[:\s\-]*(From\s+[^\n\r]+?\s+to\s+[^\n\r]+|[\d]{1,2}[\/\-][\d]{1,2}[\/\-][\d]{2,4}\s*to\s*[\d]{1,2}[\/\-][\d]{1,2}[\/\-][\d]{2,4})', text, re.I)
    if m:
        cand['period_of_insurance'] = m.group(1).strip()

    # Country
    m = re.search(r'(?:Risk Location|Country|Territorial Limit|State of)[:\s\-]*([A-Za-z \-]{2,60})', text, re.I)
    if m:
        cand['country'] = m.group(1).strip()

    # Retention percentage
    m = re.search(r'(?:Cedant’s retention|Cedant retention|Retention of Cedant|Cedant’s retention in %|Cedant retention in %)[^\d]{0,10}([\d]{1,3}\s*%?)', text, re.I)
    if m:
        cand['retention_of_cedant'] = m.group(1).strip()

    # Share offered
    m = re.search(r'(?:Share Offered|Offered Share|Share)[:\s\-]*([\d]{1,3}\s*%?)', text, re.I)
    if m:
        cand['share_offered'] = m.group(1).strip()

    return cand

MASTER_PROMPT = """INSTRUCTIONS FOR STRUCTURED EXTRACTION (READ CAREFULLY)

You are given the text of a single email and any OCR-extracted text from its attachments (only regions with confidence ≥ 0.7). Use this context to extract reinsurance information and output it as a JSON object.

Fields to extract

insured: Name of the insured party (string).

cedant: Name of the ceding insurer or company (string).

broker: Name of the broker or intermediary (string).

occupation_of_insured: Occupation or role of the insured (string).

main_activities: Main business activities (string).

perils_covered: Covered perils or risks (string).

geographical_limit: Geographical territory or limits (string).

situation_of_risk: Location or situation of the risk (string).

total_sum_insured: Total insured sum (numeric part only, as string).

total_sum_insured_float: Total insured sum as a number (float).

currency: Currency of the insured sum (string).

period_of_insurance: Insurance coverage period (dates or description, string).

excess_deductible: Excess or deductible (string; use "TBD" if unknown).

retention_of_cedant: Cedant's retention (string; use "TBD" if unknown).

possible_maximum_loss: Possible maximum loss (string; use "TBD" if unknown).

cat_exposure: Catastrophe exposure (string; use "TBD" if unknown).

claims_experience: Claims experience details (string; use "TBD" if unknown).

reinsurance_deductions: Reinsurance deductions (string; use "TBD" if unknown).

share_offered: Share offered (string; use "TBD" if unknown).

inward_acceptances: Inward acceptances (string; use "TBD" if unknown).

risk_surveyor_report: Risk surveyor’s report (string; use "TBD" if unknown).

premium_rates: Premium rates (string; use "TBD" if unknown).

premium: Premium amount (string; use "TBD" if unknown).

climate_change_risk_factors: Climate change risk factors (string; use "TBD" if unknown).

esg_risk_assessment: ESG risk assessment (string; use "TBD" if unknown).

country: Country of the risk or insured (string).

Extraction Instructions

Output Format: Provide exactly one valid JSON object with all the fields above. Do not include any additional keys or text. Do not output markdown or code fences—only raw JSON.

Missing Values: If a field’s value is not present or is unclear in the text, set it to "TBD" (the string).

Value Types: All values should be strings except for total_sum_insured_float, which must be a numeric value (no quotes). For total_sum_insured, strip any formatting so it’s a plain number as a string. Use a standard currency code or symbol for currency.

Semantic Mapping: The text may use different terms or synonyms. Use context to map them to the correct fields. For example:

Words like “Insured”, “Assured”, or the policyholder’s name → insured.

“Cedant”, “Ceding insurer”, or similar → cedant.

“Broker” or “Intermediary” → broker.

Descriptions of the insured’s job or role → occupation_of_insured.

Descriptions of business or operations → main_activities.

Phrases like “covered perils”, “risks” → perils_covered.

Territory or country names → geographical_limit or country as appropriate.

Locations or site descriptions → situation_of_risk.

Amounts with currency (e.g. “USD 100000”) → total_sum_insured (e.g. "100000"), currency (e.g. "USD"), total_sum_insured_float (e.g. 100000.0).

Date ranges or terms like “from”/“to” → period_of_insurance.

Words like “Excess”, “Deductible” → excess_deductible.

“Retention” → retention_of_cedant.

“Maximum loss”, “PML” → possible_maximum_loss.

“Cat exposure” or “Catastrophe” → cat_exposure.

“Claims experience” → claims_experience.

“Reinsurance deductions” → reinsurance_deductions.

“Share offered” → share_offered.

“Inward acceptances” → inward_acceptances.

“Surveyor report” → risk_surveyor_report or survey report, the report itself might be one of the attachments so look for the file name; risk_surveyor_report or survey report or risk report .

“Premium rate” → premium_rates.

“Premium” (amount) → premium.

“Climate risk factors” → climate_change_risk_factors.

“ESG” or “sustainability” terms → esg_risk_assessment.

No Extra Text: The model should only output the JSON object. It must not output any explanatory text or additional formatting. Ensure the JSON is syntactically correct and exactly matches the fields above.

---TEXT BLOCK STARTS BELOW---
{LLM_INPUT}
---TEXT BLOCK ENDS ABOVE---
"""

def _truncate_keep_ends(text, max_len=PROCESSING_CONFIG['max_text_length']):
    """If text > max_len, keep start and end with a truncation marker in the middle."""
    if not text or len(text) <= max_len:
        return text
    head = text[: int(max_len * 0.7)]
    tail = text[-int(max_len * 0.3):]
    return head + "\n\n...[TRUNCATED]...\n\n" + tail

def create_llm_context(email_data, attachment_texts):
    """
    Build LLM context and fill the MASTER_PROMPT with the combined text.
    - Skips attachments with empty text.
    - Returns (context_dict, llm_prompt_string).
    """

    # keep only attachments that have non-empty text (already filtered by OCR threshold earlier)
    kept_attachments = []
    for att in attachment_texts:
        txt = att.get('text', '')
        if isinstance(txt, str) and txt.strip():
            kept_attachments.append(att)

    # Build context (same shape as before but with kept attachments)
    context = {
        'email': {
            'subject': email_data['metadata'].get('subject', ''),
            'from': email_data['metadata'].get('sender', ''),
            'date': email_data['metadata'].get('date', ''),
            'body': email_data['body'].get('plain_text', '') if 'body' in email_data else ''
        },
        'attachments': kept_attachments,
        'total_attachments': len(kept_attachments),
        'processing_times': {
            'email': 0,
            'attachments': sum(a.get('time', 0) for a in kept_attachments)
        }
    }

    # Build combined human-readable text
    parts = []
    parts.append(f"Email Subject: {context['email']['subject']}")
    parts.append(f"From: {context['email']['from']}")
    parts.append(f"Date: {context['email']['date']}")
    parts.append("\nEmail Body:\n" + (context['email']['body'].strip() or ""))

    if kept_attachments:
        parts.append("\nAttachments included for extraction:\n")
        for i, att in enumerate(kept_attachments, start=1):
            header = f"--- Attachment {i}: {att['file']} ---"
            page_info = f" (page {att.get('page')})" if att.get('page') else ""
            body_text = att.get('text', '').strip()
            # if extremely long, keep the head and tail
            max_att_len = PROCESSING_CONFIG['max_attachment_length']
            if len(body_text) > max_att_len:
                body_text = body_text[:int(max_att_len*0.7)] + "\n\n...[ATTACHMENT TRUNCATED]...\n\n" + body_text[-int(max_att_len*0.3):]
            parts.append(header + page_info + "\n" + body_text)

    combined_text = "\n\n".join(parts).strip()

    # Add candidate hints from combined_text and email body to help mapping
    try:
        candidates = extract_candidates_from_text(combined_text + "\n" + context['email']['body'])
    except Exception:
        candidates = {}

    if candidates:
        hint_lines = ["\nCandidate Hints (auto-extracted):"]
        for k, v in candidates.items():
            hint_lines.append(f"{k}: {v}")
        combined_text += "\n\n" + "\n".join(hint_lines)

    # Truncate overall combined_text to avoid overlong input
    combined_text = _truncate_keep_ends(combined_text, max_len=11000)

    # Fill the MASTER_PROMPT with the combined text
    llm_prompt = MASTER_PROMPT.replace("{LLM_INPUT}", combined_text)

    # Return both the structured context and the full prompt to send to the LLM
    return context, llm_prompt

def search_emails(all_results, search_term, field='subject'):
    """Search through extracted email data"""
    matches = []
    for folder_name, data in all_results.items():
        if field in data['metadata']:
            if search_term.lower() in str(data['metadata'][field]).lower():
                matches.append({
                    'folder': folder_name,
                    'subject': data['metadata']['subject'],
                    'sender': data['metadata']['sender'],
                    'date': data['metadata']['date']
                })
    return matches


def get_email_body_by_subject(all_results, subject_keyword):
    """Get full email body by searching subject"""
    for folder_name, data in all_results.items():
        if subject_keyword.lower() in data['metadata']['subject'].lower():
            return {
                'folder': folder_name,
                'subject': data['metadata']['subject'],
                'body_text': data['body']['plain_text'],
                'body_html': data['body']['html'],
                'attachments': data['saved_attachments']
            }
    return None


def print_email_summary(results):
    """Print a nice summary of all emails"""
    print("\n" + "=" * 80)
    print(f"EMAIL SUMMARY - Total: {len(results)} emails")
    print("=" * 80)

    for i, (folder, data) in enumerate(results.items(), 1):
        print(f"\n[{i}] {folder}")
        print(f"    Subject: {data['metadata']['subject']}")
        print(f"    From: {data['metadata']['sender']}")
        print(f"    Date: {data['metadata']['date']}")
        print(f"    Attachments: {len(data['saved_attachments'])} files")
        if data['saved_attachments']:
            for att in data['saved_attachments'][:3]:
                print(f"      - {att['filename']}")
            if len(data['saved_attachments']) > 3:
                print(f"      ... and {len(data['saved_attachments']) - 3} more")

class Detect_and_Recognize:
    def __init__(self):
        """
                Initialize PaddleOCR with custom parameters
                Args:
                    device: 'cpu' or 'gpu' or 'gpu:0,1' for multiple GPUs
                    enable_mkldnn: Enable MKL-DNN acceleration on CPU
                    cpu_threads: Number of CPU threads
                    det_limit_side_len: Max/min side length for detection
                    det_limit_type: 'max' or 'min' - how to apply side length limit
                    det_thresh: Detection threshold for text pixels
                    det_box_thresh: Threshold for text region boxes
                    det_unclip_ratio: Box expansion ratio
                    rec_batch_size: Batch size for recognition
                    use_doc_orientation: Enable document orientation classification
                    use_doc_unwarping: Enable document unwarping
                    use_textline_orientation: Enable text line orientation
        """
        self.ocr = PaddleOCR(
            # doc_orientation_classify_model_name='PP-LCNet_x1_0_doc_ori',
            # doc_orientation_classify_model_dir='PP-LCNet_x1_0_doc_ori',
            # textline_orientation_model_name='PP-LCNet_x1_0_textline_ori',
            # textline_orientation_model_dir='PP-LCNet_x1_0_textline_ori',
            text_detection_model_name='PP-OCRv5_mobile_det',
            text_recognition_model_name='PP-OCRv5_mobile_rec',
            text_detection_model_dir="PP-OCRv5_mobile_det_infer",
            text_recognition_model_dir="PP-OCRv5_mobile_rec_infer",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=OCR_CONFIG['device'],
            cpu_threads=OCR_CONFIG['cpu_threads'],
            enable_mkldnn=OCR_CONFIG['enable_mkldnn'],
            text_det_limit_side_len=OCR_CONFIG['det_limit_side_len'],
            text_det_limit_type='max',
            text_det_thresh=OCR_CONFIG['text_det_thresh'],
            text_det_box_thresh=OCR_CONFIG['text_det_box_thresh'],
            text_det_unclip_ratio=1.5,
            text_recognition_batch_size=OCR_CONFIG['text_recognition_batch_size'],
            # ocr_version="PP-OCRv5",
        )

    def ocr_with_detection_and_recognition(self, input_path):
        """
        Perform OCR with both detection and recognition

        Args:
            input_path: Path to image, PDF, or directory
            batch_size: Override default batch size

        Returns:
            PaddleOCR result object(s) with detection boxes and recognized text
        """

        return self.ocr.predict(input_path)

    def pdf_to_images(self, pdf_path, dpi=300):
        """
        Convert PDF pages to images for OCR processing

        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for rendering (higher = better quality, slower)

        Returns:
            tuple: (processed_images, original_images)
                - processed_images: List of numpy arrays ready for OCR
                - original_images: List of original numpy arrays
        """
        doc = fitz.open(pdf_path)
        processed_images = []
        original_images = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            # Render at specified DPI
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)

            # Convert to numpy array
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )

            # Convert RGBA to RGB if needed
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

            original_images.append(img.copy())
            processed_images.append(img)

        doc.close()
        return processed_images, original_images

def check_ollama_installed():
    """Check if Ollama service is responding"""
    try:
        models = ollama.list()

        # ListResponse object has a 'models' attribute
        if hasattr(models, 'models'):
            model_list = models.models
        else:
            model_list = []

        # Extract model names
        model_names = []
        for m in model_list:
            if hasattr(m, 'model'):
                model_names.append(m.model)
            elif hasattr(m, 'name'):
                model_names.append(m.name)

        print(f"Ollama service is running")
        print(f"   Available models: {model_names if model_names else 'None found'}")
        return True
    except Exception as e:
        print(f"Cannot connect to Ollama service: {e}")
        print("   Make sure Ollama is running")
        return False

def save_processing_results(folder_path: Path, context: Dict, llm_prompt: str, llm_result: Dict):
    """Save all processing results to JSON files with proper timing"""

    timestamp = datetime.datetime.now().isoformat()

    # Save LLM context
    context_path = folder_path / 'llm_context.json'
    context_data = {
        "context": context,
        "timestamp": timestamp,
        "metadata": {
            "email_subject": context.get('email', {}).get('subject', ''),
            "total_attachments": context.get('total_attachments', 0),
            "processing_times": context.get('processing_times', {})
        }
    }
    with open(context_path, 'w', encoding='utf-8') as f:
        json.dump(context_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved LLM context: {context_path.name}")

    # Save master prompt with LLM context
    prompt_path = folder_path / 'master_prompt.json'
    prompt_data = {
        "master_prompt": llm_prompt,
        "llm_context": context,
        "timestamp": timestamp,
        "prompt_length": len(llm_prompt),
        "context_summary": {
            "email_subject": context.get('email', {}).get('subject', ''),
            "total_attachments": context.get('total_attachments', 0),
            "processing_times": context.get('processing_times', {})
        }
    }
    with open(prompt_path, 'w', encoding='utf-8') as f:
        json.dump(prompt_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved master prompt with LLM context: {prompt_path.name}")

    # Save LLM response with timing
    if llm_result["success"]:
        result_path = folder_path / 'llm_response.json'
        result_data = {
            "success": True,
            "extracted_data": llm_result["extracted_data"],
            "raw_response": llm_result["raw_response"],
            "timing": {
                "generation_time_seconds": llm_result["generation_time"],
                "tokens_generated": llm_result["tokens_generated"],
                "timestamp": timestamp
            },
            "model_info": {
                "model_used": llm_result["model_used"],
                "prompt_length": llm_result.get("metadata", {}).get("prompt_length", 0),
                "temperature": llm_result.get("metadata", {}).get("temperature", 0.1),
                "max_tokens": llm_result.get("metadata", {}).get("max_tokens", 2000)
            }
        }
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved LLM response with timing: {result_path.name}")

        # Also save a clean version with just the extracted data
        clean_path = folder_path / 'extracted_reinsurance_data.json'
        with open(clean_path, 'w', encoding='utf-8') as f:
            json.dump(llm_result["extracted_data"], f, indent=2, ensure_ascii=False)
        print(f"✓ Saved clean extraction: {clean_path.name}")

    else:
        error_path = folder_path / 'llm_error.json'
        error_data = {
            "success": False,
            "error": llm_result["error"],
            "raw_response": llm_result.get("raw_response", ""),
            "timing": {
                "generation_time_seconds": llm_result.get("generation_time", 0),
                "timestamp": timestamp
            }
        }
        with open(error_path, 'w', encoding='utf-8') as f:
            json.dump(error_data, f, indent=2, ensure_ascii=False)
        print(f"✗ Saved error with timing: {error_path.name}")


def process_with_llm_integration(root_folder: str, model_name: str = None):
    """
    Complete pipeline: Email processing + OCR + LLM extraction

    Args:
        root_folder: Path to folder containing email subfolders
        model_name: Local Ollama model name to use
    """
    check_ollama_installed()
    # Initialize components
    print("🚀 Initializing components...")

    # Test local Ollama connection
    llm_processor = LocalLLMProcessor(model_name)
    if not llm_processor.test_connection():
        print("❌ Cannot proceed without local Ollama connection")
        return

    print("Initializing OCR engine...")
    ocr_engine = Detect_and_Recognize()

    root_path = Path(root_folder)
    subfolders = [f for f in root_path.iterdir() if f.is_dir()]

    print(f"📁 Found {len(subfolders)} email folders to process")

    for i, subfolder in enumerate(sorted(subfolders), 1):
        print(f"\n{'=' * 60}")
        print(f"📧 Processing folder {i}/{len(subfolders)}: {subfolder.name}")
        print(f"{'=' * 60}")

        try:
            # Step 1: Extract email data
            print("📨 Extracting email data...")
            processor = EmailProcessor(subfolder)
            email_data = processor.get_complete_data()

            if not email_data:
                print("❌ No email data found")
                continue

            print(f"✓ Email: {email_data['metadata']['subject']}")
            print(f"✓ Attachments: {len(email_data['saved_attachments'])}")

            print("📄 Processing attachments...")
            attachment_texts = processor.process_attachments_with_ocr(
                email_data, ocr_engine, save_visuals=PROCESSING_CONFIG['save_visuals']
            )

            # Step 3: Create LLM context
            print("🧠 Creating LLM context...")
            context, llm_prompt = create_llm_context(email_data, attachment_texts)

            # Step 4: LLM extraction
            print("🤖 Running LLM extraction...")
            x = time.time()
            llm_result = llm_processor.extract_structured_data(llm_prompt)
            print(f'Response generation took; {time.time() - x} seconds')
            # Step 5: Save results
            print("💾 Saving results...")
            save_processing_results(subfolder, context, llm_prompt, llm_result)

            if llm_result["success"]:
                print(
                    f"✅ Success! Generated {llm_result['tokens_generated']} tokens in {llm_result['generation_time']:.2f}s")
                # Print summary of extracted data
                extracted = llm_result["extracted_data"]
                print(
                    f"📊 Extracted: {extracted.get('insured', 'N/A')} | {extracted.get('cedant', 'N/A')} | {extracted.get('total_sum_insured', 'N/A')}")
            else:
                print(f"❌ LLM extraction failed: {llm_result['error']}")

        except Exception as e:
            print(f"❌ Error processing {subfolder.name}: {e}")
            traceback.print_exc()

    print(f"\n🎉 Processing complete!")


if __name__ == "__main__":
    root_folder = "tuesday test data"
    model_name = OLLAMA_CONFIG['model_name']
    process_with_llm_integration(root_folder, model_name)







