import time
import datetime
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
from openpyxl.utils import get_column_letter
import tempfile
import os
from PIL import Image

class OfficeDocumentProcessor:
    """Process Office documents (docx, pptx, xlsx, csv) with table extraction"""

    @staticmethod
    def extract_docx(file_path):
        """Extract text and tables from Word documents"""
        doc = Document(file_path)
        content = {
            'paragraphs': [],
            'tables': [],
            'metadata': {
                'total_paragraphs': 0,
                'total_tables': 0
            }
        }

        # Extract paragraphs
        for para in doc.paragraphs:
            if para.text.strip():
                content['paragraphs'].append(para.text)

        # Extract tables with proper column/row alignment
        for table_idx, table in enumerate(doc.tables):
            table_data = {
                'table_number': table_idx + 1,
                'rows': len(table.rows),
                'columns': len(table.columns),
                'data': [],
                'dataframe': None
            }

            # Extract table data
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells]
                table_data['data'].append(row_data)

            # Convert to DataFrame if possible
            try:
                if len(table_data['data']) > 1:
                    # Assume first row is header
                    df = pd.DataFrame(table_data['data'][1:], columns=table_data['data'][0])
                    table_data['dataframe'] = df.to_dict('records')
                    table_data['dataframe_shape'] = df.shape
                else:
                    df = pd.DataFrame(table_data['data'])
                    table_data['dataframe'] = df.to_dict('records')
                    table_data['dataframe_shape'] = df.shape
            except Exception as e:
                print(f"Could not convert table {table_idx + 1} to DataFrame: {e}")

            content['tables'].append(table_data)

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
        """Extract data from Excel files with sheet and table info"""
        wb = openpyxl.load_workbook(file_path, data_only=True)
        content = {
            'sheets': [],
            'metadata': {
                'total_sheets': len(wb.sheetnames),
                'sheet_names': wb.sheetnames
            }
        }

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            sheet_data = {
                'sheet_name': sheet_name,
                'used_range': f"{ws.dimensions}",
                'data': [],
                'dataframe': None
            }

            # Extract all rows
            for row in ws.iter_rows(values_only=True):
                sheet_data['data'].append(list(row))

            # Convert to DataFrame
            try:
                if len(sheet_data['data']) > 1:
                    # Try to use first row as headers
                    df = pd.DataFrame(sheet_data['data'][1:], columns=sheet_data['data'][0])
                else:
                    df = pd.DataFrame(sheet_data['data'])

                # Remove completely empty rows/columns
                df = df.dropna(how='all').dropna(axis=1, how='all')

                sheet_data['dataframe'] = df.to_dict('records')
                sheet_data['dataframe_shape'] = df.shape
                sheet_data['columns'] = df.columns.tolist()
            except Exception as e:
                print(f"Could not convert sheet '{sheet_name}' to DataFrame: {e}")

            content['sheets'].append(sheet_data)

        return content

    @staticmethod
    def extract_csv(file_path):
        """Extract data from CSV files"""
        content = {
            'filename': Path(file_path).name,
            'data': [],
            'dataframe': None,
            'metadata': {}
        }

        try:
            # Try reading with pandas (handles various encodings and delimiters)
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')

            content['dataframe'] = df.to_dict('records')
            content['dataframe_shape'] = df.shape
            content['columns'] = df.columns.tolist()
            content['metadata']['rows'] = len(df)
            content['metadata']['columns'] = len(df.columns)

            # Also store raw data
            content['data'] = [df.columns.tolist()] + df.values.tolist()

        except Exception as e:
            print(f"Error reading CSV with pandas: {e}")
            # Fallback to basic reading
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    import csv
                    reader = csv.reader(f)
                    content['data'] = list(reader)
                    if content['data']:
                        content['metadata']['rows'] = len(content['data'])
                        content['metadata']['columns'] = len(content['data'][0]) if content['data'] else 0
            except Exception as e2:
                print(f"Error reading CSV file: {e2}")

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

            # PDF - OCR Processing with timing
            if ext in ['.pdf']:
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

                        # OCR with PaddleOCR predict
                        ocr_start = time.time()
                        results = ocr_engine.ocr_with_detection_and_recognition(tmp_path)
                        ocr_time = time.time() - ocr_start

                        # Extract text from result objects using .json attribute
                        text_parts = []
                        for res in results:
                            result_data = res.json  # Access JSON data
                            # Extract text from the structured result
                            if 'dt_polys' in result_data and 'rec_text' in result_data:
                                text_parts.extend(result_data['rec_text'])

                        full_text = ' '.join(text_parts)

                        attachment_texts.append({
                            'file': attachment['filename'],
                            'page': page_num + 1,
                            'text': full_text,
                            'method': 'ocr',
                            'time': time.time() - page_start,
                            'ocr_time': ocr_time
                        })

                        # Save using PaddleOCR's built-in methods
                        if save_visuals and out_root is not None:
                            save_start = time.time()
                            for res in results:
                                res.save_to_img(str(out_root / f"{base_name}_page{page_num + 1}_annotated.png"))
                                res.save_to_json(str(out_root / f"{base_name}_page{page_num + 1}_ocr.json"))
                            save_time = time.time() - save_start
                            print(f"  [PDF] Page {page_num + 1} - OCR: {ocr_time:.2f}s, Save: {save_time:.2f}s")

                        # Cleanup temp file
                        os.unlink(tmp_path)

                    print(f"  [PDF] Total: {time.time() - pdf_start:.2f}s")

                except Exception as e:
                    print(f"[OCR] Error processing PDF {file_path}: {e}")
                    traceback.print_exc()

            # IMAGE FILES - OCR Processing
            # IMAGE FILES - OCR Processing with timing
            elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
                try:
                    img_start = time.time()

                    # OCR with timing
                    ocr_start = time.time()
                    results = ocr_engine.ocr_with_detection_and_recognition(file_path)
                    ocr_time = time.time() - ocr_start

                    # Extract text from result objects
                    text_parts = []
                    for res in results:
                        result_data = res.json
                        if 'dt_polys' in result_data and 'rec_text' in result_data:
                            text_parts.extend(result_data['rec_text'])

                    full_text = ' '.join(text_parts)

                    attachment_texts.append({
                        'file': attachment['filename'],
                        'text': full_text,
                        'method': 'ocr',
                        'time': time.time() - img_start,
                        'ocr_time': ocr_time
                    })

                    # Save using PaddleOCR's built-in methods
                    if save_visuals and out_root is not None:
                        save_start = time.time()
                        for res in results:
                            res.save_to_img(str(out_root / f"{base_name}_annotated.png"))
                            res.save_to_json(str(out_root / f"{base_name}_ocr.json"))
                        save_time = time.time() - save_start
                        print(f"  [IMG] OCR: {ocr_time:.2f}s, Save: {save_time:.2f}s")

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

def create_llm_context(email_data, attachment_texts):
    """Combine email and attachment data for LLM"""
    context = {
        'email': {
            'subject': email_data['metadata']['subject'],
            'from': email_data['metadata']['sender'],
            'date': email_data['metadata']['date'],
            'body': email_data['body']['plain_text']
        },
        'attachments': attachment_texts,
        'total_attachments': len(attachment_texts),
        'processing_times': {
            'email': 0,  # Add timing
            'attachments': sum(a['time'] for a in attachment_texts)
        }
    }

    # Create a single text representation for LLM
    llm_text = f"""
        Email Subject: {context['email']['subject']}
        From: {context['email']['from']}
        Date: {context['email']['date']}

        Email Body:
        {context['email']['body']}

        Attachments ({context['total_attachments']}):
    """

    for att in attachment_texts:
        llm_text += f"\n\n--- {att['file']} ---\n{att['text'][:500]}..."  # Truncate for preview

    return context, llm_text


def process_all_subfolders(root_folder, save_json=True):
    """Process all subfolders in root folder with progress indicators"""
    root_path = Path(root_folder)

    # Count total folders first
    subfolders = [f for f in root_path.iterdir() if f.is_dir()]
    total_folders = len(subfolders)

    if total_folders == 0:
        print(f"No subfolders found in {root_folder}")
        return {}

    all_results = {}
    processed = 0
    errors = 0

    for subfolder in sorted(subfolders):
        processed += 1
        print(f"\n[{processed}/{total_folders}] Processing: {subfolder.name}")

        try:
            processor = EmailProcessor(subfolder)
            data = processor.get_complete_data()

            if data:
                all_results[subfolder.name] = data

                print(f"  ✓ Subject: {data['metadata']['subject']}")
                print(f"  ✓ From: {data['metadata']['sender']}")
                print(f"  ✓ Date: {data['metadata']['date']}")
                print(f"  ✓ Saved attachments: {len(data['saved_attachments'])}")
                print(f"  ✓ Recipients (To): {len(data['recipients']['to'])}")

                if save_json:
                    json_path = subfolder / 'email_data.json'
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    print(f"  ✓ Saved: {json_path.name}")
            else:
                print(f"  ✗ No .msg file found or error processing")
                errors += 1

        except Exception as e:
            print(f"  ✗ Error processing folder: {e}")
            errors += 1
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 80}")
    print(f"Processing complete: {len(all_results)} successful, {errors} errors")
    print(f"{'=' * 80}")

    return all_results


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
                    use_tensorrt: Use TensorRT acceleration (GPU only)
                    precision: TensorRT precision ('fp32' or 'fp16')
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
            doc_orientation_classify_model_name='PP-LCNet_x1_0_doc_ori',
            doc_orientation_classify_model_dir='PP-LCNet_x1_0_doc_ori',
            textline_orientation_model_name='PP-LCNet_x1_0_textline_ori',
            textline_orientation_model_dir='PP-LCNet_x1_0_textline_ori',
            text_detection_model_name='PP-OCRv5_mobile_det',
            text_recognition_model_name='PP-OCRv5_server_rec',
            text_detection_model_dir="PP-OCRv5_mobile_det_infer",
            text_recognition_model_dir="PP-OCRv5_server_rec_infer",
            use_doc_orientation_classify=True,
            use_doc_unwarping=True,
            use_textline_orientation=True,
            device='cpu',
            cpu_threads=4,
            enable_mkldnn=True,
            use_tensorrt=False,
            precision='fp16',
            det_limit_side_len=960,
            det_limit_type='max',
            text_det_thresh=0.3,
            text_det_box_thresh=0.6,
            text_det_unclip_ratio=1.5,
            text_recognition_batch_size=4,
            ocr_version="PP-OCRv5",
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


if __name__ == "__main__":
    root_folder = "test datasets"

    # Initialize OCR engine
    print("Initializing OCR engine...")
    init_start = time.time()
    ocr_engine = Detect_and_Recognize()
    print(f"OCR initialization: {time.time() - init_start:.2f}s\n")

    total_start = time.time()

    # Step 1: Extract email data
    print("Step 1: Processing emails...")
    email_start = time.time()
    results = process_all_subfolders(root_folder, save_json=True)
    print(f"Email processing: {time.time() - email_start:.2f}s\n")

    # Step 2: OCR attachments
    print("Step 2: Processing attachments with OCR...")
    ocr_start = time.time()

    for folder_name, email_data in results.items():
        folder_start = time.time()
        print(f"\n[Folder: {folder_name}]")

        processor = EmailProcessor(Path(root_folder) / folder_name)
        attachment_texts = processor.process_attachments_with_ocr(
            email_data,
            ocr_engine,
            save_visuals=True
        )

        # Step 3: Create LLM context
        context_start = time.time()
        context, llm_text = create_llm_context(email_data, attachment_texts)
        context_time = time.time() - context_start

        # Save combined context
        output_path = Path(root_folder) / folder_name / 'llm_context.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(context, f, indent=2, ensure_ascii=False)

        folder_time = time.time() - folder_start
        print(f"  Context creation: {context_time:.2f}s")
        print(f"  Folder total: {folder_time:.2f}s")

    print(f"\nOCR processing: {time.time() - ocr_start:.2f}s")
    print(f"Total processing time: {time.time() - total_start:.2f}s")






