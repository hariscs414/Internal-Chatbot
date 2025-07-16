# document_processor.py
import re
import pickle
import os
import base64
from io import BytesIO
import numpy as np
import sqlite3
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import fitz
import streamlit as st
from typing import Dict, List, Optional
from docx import Document
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
import json

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab')
    except:
        pass

class DocumentProcessor:
    """Handles document ingestion and processing from uploads and project folders"""
    
    def __init__(self, db_manager, documents_folder: str = "documents"):
        self.db_manager = db_manager
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents_folder = documents_folder
        
        # Create documents folder if it doesn't exist
        if not os.path.exists(self.documents_folder):
            os.makedirs(self.documents_folder)
    
    def load_documents_from_folder(self) -> bool:
        """Load all documents from the project documents folder"""
        if not os.path.exists(self.documents_folder):
            st.warning(f"Documents folder '{self.documents_folder}' not found")
            return False
        
        processed_count = 0
        error_count = 0
        
        # Get list of already processed documents
        processed_docs = self._get_processed_documents()
        
        for filename in os.listdir(self.documents_folder):
            file_path = os.path.join(self.documents_folder, filename)
            
            # Skip if already processed
            if filename in processed_docs:
                continue
                
            if os.path.isfile(file_path):
                file_extension = filename.lower().split('.')[-1]
                
                if file_extension in ['pdf', 'docx', 'pptx','ppt']:
                    try:
                        with open(file_path, 'rb') as file:
                            file_content = file.read()
                        
                        if self.ingest_document(file_content, filename, file_extension):
                            processed_count += 1
                            st.success(f"Processed: {filename}")
                        else:
                            error_count += 1
                            st.error(f"Failed to process: {filename}")
                            
                    except Exception as e:
                        error_count += 1
                        st.error(f"Error reading {filename}: {str(e)}")
        
        if processed_count > 0:
            st.success(f"Successfully processed {processed_count} documents from folder")
        if error_count > 0:
            st.warning(f"Failed to process {error_count} documents")
            
        return processed_count > 0
    
    def _get_processed_documents(self) -> set:
        """Get list of already processed document filenames"""
        try:
            conn = sqlite3.connect(self.db_manager.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT filename FROM documents")
            processed = {row[0] for row in cursor.fetchall()}
            conn.close()
            return processed
        except Exception:
            return set()
    
    def process_pdf(self, file_content: bytes, filename: str) -> Dict:
        """Extract text and images from PDF with improved error code association"""
        text_content = ""
        images = []
        
        try:
            pdf_doc = fitz.open(stream=file_content, filetype="pdf")
            
            for page_num in range(len(pdf_doc)):
                page = pdf_doc.load_page(page_num)
                page_text = page.get_text()
                text_content += page_text + "\n"
                
                # Extract images with better error code association
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(pdf_doc, xref)
                        if pix.n < 5:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            img_b64 = base64.b64encode(img_data).decode('utf-8')
                            
                            # Improved error code association
                            # Look for error codes in a wider context around the image
                            context_text = ""
                            
                            # Get text from current page
                            current_page_text = page_text
                            
                            # Get text from previous page if available
                            if page_num > 0:
                                prev_page = pdf_doc.load_page(page_num - 1)
                                prev_text = prev_page.get_text()
                                context_text = prev_text[-500:] + " " + current_page_text  # Last 500 chars of previous page
                            else:
                                context_text = current_page_text
                            
                            # Get text from next page if available
                            if page_num < len(pdf_doc) - 1:
                                next_page = pdf_doc.load_page(page_num + 1)
                                next_text = next_page.get_text()
                                context_text += " " + next_text[:500]  # First 500 chars of next page
                            
                            # Find error codes in this extended context
                            associated_codes = self.find_error_codes_on_page(context_text)
                            
                            # If no codes found in extended context, look for codes in image vicinity
                            if not associated_codes:
                                # Try to find error codes in the immediate text around the image position
                                # This is a simplified approach - in a real implementation, 
                                # you'd analyze the image position relative to text blocks
                                text_blocks = page_text.split('\n')
                                for block in text_blocks:
                                    if len(block.strip()) > 10:  # Meaningful text block
                                        block_codes = self.find_error_codes_on_page(block)
                                        if block_codes:
                                            associated_codes.extend(block_codes)
                                            break
                            
                            # Remove duplicates and limit to most relevant codes
                            associated_codes = list(set(associated_codes))[:3]
                            
                            images.append({
                                "data": img_b64,
                                "page": page_num + 1,
                                "index": img_index,
                                "associated_codes": associated_codes,
                                "page_text": page_text[:500]  # Store snippet for context
                            })
                        pix = None
                    except Exception as e:
                        print(f"Error processing image {img_index} on page {page_num}: {e}")
                        continue
            
            pdf_doc.close()
        except Exception as e:
            st.error(f"Error processing PDF {filename}: {str(e)}")
            return {"text": "", "images": [], "error": str(e)}
        
        return {"text": text_content, "images": images}
    
    def process_pptx(self, file_content: bytes, filename: str) -> Dict:
        """Extract text and images from PowerPoint presentation"""
        text_content = ""
        images = []
        
        try:
            prs = Presentation(BytesIO(file_content))
            
            for slide_num, slide in enumerate(prs.slides):
                slide_text = ""
                
                # Extract text from all shapes
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        slide_text += shape.text + "\n"
                    
                    # Extract images from shapes
                    if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                        try:
                            image = shape.image
                            img_data = image.blob
                            img_b64 = base64.b64encode(img_data).decode('utf-8')
                            
                            # Try to associate with error codes found on this slide
                            associated_codes = self.find_error_codes_on_page(slide_text)
                            
                            images.append({
                                "data": img_b64,
                                "page": slide_num + 1,  # Use slide number as page
                                "index": len(images),
                                "associated_codes": associated_codes,
                                "page_text": slide_text[:500]
                            })
                        except Exception as e:
                            print(f"Error processing image on slide {slide_num}: {e}")
                            continue
                
                # Add slide header
                text_content += f"=== Slide {slide_num + 1} ===\n"
                text_content += slide_text + "\n\n"
                
                # Extract text from notes
                if slide.notes_slide and slide.notes_slide.notes_text_frame:
                    notes_text = slide.notes_slide.notes_text_frame.text
                    if notes_text.strip():
                        text_content += f"Notes: {notes_text}\n\n"
        
        except Exception as e:
            st.error(f"Error processing PowerPoint {filename}: {str(e)}")
            return {"text": "", "images": [], "error": str(e)}
        
        return {"text": text_content, "images": images}
    
    def find_error_codes_on_page(self, text: str) -> List[str]:
        """Find error codes mentioned on a specific page with improved detection"""
        # Multiple patterns for better error code detection
        patterns = [
            r'\b([A-Fa-f0-9]{4})\b',  # Basic hex pattern
            r'Code\s+([A-Fa-f0-9]{4})',  # Code XXXX
            r'Error\s+([A-Fa-f0-9]{4})',  # Error XXXX
            r'([A-Fa-f0-9]{4})\s*[-:]\s*',  # XXXX: or XXXX-
        ]
        
        codes = set()
        for pattern in patterns:
            matches = re.findall(pattern, text.upper())
            codes.update(matches)
        
        return list(codes)
    
    def process_docx(self, file_content: bytes, filename: str) -> Dict:
        """Extract text from Word document"""
        try:
            doc = Document(BytesIO(file_content))
            text_content = ""
            
            for paragraph in doc.paragraphs:
                text_content += paragraph.text + "\n"
            
            # Extract tables
            tables_text = ""
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        row_text.append(cell.text.strip())
                    tables_text += " | ".join(row_text) + "\n"
            
            return {"text": text_content + "\n" + tables_text, "images": []}
        except Exception as e:
            st.error(f"Error processing DOCX {filename}: {str(e)}")
            return {"text": "", "images": [], "error": str(e)}
    
    def extract_error_codes_with_context(self, text: str) -> List[Dict]:
        """Extract error codes with their descriptions and surrounding context"""
        error_codes = []
        
        # Multiple patterns to catch different formats
        patterns = [
            r'([A-Fa-f0-9]{4})\s*[-:=]\s*(.{10,200})',  # Code: Description
            r'Code\s+([A-Fa-f0-9]{4})\s*[-:]\s*(.{10,200})',  # Code XXXX: Description
            r'Error\s+([A-Fa-f0-9]{4})\s*[-:]\s*(.{10,200})',  # Error XXXX: Description
            r'([A-Fa-f0-9]{4})\s+(.{20,200})',  # Code followed by description
        ]
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            for pattern in patterns:
                matches = re.findall(pattern, line, re.IGNORECASE)
                for match in matches:
                    code = match[0].upper()
                    description = match[1].strip()
                    
                    # Clean up description
                    description = re.sub(r'\s+', ' ', description)
                    
                    if len(description) > 10:  # Filter out short descriptions
                        # Get surrounding context (previous and next lines)
                        context_lines = []
                        for j in range(max(0, i-2), min(len(lines), i+3)):
                            if lines[j].strip():
                                context_lines.append(lines[j].strip())
                        
                        error_codes.append({
                            "code": code,
                            "description": description,
                            "context": " ".join(context_lines),
                            "line_number": i + 1
                        })
        
        return error_codes
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create vector embeddings for text chunks"""
        return self.model.encode(texts)
    
    def ingest_document(self, file_content: bytes, filename: str, file_type: str) -> bool:
        """Process and store document with improved image-error code association"""
        conn = None
        try:
            # Process document based on type
            if file_type == "pdf":
                result = self.process_pdf(file_content, filename)
            elif file_type == "docx":
                result = self.process_docx(file_content, filename)
            elif file_type == "pptx":
                result = self.process_pptx(file_content, filename)
            else:
                st.error(f"Unsupported file type: {file_type}")
                return False
            
            if "error" in result:
                return False
            
            text_content = result["text"]
            images = result.get("images", [])
            
            # Store document
            conn = sqlite3.connect(self.db_manager.db_path, timeout=30.0)
            cursor = conn.cursor()
            
            conn.execute("BEGIN IMMEDIATE")
            
            # Add metadata
            metadata = {"image_count": len(images)}
            if file_type == "pptx":
                slide_count = text_content.count("=== Slide")
                metadata["slide_count"] = slide_count
            
            cursor.execute("""
                INSERT INTO documents (filename, content, doc_type, metadata) 
                VALUES (?, ?, ?, ?)
            """, (filename, text_content, file_type, json.dumps(metadata)))
            
            doc_id = cursor.lastrowid
            
            # Extract and store error codes
            error_codes = self.extract_error_codes_with_context(text_content)
            stored_error_codes = {}  # Track stored error codes for better image association
            
            for error_code in error_codes:
                cursor.execute("""
                    INSERT OR REPLACE INTO error_codes (code, description, source_doc, procedure_steps, category) 
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    error_code["code"], 
                    error_code["description"], 
                    filename,
                    error_code["context"],
                    "extracted"
                ))
                stored_error_codes[error_code["code"]] = error_code
            
            # Store images with improved error code association
            for i, img in enumerate(images):
                if file_type == "pptx":
                    img_filename = f"{filename}_slide_{img['page']}_img_{i}"
                else:
                    img_filename = f"{filename}_page_{img['page']}_img_{i}"
                
                # Determine the best error code association
                best_associated_code = None
                if img.get("associated_codes"):
                    # Use the first associated code if it exists in our stored codes
                    for code in img["associated_codes"]:
                        if code in stored_error_codes:
                            best_associated_code = code
                            break
                    
                    # If no match in stored codes, use the first found code
                    if not best_associated_code:
                        best_associated_code = img["associated_codes"][0]
                
                # Enhanced description with context
                enhanced_description = f"Image from {filename} {'slide' if file_type == 'pptx' else 'page'} {img['page']}"
                
                # Add page text context to description for better search
                if img.get('page_text'):
                    page_text_clean = re.sub(r'\s+', ' ', img['page_text'][:200])  # Clean and truncate
                    enhanced_description += f" - Context: {page_text_clean}"
                
                # Add error code context if available
                if best_associated_code and best_associated_code in stored_error_codes:
                    error_context = stored_error_codes[best_associated_code]['description'][:100]
                    enhanced_description += f" - Related to: {error_context}"
                
                # Store image in database
                cursor.execute("""
                    INSERT INTO images (filename, image_data, description, step_number, associated_code) 
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    img_filename, 
                    img["data"],
                    enhanced_description,
                    img['page'],
                    best_associated_code
                ))
            
            # Create embeddings for meaningful sentences only
            try:
                sentences = sent_tokenize(text_content)
            except LookupError:
                sentences = text_content.split('. ')
                sentences = [s.strip() + '.' for s in sentences if s.strip()]
            
            if sentences:
                # Filter for meaningful sentences
                meaningful_sentences = [
                    s for s in sentences 
                    if len(s.strip()) > 30 and  # Longer threshold
                    not s.strip().startswith("=== Slide") and
                    not s.strip().startswith("Page") and
                    len(s.split()) > 5  # At least 5 words
                ]
                
                if meaningful_sentences:
                    # Create embeddings in batches for better performance
                    batch_size = 50
                    for i in range(0, len(meaningful_sentences), batch_size):
                        batch = meaningful_sentences[i:i+batch_size]
                        embeddings = self.create_embeddings(batch)
                        
                        for sentence, embedding in zip(batch, embeddings):
                            cursor.execute("""
                                INSERT INTO embeddings (content_id, content_type, embedding, text_content) 
                                VALUES (?, ?, ?, ?)
                            """, (str(doc_id), "document", pickle.dumps(embedding), sentence))
            
            conn.commit()
            
            # Print summary of extracted images
            if images:
                st.success(f"Extracted {len(images)} images from {filename}")
            
            return True
            
        except Exception as e:
            if conn:
                conn.rollback()
            st.error(f"Error ingesting document {filename}: {str(e)}")
            print(f"Detailed error: {e}")
            return False
        finally:
            if conn:
                conn.close()
    
    def get_document_stats(self) -> Dict:
        """Get statistics about processed documents"""
        try:
            conn = sqlite3.connect(self.db_manager.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM error_codes")
            error_code_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM images")
            image_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM embeddings")
            embedding_count = cursor.fetchone()[0]
            
            # Get document type breakdown
            cursor.execute("SELECT doc_type, COUNT(*) FROM documents GROUP BY doc_type")
            doc_types = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                "documents": doc_count,
                "error_codes": error_code_count,
                "images": image_count,
                "embeddings": embedding_count,
                "document_types": doc_types
            }
        except Exception:
            return {
                "documents": 0, 
                "error_codes": 0, 
                "images": 0, 
                "embeddings": 0,
                "document_types": {}
            }
