#document_processor.py
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
                
                if file_extension in ['pdf', 'docx']:
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
        """Extract text and images from PDF with proper base64 encoding"""
        text_content = ""
        images = []
        
        try:
            pdf_doc = fitz.open(stream=file_content, filetype="pdf")
            
            for page_num in range(len(pdf_doc)):
                page = pdf_doc.load_page(page_num)
                page_text = page.get_text()
                text_content += page_text + "\n"
                
                # Extract images with proper encoding
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(pdf_doc, xref)
                        if pix.n < 5:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            # Convert to base64 immediately
                            img_b64 = base64.b64encode(img_data).decode('utf-8')
                            
                            # Try to associate with error codes found on this page
                            associated_codes = self.find_error_codes_on_page(page_text)
                            
                            images.append({
                                "data": img_b64,  # Store as base64 string
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
    
    def find_error_codes_on_page(self, text: str) -> List[str]:
        """Find error codes mentioned on a specific page"""
        hex_pattern = r'\b([A-Fa-f0-9]{4})\b'
        codes = re.findall(hex_pattern, text.upper())
        return list(set(codes))  # Remove duplicates
    
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
        """Process and store document in database with proper image handling"""
        conn = None
        try:
            # Process document based on type
            if file_type == "pdf":
                result = self.process_pdf(file_content, filename)
            elif file_type == "docx":
                result = self.process_docx(file_content, filename)
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
            
            cursor.execute("""
                INSERT INTO documents (filename, content, doc_type, metadata) 
                VALUES (?, ?, ?, ?)
            """, (filename, text_content, file_type, json.dumps({"image_count": len(images)})))
            
            doc_id = cursor.lastrowid
            
            # Extract and store error codes with enhanced context
            error_codes = self.extract_error_codes_with_context(text_content)
            for error_code in error_codes:
                # Store main error code entry
                cursor.execute("""
                    INSERT OR REPLACE INTO error_codes (code, description, source_doc, procedure_steps, category) 
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    error_code["code"], 
                    error_code["description"], 
                    filename,
                    error_code["context"],  # Store context as procedure steps
                    "extracted"  # Mark as extracted from document
                ))
            
            # Store images with proper error code associations
            for i, img in enumerate(images):
                img_filename = f"{filename}_page_{img['page']}_img_{i}"
                
                # Store image with base64 data (already encoded)
                cursor.execute("""
                    INSERT INTO images (filename, image_data, description, step_number) 
                    VALUES (?, ?, ?, ?)
                """, (
                    img_filename, 
                    img["data"],  # Already base64 encoded
                    f"Image from {filename} page {img['page']}", 
                    img['page']
                ))
                
                image_id = cursor.lastrowid
                
                # Associate image with error codes found on the same page
                for code in img.get("associated_codes", []):
                    cursor.execute("""
                        UPDATE images 
                        SET associated_code = ? 
                        WHERE id = ?
                    """, (code, image_id))
                    
                    # If no specific association, also try to link by proximity
                    if not img.get("associated_codes"):
                        # Find the most recent error code in the document
                        recent_codes = [ec["code"] for ec in error_codes[-3:]]  # Last 3 codes
                        if recent_codes:
                            cursor.execute("""
                                UPDATE images 
                                SET associated_code = ? 
                                WHERE id = ?
                            """, (recent_codes[0], image_id))
            
            # Create embeddings for text chunks
            try:
                sentences = sent_tokenize(text_content)
            except LookupError:
                sentences = text_content.split('. ')
                sentences = [s.strip() + '.' for s in sentences if s.strip()]
            
            if sentences:
                # Filter out very short sentences
                meaningful_sentences = [s for s in sentences if len(s.strip()) > 20]
                
                if meaningful_sentences:
                    embeddings = self.create_embeddings(meaningful_sentences)
                    
                    for sentence, embedding in zip(meaningful_sentences, embeddings):
                        cursor.execute("""
                            INSERT INTO embeddings (content_id, content_type, embedding, text_content) 
                            VALUES (?, ?, ?, ?)
                        """, (str(doc_id), "document", pickle.dumps(embedding), sentence))
            
            conn.commit()
            return True
            
        except Exception as e:
            if conn:
                conn.rollback()
            st.error(f"Error ingesting document {filename}: {str(e)}")
            print(f"Detailed error: {e}")  # For debugging
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
            
            conn.close()
            
            return {
                "documents": doc_count,
                "error_codes": error_code_count,
                "images": image_count,
                "embeddings": embedding_count
            }
        except Exception:
            return {"documents": 0, "error_codes": 0, "images": 0, "embeddings": 0}