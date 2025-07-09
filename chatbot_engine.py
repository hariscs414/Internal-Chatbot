# chatbot_engine.py
import sqlite3
import re
import pickle
import base64
import io
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv
class ChatbotEngine:
    """Main chatbot logic with context memory, retrieval, and AI-powered understanding"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.context_memory = {}
        
        # Initialize OpenAI client with environment variables
        api_key = os.getenv('OPENAI_API_KEY')
        base_url = os.getenv('OPENAI_BASE_URL', 'https://openrouter.ai/api/v1')
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.ai_client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
    
    def clear_user_context(self, username: str):
        """Clear context memory for a specific user"""
        if username in self.context_memory:
            self.context_memory[username] = []
    
    def search_error_code(self, code: str) -> Optional[Dict]:
        """Search for specific error code in database with better debugging"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_manager.db_path, timeout=10.0)
            cursor = conn.cursor()
            
            # First try exact match
            cursor.execute("""
                SELECT code, description, procedure_steps, category, source_doc 
                FROM error_codes 
                WHERE UPPER(code) = UPPER(?)
            """, (code,))
            result = cursor.fetchone()
            
            if result:
                return {
                    "code": result[0],
                    "description": result[1],
                    "procedure_steps": result[2] or "",
                    "category": result[3] or "",
                    "source_doc": result[4] or ""
                }
            
            # If no exact match, try partial match
            cursor.execute("""
                SELECT code, description, procedure_steps, category, source_doc 
                FROM error_codes 
                WHERE UPPER(code) LIKE UPPER(?) OR UPPER(description) LIKE UPPER(?)
            """, (f"%{code}%", f"%{code}%"))
            result = cursor.fetchone()
            
            if result:
                return {
                    "code": result[0],
                    "description": result[1],
                    "procedure_steps": result[2] or "",
                    "category": result[3] or "",
                    "source_doc": result[4] or ""
                }
            
            return None
            
        except Exception as e:
            print(f"Error searching for code {code}: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    def get_repair_procedure_images(self, error_code: str) -> List[Dict]:
        """Get images associated with error code using multiple strategies"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_manager.db_path, timeout=10.0)
            cursor = conn.cursor()
            
            images = []
            
            # Strategy 1: Direct association with error code
            cursor.execute("""
                SELECT filename, image_data, description, step_number 
                FROM images 
                WHERE UPPER(associated_code) = UPPER(?) 
                ORDER BY step_number, filename
            """, (error_code,))
            
            direct_results = cursor.fetchall()
            
            # Strategy 2: If no direct association, search by error code in filename or description
            if not direct_results:
                cursor.execute("""
                    SELECT filename, image_data, description, step_number 
                    FROM images 
                    WHERE UPPER(filename) LIKE UPPER(?) 
                       OR UPPER(description) LIKE UPPER(?)
                    ORDER BY step_number, filename
                """, (f"%{error_code}%", f"%{error_code}%"))
                direct_results = cursor.fetchall()
            
            # Strategy 3: If still no results, get images from documents that contain this error code
            if not direct_results:
                cursor.execute("""
                    SELECT i.filename, i.image_data, i.description, i.step_number
                    FROM images i
                    JOIN documents d ON i.filename LIKE d.filename || '%'
                    WHERE UPPER(d.content) LIKE UPPER(?)
                    ORDER BY i.step_number, i.filename
                    LIMIT 5
                """, (f"%{error_code}%",))
                direct_results = cursor.fetchall()
            
            for row in direct_results:
                if row[1]:  # Check if image_data exists
                    try:
                        image_data = row[1]
                        
                        # Handle both base64 string and binary data
                        if isinstance(image_data, str):
                            # Already base64 encoded
                            image_b64 = image_data
                        elif isinstance(image_data, bytes):
                            # Need to encode binary data
                            image_b64 = base64.b64encode(image_data).decode('utf-8')
                        else:
                            print(f"Unexpected image data type: {type(image_data)}")
                            continue
                        
                        images.append({
                            "filename": row[0],
                            "data": image_b64,
                            "description": row[2] or f"Repair procedure for {error_code}",
                            "step_number": row[3] or 0,
                            "error_code": error_code
                        })
                    except Exception as e:
                        print(f"Error processing image {row[0]}: {e}")
                        continue
            
            print(f"Found {len(images)} images for error code {error_code}")
            return images
            
        except Exception as e:
            print(f"Error getting repair procedure for {error_code}: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def get_related_images(self, query: str) -> List[Dict]:
        """Get images related to general queries (not specific error codes)"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_manager.db_path, timeout=10.0)
            cursor = conn.cursor()
            
            # Extract key terms from query
            query_terms = [word.lower() for word in query.split() if len(word) > 3]
            
            images = []
            
            for term in query_terms:
                cursor.execute("""
                    SELECT filename, image_data, description, step_number 
                    FROM images 
                    WHERE LOWER(description) LIKE ? 
                       OR LOWER(filename) LIKE ?
                    ORDER BY step_number, filename
                    LIMIT 3
                """, (f"%{term}%", f"%{term}%"))
                
                for row in cursor.fetchall():
                    if row[1] and row[0] not in [img["filename"] for img in images]:
                        try:
                            image_data = row[1]
                            
                            if isinstance(image_data, str):
                                image_b64 = image_data
                            elif isinstance(image_data, bytes):
                                image_b64 = base64.b64encode(image_data).decode('utf-8')
                            else:
                                continue
                            
                            images.append({
                                "filename": row[0],
                                "data": image_b64,
                                "description": row[2] or "Related image",
                                "step_number": row[3] or 0,
                                "relevance": term
                            })
                        except Exception as e:
                            print(f"Error processing related image {row[0]}: {e}")
                            continue
            
            return images[:5]  # Limit to 5 related images
            
        except Exception as e:
            print(f"Error getting related images: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def vector_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Perform vector similarity search with improved filtering"""
        try:
            query_embedding = self.model.encode([query])
            
            conn = sqlite3.connect(self.db_manager.db_path, timeout=10.0)
            cursor = conn.cursor()
            cursor.execute("SELECT content_id, content_type, embedding, text_content FROM embeddings")
            
            results = []
            for row in cursor.fetchall():
                try:
                    stored_embedding = pickle.loads(row[2])
                    if not isinstance(stored_embedding, np.ndarray):
                        stored_embedding = np.array(stored_embedding)
                    
                    similarity = cosine_similarity(query_embedding, [stored_embedding])[0][0]
                    
                    if similarity > 0.3:  # Lower threshold for better recall
                        results.append({
                            "content_id": row[0],
                            "content_type": row[1],
                            "text_content": row[3],
                            "similarity": float(similarity)
                        })
                except Exception as e:
                    continue
            
            conn.close()
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            print(f"Error in vector search: {e}")
            return []
    
    def search_documents_by_keywords(self, query: str) -> List[Dict]:
        """Search documents using keyword matching as fallback"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_manager.db_path, timeout=10.0)
            cursor = conn.cursor()
            
            keywords = [word.strip().lower() for word in query.split() if len(word) > 2]
            
            if not keywords:
                return []
            
            search_pattern = '%' + '%'.join(keywords) + '%'
            
            cursor.execute("""
                SELECT filename, content, doc_type 
                FROM documents 
                WHERE LOWER(content) LIKE ? 
                LIMIT 5
            """, (search_pattern,))
            
            results = []
            for row in cursor.fetchall():
                content = row[1]
                sentences = content.split('.')
                relevant_sentences = []
                
                for sentence in sentences:
                    if any(keyword in sentence.lower() for keyword in keywords):
                        relevant_sentences.append(sentence.strip())
                        if len(relevant_sentences) >= 3:
                            break
                
                if relevant_sentences:
                    results.append({
                        "filename": row[0],
                        "content": '. '.join(relevant_sentences),
                        "doc_type": row[2]
                    })
            
            return results
            
        except Exception as e:
            print(f"Error in keyword search: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def analyze_query_with_ai(self, query: str, context_data: List[Dict], has_images: bool = False) -> str:
        """Use DeepSeek to analyze query and generate intelligent response"""
        try:
            # Prepare context from search results
            context_text = ""
            for i, item in enumerate(context_data[:3], 1):
                if 'text_content' in item:
                    context_text += f"\n{i}. {item['text_content']}\n"
                elif 'content' in item:
                    context_text += f"\n{i}. From {item.get('filename', 'document')}: {item['content']}\n"
            
            image_note = "\n\nNote: Relevant images are also included with this response." if has_images else ""
            
            system_prompt = """You are a specialized technical support AI for machine repair and troubleshooting. 
            Your role is to provide accurate, helpful responses based ONLY on the provided documentation context.
            
            Guidelines:
            - Be precise and technical when discussing error codes, procedures, or repairs
            - If the query is about an error code, focus on the specific problem and solution
            - Provide step-by-step guidance when available
            - If information is not in the context, clearly state that
            - Use clear, professional language suitable for technicians
            - Reference specific parts, tools, or procedures mentioned in the documentation
            - Always prioritize safety considerations when mentioned in the docs
            - When images are available, mention that visual guides are provided"""
            
            user_prompt = f"""Query: {query}
            
            Available Documentation Context:
            {context_text}
            {image_note}
            
            Based on this context, provide a comprehensive and helpful response. If the context doesn't contain enough information to fully answer the query, explain what information is available and what might be missing."""
            
            completion = self.ai_client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://machine-repair-chatbot.local",
                    "X-Title": "Machine Repair Chatbot",
                },
                model="deepseek/deepseek-chat-v3-0324:free",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            return completion.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error with AI analysis: {e}")
            return self.format_response_fallback(context_data, query)
    
    def format_response_fallback(self, results: List[Dict], query: str) -> str:
        """Fallback response formatting if AI analysis fails"""
        if not results:
            return "I couldn't find specific information about your query in the available documentation. Could you try rephrasing your question or provide more specific details?"
        
        response_parts = [f"Here's what I found about '{query}':"]
        
        for i, result in enumerate(results[:3], 1):
            if 'similarity' in result:
                content = result['text_content']
            else:
                content = result['content']
                filename = result.get('filename', 'document')
                response_parts.append(f"\n**{i}. From {filename}:**")
            
            content = ' '.join(content.split())
            if len(content) > 400:
                sentences = content.split('.')
                truncated = []
                char_count = 0
                for sentence in sentences:
                    if char_count + len(sentence) > 400:
                        break
                    truncated.append(sentence)
                    char_count += len(sentence)
                content = '. '.join(truncated)
                if not content.endswith('.'):
                    content += '.'
            
            if 'similarity' not in result:
                response_parts.append(content)
            else:
                response_parts.append(f"\n**{i}.** {content}")
        
        return '\n'.join(response_parts)
    
    def update_context_memory(self, username: str, query: str, response: str):
        """Update user's conversation context memory"""
        if username not in self.context_memory:
            self.context_memory[username] = []
        
        # Add current exchange to memory
        self.context_memory[username].append({
            "query": query,
            "response": response,
            "timestamp": None  # Could add timestamp if needed
        })
        
        # Keep only last 5 exchanges to prevent memory bloat
        if len(self.context_memory[username]) > 5:
            self.context_memory[username] = self.context_memory[username][-5:]
    
    def get_conversation_context(self, username: str) -> str:
        """Get formatted conversation context for AI analysis"""
        if username not in self.context_memory or not self.context_memory[username]:
            return ""
        
        context_parts = ["Previous conversation context:"]
        for exchange in self.context_memory[username][-3:]:  # Last 3 exchanges
            context_parts.append(f"User: {exchange['query']}")
            context_parts.append(f"Assistant: {exchange['response'][:200]}...")  # Truncate response
        
        return "\n".join(context_parts)
    
    def generate_response(self, query: str, username: str) -> Dict:
        """Generate chatbot response with AI-powered understanding and proper image handling"""
        if username not in self.context_memory:
            self.context_memory[username] = []
        
        query = query.strip()
        if not query:
            return {
                "response": "Please ask me something specific. I can help you with error codes, troubleshooting, or information from the uploaded documents.",
                "images": []
            }
        
        response_text = ""
        images_data = []
        
        # Check for error code patterns first
        hex_codes = re.findall(r'\b([A-Fa-f0-9]{4})\b', query.upper())
        
        if hex_codes:
            # Handle error code queries
            error_responses = []
            all_error_codes = []
            
            for code in hex_codes:
                error_info = self.search_error_code(code)
                if error_info:
                    error_responses.append(error_info)
                    all_error_codes.append(code)
                    
                    # Get associated images for this error code
                    error_images = self.get_repair_procedure_images(code)
                    images_data.extend(error_images)
            
            if error_responses:
                # Use AI to analyze error codes with context
                context_data = []
                for error in error_responses:
                    context_data.append({
                        "text_content": f"Error Code {error['code']}: {error['description']}. {error['procedure_steps']}",
                        "filename": error.get('source_doc', 'database'),
                        "similarity": 1.0  # Perfect match for error codes
                    })
                
                # Get conversation context for better AI responses
                conversation_context = self.get_conversation_context(username)
                enhanced_query = f"{conversation_context}\n\nCurrent query: {query}" if conversation_context else query
                
                response_text = self.analyze_query_with_ai(enhanced_query, context_data, bool(images_data))
                
                # Update context memory
                self.update_context_memory(username, query, response_text)
                
                return {
                    "response": response_text,
                    "images": images_data
                }
            else:
                # No error codes found, but codes were detected in query
                response_text = f"I couldn't find information about the error code(s) {', '.join(hex_codes)} in the database. Please check if the code is correct or try searching for related terms."
        
        # If no error codes or no matches found, perform general search
        if not response_text:
            # Try vector search first
            vector_results = self.vector_search(query)
            
            # If vector search doesn't yield good results, try keyword search
            if not vector_results or max([r.get('similarity', 0) for r in vector_results]) < 0.5:
                keyword_results = self.search_documents_by_keywords(query)
                # Convert keyword results to match vector results format
                for result in keyword_results:
                    vector_results.append({
                        "content_id": "keyword_search",
                        "content_type": "document",
                        "text_content": result["content"],
                        "filename": result["filename"],
                        "similarity": 0.6  # Assign moderate similarity
                    })
            
            # Get related images for general queries
            related_images = self.get_related_images(query)
            images_data.extend(related_images)
            
            if vector_results:
                # Get conversation context for better AI responses
                conversation_context = self.get_conversation_context(username)
                enhanced_query = f"{conversation_context}\n\nCurrent query: {query}" if conversation_context else query
                
                response_text = self.analyze_query_with_ai(enhanced_query, vector_results, bool(images_data))
            else:
                response_text = ("I couldn't find specific information about your query in the available documentation. "
                               "Could you try rephrasing your question or provide more specific details? "
                               "You can also try asking about specific error codes or machine components.")
        
        # Update context memory
        self.update_context_memory(username, query, response_text)
        
        return {
            "response": response_text,
            "images": images_data
        }
    
    def get_database_stats(self) -> Dict:
        """Get statistics about the current database"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_manager.db_path, timeout=10.0)
            cursor = conn.cursor()
            
            stats = {}
            
            # Count documents
            cursor.execute("SELECT COUNT(*) FROM documents")
            stats["documents"] = cursor.fetchone()[0]
            
            # Count error codes
            cursor.execute("SELECT COUNT(*) FROM error_codes")
            stats["error_codes"] = cursor.fetchone()[0]
            
            # Count images
            cursor.execute("SELECT COUNT(*) FROM images")
            stats["images"] = cursor.fetchone()[0]
            
            # Count embeddings
            cursor.execute("SELECT COUNT(*) FROM embeddings")
            stats["embeddings"] = cursor.fetchone()[0]
            
            # Get recent documents
            cursor.execute("SELECT filename FROM documents ORDER BY id DESC LIMIT 5")
            stats["recent_documents"] = [row[0] for row in cursor.fetchall()]
            
            return stats
            
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {"documents": 0, "error_codes": 0, "images": 0, "embeddings": 0, "recent_documents": []}
        finally:
            if conn:
                conn.close()
    
    def search_similar_error_codes(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for error codes similar to the query using text similarity"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_manager.db_path, timeout=10.0)
            cursor = conn.cursor()
            
            # Get all error codes with descriptions
            cursor.execute("SELECT code, description, procedure_steps, source_doc FROM error_codes")
            all_codes = cursor.fetchall()
            
            if not all_codes:
                return []
            
            # Create embeddings for query and all error descriptions
            query_embedding = self.model.encode([query])
            
            descriptions = []
            code_info = []
            
            for code_data in all_codes:
                # Combine code, description, and procedure for better matching
                full_text = f"{code_data[0]} {code_data[1]} {code_data[2] or ''}"
                descriptions.append(full_text)
                code_info.append({
                    "code": code_data[0],
                    "description": code_data[1],
                    "procedure_steps": code_data[2],
                    "source_doc": code_data[3]
                })
            
            if descriptions:
                desc_embeddings = self.model.encode(descriptions)
                similarities = cosine_similarity(query_embedding, desc_embeddings)[0]
                
                # Get top matches
                similar_indices = np.argsort(similarities)[::-1][:limit]
                
                results = []
                for idx in similar_indices:
                    if similarities[idx] > 0.3:  # Threshold for relevance
                        code_data = code_info[idx]
                        code_data["similarity"] = float(similarities[idx])
                        results.append(code_data)
                
                return results
            
            return []
            
        except Exception as e:
            print(f"Error searching similar error codes: {e}")
            return []
        finally:
            if conn:
                conn.close()