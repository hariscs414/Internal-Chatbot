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
    
    def should_include_images(self, query: str) -> bool:
        """Determine if images should be included based on query intent - more restrictive"""
        # Keywords that strongly suggest need for visual guidance
        visual_keywords = [
            'how to', 'procedure', 'steps', 'repair', 'fix', 'install', 'replace', 
            'remove', 'disassemble', 'assembly', 'diagram', 'visual', 'show me',
            'guide', 'instruction', 'troubleshoot', 'maintenance', 'service',
            'location', 'where is', 'position', 'parts', 'component', 'assembly',
            'installation', 'removal', 'adjustment', 'calibration'
        ]
        
        # Keywords that suggest information-only queries (no images needed)
        info_only_keywords = [
            'what is', 'define', 'meaning', 'explanation', 'why does', 'cause',
            'reason', 'theory', 'concept', 'specification', 'rating', 'value',
            'temperature', 'pressure', 'voltage', 'current', 'frequency',
            'function', 'functioning', 'operation', 'works', 'purpose'
        ]
        
        query_lower = query.lower()
        
        # Check for info-only patterns first - these should NOT have images
        for keyword in info_only_keywords:
            if keyword in query_lower:
                return False
        
        # Check for visual guidance patterns
        for keyword in visual_keywords:
            if keyword in query_lower:
                return True
        
        # For error codes, only include images if repair/procedure keywords are present
        if re.search(r'\b[A-Fa-f0-9]{4}\b', query):
            # Check if it's asking for repair procedures vs just error info
            repair_context = any(word in query_lower for word in ['fix', 'repair', 'solve', 'procedure', 'steps', 'how'])
            return repair_context
        
        # Default to no images for general questions
        return False

    def get_repair_procedure_images(self, error_code: str) -> List[Dict]:
        """Get images associated with error code using improved relevance scoring"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_manager.db_path, timeout=10.0)
            cursor = conn.cursor()
            
            images = []
            seen_images = set()
            
            # Strategy 1: Direct association by error code (highest priority)
            cursor.execute("""
                SELECT filename, image_data, description, step_number 
                FROM images 
                WHERE UPPER(associated_code) = UPPER(?) 
                AND image_data IS NOT NULL
                ORDER BY step_number, filename
            """, (error_code,))
            
            for row in cursor.fetchall():
                if row[1] and row[0] not in seen_images:
                    seen_images.add(row[0])
                    try:
                        image_data = row[1]
                        # Handle both base64 string and bytes
                        if isinstance(image_data, bytes):
                            try:
                                # Try to decode as base64
                                image_b64 = base64.b64encode(image_data).decode('utf-8')
                            except:
                                # If it fails, use the bytes directly
                                image_b64 = base64.b64encode(image_data).decode('utf-8')
                        else:
                            # Already base64 string
                            image_b64 = image_data
                        
                        images.append({
                            "filename": row[0],
                            "data": image_b64,
                            "description": row[2] or f"Repair procedure for {error_code}",
                            "step_number": row[3] or 0,
                            "error_code": error_code,
                            "relevance": "direct_association",
                            "score": 1.0
                        })
                    except Exception as e:
                        print(f"Error processing image {row[0]}: {e}")
                        continue
            
            # If we have direct matches, return only those - don't look for more
            if len(images) >= 1:
                return sorted(images, key=lambda x: x['step_number'])[:2]  # Limit to 2 direct matches
            
            # Only if NO direct matches found, look for filename matches with stricter criteria
            if len(images) == 0:
                # Strategy 2: Exact filename match with error code
                cursor.execute("""
                    SELECT filename, image_data, description, step_number 
                    FROM images 
                    WHERE (UPPER(filename) LIKE UPPER(?) OR UPPER(description) LIKE UPPER(?))
                    AND image_data IS NOT NULL
                    ORDER BY 
                        CASE 
                            WHEN UPPER(filename) LIKE UPPER(?) THEN 1
                            WHEN UPPER(description) LIKE UPPER(?) THEN 2
                            ELSE 3
                        END,
                        step_number, filename
                    LIMIT 1
                """, (f"%{error_code}%", f"%{error_code}%", f"%{error_code}%", f"%{error_code}%"))
                
                for row in cursor.fetchall():
                    if row[1] and row[0] not in seen_images:
                        # Verify the match is actually relevant
                        filename_match = error_code.upper() in row[0].upper()
                        desc_match = row[2] and error_code.upper() in row[2].upper()
                        
                        if filename_match or desc_match:
                            seen_images.add(row[0])
                            try:
                                image_data = row[1]
                                if isinstance(image_data, bytes):
                                    image_b64 = base64.b64encode(image_data).decode('utf-8')
                                else:
                                    image_b64 = image_data
                                
                                images.append({
                                    "filename": row[0],
                                    "data": image_b64,
                                    "description": row[2] or f"Repair procedure for {error_code}",
                                    "step_number": row[3] or 0,
                                    "error_code": error_code,
                                    "relevance": "filename_match" if filename_match else "description_match",
                                    "score": 0.9 if filename_match else 0.8
                                })
                            except Exception as e:
                                print(f"Error processing image {row[0]}: {e}")
                                continue
            
            # Sort by relevance score and return
            images.sort(key=lambda x: (x['score'], -x['step_number']), reverse=True)
            return images[:2]  # Maximum 2 images, only highly relevant ones
            
        except Exception as e:
            print(f"Error getting repair procedure for {error_code}: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def get_related_images(self, query: str) -> List[Dict]:
        """Get images related to general queries with very strict relevance filtering"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_manager.db_path, timeout=10.0)
            cursor = conn.cursor()
            
            # Very selective keyword extraction - focus on technical terms
            stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'how', 'what', 'when', 'where', 'why', 'can', 'could', 'would', 'should', 'will', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'a', 'an', 'this', 'that', 'these', 'those', 'type', 'method', 'functioning'}
            
            # Extract only highly relevant technical terms
            query_terms = [word.lower() for word in query.split() 
                          if len(word) > 5 and word.lower() not in stop_words 
                          and not word.isdigit() and word.isalpha()]
            
            if not query_terms:
                return []
            
            images = []
            seen_images = set()
            
            # Strategy 1: Look for exact matches in image descriptions with stricter criteria
            for term in query_terms[:1]:  # Only use the most relevant term
                cursor.execute("""
                    SELECT filename, image_data, description, step_number 
                    FROM images 
                    WHERE LOWER(description) LIKE ? 
                    AND image_data IS NOT NULL
                    AND LENGTH(description) > 20
                    AND (LOWER(filename) LIKE ? OR LOWER(description) LIKE ?)
                    ORDER BY LENGTH(description) DESC, step_number
                    LIMIT 1
                """, (f"%{term}%", f"%{term}%", f"%{term}%"))
                
                result_rows = cursor.fetchall()
                for row in result_rows:
                    if row[1] and row[0] not in seen_images:
                        # Verify the image description is actually highly relevant
                        description = row[2] or ""
                        if term in description.lower() and len(description) > 30:
                            seen_images.add(row[0])
                            try:
                                image_data = row[1]
                                if isinstance(image_data, bytes):
                                    image_b64 = base64.b64encode(image_data).decode('utf-8')
                                else:
                                    image_b64 = image_data
                                
                                images.append({
                                    "filename": row[0],
                                    "data": image_b64,
                                    "description": description,
                                    "step_number": row[3] or 0,
                                    "relevance": f"exact_description_match_{term}",
                                    "score": 0.9
                                })
                            except Exception as e:
                                print(f"Error processing related image {row[0]}: {e}")
                                continue
            
            # Return only if we found highly relevant matches
            images.sort(key=lambda x: x['score'], reverse=True)
            return images[:1]  # Maximum 1 image for general queries, only if highly relevant
            
        except Exception as e:
            print(f"Error getting related images: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def search_tables(self, query: str) -> List[Dict]:
        """Enhanced table search with better relevance and formatting"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_manager.db_path, timeout=10.0)
            cursor = conn.cursor()
            
            # Extract key search terms
            search_terms = [term.strip().lower() for term in query.split() if len(term) > 2]
            
            if not search_terms:
                return []
            
            results = []
            
            # Strategy 1: Multi-term search with ranking
            for term in search_terms[:3]:  # Use top 3 terms
                cursor.execute("""
                    SELECT t.table_id, t.raw_content, t.formatted_content, t.page_number, d.filename, d.doc_type
                    FROM tables t
                    JOIN documents d ON t.doc_id = d.id
                    WHERE (LOWER(t.raw_content) LIKE ? OR LOWER(t.formatted_content) LIKE ?)
                    AND LENGTH(COALESCE(t.formatted_content, t.raw_content)) > 50
                    ORDER BY t.page_number
                    LIMIT 5
                """, (f"%{term}%", f"%{term}%"))
                
                for row in cursor.fetchall():
                    table_id = row[0]
                    # Avoid duplicates
                    if not any(r["table_id"] == table_id for r in results):
                        content = row[2] if row[2] else row[1]  # Prefer formatted content
                        if content and len(content.strip()) > 50:
                            # Calculate relevance score
                            content_lower = content.lower()
                            score = sum(1 for term in search_terms if term in content_lower)
                            score = score / len(search_terms)  # Normalize
                            
                            results.append({
                                "table_id": table_id,
                                "content": content.strip(),
                                "page_number": row[3],
                                "filename": row[4],
                                "doc_type": row[5],
                                "content_type": "table",
                                "relevance_score": score
                            })
            
            # Sort by relevance score and return best matches
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            return results[:4]  # Return top 4 table results
            
        except Exception as e:
            print(f"Error searching tables: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def vector_search(self, query: str, top_k: int = 6) -> List[Dict]:
        """Enhanced vector similarity search with better filtering and context scoring"""
        try:
            query_embedding = self.model.encode([query])
            
            conn = sqlite3.connect(self.db_manager.db_path, timeout=10.0)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT content_id, content_type, embedding, text_content 
                FROM embeddings 
                WHERE LENGTH(text_content) > 20
            """)
            
            results = []
            for row in cursor.fetchall():
                try:
                    stored_embedding = pickle.loads(row[2])
                    if not isinstance(stored_embedding, np.ndarray):
                        stored_embedding = np.array(stored_embedding)
                    
                    similarity = cosine_similarity(query_embedding, [stored_embedding])[0][0]
                    
                    # Improved scoring with content quality factors
                    content_length = len(row[3])
                    content_quality_bonus = min(0.1, content_length / 2000)  # Bonus for longer, detailed content
                    adjusted_similarity = similarity + content_quality_bonus
                    
                    # More nuanced threshold based on content type
                    min_threshold = 0.5 if row[1] == 'table' else 0.55  # Lower threshold for tables
                    
                    if adjusted_similarity > min_threshold:
                        results.append({
                            "content_id": row[0],
                            "content_type": row[1],
                            "text_content": row[3],
                            "similarity": float(adjusted_similarity),
                            "raw_similarity": float(similarity),
                            "content_length": content_length
                        })
                except Exception as e:
                    continue
            
            conn.close()
            
            # Sort by adjusted similarity and return diverse results
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Ensure content diversity - avoid too many similar results
            diverse_results = []
            seen_content_types = {}
            
            for result in results:
                content_type = result["content_type"]
                if content_type not in seen_content_types:
                    seen_content_types[content_type] = 0
                
                # Limit results per content type for diversity
                if seen_content_types[content_type] < 3:  # Max 3 per type
                    diverse_results.append(result)
                    seen_content_types[content_type] += 1
                    
                    if len(diverse_results) >= top_k:
                        break
            
            return diverse_results
            
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
        """Use AI to analyze query with strict document-only responses and detailed explanations"""
        try:
            # Select all relevant context for comprehensive answers
            context_text = ""
            if context_data:
                context_parts = []
                for i, result in enumerate(context_data[:6]):  # Increased to top 6 results for more detailed context
                    if 'text_content' in result:
                        content = result['text_content']
                    elif 'content' in result:
                        content = result['content']
                    else:
                        continue
                    
                    # Better context formatting with more detail preservation
                    if result.get('content_type') == 'table':
                        context_parts.append(f"Technical specifications and data: {content}")
                    elif result.get('content_type') == 'error_code':
                        context_parts.append(f"Error code information: {content}")
                    else:
                        context_parts.append(content)
                
                context_text = "\n\n".join(context_parts)
            
            # If no context found, return a human-like response asking for more info
            if not context_text.strip():
                return ("I don't have specific information about that in my experience. Could you give me a bit more detail? What type of machine are you working with and what exactly is happening? The more specifics you can share, the better I can help you out.")
            
            # Don't truncate context - allow more comprehensive information
            if len(context_text) > 4000:  # Increased limit for more detailed responses
                # Smart truncation - try to keep complete sentences
                sentences = context_text.split('. ')
                truncated_sentences = []
                char_count = 0
                for sentence in sentences:
                    if char_count + len(sentence) > 4000:
                        break
                    truncated_sentences.append(sentence)
                    char_count += len(sentence)
                context_text = '. '.join(truncated_sentences)
                if not context_text.endswith('.'):
                    context_text += '.'
            
            image_note = " I've also pulled up some relevant technical diagrams that should help illustrate the procedures." if has_images else ""
            
            system_prompt = """You are an experienced repair technician with decades of hands-on experience. Provide comprehensive, detailed answers based strictly on the technical information provided.

            CRITICAL RULES:
            1. ONLY use information from the provided context - no external knowledge
            2. Give detailed, step-by-step explanations when procedures are involved
            3. Include ALL relevant technical details from the context - don't summarize too much
            4. Use conversational phrases like "From what I've seen" or "In my experience" naturally
            5. NEVER mention documents, manuals, databases, or sources
            6. If context lacks specific info, say "I'd need to know [specific detail]" but provide what you can first
            7. For procedures, list steps clearly with explanations
            8. Include technical specifications, measurements, and part numbers when available
            9. Explain not just what to do, but why it's done that way
            10. Be thorough and educational - share the technical knowledge comprehensively
            11. If multiple related issues or procedures are mentioned in context, explain them all
            12. Include safety considerations and precautions when mentioned in the context"""

            user_prompt = f"""Question: {query}

        Available technical information:
        {context_text}
        {image_note}

        Provide a comprehensive, detailed answer based on this technical information. Include all relevant procedures, specifications, and explanations. If steps are involved, explain each one thoroughly with reasoning. Share all the technical knowledge available in the context."""
            
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
                temperature=0.1,  # Lower temperature for more consistent, detailed responses
                max_tokens=1500   # Increased significantly for comprehensive responses
            )
            
            return completion.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error with AI analysis: {e}")
            return self.format_response_fallback(context_data, query)

    def format_response_fallback(self, results: List[Dict], query: str) -> str:
        """Fallback response formatting with human-like tone"""
        if not results:
            return ("I don't have experience with that specific issue. Can you tell me more about what's going on? "
                   "What type of machine is it, and what symptoms are you seeing? Sometimes the little details "
                   "make all the difference in figuring out what's wrong.")
        
        # Create a natural response as if sharing personal experience
        response_parts = []
        
        for i, result in enumerate(results[:2], 1):  # Limit to 2 results for cleaner response
            if 'text_content' in result:
                content = result['text_content']
            else:
                content = result['content']
            
            content = ' '.join(content.split())
            if len(content) > 350:  # Slightly increased for better context
                sentences = content.split('.')
                truncated = []
                char_count = 0
                for sentence in sentences:
                    if char_count + len(sentence) > 350:
                        break
                    truncated.append(sentence)
                    char_count += len(sentence)
                content = '. '.join(truncated)
                if not content.endswith('.'):
                    content += '.'
            
            response_parts.append(content)
        
        # Join with natural, personal transitions
        if len(response_parts) > 1:
            return f"From what I've seen, {response_parts[0].lower()}\n\nAlso, in my experience, {response_parts[1].lower()}"
        else:
            intro_phrases = [
                "From what I've learned working on these machines, ",
                "In my experience, ",
                "What I've seen before is that ",
                "From the work I've done on similar issues, "
            ]
            intro = random.choice(intro_phrases)
            return f"{intro}{response_parts[0].lower()}" if response_parts else ("I don't have specific experience with that. "
                   "Could you give me more details about what's happening?")
    
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
            context_parts.append(f"You: {exchange['response'][:200]}...")  # Truncate response
        
        return "\n".join(context_parts)
    
    def generate_response(self, query: str, username: str) -> Dict:
        """Generate chatbot response with improved search including tables - UPDATED VERSION"""
        if username not in self.context_memory:
            self.context_memory[username] = []
        
        query = query.strip()
        if not query:
            return {
                "response": "Hello! What can I help you with today? I'm here to assist with any machine issues, error codes, or troubleshooting questions you might have.",
                "images": []
            }
        
        response_text = ""
        images_data = []
        
        # Determine if images should be included based on query intent
        should_include_images = self.should_include_images(query)
        
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
                    
                    # Get associated images only if the query suggests need for visual guidance
                    if should_include_images:
                        error_images = self.get_repair_procedure_images(code)
                        # Filter images based on relevance score
                        relevant_images = [img for img in error_images if img.get('score', 0) > 0.6]
                        images_data.extend(relevant_images)
            
            if error_responses:
                # Create focused context for AI
                context_data = []
                for error in error_responses:
                    context_data.append({
                        "text_content": f"Error Code {error['code']}: {error['description']}. {error['procedure_steps']}",
                        "similarity": 1.0
                    })
                
                response_text = self.analyze_query_with_ai(query, context_data, bool(images_data))
                
                # Update context memory
                self.update_context_memory(username, query, response_text)
                
                return {
                    "response": response_text,
                    "images": images_data
                }
        
        # General search for non-error-code queries
        if not response_text:
            # Try vector search first
            vector_results = self.vector_search(query)
            
            # Try table search
            table_results = self.search_tables(query)
            
            # Convert table results to match vector result format
            for table_result in table_results:
                vector_results.append({
                    "content_id": table_result["table_id"],
                    "content_type": "table",
                    "text_content": table_result["content"],
                    "similarity": 0.8  # Give tables high relevance
                })
            
            # Filter results to only include relevant ones (lowered threshold)
            relevant_results = [r for r in vector_results if r.get('similarity', 0) > 0.4]
            
            # Use keyword search as additional fallback
            if len(relevant_results) < 2:
                keyword_results = self.search_documents_by_keywords(query)
                
                for result in keyword_results:
                    relevant_results.append({
                        "content_id": "keyword_search",
                        "content_type": "document", 
                        "text_content": result["content"],
                        "similarity": 0.5
                    })
            
            # Get related images if query suggests need for visual guidance
            if should_include_images and relevant_results:
                related_images = self.get_related_images(query)
                # Less strict filtering for images
                relevant_images = [img for img in related_images if img.get('score', 0) > 0.6]
                images_data.extend(relevant_images)
            
            if relevant_results:
                response_text = self.analyze_query_with_ai(query, relevant_results, bool(images_data))
            else:
                response_text = ("What specific issue are you having? If you can tell me more about the machine type "
                               "and what symptoms you're seeing, I can give you better guidance.")
        
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
