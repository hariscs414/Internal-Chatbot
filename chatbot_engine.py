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
        """Determine if images should be included based on query intent"""
        # Keywords that strongly suggest need for visual guidance
        visual_keywords = [
            'how to', 'procedure', 'steps', 'repair', 'fix', 'install', 'replace', 
            'remove', 'disassemble', 'assembly', 'diagram', 'visual', 'show me',
            'guide', 'instruction', 'troubleshoot', 'maintenance', 'service',
            'location', 'where is', 'position', 'parts', 'component'
        ]
        
        # Keywords that suggest information-only queries (no images needed)
        info_only_keywords = [
            'what is', 'define', 'meaning', 'explanation', 'why does', 'cause',
            'reason', 'theory', 'concept', 'specification', 'rating', 'value',
            'temperature', 'pressure', 'voltage', 'current', 'frequency'
        ]
        
        query_lower = query.lower()
        
        # Check for info-only patterns first
        for keyword in info_only_keywords:
            if keyword in query_lower:
                return False
        
        # Check for visual guidance patterns
        for keyword in visual_keywords:
            if keyword in query_lower:
                return True
        
        # Check for error codes - these often need visual guidance
        if re.search(r'\b[A-Fa-f0-9]{4}\b', query):
            return True
        
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
                            "relevance": "direct_association",
                            "score": 1.0
                        })
                    except Exception as e:
                        print(f"Error processing image {row[0]}: {e}")
                        continue
            
            # If we have direct matches, prefer them and only add more if needed
            if len(images) >= 2:
                return sorted(images, key=lambda x: x['step_number'])[:3]
            
            # Strategy 2: Exact filename match (high priority)
            cursor.execute("""
                SELECT filename, image_data, description, step_number 
                FROM images 
                WHERE UPPER(filename) = UPPER(?) 
                AND image_data IS NOT NULL
                ORDER BY step_number, filename
            """, (f"{error_code}.jpg",))  # Try exact filename first
            
            for row in cursor.fetchall():
                if row[1] and row[0] not in seen_images:
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
                            "relevance": "exact_filename",
                            "score": 0.9
                        })
                    except Exception as e:
                        print(f"Error processing image {row[0]}: {e}")
                        continue
            
            # Strategy 3: Filename contains error code (medium priority)
            if len(images) < 2:
                cursor.execute("""
                    SELECT filename, image_data, description, step_number 
                    FROM images 
                    WHERE UPPER(filename) LIKE UPPER(?) 
                    AND image_data IS NOT NULL
                    ORDER BY step_number, filename
                    LIMIT 2
                """, (f"%{error_code}%",))
                
                for row in cursor.fetchall():
                    if row[1] and row[0] not in seen_images:
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
                                "relevance": "filename_contains",
                                "score": 0.7
                            })
                        except Exception as e:
                            print(f"Error processing image {row[0]}: {e}")
                            continue
            
            # Strategy 4: Semantic search only if we have very few results
            if len(images) < 1:
                error_info = self.search_error_code(error_code)
                if error_info and error_info.get('description'):
                    description = error_info['description']
                    description_embedding = self.model.encode([description])
                    
                    # Get only images with good descriptions
                    cursor.execute("""
                        SELECT filename, image_data, description, step_number 
                        FROM images 
                        WHERE description IS NOT NULL 
                        AND LENGTH(description) > 10
                        AND image_data IS NOT NULL
                    """)
                    all_images = cursor.fetchall()
                    
                    # Calculate similarity scores
                    similarities = []
                    for filename, image_data, img_desc, step_num in all_images:
                        if filename not in seen_images and img_desc and image_data:
                            img_embedding = self.model.encode([img_desc])
                            similarity = cosine_similarity(description_embedding, img_embedding)[0][0]
                            if similarity > 0.6:  # Higher threshold for semantic matching
                                similarities.append((filename, image_data, img_desc, step_num, similarity))
                    
                    # Sort by similarity and get only the best match
                    similarities.sort(key=lambda x: x[4], reverse=True)
                    
                    for filename, image_data, img_desc, step_num, sim in similarities[:1]:  # Only best match
                        if filename not in seen_images:
                            seen_images.add(filename)
                            try:
                                if isinstance(image_data, bytes):
                                    image_b64 = base64.b64encode(image_data).decode('utf-8')
                                else:
                                    image_b64 = image_data
                                
                                images.append({
                                    "filename": filename,
                                    "data": image_b64,
                                    "description": img_desc,
                                    "step_number": step_num or 0,
                                    "error_code": error_code,
                                    "relevance": "semantic_match",
                                    "score": sim
                                })
                            except Exception as e:
                                print(f"Error processing semantic match image {filename}: {e}")
                                continue
            
            # Sort by relevance score and return top results
            images.sort(key=lambda x: (x['score'], -x['step_number']), reverse=True)
            return images[:3]  # Maximum 3 images
            
        except Exception as e:
            print(f"Error getting repair procedure for {error_code}: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def get_related_images(self, query: str) -> List[Dict]:
        """Get images related to general queries with strict relevance filtering"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_manager.db_path, timeout=10.0)
            cursor = conn.cursor()
            
            # More selective keyword extraction
            stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'how', 'what', 'when', 'where', 'why', 'can', 'could', 'would', 'should', 'will', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'a', 'an', 'this', 'that', 'these', 'those'}
            
            # Extract technical terms (longer words that are likely to be relevant)
            query_terms = [word.lower() for word in query.split() 
                          if len(word) > 4 and word.lower() not in stop_words 
                          and not word.isdigit()]
            
            if not query_terms:
                return []
            
            images = []
            seen_images = set()
            
            # Strategy 1: Look for exact matches in image descriptions
            for term in query_terms[:2]:  # Only use top 2 most relevant terms
                cursor.execute("""
                    SELECT filename, image_data, description, step_number 
                    FROM images 
                    WHERE LOWER(description) LIKE ? 
                    AND image_data IS NOT NULL
                    AND LENGTH(description) > 15
                    ORDER BY LENGTH(description) DESC, step_number
                    LIMIT 1
                """, (f"%{term}%",))
                
                for row in cursor.fetchall():
                    if row[1] and row[0] not in seen_images:
                        # Verify the image description is actually relevant
                        description = row[2] or ""
                        if any(term in description.lower() for term in query_terms):
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
                                    "relevance": f"description_match_{term}",
                                    "score": 0.8
                                })
                            except Exception as e:
                                print(f"Error processing related image {row[0]}: {e}")
                                continue
            
            # Strategy 2: Semantic search with high threshold
            if len(images) < 1:
                try:
                    query_embedding = self.model.encode([query])
                    
                    cursor.execute("""
                        SELECT filename, image_data, description, step_number 
                        FROM images 
                        WHERE description IS NOT NULL 
                        AND LENGTH(description) > 20
                        AND image_data IS NOT NULL
                    """)
                    all_images = cursor.fetchall()
                    
                    similarities = []
                    for filename, image_data, img_desc, step_num in all_images:
                        if filename not in seen_images and img_desc:
                            img_embedding = self.model.encode([img_desc])
                            similarity = cosine_similarity(query_embedding, img_embedding)[0][0]
                            if similarity > 0.7:  # High threshold for general queries
                                similarities.append((filename, image_data, img_desc, step_num, similarity))
                    
                    similarities.sort(key=lambda x: x[4], reverse=True)
                    
                    # Only take the best semantic match
                    for filename, image_data, img_desc, step_num, sim in similarities[:1]:
                        if filename not in seen_images:
                            seen_images.add(filename)
                            try:
                                if isinstance(image_data, bytes):
                                    image_b64 = base64.b64encode(image_data).decode('utf-8')
                                else:
                                    image_b64 = image_data
                                
                                images.append({
                                    "filename": filename,
                                    "data": image_b64,
                                    "description": img_desc,
                                    "step_number": step_num or 0,
                                    "relevance": "semantic_match",
                                    "score": sim
                                })
                            except Exception as e:
                                print(f"Error processing semantic image {filename}: {e}")
                                continue
                except Exception as e:
                    print(f"Error in semantic search: {e}")
            
            # Sort by relevance score and return limited results
            images.sort(key=lambda x: x['score'], reverse=True)
            return images[:2]  # Maximum 2 images for general queries
            
        except Exception as e:
            print(f"Error getting related images: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def vector_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Perform vector similarity search with improved filtering and ranking"""
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
                    
                    # Increased threshold for better quality results
                    if similarity > 0.6:  # Higher threshold for more relevant results
                        results.append({
                            "content_id": row[0],
                            "content_type": row[1],
                            "text_content": row[3],
                            "similarity": float(similarity)
                        })
                    elif similarity > 0.4:  # Medium threshold for backup results
                        results.append({
                            "content_id": row[0],
                            "content_type": row[1],
                            "text_content": row[3],
                            "similarity": float(similarity)
                        })
                except Exception as e:
                    continue
            
            conn.close()
            
            # Sort by similarity and return top results
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            # If we have high-quality results (>0.6), only return those
            high_quality = [r for r in results if r["similarity"] > 0.6]
            if high_quality:
                return high_quality[:top_k]
            
            # Otherwise return top medium-quality results
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
        """Use AI to analyze query with improved context management"""
        try:
            # Select only the most relevant context (avoid information overload)
            context_text = ""
            if context_data:
                # Use only the best result for context
                best_result = context_data[0]
                if 'text_content' in best_result:
                    context_text = best_result['text_content']
                elif 'content' in best_result:
                    context_text = best_result['content']
                
                # Add second result only if it's very relevant (high similarity)
                if len(context_data) > 1 and context_data[1].get('similarity', 0) > 0.7:
                    second_result = context_data[1]
                    second_text = second_result.get('text_content', '') or second_result.get('content', '')
                    if second_text:
                        context_text += f"\n\nAdditional context: {second_text}"
            
            # Truncate context if too long
            if len(context_text) > 1000:
                context_text = context_text[:1000] + "..."
            
            image_note = "\n\nNote: I've included relevant visual guides to help illustrate the process." if has_images else ""
            
            system_prompt = """You are an experienced machine repair technician with years of hands-on experience. 
            You have deep knowledge of various machines, error codes, troubleshooting procedures, and repair techniques.
            
            CRITICAL INSTRUCTIONS:
            - NEVER mention documents, databases, manuals, documentation, or any source materials
            - NEVER ask users to "refer to documentation" or "check the manual"
            - NEVER use phrases like "based on the provided information", "according to the documentation", "based on the provided documentation context", or "the documentation references"
            - NEVER say "Missing Information" or ask to "check other documentation sections"
            - Respond as if you personally know this information from your experience
            - Be conversational and natural, like talking to a colleague
            - Provide detailed explanations and expand on ideas when helpful
            - Use your expertise to give context and background information
            - If you don't have complete information, say "I'd need to see the unit to give you more specific details" or "That would depend on the specific model"
            - Be confident in your knowledge while remaining helpful and approachable
            - Never reveal that you're using stored information or databases
            - Focus on practical, actionable advice
            - Keep responses concise but comprehensive
            
            Respond naturally as an expert technician would, drawing from your extensive experience with machines and repairs."""
            
            user_prompt = f"""A colleague asks you: {query}

Your knowledge and experience tells you:
{context_text}
{image_note}

Respond naturally as an experienced technician would, providing comprehensive help and practical advice."""
            
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
                temperature=0.3,  # Reduced temperature for more consistent responses
                max_tokens=800    # Reduced tokens for more focused responses
            )
            
            return completion.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error with AI analysis: {e}")
            return self.format_response_fallback(context_data, query)
    
    def format_response_fallback(self, results: List[Dict], query: str) -> str:
        """Fallback response formatting if AI analysis fails"""
        if not results:
            return "I'm not familiar with that specific issue. Could you provide more details about what you're experiencing? What type of machine are you working with and what symptoms are you seeing?"
        
        # Create a natural response as if the technician knows this information
        response_parts = []
        
        for i, result in enumerate(results[:2], 1):  # Limit to 2 results for cleaner response
            if 'similarity' in result:
                content = result['text_content']
            else:
                content = result['content']
            
            content = ' '.join(content.split())
            if len(content) > 300:
                sentences = content.split('.')
                truncated = []
                char_count = 0
                for sentence in sentences:
                    if char_count + len(sentence) > 300:
                        break
                    truncated.append(sentence)
                    char_count += len(sentence)
                content = '. '.join(truncated)
                if not content.endswith('.'):
                    content += '.'
            
            response_parts.append(content)
        
        # Join with natural transitions
        if len(response_parts) > 1:
            return f"{response_parts[0]}\n\nAdditionally, {response_parts[1].lower()}"
        else:
            return response_parts[0] if response_parts else "I'm not familiar with that specific issue. Could you provide more details?"
    
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
        """Generate chatbot response with improved image selection logic"""
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
            else:
                # No error codes found
                response_text = f"I'm not familiar with error code {', '.join(hex_codes)}. Could you double-check the code? Sometimes they can be misread. What symptoms are you seeing with the machine?"
        
        # General search for non-error-code queries
        if not response_text:
            # Try vector search first
            vector_results = self.vector_search(query)
            
            # Only use keyword search if vector search yields poor results
            if not vector_results or (vector_results and max([r.get('similarity', 0) for r in vector_results]) < 0.5):
                keyword_results = self.search_documents_by_keywords(query)
                
                # Only add keyword results if they're potentially relevant
                if keyword_results and not vector_results:
                    for result in keyword_results:
                        vector_results.append({
                            "content_id": "keyword_search",
                            "content_type": "document",
                            "text_content": result["content"],
                            "similarity": 0.5
                        })
            
            # Get related images only if query suggests need for visual guidance AND we have good text results
            if should_include_images and vector_results and max([r.get('similarity', 0) for r in vector_results]) > 0.6:
                related_images = self.get_related_images(query)
                # Additional filtering for relevance
                relevant_images = [img for img in related_images if img.get('score', 0) > 0.7]
                images_data.extend(relevant_images)
            
            if vector_results:
                response_text = self.analyze_query_with_ai(query, vector_results, bool(images_data))
            else:
                response_text = ("I'm not sure about that specific issue. Could you give me more details about what you're working on? "
                               "What type of machine is it, and what exactly is happening? The more information you can provide, "
                               "the better I can help you troubleshoot the problem.")
        
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
