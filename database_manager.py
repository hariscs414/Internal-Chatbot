# database_manager.py
import sqlite3
import hashlib
import json
from typing import Dict, List, Optional
from datetime import datetime

class DatabaseManager:
    """Handles all database operations for users, documents, and chat history"""
    
    def __init__(self, db_path: str = "chatbot_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        
        # Documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                content TEXT,
                doc_type TEXT,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                file_hash TEXT,
                file_size INTEGER
            )
        ''')
        
        # Error codes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS error_codes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT UNIQUE NOT NULL,
                description TEXT NOT NULL,
                category TEXT,
                procedure_steps TEXT,
                source_doc TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Images table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                image_data BLOB,
                associated_code TEXT,
                step_number INTEGER,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Chat history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                message TEXT NOT NULL,
                response TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT,
                FOREIGN KEY (username) REFERENCES users(username)
            )
        ''')
        
        # Vector embeddings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_id TEXT NOT NULL,
                content_type TEXT NOT NULL,
                embedding BLOB NOT NULL,
                text_content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_error_codes_code ON error_codes(code)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_associated_code ON images(associated_code)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_history_username ON chat_history(username)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_content ON embeddings(content_id, content_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename)")
        
        # Create default admin user if not exists
        cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
        if cursor.fetchone()[0] == 0:
            admin_hash = hashlib.sha256("admin123".encode()).hexdigest()
            cursor.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                         ("admin", admin_hash, "admin"))
        
        conn.commit()
        conn.close()
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user credentials and update last login"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        cursor.execute("SELECT id, username, role FROM users WHERE username = ? AND password_hash = ?",
                      (username, password_hash))
        result = cursor.fetchone()
        
        if result:
            # Update last login timestamp
            cursor.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE username = ?", (username,))
            conn.commit()
            
            user_info = {"id": result[0], "username": result[1], "role": result[2]}
        else:
            user_info = None
            
        conn.close()
        return user_info
    
    def add_user(self, username: str, password: str, role: str = "user") -> bool:
        """Add new user to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            cursor.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                         (username, password_hash, role))
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False
        except Exception as e:
            print(f"Error adding user: {e}")
            return False
    
    def update_user_password(self, username: str, new_password: str) -> bool:
        """Update user password"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            password_hash = hashlib.sha256(new_password.encode()).hexdigest()
            cursor.execute("UPDATE users SET password_hash = ? WHERE username = ?",
                         (password_hash, username))
            
            updated = cursor.rowcount > 0
            conn.commit()
            conn.close()
            return updated
        except Exception as e:
            print(f"Error updating password: {e}")
            return False
    
    def delete_user(self, username: str, current_username: str) -> bool:
        """Delete user from database (admin cannot delete themselves)"""
        if username == current_username:
            return False  # Cannot delete self
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete user's chat history first (due to foreign key constraint)
            cursor.execute("DELETE FROM chat_history WHERE username = ?", (username,))
            cursor.execute("DELETE FROM users WHERE username = ?", (username,))
            
            deleted_rows = cursor.rowcount
            conn.commit()
            conn.close()
            
            return deleted_rows > 0
        except Exception as e:
            print(f"Error deleting user: {e}")
            return False
    
    def get_all_users(self) -> List[Dict]:
        """Get all users from database with enhanced information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT u.username, u.role, u.created_at, u.last_login,
                   COUNT(c.id) as message_count
            FROM users u
            LEFT JOIN chat_history c ON u.username = c.username
            GROUP BY u.username, u.role, u.created_at, u.last_login
            ORDER BY u.created_at DESC
        """)
        
        users = []
        for row in cursor.fetchall():
            users.append({
                "username": row[0],
                "role": row[1],
                "created_at": row[2],
                "last_login": row[3],
                "message_count": row[4]
            })
        
        conn.close()
        return users
    
    def save_chat_message(self, username: str, message: str, response: str, session_id: str = None):
        """Save chat interaction to database with session tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chat_history (username, message, response, session_id) 
            VALUES (?, ?, ?, ?)
        """, (username, message, response, session_id))
        conn.commit()
        conn.close()
    
    def save_chat_history(self, username: str, message: str, response: str, session_id: str = None):
        """Alias for save_chat_message for backward compatibility"""
        return self.save_chat_message(username, message, response, session_id)
    
    def get_chat_history(self, username: str, limit: int = 10) -> List[Dict]:
        """Retrieve recent chat history for user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT message, response, timestamp, session_id
            FROM chat_history 
            WHERE username = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (username, limit))
        
        history = []
        for row in cursor.fetchall():
            history.append({
                "message": row[0],
                "response": row[1],
                "timestamp": row[2],
                "session_id": row[3]
            })
        
        conn.close()
        return list(reversed(history))
    
    def get_chat_sessions(self, username: str) -> List[Dict]:
        """Get chat sessions for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT session_id, MIN(timestamp) as start_time, 
                   MAX(timestamp) as end_time, COUNT(*) as message_count
            FROM chat_history 
            WHERE username = ? AND session_id IS NOT NULL
            GROUP BY session_id
            ORDER BY start_time DESC
        """, (username,))
        
        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                "session_id": row[0],
                "start_time": row[1],
                "end_time": row[2],
                "message_count": row[3]
            })
        
        conn.close()
        return sessions
    
    def clear_user_chat_history(self, username: str) -> bool:
        """Clear all chat history for a specific user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chat_history WHERE username = ?", (username,))
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            return deleted_count > 0
        except Exception as e:
            print(f"Error clearing chat history: {e}")
            return False
    
    def get_database_info(self) -> Dict:
        """Get comprehensive database information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        info = {}
        
        # Table counts
        tables = ['users', 'documents', 'error_codes', 'images', 'chat_history', 'embeddings']
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            info[f"{table}_count"] = cursor.fetchone()[0]
        
        # Database size
        cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
        db_size = cursor.fetchone()[0]
        info['database_size_bytes'] = db_size
        info['database_size_mb'] = round(db_size / (1024 * 1024), 2)
        
        # Recent activity
        cursor.execute("SELECT COUNT(*) FROM chat_history WHERE datetime(timestamp) > datetime('now', '-24 hours')")
        info['messages_last_24h'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM documents WHERE datetime(upload_date) > datetime('now', '-7 days')")
        info['documents_last_7d'] = cursor.fetchone()[0]
        
        conn.close()
        return info
    
    def backup_database(self, backup_path: str) -> bool:
        """Create a backup of the database"""
        try:
            source = sqlite3.connect(self.db_path)
            backup = sqlite3.connect(backup_path)
            source.backup(backup)
            backup.close()
            source.close()
            return True
        except Exception as e:
            print(f"Error creating backup: {e}")
            return False
    
    def clean_old_embeddings(self, days_old: int = 30) -> int:
        """Clean up old embeddings that are no longer referenced"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete embeddings older than specified days that don't have corresponding documents
            cursor.execute("""
                DELETE FROM embeddings 
                WHERE datetime(created_at) < datetime('now', '-{} days')
                AND content_id NOT IN (SELECT CAST(id AS TEXT) FROM documents)
            """.format(days_old))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            return deleted_count
        except Exception as e:
            print(f"Error cleaning embeddings: {e}")
            return 0
    
    def get_user_statistics(self, username: str) -> Dict:
        """Get detailed statistics for a specific user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Total messages
        cursor.execute("SELECT COUNT(*) FROM chat_history WHERE username = ?", (username,))
        stats['total_messages'] = cursor.fetchone()[0]
        
        # Messages this week
        cursor.execute("""
            SELECT COUNT(*) FROM chat_history 
            WHERE username = ? AND datetime(timestamp) > datetime('now', '-7 days')
        """, (username,))
        stats['messages_this_week'] = cursor.fetchone()[0]
        
        # First message date
        cursor.execute("""
            SELECT MIN(timestamp) FROM chat_history WHERE username = ?
        """, (username,))
        first_message = cursor.fetchone()[0]
        stats['first_message_date'] = first_message
        
        # Last message date
        cursor.execute("""
            SELECT MAX(timestamp) FROM chat_history WHERE username = ?
        """, (username,))
        last_message = cursor.fetchone()[0]
        stats['last_message_date'] = last_message
        
        # Average message length
        cursor.execute("""
            SELECT AVG(LENGTH(message)) FROM chat_history WHERE username = ?
        """, (username,))
        avg_length = cursor.fetchone()[0]
        stats['avg_message_length'] = round(avg_length, 2) if avg_length else 0
        
        conn.close()
        return stats
    
    def optimize_database(self) -> bool:
        """Optimize database performance"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Analyze tables for better query planning
            cursor.execute("ANALYZE")
            
            # Vacuum to reclaim space and defragment
            cursor.execute("VACUUM")
            
            conn.close()
            return True
        except Exception as e:
            print(f"Error optimizing database: {e}")
            return False