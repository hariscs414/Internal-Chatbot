# main_interface.py
import streamlit as st
import pandas as pd
import re
import base64
import io
import os
import uuid
from PIL import Image
from datetime import datetime
from db.database_manager import *
from processing.document_processor import *
from engine.chatbot_engine import *

# Page configuration
st.set_page_config(
    page_title="Machine Repair Chatbot",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components with error handling
@st.cache_resource
def init_app():
    """Initialize application components with proper error handling"""
    try:
        db_manager = DatabaseManager()
        doc_processor = DocumentProcessor(db_manager)
        chatbot = ChatbotEngine(db_manager)
        
        # Verify database connection
        stats = db_manager.get_database_info()
        st.success(f"✅ Database initialized - {stats.get('documents_count', 0)} documents, {stats.get('error_codes_count', 0)} error codes")
        
        return db_manager, doc_processor, chatbot
    except Exception as e:
        st.error(f"❌ Failed to initialize application: {e}")
        st.stop()

def decode_image(image_data):
    """Safely decode base64 image data with enhanced error handling"""
    try:
        if isinstance(image_data, str):
            # Handle base64 string with or without data URL prefix
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
        elif isinstance(image_data, bytes):
            image_bytes = image_data
        else:
            st.warning(f"Unexpected image data type: {type(image_data)}")
            return None
        
        image = Image.open(io.BytesIO(image_bytes))
        return image
    except Exception as e:
        st.error(f"Error decoding image: {e}")
        return None

def display_image_from_data(image_data, caption="Repair Step", width=None):
    """Display image from base64 or binary data with better error handling"""
    try:
        image = decode_image(image_data)
        if image:
            st.image(image, caption=caption, width=width, use_column_width=True)
            return True
        else:
            st.warning(f"Could not display image: {caption}")
            return False
    except Exception as e:
        st.error(f"Error displaying image {caption}: {e}")
        return False

def login_page(db_manager):
    """Enhanced login page with session management"""
    st.title("🔐 Machine Repair Chatbot - Login")
    
    # Add some styling
    st.markdown("""
    <style>
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        background-color: #f8f9fa;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.container():
            st.markdown('<div class="login-container">', unsafe_allow_html=True)
            
            with st.form("login_form", clear_on_submit=True):
                st.markdown("### Please sign in")
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                remember_me = st.checkbox("Remember me")
                
                col_login, col_register = st.columns(2)
                with col_login:
                    login_clicked = st.form_submit_button("🔐 Login", use_container_width=True)
                with col_register:
                    register_clicked = st.form_submit_button("📝 Register", use_container_width=True)
                
                if login_clicked and username and password:
                    user = db_manager.authenticate_user(username, password)
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.user = user
                        st.session_state.session_id = str(uuid.uuid4())
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
                
                elif register_clicked and username and password:
                    if len(password) < 6:
                        st.error("Password must be at least 6 characters long")
                    elif db_manager.add_user(username, password, "user"):
                        st.success("✅ Account created successfully! Please login.")
                        st.rerun()
                    else:
                        st.error("❌ Username already exists or registration failed")
                
                elif (login_clicked or register_clicked) and (not username or not password):
                    st.error("Please fill in both username and password")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            
        
            
            # System status
            with st.expander("📊 System Status"):
                try:
                    stats = db_manager.get_database_info()
                    col_stat1, col_stat2 = st.columns(2)
                    with col_stat1:
                        st.metric("Users", stats.get('users_count', 0))
                        st.metric("Documents", stats.get('documents_count', 0))
                    with col_stat2:
                        st.metric("Error Codes", stats.get('error_codes_count', 0))
                        st.metric("Images", stats.get('images_count', 0))
                except Exception as e:
                    st.error(f"Error loading system status: {e}")

def admin_panel(db_manager, doc_processor):
    """Enhanced admin control panel with better error handling and features"""
    st.title("🛠️ Admin Control Panel")
    st.markdown(f"Logged in as: **{st.session_state.user['username']}** ({st.session_state.user['role']})")
    
    # System overview metrics
    with st.container():
        try:
            stats = db_manager.get_database_info()
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("📄 Documents", stats.get('documents_count', 0))
            with col2:
                st.metric("🔢 Error Codes", stats.get('error_codes_count', 0))
            with col3:
                st.metric("🖼️ Images", stats.get('images_count', 0))
            with col4:
                st.metric("💬 Total Messages", stats.get('chat_history_count', 0))
            with col5:
                st.metric("👥 Users", stats.get('users_count', 0))
                
        except Exception as e:
            st.error(f"Error loading system overview: {e}")
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📚 Document Management", 
        "👥 User Management", 
        "📊 System Analytics", 
        "📁 Bulk Upload", 
        "⚙️ System Settings"
    ])
    
    with tab1:
        st.header("📚 Document Management")
        
        # File upload section
        with st.expander("📤 Upload New Documents", expanded=True):
            uploaded_files = st.file_uploader(
                "Choose files to upload",
                accept_multiple_files=True,
                type=['pdf', 'docx', 'txt', 'jpg', 'jpeg', 'png'],
                help="Supported formats: PDF, DOCX, TXT, JPG, PNG"
            )
            
            if uploaded_files:
                st.info(f"Selected {len(uploaded_files)} file(s) for upload")
                
                col_process, col_clear = st.columns([1, 1])
                with col_process:
                    process_clicked = st.button("🚀 Process Documents", type="primary", use_container_width=True)
                with col_clear:
                    if st.button("🗑️ Clear Selection", use_container_width=True):
                        st.rerun()
                
                if process_clicked:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    success_count = 0
                    error_files = []
                    
                    for i, file in enumerate(uploaded_files):
                        try:
                            status_text.text(f"Processing: {file.name}")
                            file_type = file.name.split('.')[-1].lower()
                            file_content = file.read()
                            
                            if doc_processor.ingest_document(file_content, file.name, file_type):
                                success_count += 1
                                st.success(f"✅ {file.name} processed successfully")
                            else:
                                error_files.append(file.name)
                                st.error(f"❌ Failed to process {file.name}")
                            
                        except Exception as e:
                            error_files.append(f"{file.name} ({str(e)})")
                            st.error(f"❌ Error processing {file.name}: {e}")
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    status_text.empty()
                    
                    if success_count > 0:
                        st.balloons()
                        st.success(f"🎉 Successfully processed {success_count}/{len(uploaded_files)} documents!")
                    
                    if error_files:
                        with st.expander("❌ Failed Files"):
                            for error_file in error_files:
                                st.write(f"• {error_file}")
        
        # Existing documents section
        with st.expander("📋 Document Library"):
            try:
                conn = sqlite3.connect(db_manager.db_path)
                
                # Check available columns
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(documents)")
                columns = [column[1] for column in cursor.fetchall()]
                
                # Build query based on available columns
                base_columns = "id, filename, doc_type"
                if 'upload_date' in columns:
                    query = f"SELECT {base_columns}, upload_date FROM documents ORDER BY upload_date DESC"
                elif 'created_at' in columns:
                    query = f"SELECT {base_columns}, created_at FROM documents ORDER BY created_at DESC"
                else:
                    query = f"SELECT {base_columns}, 'N/A' as date_column FROM documents ORDER BY id DESC"
                
                docs_df = pd.read_sql_query(query, conn)
                conn.close()
                
                if not docs_df.empty:
                    # Add search functionality
                    search_term = st.text_input("🔍 Search documents", placeholder="Enter filename or type...")
                    
                    if search_term:
                        mask = docs_df['filename'].str.contains(search_term, case=False, na=False) | \
                               docs_df['doc_type'].str.contains(search_term, case=False, na=False)
                        filtered_df = docs_df[mask]
                    else:
                        filtered_df = docs_df
                    
                    st.dataframe(
                        filtered_df, 
                        use_container_width=True,
                        column_config={
                            "id": st.column_config.NumberColumn("ID", width="small"),
                            "filename": st.column_config.TextColumn("Filename", width="large"),
                            "doc_type": st.column_config.TextColumn("Type", width="small"),
                        }
                    )
                    
                    st.caption(f"Showing {len(filtered_df)} of {len(docs_df)} documents")
                else:
                    st.info("📝 No documents uploaded yet. Upload some documents to get started!")
                    
            except Exception as e:
                st.error(f"Error loading documents: {e}")
    
    with tab2:
        st.header("👥 User Management")
        
        # Add new user section
        with st.expander("➕ Add New User"):
            with st.form("add_user_form"):
                col_user, col_pass, col_role = st.columns(3)
                
                with col_user:
                    new_username = st.text_input("Username", placeholder="Enter username")
                with col_pass:
                    new_password = st.text_input("Password", type="password", placeholder="Enter password")
                with col_role:
                    new_role = st.selectbox("Role", ["user", "admin"])
                
                if st.form_submit_button("👤 Add User", use_container_width=True):
                    if not new_username or not new_password:
                        st.error("Please fill in both username and password")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters long")
                    elif db_manager.add_user(new_username, new_password, new_role):
                        st.success(f"✅ User '{new_username}' added successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to add user (username might already exist)")
        
        # User list and management
        st.subheader("👥 Current Users")
        try:
            users = db_manager.get_all_users()
            
            if users:
                for user in users:
                    with st.container():
                        col1, col2, col3, col4, col5 = st.columns([3, 1, 2, 2, 1])
                        
                        with col1:
                            st.write(f"**{user['username']}**")
                        with col2:
                            role_color = "🔴" if user['role'] == 'admin' else "🟢"
                            st.write(f"{role_color} {user['role']}")
                        with col3:
                            created_date = user['created_at'][:10] if user['created_at'] else 'N/A'
                            st.write(f"📅 {created_date}")
                        with col4:
                            last_login = user.get('last_login', 'Never')
                            if last_login and last_login != 'Never':
                                last_login = last_login[:10]
                            st.write(f"🕒 {last_login}")
                        with col5:
                            if user['username'] != st.session_state.user['username']:
                                if st.button("🗑️", key=f"delete_{user['username']}", help="Delete user"):
                                    if db_manager.delete_user(user['username'], st.session_state.user['username']):
                                        st.success(f"✅ User '{user['username']}' deleted!")
                                        st.rerun()
                                    else:
                                        st.error(f"❌ Failed to delete user '{user['username']}'")
                            else:
                                st.write("👤 You")
                        
                        # Show user statistics
                        if hasattr(db_manager, 'get_user_statistics'):
                            user_stats = db_manager.get_user_statistics(user['username'])
                            with st.expander(f"📊 Stats for {user['username']}", expanded=False):
                                stat_col1, stat_col2 = st.columns(2)
                                with stat_col1:
                                    st.metric("Total Messages", user_stats.get('total_messages', 0))
                                with stat_col2:
                                    st.metric("This Week", user_stats.get('messages_this_week', 0))
                        
                        st.divider()
            else:
                st.info("👤 No users found in the system.")
                
        except Exception as e:
            st.error(f"Error loading users: {e}")
    
    with tab3:
        st.header("📊 System Analytics")
        
        try:
            # Database statistics
            stats = db_manager.get_database_info()
            
            # System health metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Activity Metrics")
                st.metric("Messages (24h)", stats.get('messages_last_24h', 0))
                st.metric("New Documents (7d)", stats.get('documents_last_7d', 0))
                st.metric("Database Size", f"{stats.get('database_size_mb', 0)} MB")
            
            with col2:
                st.subheader("🎯 Content Distribution")
                
                # Create a simple chart if we have data
                if stats.get('documents_count', 0) > 0:
                    chart_data = {
                        'Content Type': ['Documents', 'Error Codes', 'Images', 'Embeddings'],
                        'Count': [
                            stats.get('documents_count', 0),
                            stats.get('error_codes_count', 0),
                            stats.get('images_count', 0),
                            stats.get('embeddings_count', 0)
                        ]
                    }
                    st.bar_chart(data=pd.DataFrame(chart_data).set_index('Content Type'))
            
            # Recent activity
            st.subheader("📋 Recent Error Codes")
            conn = sqlite3.connect(db_manager.db_path)
            recent_codes = pd.read_sql_query("""
                SELECT code, description, category, source_doc, created_at
                FROM error_codes 
                ORDER BY created_at DESC 
                LIMIT 10
            """, conn)
            conn.close()
            
            if not recent_codes.empty:
                st.dataframe(recent_codes, use_container_width=True)
            else:
                st.info("🔍 No error codes found in the database.")
                
        except Exception as e:
            st.error(f"Error loading analytics: {e}")
    
    with tab4:
        st.header("📁 Bulk Document Processing")
        
        # Load from project folder
        with st.expander("📂 Load from Project Directory", expanded=True):
            folder_path = getattr(doc_processor, 'documents_folder', './documents')
            st.info(f"📁 Documents folder: `{folder_path}`")
            
            col_load, col_refresh = st.columns(2)
            with col_load:
                if st.button("🔄 Load All Documents", type="primary", use_container_width=True):
                    with st.spinner("Processing documents from folder..."):
                        if hasattr(doc_processor, 'load_documents_from_folder'):
                            success = doc_processor.load_documents_from_folder()
                            if success:
                                st.balloons()
                                st.success("🎉 Documents loaded successfully!")
                            else:
                                st.warning("⚠️ No new documents found or processing failed.")
                        else:
                            st.error("Bulk loading not supported by current document processor")
            
            with col_refresh:
                if st.button("🔍 Show Folder Contents", use_container_width=True):
                    try:
                        if os.path.exists(folder_path):
                            files = [f for f in os.listdir(folder_path) 
                                   if f.lower().endswith(('.pdf', '.docx', '.txt'))]
                            
                            if files:
                                st.subheader("📄 Available Files")
                                for file in files:
                                    file_path = os.path.join(folder_path, file)
                                    if os.path.exists(file_path):
                                        file_size = os.path.getsize(file_path)
                                        size_str = f"{file_size:,} bytes"
                                        if file_size > 1024*1024:
                                            size_str = f"{file_size/(1024*1024):.1f} MB"
                                        elif file_size > 1024:
                                            size_str = f"{file_size/1024:.1f} KB"
                                        
                                        st.write(f"📄 **{file}** ({size_str})")
                            else:
                                st.info("📂 No compatible files found in the documents folder.")
                        else:
                            st.warning(f"📂 Documents folder '{folder_path}' does not exist.")
                    except Exception as e:
                        st.error(f"Error reading folder contents: {e}")
    
    with tab5:
        st.header("⚙️ System Settings")
        
        # Database management
        with st.expander("🗄️ Database Management"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🧹 Optimize Database", use_container_width=True):
                    if hasattr(db_manager, 'optimize_database'):
                        with st.spinner("Optimizing database..."):
                            if db_manager.optimize_database():
                                st.success("✅ Database optimized successfully!")
                            else:
                                st.error("❌ Database optimization failed!")
                    else:
                        st.info("Database optimization not available")
            
            with col2:
                if st.button("📦 Create Backup", use_container_width=True):
                    if hasattr(db_manager, 'backup_database'):
                        backup_path = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
                        with st.spinner("Creating backup..."):
                            if db_manager.backup_database(backup_path):
                                st.success(f"✅ Backup created: {backup_path}")
                            else:
                                st.error("❌ Backup creation failed!")
                    else:
                        st.info("Database backup not available")
            
            with col3:
                if st.button("🧽 Clean Old Data", use_container_width=True):
                    if hasattr(db_manager, 'clean_old_embeddings'):
                        with st.spinner("Cleaning old data..."):
                            cleaned = db_manager.clean_old_embeddings(30)
                            if cleaned > 0:
                                st.success(f"✅ Cleaned {cleaned} old embeddings!")
                            else:
                                st.info("No old data to clean")
                    else:
                        st.info("Data cleaning not available")
        
        # System information
        with st.expander("ℹ️ System Information"):
            try:
                import sys
                import platform
                
                system_info = {
                    "Python Version": sys.version,
                    "Platform": platform.platform(),
                    "Streamlit Version": st.__version__,
                    "Database Path": db_manager.db_path,
                }
                
                for key, value in system_info.items():
                    st.write(f"**{key}:** {value}")
                    
            except Exception as e:
                st.error(f"Error getting system information: {e}")

def chat_interface(db_manager, chatbot):
    """Enhanced chat interface with better UX and functionality"""
    st.title("🤖 Machine Repair Assistant")
    
    # User welcome header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"Welcome back, **{st.session_state.user['username']}**! 👋")
        st.caption("Ask me about error codes, repair procedures, or technical questions...")
    with col2:
        if st.button("🔄 New Conversation", help="Start a fresh conversation"):
            if "messages" in st.session_state:
                st.session_state.messages = []
            if hasattr(chatbot, 'clear_user_context'):
                chatbot.clear_user_context(st.session_state.user['username'])
            st.rerun()
    
    st.markdown("---")
    
    # Sidebar with enhanced features
    with st.sidebar:
        st.header("💬 Chat Controls")
        
        # Quick help
        with st.expander("💡 Quick Help", expanded=False):
            st.markdown("""
            **Example queries:**
            - `What is error code 1234?`
            - `How to repair hydraulic pump?`
            - `Troubleshooting steps for motor failure`
            - `Show me images for error A0B1`
            """)
        
        # Chat history
        st.header("📚 Recent Conversations")
        try:
            history = db_manager.get_chat_history(st.session_state.user['username'], 5)
            
            if history:
                for i, chat in enumerate(history):
                    with st.expander(f"💬 {chat['message'][:25]}...", expanded=False):
                        st.write(f"**❓ You:** {chat['message']}")
                        st.write(f"**🤖 Bot:** {chat['response'][:200]}...")
                        st.caption(f"🕒 {chat['timestamp']}")
            else:
                st.info("No conversation history yet")
                
        except Exception as e:
            st.error(f"Error loading chat history: {e}")
        
        st.markdown("---")
        
        # Context and history management
        if st.button("🧹 Clear Context & History", use_container_width=True):
            try:
                # Clear context memory
                if hasattr(chatbot, 'clear_user_context'):
                    chatbot.clear_user_context(st.session_state.user['username'])
                
                # Clear chat history from database
                if db_manager.clear_user_chat_history(st.session_state.user['username']):
                    if "messages" in st.session_state:
                        st.session_state.messages = []
                    
                    st.success("✅ Context and history cleared!")
                    st.rerun()
                else:
                    st.error("❌ Failed to clear history")
            except Exception as e:
                st.error(f"Error clearing history: {e}")
        
        # System stats for chat
        if hasattr(chatbot, 'get_database_stats'):
            with st.expander("📊 System Stats"):
                try:
                    stats = chatbot.get_database_stats()
                    st.metric("Documents", stats.get('documents', 0))
                    st.metric("Error Codes", stats.get('error_codes', 0)) 
                    st.metric("Images", stats.get('images', 0))
                except Exception as e:
                    st.error(f"Error loading stats: {e}")
    
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages with enhanced formatting
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display images if present
                if "images" in message and message["images"]:
                    st.markdown("**🖼️ Repair Procedure Images:**")
                    
                    # Display images in a grid layout
                    num_images = len(message["images"])
                    if num_images == 1:
                        cols = [1]
                    elif num_images == 2:
                        cols = st.columns(2)
                    else:
                        cols = st.columns(min(3, num_images))
                    
                    for i, img_info in enumerate(message["images"]):
                        try:
                            col_idx = i % len(cols) if isinstance(cols, list) else 0
                            current_col = cols[col_idx] if isinstance(cols, list) else cols
                            
                            with current_col:
                                display_image_from_data(
                                    img_info['data'], 
                                    f"Step {img_info.get('step_number', i+1)}"
                                )
                                
                                # Image details
                                st.caption(f"**Step {img_info.get('step_number', i+1)}**")
                                st.caption(img_info.get('description', 'Repair procedure step'))
                                if 'filename' in img_info:
                                    st.caption(f"*{img_info['filename']}*")
                        
                        except Exception as e:
                            st.error(f"Error displaying image {i+1}: {e}")
    
    # Chat input with enhanced handling
    if prompt := st.chat_input("💬 Ask about error codes, repair procedures, or technical questions..."):
        # Validate input
        if not prompt.strip():
            st.warning("Please enter a question or message")
            return
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing your query with AI..."):
                try:
                    # Get session ID for this conversation
                    session_id = st.session_state.get('session_id', str(uuid.uuid4()))
                    st.session_state.session_id = session_id
                    
                    # Get response from chatbot
                    result = chatbot.generate_response(prompt, st.session_state.user['username'])
                    
                    if isinstance(result, dict):
                        response = result.get("response", "I apologize, but I couldn't generate a proper response.")
                        images_data = result.get("images", [])
                    else:
                        response = str(result) if result else "I apologize, but I couldn't generate a response."
                        images_data = []
                    
                    # Display text response
                    st.markdown(response)
                    
                    # Display images if available
                    if images_data:
                        st.markdown("**🖼️ Repair Procedure Images:**")
                        num_images = len(images_data)
                        if num_images == 1:
                            display_image_from_data(
                                images_data[0]['data'], 
                                f"Step {images_data[0].get('step_number', 1)}: {images_data[0].get('description', 'Repair procedure')}"
                            )
                        else:
                            # Create responsive columns for multiple images
                            cols_per_row = min(3, num_images)
                            for i in range(0, num_images, cols_per_row):
                                cols = st.columns(cols_per_row)
                                for j, col in enumerate(cols):
                                    img_idx = i + j
                                    if img_idx < num_images:
                                        with col:
                                            img_info = images_data[img_idx]
                                            display_image_from_data(
                                                img_info['data'],
                                                f"Step {img_info.get('step_number', img_idx + 1)}"
                                            )
                                            st.caption(f"**Step {img_info.get('step_number', img_idx + 1)}**")
                                            st.caption(img_info.get('description', 'Repair procedure step'))
                    
                    # Add assistant message to chat history
                    message_data = {"role": "assistant", "content": response}
                    if images_data:
                        message_data["images"] = images_data
                    st.session_state.messages.append(message_data)
                    
                    # Save to database
                    try:
                        db_manager.save_chat_history(
                            st.session_state.user['username'],
                            prompt,
                            response,
                            session_id
                        )
                    except Exception as e:
                        st.error(f"Warning: Could not save chat history: {e}")
                        
                except Exception as e:
                    st.error(f"❌ Error generating response: {e}")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": "I apologize, but I encountered an error while processing your request. Please try again or contact support if the issue persists."
                    })



