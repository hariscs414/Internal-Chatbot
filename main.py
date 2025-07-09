import streamlit as st
from ui.main_interface import *
def main():
    """Main application function with enhanced routing and session management"""
    
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    
    # Initialize app components
    db_manager, doc_processor, chatbot = init_app()
    
    # Authentication check
    if not st.session_state.authenticated:
        login_page(db_manager)
        return
    
    # Main navigation
    st.sidebar.title("🔧 Navigation")
    st.sidebar.markdown(f"👤 **{st.session_state.user['username']}** ({st.session_state.user['role']})")
    
    # Navigation options based on user role
    if st.session_state.user['role'] == 'admin':
        page_options = ["💬 Chat Assistant", "🛠️ Admin Panel"]
        icons = ["💬", "🛠️"]
    else:
        page_options = ["💬 Chat Assistant"]
        icons = ["💬"]
    
    # Page selection
    selected_page = st.sidebar.selectbox(
        "Choose a page:",
        page_options,
        index=0
    )
    
    # Logout button
    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🚪 Logout", use_container_width=True):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    with col2:
        if st.button("ℹ️ About", use_container_width=True):
            st.sidebar.info("""
            **Machine Repair Chatbot**
            
            🤖 AI-powered assistant for machine repair and maintenance
            
            **Features:**
            - Error code lookup
            - Repair procedures
            - Technical documentation
            - Image-guided repairs
            - Chat history
            
            """)
    
    # Route to appropriate page
    try:
        if selected_page == "💬 Chat Assistant":
            chat_interface(db_manager, chatbot)
        elif selected_page == "🛠️ Admin Panel" and st.session_state.user['role'] == 'admin':
            admin_panel(db_manager, doc_processor)
        else:
            st.error("❌ Access denied or invalid page selection")
            
    except Exception as e:
        st.error(f"❌ Application error: {e}")
        st.info("Please try refreshing the page or contact support if the issue persists.")

# Application styling
def apply_custom_css():
    """Apply custom CSS for better UI/UX"""
    st.markdown("""
    <style>
    /* Main app styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f1f3f6;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 20px;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Success/Error message styling */
    .stSuccess {
        border-radius: 10px;
        border-left: 4px solid #28a745;
    }
    
    .stError {
        border-radius: 10px;
        border-left: 4px solid #dc3545;
    }
    
    .stWarning {
        border-radius: 10px;
        border-left: 4px solid #ffc107;
    }
    
    .stInfo {
        border-radius: 10px;
        border-left: 4px solid #17a2b8;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* File uploader */
    .stFileUploader {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Tables */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    </style>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()