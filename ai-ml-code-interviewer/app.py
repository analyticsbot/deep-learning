"""
AI-ML Code Interviewer - Main Application
"""
import streamlit as st
import logging
import os
from dotenv import load_dotenv

from coding_module import CodingModule
from quiz_module import QuizModule
from settings_module import SettingsModule
import config
import utils

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def setup_page():
    """Set up the Streamlit page configuration."""
    st.set_page_config(
        page_title="ML/DL Interview Preparation",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS
    st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f0ff;
        border-bottom: 2px solid #4e89e8;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Main application entry point."""
    # Set up the page
    setup_page()
    
    # App title and description
    st.title("Machine Learning & Deep Learning Interview Preparation")
    
    st.markdown("""
    This app helps you prepare for machine learning and deep learning interviews by providing:
    - **Coding Practice**: Implement ML/DL algorithms with adjustable difficulty
    - **Multiple Choice Questions**: Test your knowledge with quizzes
    """)
    
    # Initialize modules
    coding_module = CodingModule()
    quiz_module = QuizModule()
    settings_module = SettingsModule()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Coding Practice", "Multiple Choice Questions", "Settings"])
    
    # Tab 1: Coding Practice
    with tab1:
        coding_module.render()
    
    # Tab 2: Multiple Choice Questions
    with tab2:
        quiz_module.render()
    
    # Tab 3: Settings
    with tab3:
        settings_module.render()
        
        # About section
        st.subheader("About")
        
        st.markdown("""
        **AI-ML Code Interviewer** is an interactive tool designed to help you prepare for machine learning and deep learning interviews.
        
        - Practice implementing algorithms from scratch or using standard libraries
        - Test your knowledge with multiple-choice questions
        - Get explanations and feedback to improve your understanding
        
        This application uses a Large Language Model to generate code examples and questions.
        """)

if __name__ == "__main__":
    main()