"""
AI-ML Code Interviewer - Main Application

This Streamlit application helps users prepare for machine learning and deep learning interviews by:
- Providing coding practice with adjustable difficulty levels
- Offering multiple-choice questions across various topics
- Generating explanations and feedback using Large Language Models
- Allowing configuration of LLM providers and settings

The application is designed to be modular and extensible, with separate components for:
- Coding practice
- Quiz generation
- Settings management
- Help documentation

Features:
- Code generation and evaluation
- Multiple-choice question generation
- Detailed explanations and feedback
- Configurable LLM providers (LM Studio, OpenAI, Anthropic, Google Gemini)
- Code execution control
- Session history tracking
"""
import logging
import os

import config
import streamlit as st
from coding_module import CodingModule
from dotenv import load_dotenv
from help_module import HelpModule
from llm_service import LLMService
from quiz_module import QuizModule
from settings_module import SettingsModule

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def setup_page():
    """Set up the Streamlit page configuration."""
    st.set_page_config(
        page_title="ML/DL Interview Preparation",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Add custom CSS
    # Add custom CSS from external file
    with open("styles/app.css", "r") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def check_lm_studio_models():
    """
    Check if LM Studio has loaded models and display appropriate message.
    """
    llm_service = LLMService()
    loaded_models = llm_service.check_loaded_models()

    if not loaded_models:
        st.error(
            """
        No models are loaded in LM Studio. Please:
        1. Start LM Studio
        2. Load a model
        3. Restart this application
        """
        )
        st.stop()

    # Update the model selection in settings if needed
    available_models = [model.get("id", "") for model in loaded_models["data"]]
    if config.LLM_MODEL not in available_models:
        if available_models:
            config.LLM_MODEL = available_models[0]  # Use the first available model
            st.info(f"Switched to model: {config.LLM_MODEL}")
            # Update environment variable
            os.environ["LLM_MODEL"] = config.LLM_MODEL
        else:
            st.error("No models available in LM Studio. Please load a model.")
            st.stop()

    # Verify the model is properly set in the LLM service
    try:
        llm_service = LLMService(
            provider=config.LLM_PROVIDER,
            model=config.LLM_MODEL,
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
        )
        logger.info(f"Initialized LLM service with model: {config.LLM_MODEL}")
    except Exception as e:
        logger.error(f"Error initializing LLM service: {str(e)}")
        st.error("Failed to initialize LLM service. Please check your settings and try again.")
        st.stop()


def main():
    """Main application entry point."""
    # Set up the page
    setup_page()

    # Check LM Studio models at startup
    if config.LLM_PROVIDER == "lmstudio":
        check_lm_studio_models()

    # App title and description
    st.title("Machine Learning & Deep Learning Interview Preparation")

    st.markdown(
        """
    This app helps you prepare for machine learning and deep learning interviews by providing:
    - **Coding Practice**: Implement ML/DL algorithms with adjustable difficulty
    - **Multiple Choice Questions**: Test your knowledge with ML/DL quizzes
    """
    )

    # Initialize modules
    coding_module = CodingModule()
    quiz_module = QuizModule()
    settings_module = SettingsModule()
    help_module = HelpModule()

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Coding Practice", "Multiple Choice Questions", "Settings", "Help"]
    )

    # Tab 1: Coding Practice
    with tab1:
        coding_module.render()

    # Tab 2: Multiple Choice Questions
    with tab2:
        quiz_module.render()

    # Tab 3: Settings
    with tab3:
        settings_module.render()

    # Tab 4: Help
    with tab4:
        help_module.render()


if __name__ == "__main__":
    main()
