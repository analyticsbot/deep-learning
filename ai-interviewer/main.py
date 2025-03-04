import streamlit as st
import os
import sys
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables
load_dotenv()

# Run the application
if __name__ == "__main__":
    # Import and run the main application
    from src.app import AIInterviewer
    app = AIInterviewer()
    app.run()
