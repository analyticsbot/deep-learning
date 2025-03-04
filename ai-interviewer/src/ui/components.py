import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time

class UIComponents:
    """
    A class to handle UI components for the AI Interviewer app
    """
    
    def __init__(self):
        # Initialize theme settings
        self.theme = self._get_theme_setting()
    
    def _get_theme_setting(self):
        """Get the current theme setting from session state"""
        if "theme" not in st.session_state:
            st.session_state["theme"] = "light"
        return st.session_state["theme"]
    
    def toggle_theme(self):
        """Toggle between light and dark themes"""
        if self.theme == "light":
            self.theme = "dark"
        else:
            self.theme = "light"
        st.session_state["theme"] = self.theme
        
        # Apply theme CSS
        self._apply_theme_css()
    
    def _apply_theme_css(self):
        """Apply CSS for the current theme"""
        if self.theme == "dark":
            st.markdown("""
            <style>
                .stApp {
                    background-color: #1E1E1E;
                    color: #FFFFFF;
                }
                .stTextInput, .stTextArea, .stSelectbox, .stMultiselect {
                    background-color: #2D2D2D;
                    color: #FFFFFF;
                }
                .stButton > button {
                    background-color: #4CAF50;
                    color: white;
                }
                .stProgress > div > div {
                    background-color: #4CAF50;
                }
                .stMarkdown {
                    color: #FFFFFF;
                }
            </style>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <style>
                .stApp {
                    background-color: #FFFFFF;
                    color: #000000;
                }
                .stButton > button {
                    background-color: #4CAF50;
                    color: white;
                }
                .stProgress > div > div {
                    background-color: #4CAF50;
                }
            </style>
            """, unsafe_allow_html=True)
    
    def display_header(self, title="AI Interviewer"):
        """Display the app header"""
        col1, col2 = st.columns([6, 1])
        with col1:
            st.title(title)
        with col2:
            theme_label = "🌙 Dark" if self.theme == "light" else "☀️ Light"
            if st.button(theme_label):
                self.toggle_theme()
        
        st.markdown("---")
    
    def display_sidebar(self, available_llm_providers):
        """Display the sidebar with settings"""
        with st.sidebar:
            st.header("Settings")
            
            # LLM Provider selection
            st.subheader("LLM Provider")
            provider = st.selectbox(
                "Select LLM Provider",
                options=available_llm_providers,
                index=0
            )
            
            # Language selection
            st.subheader("Language")
            language = st.selectbox(
                "Select Language",
                options=["English", "Spanish", "French", "German", "Chinese", "Japanese"],
                index=0
            )
            
            # Audio settings
            st.subheader("Audio Settings")
            enable_tts = st.checkbox("Enable Text-to-Speech", value=True)
            
            # Export settings
            st.subheader("Export Settings")
            export_format = st.selectbox(
                "Export Format",
                options=["PDF", "Text", "JSON"],
                index=0
            )
            
            # About section
            st.markdown("---")
            st.markdown("### About")
            st.markdown("AI Interviewer helps you practice for job interviews with AI-powered feedback.")
            st.markdown("Version 2.0.0")
            
            return {
                "provider": provider,
                "language": language,
                "enable_tts": enable_tts,
                "export_format": export_format
            }
    
    def display_recording_controls(self, speech_handler):
        """Display recording controls with visual feedback"""
        st.subheader("Recording")
        
        # Display current status
        status = speech_handler.get_status()
        status_color = {
            "idle": "blue",
            "listening": "green",
            "transcribing": "orange",
            "processing": "purple",
            "speaking": "teal"
        }.get(status, "gray")
        
        st.markdown(f"<h4 style='color: {status_color};'>Status: {status.capitalize()}</h4>", unsafe_allow_html=True)
        
        # Recording controls
        col1, col2 = st.columns(2)
        with col1:
            start_disabled = speech_handler.is_listening
            if st.button("🎤 Start Recording", disabled=start_disabled, key="start_recording"):
                return "start"
        
        with col2:
            stop_disabled = not speech_handler.is_listening
            if st.button("⏹️ Stop Recording", disabled=stop_disabled, key="stop_recording"):
                return "stop"
        
        # Display audio visualization if listening
        if speech_handler.is_listening and speech_handler.energy_levels:
            fig = speech_handler.get_audio_visualization()
            if fig:
                st.pyplot(fig)
        
        return None
    
    def display_progress_bar(self, current, total):
        """Display a progress bar for the interview progress"""
        progress = current / total
        st.progress(progress)
        st.markdown(f"Question {current} of {total}")
    
    def display_question(self, question):
        """Display the current interview question"""
        st.markdown(f"## Question")
        st.markdown(f"### {question}")
    
    def display_response(self, response_text):
        """Display the user's response"""
        if response_text:
            st.markdown("## Your Response")
            st.markdown(f"{response_text}")
    
    def display_feedback(self, feedback):
        """Display the feedback from the LLM"""
        if feedback:
            st.markdown("## Feedback")
            st.markdown(f"{feedback}")
    
    def display_export_button(self, interview_data, export_manager, export_format="PDF"):
        """Display a button to export the interview results"""
        if st.button(f"Export as {export_format}"):
            if export_format == "PDF":
                file_path = export_manager.export_to_pdf(interview_data)
            elif export_format == "Text":
                file_path = export_manager.export_to_text(interview_data)
            elif export_format == "JSON":
                file_path = export_manager.export_to_json(interview_data)
            else:
                file_path = None
            
            if file_path:
                st.success(f"Exported to {file_path}")
                return file_path
            else:
                st.error("Export failed")
        
        return None
    
    def display_tutorial(self):
        """Display a tutorial for first-time users"""
        if "tutorial_shown" not in st.session_state:
            st.session_state["tutorial_shown"] = False
        
        if not st.session_state["tutorial_shown"]:
            with st.expander("How to use AI Interviewer", expanded=True):
                st.markdown("""
                ### Welcome to AI Interviewer!
                
                Here's how to use this application:
                
                1. **Setup**: Choose a question template or enter your own questions.
                2. **Configure**: Select the company and job role you're interviewing for.
                3. **Interview**: Click 'Start Recording' to answer each question.
                4. **Feedback**: Receive AI-powered feedback on your responses.
                5. **Export**: Save your interview results for later review.
                
                Click the 'X' on this box to close the tutorial.
                """)
            
            st.session_state["tutorial_shown"] = True
    
    def display_question_templates(self, templates, selected_role=None, selected_category=None):
        """Display question template selection UI"""
        st.subheader("Question Templates")
        
        # Get unique roles and categories
        roles = sorted(list(set([t["job_role"] for t in templates if t["job_role"]])))
        
        # Role selection
        role = st.selectbox("Select Job Role", ["All Roles"] + roles, index=0)
        
        # Filter templates by role
        filtered_templates = templates
        if role != "All Roles":
            filtered_templates = [t for t in templates if t["job_role"] == role]
        
        # Get categories for the selected role
        categories = sorted(list(set([t["industry"] for t in filtered_templates if t["industry"]])))
        
        # Category selection
        category = st.selectbox("Select Category", ["All Categories"] + categories, index=0)
        
        # Filter templates by category
        if category != "All Categories":
            filtered_templates = [t for t in filtered_templates if t["industry"] == category]
        
        # Template selection
        template_names = [t["template_name"] for t in filtered_templates]
        if template_names:
            selected_template = st.selectbox("Select Template", template_names, index=0)
            
            # Get questions for the selected template
            selected_template_data = next((t for t in filtered_templates if t["template_name"] == selected_template), None)
            
            if selected_template_data:
                questions = selected_template_data["questions"]
                
                # Display questions
                st.markdown("### Template Questions")
                for i, q in enumerate(questions, 1):
                    st.markdown(f"{i}. {q}")
                
                # Use template button
                if st.button("Use This Template"):
                    return questions
        else:
            st.info("No templates found for the selected criteria.")
        
        return None
