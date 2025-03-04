import streamlit as st
import os
import random
import time
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# Import custom modules
from audio.speech_recognition import SpeechHandler
from llm.llm_provider import LLMProvider
from utils.database import Database
from utils.export import ExportManager
from templates.question_templates import QuestionTemplates
from ui.components import UIComponents

# Load environment variables
load_dotenv()

class AIInterviewer:
    """
    Main application class for the AI Interviewer
    """
    
    def __init__(self):
        # Initialize components
        self.db = Database()
        self.speech_handler = SpeechHandler()
        self.llm_provider = LLMProvider()
        self.export_manager = ExportManager()
        self.templates = QuestionTemplates(self.db)
        self.ui = UIComponents()
        
        # Initialize session state
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state variables"""
        if "asked_questions" not in st.session_state:
            st.session_state["asked_questions"] = self.db.fetch_asked_questions()
        
        if "selected_questions" not in st.session_state:
            st.session_state["selected_questions"] = []
        
        if "reordered_questions" not in st.session_state:
            st.session_state["reordered_questions"] = []
        
        if "question_index" not in st.session_state:
            st.session_state["question_index"] = 0
        
        if "response_text" not in st.session_state:
            st.session_state["response_text"] = ""
        
        if "interview_id" not in st.session_state:
            st.session_state["interview_id"] = None
        
        if "interview_results" not in st.session_state:
            st.session_state["interview_results"] = []
        
        if "session_id" not in st.session_state:
            st.session_state["session_id"] = self.db.create_user_session()
    
    def reorder_questions(self, questions, order_choice, company_name, job_role):
        """Reorder questions based on the selected order choice"""
        if not questions:
            return []
        
        if order_choice == "Random":
            # Create a copy to avoid modifying the original list
            shuffled = questions.copy()
            random.shuffle(shuffled)
            return shuffled
        
        elif order_choice == "Prioritize Based on Role":
            prompt = f"Prioritize these interview questions for a candidate applying for '{job_role}' at '{company_name}':\n"
            prompt += "\n".join([f"- {q}" for q in questions])
            
            # Call LLM to prioritize questions
            reordered_text = self.llm_provider.call_llm(
                prompt=prompt,
                provider=st.session_state.get("llm_provider", "lmstudio"),
                stream=False
            )
            
            # Extract questions from the response
            reordered = [q.strip("- ") for q in reordered_text.split("\n") if q.strip()]
            
            # Make sure all original questions are included
            for q in questions:
                if q not in reordered:
                    reordered.append(q)
            
            return reordered
        
        # Default: return questions in original order
        return questions
    
    async def process_response(self, question, response, company_name, job_role, behavior_role, response_criteria):
        """Process the user's response and generate feedback"""
        # Save the response to the database
        if st.session_state["interview_id"]:
            # Prepare the prompt for the LLM
            prompt = f"""As a {behavior_role} at {company_name}, interviewing for {job_role}, 
            imagine you asked the question '{question}', analyze this answer: '{response}'. 
            Focus on: {response_criteria}. 
            Keep the response brief and use bullet points."""
            
            # Call the LLM for feedback (non-blocking)
            feedback = self.llm_provider.call_llm(
                prompt=prompt,
                provider=st.session_state.get("llm_provider", "lmstudio"),
                stream=False
            )
            
            # Save response and feedback to database
            self.db.save_response(st.session_state["interview_id"], question, response, feedback)
            
            # Add to interview results
            st.session_state["interview_results"].append({
                "question": question,
                "response": response,
                "feedback": feedback,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            return feedback
        
        return "Error: No active interview session."
    
    def run(self):
        """Run the main application"""
        # Display header and sidebar
        self.ui.display_header()
        settings = self.ui.display_sidebar(self.llm_provider.get_available_providers())
        
        # Store selected provider in session state
        st.session_state["llm_provider"] = settings["provider"]
        
        # Display tutorial for first-time users
        self.ui.display_tutorial()
        
        # Initialize selected tab if not present
        if "selected_tab" not in st.session_state:
            st.session_state["selected_tab"] = "Interview Setup"
            
        # Main application tabs
        tab_names = ["Interview Setup", "Interview Session", "Results"]
        tab1, tab2, tab3 = st.tabs(tab_names)
        
        # Get the index of the selected tab
        tab_index = tab_names.index(st.session_state["selected_tab"])
        st.info(f"Current tab: {st.session_state['selected_tab']} (index: {tab_index})")
        
        with tab1:
            if st.session_state["selected_tab"] == "Interview Setup":
                self._run_setup_tab(settings)
        
        with tab2:
            if st.session_state["selected_tab"] == "Interview Session":
                self._run_interview_tab(settings)
        
        with tab3:
            if st.session_state["selected_tab"] == "Results":
                self._run_results_tab(settings)
        
        # Clean up resources when the app is closed
        self.db.close()
    
    def _run_setup_tab(self, settings):
        """Run the Interview Setup tab"""
        st.header("Interview Setup")
        
        # Company and role information
        st.subheader("Company and Role Information")
        company_name = st.text_input("Company Name:", "Google")
        job_role = st.text_input("Job Role:", "Software Engineer")
        
        # Question input options
        st.subheader("Question Selection")
        question_source = st.radio(
            "Select question source:",
            ["Use Template", "Enter Custom Questions"]
        )
        
        if question_source == "Use Template":
            # Get templates from database
            templates = self.templates.get_templates()
            
            # Display template selection UI
            template_questions = self.ui.display_question_templates(templates)
            
            if template_questions:
                st.session_state["selected_questions"] = template_questions
        
        else:  # Enter Custom Questions
            default_questions = """Tell me about yourself.
Why do you want to join this company?
What are your strengths?
Describe a challenging project you've worked on.
Where do you see yourself in five years?"""
            
            question_input = st.text_area(
                "Enter questions, each on a new line:",
                value=default_questions,
                height=200
            )
            
            # Parse questions
            questions = [q.strip() for q in question_input.split("\n") if q.strip()]
            
            if st.button("Save Questions"):
                st.session_state["selected_questions"] = questions
                st.success(f"Saved {len(questions)} questions!")
        
        # Question ordering
        st.subheader("Question Order")
        order_choice = st.radio(
            "Select question order:",
            ["In Order", "Random", "Prioritize Based on Role"]
        )
        
        # Interview behavior settings
        st.subheader("Interview Settings")
        behavior_role = st.text_input("Define the role of the LLM (e.g., HR recruiter):", "HR recruiter")
        response_criteria = st.text_area(
            "Describe response criteria (e.g., brevity, quality):",
            "Provide feedback on brevity, clarity, relevance, and overall quality."
        )
        follow_up = st.checkbox("Allow LLM to ask follow-up questions if applicable")
        
        # Start interview button
        if st.button("Start Interview"):
            st.info("Start Interview button clicked")
            if st.session_state.get("selected_questions", []):
                st.info(f"Selected questions: {len(st.session_state['selected_questions'])}")
                
                # Reorder questions if needed
                st.info(f"Reordering questions using method: {order_choice}")
                reordered = self.reorder_questions(
                    st.session_state["selected_questions"],
                    order_choice,
                    company_name,
                    job_role
                )
                st.info(f"Reordered {len(reordered)} questions")
                
                # Store reordered questions in session state
                st.session_state["reordered_questions"] = reordered
                
                # Reset question index
                st.session_state["question_index"] = 0
                st.info("Reset question index to 0")
                
                # Create a new interview session
                st.info("Creating new interview session in database")
                interview_id = self.db.create_interview_session(
                    st.session_state["session_id"],
                    company_name,
                    job_role
                )
                st.info(f"Created interview session with ID: {interview_id}")
                
                # Store interview ID in session state
                st.session_state["interview_id"] = interview_id
                
                # Clear previous results
                st.session_state["interview_results"] = []
                
                # Store settings in session state
                st.session_state["company_name"] = company_name
                st.session_state["job_role"] = job_role
                st.session_state["behavior_role"] = behavior_role
                st.session_state["response_criteria"] = response_criteria
                st.session_state["follow_up"] = follow_up
                
                # Set the selected tab to Interview Session
                st.info("Switching to Interview Session tab")
                st.session_state["selected_tab"] = "Interview Session"
                
                # Switch to the Interview tab
                st.rerun()
            else:
                st.error("Please select or enter questions first.")
    
    def _run_interview_tab(self, settings):
        """Run the Interview Session tab"""
        st.header("Interview Session")
        
        # Debug information
        st.write(f"Session state keys: {list(st.session_state.keys())}")
        st.write(f"Has reordered_questions: {bool(st.session_state.get('reordered_questions', []))}")
        st.write(f"Question index: {st.session_state.get('question_index', 0)}")
        st.write(f"Interview ID: {st.session_state.get('interview_id')}")
        
        # Check if we have questions to ask
        if not st.session_state.get("reordered_questions", []):
            st.info("Please set up your interview in the 'Interview Setup' tab first.")
            return
        
        # Get current question
        questions = st.session_state["reordered_questions"]
        question_index = st.session_state["question_index"]
        
        # Display progress
        self.ui.display_progress_bar(question_index + 1, len(questions))
        
        # Display current question
        if question_index < len(questions):
            current_question = questions[question_index]
            self.ui.display_question(current_question)
            
            # Add the question to the asked questions list if not already there
            if current_question not in st.session_state["asked_questions"]:
                self.db.add_asked_question(current_question)
                st.session_state["asked_questions"].append(current_question)
            
            # Recording controls
            action = self.ui.display_recording_controls(self.speech_handler)
            
            if action == "start":
                # Speak the question if TTS is enabled
                if settings["enable_tts"]:
                    self.speech_handler.speak_text(current_question)
                
                # Start recording
                self.speech_handler.start_listening()
                st.rerun()
            
            elif action == "stop":
                # Stop recording and get the transcript
                transcript = self.speech_handler.stop_listening()
                st.session_state["response_text"] = transcript
                
                # Display the response
                self.ui.display_response(transcript)
                
                # Process the response (async)
                if transcript:
                    with st.spinner("Generating feedback..."):
                        # Use asyncio to process the response
                        feedback = asyncio.run(self.process_response(
                            current_question,
                            transcript,
                            st.session_state.get("company_name", "Company"),
                            st.session_state.get("job_role", "Role"),
                            st.session_state.get("behavior_role", "HR recruiter"),
                            st.session_state.get("response_criteria", "Provide feedback on brevity, clarity, and relevance.")
                        ))
                        
                        # Display feedback
                        self.ui.display_feedback(feedback)
                        
                        # Move to the next question button
                        if st.button("Next Question"):
                            st.session_state["question_index"] += 1
                            st.session_state["response_text"] = ""
                            st.rerun()
            
            # Display current response if available
            elif st.session_state["response_text"]:
                self.ui.display_response(st.session_state["response_text"])
        
        else:
            # Interview complete
            st.success("Interview complete! View your results in the 'Results' tab.")
            
            # Reset button
            if st.button("Start New Interview"):
                st.session_state["question_index"] = 0
                st.session_state["response_text"] = ""
                st.session_state["reordered_questions"] = []
                st.rerun()
    
    def _run_results_tab(self, settings):
        """Run the Results tab"""
        st.header("Interview Results")
        
        # Check if we have results to display
        if not st.session_state["interview_results"]:
            st.info("Complete your interview to see results here.")
            return
        
        # Display interview details
        st.subheader("Interview Details")
        st.write(f"Company: {st.session_state.get('company_name', 'N/A')}")
        st.write(f"Role: {st.session_state.get('job_role', 'N/A')}")
        st.write(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        
        # Display results
        st.subheader("Questions and Responses")
        for i, result in enumerate(st.session_state["interview_results"], 1):
            with st.expander(f"Q{i}: {result['question']}"):
                st.markdown("**Your Response:**")
                st.write(result["response"])
                
                st.markdown("**Feedback:**")
                st.write(result["feedback"])
        
        # Export options
        st.subheader("Export Results")
        
        # Prepare interview data for export
        interview_data = {
            "company_name": st.session_state.get("company_name", "Company"),
            "job_role": st.session_state.get("job_role", "Role"),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "responses": st.session_state["interview_results"]
        }
        
        # Display export button
        self.ui.display_export_button(
            interview_data,
            self.export_manager,
            settings["export_format"]
        )

# Run the application
if __name__ == "__main__":
    app = AIInterviewer()
    app.run()
