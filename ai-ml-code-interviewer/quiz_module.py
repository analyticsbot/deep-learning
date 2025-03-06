"""
Quiz module for the AI-ML Code Interviewer application.
"""
import streamlit as st
import logging
from typing import Dict, List, Any, Optional

from llm_service import LLMService
import utils
import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuizModule:
    """
    Module for handling quiz functionality.
    """
    
    def __init__(self):
        """Initialize the quiz module."""
        self.llm_service = LLMService()
    
    def render(self):
        """Render the quiz UI."""
        st.header("Multiple Choice Practice")
        
        # Get user settings for multiple-choice questions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_topic = st.selectbox(
                "Select a topic:", 
                config.ML_DL_TOPICS,
                key="quiz_topic"
            )
        
        with col2:
            num_questions = st.number_input(
                "Number of questions:", 
                min_value=1, 
                max_value=config.MAX_QUESTIONS, 
                value=5,
                key="num_questions"
            )
        
        with col3:
            difficulty_level = st.selectbox(
                "Difficulty level:", 
                config.DIFFICULTY_LEVELS,
                key="difficulty_level"
            )
        
        # Generate questions button
        if st.button("Get Questions", key="get_questions_button"):
            with st.spinner("Generating questions..."):
                # Call the LLM service to generate questions
                questions = self.llm_service.generate_quiz(
                    selected_topic, 
                    num_questions, 
                    difficulty_level
                )
                
                if questions:
                    # Store the generated questions in session state
                    utils.save_to_session_state("quiz_questions", questions)
                    utils.save_to_session_state("user_answers", {})
                    utils.save_to_session_state("quiz_submitted", False)
                    st.success(f"Generated {len(questions)} questions!")
                else:
                    st.error("Failed to generate questions. Please try again.")
        
        # Get the generated questions from session state
        questions = utils.get_from_session_state("quiz_questions", [])
        user_answers = utils.get_from_session_state("user_answers", {})
        quiz_submitted = utils.get_from_session_state("quiz_submitted", False)
        
        if questions:
            st.subheader(f"Quiz: {selected_topic} ({difficulty_level})")
            
            # Display each question with options
            for idx, q in enumerate(questions):
                question_key = f"question_{idx}"
                
                st.markdown(f"**Q{idx + 1}: {q['question']}**")
                
                # Extract option letters (A, B, C, D) for easier reference
                option_letters = [opt.split('.')[0].strip() for opt in q['options']]
                
                # Display options with radio buttons
                if not quiz_submitted:
                    selected_option = st.radio(
                        f"Select answer for Q{idx + 1}:",
                        q['options'],
                        key=question_key
                    )
                    
                    # Store the selected option in user_answers
                    selected_letter = selected_option.split('.')[0].strip()
                    user_answers[idx] = [selected_letter]
                    
                    # Update session state
                    utils.save_to_session_state("user_answers", user_answers)
                else:
                    # Display the question with color-coded answers
                    for opt in q['options']:
                        opt_letter = opt.split('.')[0].strip()
                        is_user_selected = opt_letter in user_answers.get(idx, [])
                        is_correct = opt_letter in q['answer']
                        
                        if is_user_selected and is_correct:
                            st.markdown(f"✅ **{opt}**")
                        elif is_user_selected and not is_correct:
                            st.markdown(f"❌ **{opt}**")
                        elif not is_user_selected and is_correct:
                            st.markdown(f"✓ *{opt}*")
                        else:
                            st.markdown(f"  {opt}")
            
            # Submit button
            if not quiz_submitted and st.button("Submit Answers", key="submit_answers_button"):
                utils.save_to_session_state("quiz_submitted", True)
                st.experimental_rerun()
            
            # Show results if quiz is submitted
            if quiz_submitted:
                st.subheader("Quiz Results")
                
                # Calculate score
                correct_count = 0
                for idx, q in enumerate(questions):
                    if set(user_answers.get(idx, [])) == set(q['answer']):
                        correct_count += 1
                
                # Display score
                score_percentage = (correct_count / len(questions)) * 100
                st.markdown(f"### Score: {correct_count}/{len(questions)} ({score_percentage:.1f}%)")
                
                # Add explanation button for each question
                for idx, q in enumerate(questions):
                    if st.button(f"Explain Q{idx + 1}", key=f"explain_q{idx}"):
                        with st.spinner("Generating explanation..."):
                            prompt = f"Explain the following question and why the answer is correct:\n\nQuestion: {q['question']}\n\nOptions: {', '.join(q['options'])}\n\nCorrect Answer(s): {', '.join(q['answer'])}"
                            explanation = self.llm_service.generate_response(prompt)
                            
                            if explanation:
                                st.markdown(explanation)
                            else:
                                st.error("Failed to generate explanation. Please try again.")
                
                # Reset button
                if st.button("Start New Quiz", key="reset_quiz_button"):
                    utils.save_to_session_state("quiz_questions", [])
                    utils.save_to_session_state("user_answers", {})
                    utils.save_to_session_state("quiz_submitted", False)
                    st.experimental_rerun()
