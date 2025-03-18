"""
Coding practice module for the AI-ML Code Interviewer application.
"""
import logging

import config
import streamlit as st
import utils
from code_executor import CodeExecutor
from history_manager import HistoryManager
from llm_service import LLMService
from streamlit_ace import st_ace

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodingModule:
    """
    Module for handling coding practice functionality.
    """

    def __init__(self):
        """Initialize the coding module."""
        self.llm_service = LLMService()
        self.code_executor = CodeExecutor()
        self.history_manager = HistoryManager()

    def render(self):
        """Render the coding practice UI."""
        st.header("Coding Practice")

        # Step 1: Select a topic
        selected_topic = st.selectbox(
            "Select a topic to practice:", config.ML_DL_TOPICS, key="coding_topic"
        )

        # Step 2: Set the coding intensity
        coding_intensity = st.slider(
            "How much coding do you want to do?",
            0,
            100,
            50,
            key="coding_intensity",
            help="0% = Full code provided, 100% = Skeleton code only",
        )

        # Step 3: Choose between writing from scratch or using standard packages
        implementation_choice = st.selectbox(
            "Implementation approach:",
            ["From Scratch", "Using Standard Package"],
            key="implementation_choice",
        )

        use_standard_package = implementation_choice == "Using Standard Package"

        # Show a "Get Code" button
        if st.button("Get Code", key="get_code_button"):
            with st.spinner("Generating code..."):
                # Call the LLM service to generate code
                code = self.llm_service.generate_code(
                    selected_topic, coding_intensity, use_standard_package
                )

                if code:
                    # Store the generated code in session state
                    utils.save_to_session_state("generated_code", code)
                    st.success("Code generated successfully!")
                else:
                    st.error("Failed to generate code. Please try again.")

        # Get the generated code from session state
        generated_code = utils.get_from_session_state("generated_code", "")

        if generated_code:
            # Extract code blocks if the response is in markdown format
            code_blocks = utils.extract_code_blocks(generated_code)

            if code_blocks:
                # Use the first code block
                code_to_edit = code_blocks[0]
            else:
                # Use the entire response if no code blocks found
                code_to_edit = generated_code

            # Display a header for the code editor
            st.markdown("### Edit your code below:")

            # Use Ace Editor for a professional IDE experience
            user_code = st_ace(
                value=code_to_edit,
                language="python",
                theme="monokai",
                font_size=14,
                key="user_code",
                height=400,
                auto_update=True,
                show_gutter=True,
                wrap=False,
                show_print_margin=True,
                tab_size=4,
                keybinding="vscode",
            )

            # Show a "Run Code" button
            col1, col2 = st.columns([1, 4])

            with col1:
                if st.button("Run Code", key="run_code_button"):
                    if config.ENABLE_CODE_EXECUTION:
                        with st.spinner("Running code..."):
                            # Execute the code
                            success, output, error_msg = self.code_executor.execute_code(user_code)

                            # Store the execution results in session state
                            utils.save_to_session_state(
                                "execution_results",
                                {"success": success, "output": output, "error_msg": error_msg},
                            )
                    else:
                        st.warning(
                            "Code execution is disabled for security reasons. "
                            "You can enable it in the config.py file."
                        )

            with col2:
                if st.button("Copy to Clipboard", key="copy_code_button"):
                    st.info("Code copied to clipboard! (This would work in a deployed app)")

            # Display execution results if available
            execution_results = utils.get_from_session_state("execution_results", None)

            if execution_results:
                st.subheader("Execution Results")

                if execution_results["success"]:
                    st.success("Code executed successfully!")
                else:
                    st.error("Code execution failed!")

                # Format and display the output
                formatted_output = self.code_executor.format_output(
                    execution_results["success"],
                    execution_results["output"],
                    execution_results["error_msg"],
                )

                st.markdown(formatted_output)

                # Save to history
                session_id = self.history_manager.save_coding_session(
                    selected_topic,
                    coding_intensity,
                    implementation_choice,
                    user_code,
                    execution_results["success"],
                )

                st.info(f"Session saved to history (ID: {session_id})")

        # Add a section for code explanation
        st.subheader("Code Explanation")

        if generated_code and st.button("Explain This Code", key="explain_code_button"):
            with st.spinner("Generating explanation..."):
                prompt = (
                    f"Explain the following code in detail, focusing on the ML/DL concepts "
                    f"and implementation details:\n\n{generated_code}"
                )
                explanation = self.llm_service.generate_response(prompt)

                if explanation:
                    st.markdown(explanation)
                else:
                    st.error("Failed to generate explanation. Please try again.")

        # Add a section for history
        st.subheader("Previous Coding Sessions")

        if st.button("View Coding History", key="view_coding_history_button"):
            coding_history = self.history_manager.get_session_history("coding")

            if coding_history:
                st.write(f"Found {len(coding_history)} previous coding sessions:")

                for i, session in enumerate(
                    coding_history[:5]
                ):  # Show only the 5 most recent sessions
                    with st.expander(f"Session {i+1}: {session['topic']} ({session['timestamp']})"):
                        st.write(f"**Topic:** {session['topic']}")
                        st.write(f"**Intensity:** {session['intensity']}%")
                        st.write(f"**Implementation:** {session['implementation_type']}")

                        if session["execution_success"] is not None:
                            status = "✅ Success" if session["execution_success"] else "❌ Failed"
                            st.write(f"**Execution Status:** {status}")

                        # Button to load this session
                        if st.button(f"Load Session {i+1}", key=f"load_session_{session['id']}"):
                            session_details = self.history_manager.get_session_details(
                                session["id"]
                            )
                            if session_details:
                                utils.save_to_session_state(
                                    "generated_code", session_details["code"]
                                )
                                st.experimental_rerun()
            else:
                st.info("No previous coding sessions found.")
