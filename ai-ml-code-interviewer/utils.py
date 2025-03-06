"""
Utility functions for the AI-ML Code Interviewer application.
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional

import streamlit as st
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_code_for_display(code: str) -> str:
    """
    Format Python code with syntax highlighting for display in Streamlit.

    Args:
        code: Python code to format

    Returns:
        HTML-formatted code with syntax highlighting
    """
    formatter = HtmlFormatter(style="monokai", full=False)
    highlighted = highlight(code, PythonLexer(), formatter)

    # Create CSS for the highlighted code
    css = f"""
    <style>
        {formatter.get_style_defs('.highlight')}
        .highlight {{ background-color: #272822; padding: 10px; border-radius: 5px; }}
        .highlight pre {{ margin: 0; }}
    </style>
    """

    return css + highlighted


def extract_code_blocks(text: str) -> List[str]:
    """
    Extract Python code blocks from markdown text.

    Args:
        text: Markdown text potentially containing code blocks

    Returns:
        List of extracted code blocks
    """
    # Pattern to match Python code blocks in markdown
    pattern = r"```(?:python)?\s*([\s\S]*?)```"

    # Find all matches
    matches = re.findall(pattern, text)

    return matches


def save_to_session_state(key: str, value: Any) -> None:
    """
    Save a value to Streamlit's session state.

    Args:
        key: Key to store the value under
        value: Value to store
    """
    # Always update the value, regardless of whether it exists
    st.session_state[key] = value


def get_from_session_state(key: str, default: Any = None) -> Any:
    """
    Get a value from Streamlit's session state.

    Args:
        key: Key to retrieve
        default: Default value if key doesn't exist

    Returns:
        Value from session state or default
    """
    return st.session_state.get(key, default)


def show_code_with_line_numbers(code: str) -> None:
    """
    Display code with line numbers in Streamlit.

    Args:
        code: Code to display
    """
    lines = code.split("\n")
    line_count = len(lines)
    digits = len(str(line_count))

    # Create HTML for code with line numbers
    html_lines = []
    for i, line in enumerate(lines, 1):
        line_num = str(i).rjust(digits)
        html_lines.append(f'<span style="color:#888888">{line_num}</span> {line}')

    html_code = (
        '<pre style="background-color:#f0f0f0; padding:10px; border-radius:5px; font-family:monospace;">'
        + "<br>".join(html_lines)
        + "</pre>"
    )

    st.markdown(html_code, unsafe_allow_html=True)


def parse_quiz_response(response: str) -> Optional[List[Dict[str, Any]]]:
    """
    Parse a quiz response from the LLM into a structured format.

    Args:
        response: LLM response text

    Returns:
        List of question dictionaries or None if parsing failed
    """
    try:
        # Try to find JSON in the response
        json_start = response.find("[")
        json_end = response.rfind("]") + 1

        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            questions = json.loads(json_str)

            # Validate the structure
            for q in questions:
                if not all(k in q for k in ["question", "options", "answer"]):
                    logger.error("Invalid question format: missing required fields")
                    return None

            return questions
        logger.error("Could not find JSON array in response")
        return None
    except json.JSONDecodeError as e:
        logger.error("Failed to parse response as JSON: %s", str(e))
        return None
    except Exception as e:
        logger.error("Error parsing quiz response: %s", str(e))
        return None
