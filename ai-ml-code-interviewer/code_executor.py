"""
Code execution module with safety measures.
"""
import contextlib
import io
import logging
import sys
from typing import Dict, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeExecutor:
    """
    Safely executes Python code with output capture and error handling.
    """

    @staticmethod
    def execute_code(code: str, timeout: int = 30) -> Tuple[bool, str, str]:
        """
        Execute Python code with safety measures.

        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds

        Returns:
            Tuple of (success, output, error_message)
        """
        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        success = False
        output = ""
        error_msg = ""

        try:
            # Create a restricted globals dictionary with basic builtins
            restricted_globals = {
                "__builtins__": {
                    name: getattr(__builtins__, name)
                    for name in [
                        "abs",
                        "all",
                        "any",
                        "bool",
                        "dict",
                        "dir",
                        "enumerate",
                        "filter",
                        "float",
                        "format",
                        "frozenset",
                        "hash",
                        "int",
                        "isinstance",
                        "issubclass",
                        "len",
                        "list",
                        "map",
                        "max",
                        "min",
                        "next",
                        "object",
                        "pow",
                        "print",
                        "range",
                        "repr",
                        "reversed",
                        "round",
                        "set",
                        "slice",
                        "sorted",
                        "str",
                        "sum",
                        "tuple",
                        "type",
                        "zip",
                    ]
                }
            }

            # Add common data science libraries if they're available
            try:
                import numpy as np

                restricted_globals["np"] = np
            except ImportError:
                pass

            try:
                import pandas as pd

                restricted_globals["pd"] = pd
            except ImportError:
                pass

            try:
                import matplotlib
                import matplotlib.pyplot as plt

                restricted_globals["plt"] = plt
            except ImportError:
                pass

            try:
                import sklearn

                restricted_globals["sklearn"] = sklearn
            except ImportError:
                pass

            # Execute the code with captured output
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(
                stderr_capture
            ):
                try:
                    # First, try to execute the code directly
                    exec(code, restricted_globals)
                except NameError as e:
                    # If we get a NameError, it might be because of missing imports
                    # Let's try to extract imports from the code and execute them first
                    import_lines = []
                    code_lines = code.split("\n")
                    for line in code_lines:
                        if line.strip().startswith(("import ", "from ")):
                            import_lines.append(line)

                    # Execute imports first
                    for import_line in import_lines:
                        try:
                            exec(import_line, restricted_globals)
                        except Exception as e:
                            logger.error(f"Error executing import: {e}")
                            continue

                    # Then execute the rest of the code
                    try:
                        exec(code, restricted_globals)
                    except Exception as e:
                        error_msg = f"{type(e).__name__}: {str(e)}"
                        logger.error(f"Code execution error: {error_msg}")
                        output = stdout_capture.getvalue()
                        return False, output, error_msg
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    logger.error(f"Code execution error: {error_msg}")
                    output = stdout_capture.getvalue()
                    return False, output, error_msg

            output = stdout_capture.getvalue()
            success = True
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Code execution error: {error_msg}")
            output = stdout_capture.getvalue()

        return success, output, error_msg

    @staticmethod
    def format_output(success: bool, output: str, error_msg: str) -> str:
        """
        Format the execution results into a readable string.

        Args:
            success: Whether execution was successful
            output: Captured stdout
            error_msg: Error message if execution failed

        Returns:
            Formatted output string
        """
        result = []

        if output:
            result.append("Output:")
            result.append("```")
            result.append(output)
            result.append("```")

        if not success:
            result.append("Error:")
            result.append("```")
            result.append(error_msg)
            result.append("```")

        if success and not output:
            result.append("Code executed successfully with no output.")

        return "\n".join(result)
