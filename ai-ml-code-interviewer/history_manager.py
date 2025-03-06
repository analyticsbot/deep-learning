"""
History manager for tracking user sessions and progress.
"""
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HistoryManager:
    """
    Manages user history for coding sessions and quizzes.
    """
    
    def __init__(self, history_dir: str = "user_history"):
        """
        Initialize the history manager.
        
        Args:
            history_dir: Directory to store history files
        """
        self.history_dir = history_dir
        
        # Create history directory if it doesn't exist
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)
            logger.info(f"Created history directory: {history_dir}")
    
    def save_coding_session(self, 
                           topic: str, 
                           intensity: int, 
                           implementation_type: str, 
                           code: str,
                           execution_success: Optional[bool] = None) -> str:
        """
        Save a coding session to history.
        
        Args:
            topic: ML/DL topic
            intensity: Coding intensity (0-100)
            implementation_type: "From Scratch" or "Using Standard Package"
            code: The code content
            execution_success: Whether the code executed successfully
            
        Returns:
            ID of the saved session
        """
        timestamp = datetime.now().isoformat()
        session_id = f"coding_{timestamp.replace(':', '-')}"
        
        session_data = {
            "id": session_id,
            "timestamp": timestamp,
            "type": "coding",
            "topic": topic,
            "intensity": intensity,
            "implementation_type": implementation_type,
            "code": code,
            "execution_success": execution_success
        }
        
        # Save to file
        file_path = os.path.join(self.history_dir, f"{session_id}.json")
        with open(file_path, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        logger.info(f"Saved coding session: {session_id}")
        return session_id
    
    def save_quiz_session(self, 
                         topic: str, 
                         difficulty: str, 
                         questions: List[Dict[str, Any]],
                         user_answers: Dict[int, List[str]],
                         score: int) -> str:
        """
        Save a quiz session to history.
        
        Args:
            topic: ML/DL topic
            difficulty: Difficulty level
            questions: List of question dictionaries
            user_answers: Dictionary mapping question index to user's answers
            score: User's score
            
        Returns:
            ID of the saved session
        """
        timestamp = datetime.now().isoformat()
        session_id = f"quiz_{timestamp.replace(':', '-')}"
        
        session_data = {
            "id": session_id,
            "timestamp": timestamp,
            "type": "quiz",
            "topic": topic,
            "difficulty": difficulty,
            "questions": questions,
            "user_answers": user_answers,
            "score": score,
            "total_questions": len(questions)
        }
        
        # Save to file
        file_path = os.path.join(self.history_dir, f"{session_id}.json")
        with open(file_path, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        logger.info(f"Saved quiz session: {session_id}")
        return session_id
    
    def get_session_history(self, session_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get history of sessions.
        
        Args:
            session_type: Optional filter by session type ("coding" or "quiz")
            
        Returns:
            List of session summaries
        """
        sessions = []
        
        # List all JSON files in the history directory
        for filename in os.listdir(self.history_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.history_dir, filename)
                
                try:
                    with open(file_path, 'r') as f:
                        session_data = json.load(f)
                    
                    # Filter by session type if specified
                    if session_type is None or session_data.get("type") == session_type:
                        # Create a summary without the full code or questions
                        summary = {
                            "id": session_data.get("id"),
                            "timestamp": session_data.get("timestamp"),
                            "type": session_data.get("type"),
                            "topic": session_data.get("topic")
                        }
                        
                        if session_data.get("type") == "coding":
                            summary["intensity"] = session_data.get("intensity")
                            summary["implementation_type"] = session_data.get("implementation_type")
                            summary["execution_success"] = session_data.get("execution_success")
                        elif session_data.get("type") == "quiz":
                            summary["difficulty"] = session_data.get("difficulty")
                            summary["score"] = session_data.get("score")
                            summary["total_questions"] = session_data.get("total_questions")
                        
                        sessions.append(summary)
                except Exception as e:
                    logger.error(f"Error loading session from {file_path}: {str(e)}")
        
        # Sort by timestamp (newest first)
        sessions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return sessions
    
    def get_session_details(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get details of a specific session.
        
        Args:
            session_id: ID of the session
            
        Returns:
            Session data or None if not found
        """
        file_path = os.path.join(self.history_dir, f"{session_id}.json")
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading session from {file_path}: {str(e)}")
                return None
        else:
            logger.warning(f"Session not found: {session_id}")
            return None
