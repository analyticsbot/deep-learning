import os
import sqlite3
import json
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Database:
    """
    A class to handle database operations for the AI Interviewer app
    """
    
    def __init__(self):
        self.db_path = os.getenv("DATABASE_PATH", "session_data.db")
        self.conn = None
        self.cursor = None
        self.initialize_db()
    
    def initialize_db(self):
        """Initialize the database connection and create tables if they don't exist"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            
            # Create tables if they don't exist
            self._create_tables()
            
            return True
        except Exception as e:
            st.error(f"Database initialization error: {str(e)}")
            return False
    
    def _create_tables(self):
        """Create all necessary tables for the application"""
        # Check if the old asked_questions table exists
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='asked_questions'")
        table_exists = self.cursor.fetchone()
        
        if table_exists:
            # Check if it has the timestamp column
            try:
                self.cursor.execute("SELECT timestamp FROM asked_questions LIMIT 1")
            except sqlite3.OperationalError:
                # Old schema - add timestamp column or create a new table
                st.info("Migrating database schema...")
                try:
                    # Try to add timestamp column to existing table
                    self.cursor.execute("ALTER TABLE asked_questions ADD COLUMN timestamp DATETIME DEFAULT CURRENT_TIMESTAMP")
                    self.conn.commit()
                except sqlite3.OperationalError:
                    # If that fails, work with the existing table as is
                    pass
        else:
            # Create new table with the new schema
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS asked_questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
        
        # Table for user sessions
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                session_id TEXT UNIQUE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Table for interview sessions
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS interview_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                company_name TEXT,
                job_role TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES user_sessions(session_id)
            )
        ''')
        
        # Table for responses
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interview_id INTEGER,
                question TEXT,
                response TEXT,
                feedback TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (interview_id) REFERENCES interview_sessions(id)
            )
        ''')
        
        # Table for question templates
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS question_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                template_name TEXT UNIQUE,
                job_role TEXT,
                industry TEXT,
                questions TEXT,  -- JSON array of questions
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better performance
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_asked_questions_timestamp ON asked_questions(timestamp)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_sessions_session_id ON user_sessions(session_id)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_interview_sessions_session_id ON interview_sessions(session_id)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_responses_interview_id ON responses(interview_id)')
        
        self.conn.commit()
    
    def fetch_asked_questions(self):
        """Fetch all previously asked questions"""
        try:
            # First try with timestamp column (new schema)
            try:
                self.cursor.execute("SELECT question FROM asked_questions ORDER BY timestamp DESC")
                return [row[0] for row in self.cursor.fetchall()]
            except sqlite3.OperationalError:
                # Fall back to old schema without timestamp
                self.cursor.execute("SELECT question FROM asked_questions")
                return [row[0] for row in self.cursor.fetchall()]
        except Exception as e:
            st.error(f"Error fetching asked questions: {str(e)}")
            return []
    
    def add_asked_question(self, question):
        """Add a question to the asked_questions table"""
        try:
            self.cursor.execute("INSERT INTO asked_questions (question) VALUES (?)", (question,))
            self.conn.commit()
            return True
        except Exception as e:
            st.error(f"Error adding asked question: {str(e)}")
            return False
    
    def create_user_session(self, username=None):
        """Create a new user session and return the session ID"""
        try:
            # Generate a unique session ID
            session_id = f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            self.cursor.execute(
                "INSERT INTO user_sessions (username, session_id) VALUES (?, ?)",
                (username, session_id)
            )
            self.conn.commit()
            
            return session_id
        except Exception as e:
            st.error(f"Error creating user session: {str(e)}")
            return None
    
    def create_interview_session(self, session_id, company_name, job_role):
        """Create a new interview session and return the interview ID"""
        try:
            self.cursor.execute(
                "INSERT INTO interview_sessions (session_id, company_name, job_role) VALUES (?, ?, ?)",
                (session_id, company_name, job_role)
            )
            self.conn.commit()
            
            # Get the ID of the newly created interview session
            return self.cursor.lastrowid
        except Exception as e:
            st.error(f"Error creating interview session: {str(e)}")
            return None
    
    def save_response(self, interview_id, question, response, feedback):
        """Save a user response and feedback"""
        try:
            self.cursor.execute(
                "INSERT INTO responses (interview_id, question, response, feedback) VALUES (?, ?, ?, ?)",
                (interview_id, question, response, feedback)
            )
            self.conn.commit()
            return True
        except Exception as e:
            st.error(f"Error saving response: {str(e)}")
            return False
    
    def get_interview_results(self, interview_id):
        """Get all responses and feedback for a specific interview"""
        try:
            self.cursor.execute(
                "SELECT question, response, feedback, timestamp FROM responses WHERE interview_id = ? ORDER BY timestamp",
                (interview_id,)
            )
            results = []
            for row in self.cursor.fetchall():
                results.append({
                    "question": row[0],
                    "response": row[1],
                    "feedback": row[2],
                    "timestamp": row[3]
                })
            return results
        except Exception as e:
            st.error(f"Error getting interview results: {str(e)}")
            return []
    
    def save_question_template(self, template_name, job_role, industry, questions):
        """Save a question template"""
        try:
            # Convert questions list to JSON string
            questions_json = json.dumps(questions)
            
            # Check if template already exists
            self.cursor.execute(
                "SELECT id FROM question_templates WHERE template_name = ?",
                (template_name,)
            )
            existing = self.cursor.fetchone()
            
            if existing:
                # Update existing template
                self.cursor.execute(
                    "UPDATE question_templates SET job_role = ?, industry = ?, questions = ? WHERE template_name = ?",
                    (job_role, industry, questions_json, template_name)
                )
            else:
                # Create new template
                self.cursor.execute(
                    "INSERT INTO question_templates (template_name, job_role, industry, questions) VALUES (?, ?, ?, ?)",
                    (template_name, job_role, industry, questions_json)
                )
            
            self.conn.commit()
            return True
        except Exception as e:
            st.error(f"Error saving question template: {str(e)}")
            return False
    
    def get_question_templates(self, job_role=None, industry=None):
        """Get question templates, optionally filtered by job role and/or industry"""
        try:
            query = "SELECT template_name, job_role, industry, questions FROM question_templates"
            params = []
            
            # Add filters if provided
            if job_role or industry:
                query += " WHERE"
                if job_role:
                    query += " job_role = ?"
                    params.append(job_role)
                    if industry:
                        query += " AND"
                if industry:
                    query += " industry = ?"
                    params.append(industry)
            
            self.cursor.execute(query, params)
            templates = []
            for row in self.cursor.fetchall():
                templates.append({
                    "template_name": row[0],
                    "job_role": row[1],
                    "industry": row[2],
                    "questions": json.loads(row[3])
                })
            return templates
        except Exception as e:
            st.error(f"Error getting question templates: {str(e)}")
            return []
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
