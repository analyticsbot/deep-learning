import os
import json
import streamlit as st
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

class ExportManager:
    """
    A class to handle exporting interview results to different formats
    """
    
    def __init__(self):
        self.export_dir = "exports"
        
        # Create exports directory if it doesn't exist
        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)
    
    def export_to_pdf(self, interview_data, filename=None):
        """
        Export interview results to a PDF file
        
        Args:
            interview_data (dict): Dictionary containing interview data
            filename (str, optional): Custom filename for the PDF
            
        Returns:
            str: Path to the generated PDF file
        """
        try:
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                company = interview_data.get("company_name", "Company")
                role = interview_data.get("job_role", "Role")
                filename = f"Interview_{company}_{role}_{timestamp}.pdf"
            
            # Ensure the filename has .pdf extension
            if not filename.endswith(".pdf"):
                filename += ".pdf"
            
            # Full path to the PDF file
            pdf_path = os.path.join(self.export_dir, filename)
            
            # Create the PDF document
            doc = SimpleDocTemplate(pdf_path, pagesize=letter)
            styles = getSampleStyleSheet()
            
            # Create custom styles
            title_style = ParagraphStyle(
                'TitleStyle',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=12
            )
            
            heading_style = ParagraphStyle(
                'HeadingStyle',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=10
            )
            
            normal_style = styles['Normal']
            
            # Content elements for the PDF
            elements = []
            
            # Add title
            elements.append(Paragraph(f"Interview Report", title_style))
            elements.append(Spacer(1, 12))
            
            # Add interview details
            elements.append(Paragraph(f"Company: {interview_data.get('company_name', 'N/A')}", normal_style))
            elements.append(Paragraph(f"Role: {interview_data.get('job_role', 'N/A')}", normal_style))
            elements.append(Paragraph(f"Date: {interview_data.get('date', datetime.now().strftime('%Y-%m-%d'))}", normal_style))
            elements.append(Spacer(1, 12))
            
            # Add responses and feedback
            elements.append(Paragraph("Questions and Responses", heading_style))
            elements.append(Spacer(1, 6))
            
            # Process each question and response
            responses = interview_data.get("responses", [])
            for i, response_data in enumerate(responses, 1):
                # Question
                elements.append(Paragraph(f"Q{i}: {response_data.get('question', 'N/A')}", styles['Heading3']))
                
                # Response
                elements.append(Paragraph("Your Response:", styles['Heading4']))
                elements.append(Paragraph(response_data.get('response', 'N/A'), normal_style))
                elements.append(Spacer(1, 6))
                
                # Feedback
                elements.append(Paragraph("Feedback:", styles['Heading4']))
                elements.append(Paragraph(response_data.get('feedback', 'N/A'), normal_style))
                elements.append(Spacer(1, 12))
            
            # Build the PDF
            doc.build(elements)
            
            return pdf_path
        
        except Exception as e:
            st.error(f"Error exporting to PDF: {str(e)}")
            return None
    
    def export_to_text(self, interview_data, filename=None):
        """
        Export interview results to a text file
        
        Args:
            interview_data (dict): Dictionary containing interview data
            filename (str, optional): Custom filename for the text file
            
        Returns:
            str: Path to the generated text file
        """
        try:
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                company = interview_data.get("company_name", "Company")
                role = interview_data.get("job_role", "Role")
                filename = f"Interview_{company}_{role}_{timestamp}.txt"
            
            # Ensure the filename has .txt extension
            if not filename.endswith(".txt"):
                filename += ".txt"
            
            # Full path to the text file
            txt_path = os.path.join(self.export_dir, filename)
            
            # Create the text content
            with open(txt_path, 'w') as f:
                # Write title
                f.write("INTERVIEW REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                # Write interview details
                f.write(f"Company: {interview_data.get('company_name', 'N/A')}\n")
                f.write(f"Role: {interview_data.get('job_role', 'N/A')}\n")
                f.write(f"Date: {interview_data.get('date', datetime.now().strftime('%Y-%m-%d'))}\n\n")
                
                # Write responses and feedback
                f.write("QUESTIONS AND RESPONSES\n")
                f.write("-" * 50 + "\n\n")
                
                # Process each question and response
                responses = interview_data.get("responses", [])
                for i, response_data in enumerate(responses, 1):
                    # Question
                    f.write(f"Q{i}: {response_data.get('question', 'N/A')}\n\n")
                    
                    # Response
                    f.write("Your Response:\n")
                    f.write(f"{response_data.get('response', 'N/A')}\n\n")
                    
                    # Feedback
                    f.write("Feedback:\n")
                    f.write(f"{response_data.get('feedback', 'N/A')}\n\n")
                    f.write("-" * 50 + "\n\n")
            
            return txt_path
        
        except Exception as e:
            st.error(f"Error exporting to text: {str(e)}")
            return None
    
    def export_to_json(self, interview_data, filename=None):
        """
        Export interview results to a JSON file
        
        Args:
            interview_data (dict): Dictionary containing interview data
            filename (str, optional): Custom filename for the JSON file
            
        Returns:
            str: Path to the generated JSON file
        """
        try:
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                company = interview_data.get("company_name", "Company")
                role = interview_data.get("job_role", "Role")
                filename = f"Interview_{company}_{role}_{timestamp}.json"
            
            # Ensure the filename has .json extension
            if not filename.endswith(".json"):
                filename += ".json"
            
            # Full path to the JSON file
            json_path = os.path.join(self.export_dir, filename)
            
            # Write the data to a JSON file
            with open(json_path, 'w') as f:
                json.dump(interview_data, f, indent=4)
            
            return json_path
        
        except Exception as e:
            st.error(f"Error exporting to JSON: {str(e)}")
            return None
