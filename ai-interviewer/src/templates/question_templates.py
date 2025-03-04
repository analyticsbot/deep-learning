import json
import os
import streamlit as st

class QuestionTemplates:
    """
    A class to manage interview question templates
    """
    
    def __init__(self, db):
        self.db = db
        self.default_templates = {
            "Software Engineer": {
                "General": [
                    "Tell me about yourself.",
                    "Why do you want to join our company?",
                    "What are your strengths and weaknesses?",
                    "Describe a challenging project you've worked on.",
                    "Where do you see yourself in five years?"
                ],
                "Technical": [
                    "Explain the difference between an array and a linked list.",
                    "How would you optimize a slow-performing database query?",
                    "Describe your experience with version control systems.",
                    "What is your approach to testing code?",
                    "How do you stay updated with the latest technologies?"
                ]
            },
            "Data Scientist": {
                "General": [
                    "Tell me about yourself.",
                    "Why are you interested in data science?",
                    "Describe a data project you've worked on from start to finish.",
                    "How do you explain complex technical concepts to non-technical stakeholders?",
                    "What data science blogs, podcasts, or resources do you follow?"
                ],
                "Technical": [
                    "Explain the difference between supervised and unsupervised learning.",
                    "How would you handle missing data in a dataset?",
                    "Describe your experience with feature engineering.",
                    "How do you validate a machine learning model?",
                    "What tools and libraries do you use for data visualization?"
                ]
            },
            "Product Manager": {
                "General": [
                    "Tell me about yourself.",
                    "Why are you interested in product management?",
                    "Describe a product you launched from concept to release.",
                    "How do you prioritize features in a product roadmap?",
                    "How do you handle conflicts between engineering and design teams?"
                ],
                "Technical": [
                    "How do you gather and analyze user feedback?",
                    "Describe your experience with agile methodologies.",
                    "How do you measure the success of a product?",
                    "What tools do you use for product analytics?",
                    "How do you balance user needs with business goals?"
                ]
            },
            "UX Designer": {
                "General": [
                    "Tell me about yourself.",
                    "What inspired you to become a UX designer?",
                    "Describe your design process from research to implementation.",
                    "How do you handle feedback and criticism on your designs?",
                    "What design trends are you currently following?"
                ],
                "Technical": [
                    "How do you conduct user research?",
                    "Describe your experience with usability testing.",
                    "What tools do you use for wireframing and prototyping?",
                    "How do you ensure your designs are accessible?",
                    "How do you collaborate with developers to implement your designs?"
                ]
            },
            "Marketing Manager": {
                "General": [
                    "Tell me about yourself.",
                    "What marketing campaigns are you most proud of?",
                    "How do you stay updated with the latest marketing trends?",
                    "Describe your experience with brand development.",
                    "How do you handle marketing budget constraints?"
                ],
                "Technical": [
                    "How do you measure the ROI of marketing campaigns?",
                    "Describe your experience with SEO and SEM.",
                    "What tools do you use for marketing analytics?",
                    "How do you develop a content marketing strategy?",
                    "How do you approach A/B testing for marketing materials?"
                ]
            }
        }
        
        # Initialize default templates in the database
        self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """Initialize default templates in the database if they don't exist"""
        for role, categories in self.default_templates.items():
            for category, questions in categories.items():
                template_name = f"{role} - {category}"
                self.db.save_question_template(template_name, role, category, questions)
    
    def get_templates(self, role=None, category=None):
        """Get question templates, optionally filtered by role and/or category"""
        return self.db.get_question_templates(role, category)
    
    def save_template(self, template_name, role, category, questions):
        """Save a new question template or update an existing one"""
        return self.db.save_question_template(template_name, role, category, questions)
    
    def get_template_questions(self, template_name):
        """Get questions for a specific template"""
        templates = self.db.get_question_templates()
        for template in templates:
            if template["template_name"] == template_name:
                return template["questions"]
        return []
    
    def get_template_roles(self):
        """Get all unique roles from templates"""
        templates = self.db.get_question_templates()
        roles = set()
        for template in templates:
            if template["job_role"]:
                roles.add(template["job_role"])
        return sorted(list(roles))
    
    def get_template_categories(self, role=None):
        """Get all unique categories, optionally filtered by role"""
        templates = self.db.get_question_templates(role)
        categories = set()
        for template in templates:
            if template["industry"]:
                categories.add(template["industry"])
        return sorted(list(categories))
    
    def get_template_names(self, role=None, category=None):
        """Get all template names, optionally filtered by role and/or category"""
        templates = self.db.get_question_templates(role, category)
        return [template["template_name"] for template in templates]
