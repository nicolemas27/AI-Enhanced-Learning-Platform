from datetime import datetime
from datetime import timedelta
import hashlib
import re
import time 
from bs4 import BeautifulSoup
from flask import Flask, flash, jsonify, render_template, request, redirect, url_for, session
import numpy as np
from pytube import extract
import requests
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
import google.generativeai as genai
import json, random, logging
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
import os
from bson import ObjectId
from db import db
from pytube import YouTube
import sys
from translate import Translator
import logging
from flask import jsonify
from urllib.parse import quote
from pathlib import Path
from adaptive_learning import EnhancedMemoryModel, ABTestManager, LearningAnalyzer, MemoryModel
from learning_analyzer import ConceptAnalyzer
from flask_login import LoginManager, current_user

services_dir = str(Path(__file__).parent / "services")
if services_dir not in sys.path:
    sys.path.append(services_dir)

from recommendations import generate_recommendations


logging.getLogger('pymongo').setLevel(logging.WARNING)
logging.getLogger('werkzeug').setLevel(logging.INFO)

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

load_dotenv()

app = Flask(__name__)
app.config['DEBUG'] = True
app.config['PROPAGATE_EXCEPTIONS'] = True
app.ab_test_manager = ABTestManager()
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key')  # Fallback for development
login_manager = LoginManager(app)

from auth import auth_bp, User
from admin import admin_bp

app.register_blueprint(admin_bp, url_prefix='/admin')
app.register_blueprint(auth_bp)

from auth import init_login_manager
init_login_manager(app)

MAX_FREE_USES = 3
def get_gemini_api_key():
    """Smart key selection with free tier"""
    if current_user.is_authenticated:
        # Use user's key if available, otherwise fallback to system key
        return current_user.api_key or os.getenv('GEMINI_API_KEY')
    
    # For anonymous users
    if session.get('free_uses', 0) < MAX_FREE_USES:
        session['free_uses'] = session.get('free_uses', 0) + 1
        return os.getenv('GEMINI_API_KEY')
    
    raise PermissionError("Free uses exhausted. Please login to continue")

def validate_gemini_key(key):
    """Basic pattern check without API call"""
    return re.match(r'AIzaSy[A-Za-z0-9-_]{33}', key) is not None

@app.route('/process-key', methods=['POST'])
def handle_temp_key():
    if request.form.get('temp_key'):
        session['temp_key'] = request.form['temp_key']
    return redirect(url_for('index'))

def get_transcript(url):
    """Robust YouTube transcript fetcher with validation"""
    try:
        video_id = extract.video_id(url)
        logging.debug(f"Fetching transcript for: {video_id}")
        
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        try:
            transcript = transcript_list.find_manually_created_transcript(['en'])
        except NoTranscriptFound:
            transcript = transcript_list.find_generated_transcript(['en'])
            
        raw_transcript = transcript.fetch()
        
        # Validate transcript structure
        if not raw_transcript:
            raise ValueError("Empty transcript received")
            
        if not isinstance(raw_transcript, list) or not all(
            isinstance(item, dict) and 'text' in item 
            for item in raw_transcript
        ):
            error_msg = f"Invalid transcript format. First item: {raw_transcript[0]!r}"
            logging.error(error_msg)
            raise ValueError(error_msg)
            
        #logging.debug(f"Transcript list: {transcript_list}")
        logging.debug(f"Transcript type: {type(transcript)}")
        logging.debug(f"First item type: {type(raw_transcript[0])}")
        
        return " ".join([item['text'] for item in raw_transcript])

    except NoTranscriptFound:
        raise Exception("No English subtitles available for this video")
    except Exception as e:
        logging.error(f"Transcript Error: {str(e)}", exc_info=True)
        raise Exception(f"Could not fetch transcript: {str(e)}")
    
def analyze_feature_preference(user_id):
    # Get all relevant events
    activities = db.db.research_metrics.find({
        "user_id": user_id,
        "event_type": {
            "$in": [
                "quiz_start", 
                "quiz_complete",
                "flashcard_view",
                "summary_view",
                "graph_view"
            ]
        }
    })

    # Initialize counters
    feature_counts = {
        'quiz': 0,
        'flashcards': 0,
        'summary': 0,
        'graph': 0
    }

    # Map events to features
    event_mapping = {
        'quiz_start': 'quiz',
        'quiz_complete': 'quiz',
        'flashcard_view': 'flashcards',
        'summary_view': 'summary',
        'graph_view': 'graph'
    }

    # Count activities
    for activity in activities:
        feature = event_mapping.get(activity['event_type'])
        if feature:
            feature_counts[feature] += 1

    # Calculate percentages
    total = sum(feature_counts.values()) or 1  
    percentages = {k: round((v/total)*100) for k,v in feature_counts.items()}

    # Determine dominant feature
    max_feature = max(feature_counts, key=feature_counts.get)
    max_value = feature_counts[max_feature]
    
    # Check for ties
    if list(feature_counts.values()).count(max_value) > 1:
        dominant = 'balanced'
    else:
        dominant = max_feature

    return {
        'dominant': dominant,
        'breakdown': percentages,
        'raw_counts': feature_counts
    }

def generate_quiz(transcript, difficulty, url, previous_questions=None):
    """Generate 5 questions with 4 options with robust error handling"""
    try:
         # Get API key from session
        api_key = get_gemini_api_key()
        if not api_key:
            raise ValueError("Please provide a valid Gemini API key in your profile or temporary key input")
                    
        genai.configure(api_key=api_key)
        # Truncate transcript to prevent overflow and add format instructions
        truncated_transcript = transcript[:3000]  
        random_seed = random.randint(0, 10000)
        
        # Add previous questions to avoid repetition
        previous_questions_text = ""
        if previous_questions:
            previous_questions_text = "Previously generated questions (DO NOT REPEAT THESE):\n"
            for i, q in enumerate(previous_questions):
                previous_questions_text += f"{i+1}. {q['question']}\n"
        
        prompt = f"""Generate EXACTLY 5 NEW quiz questions (DIFFERENT from previous ones) from this transcript in PURE JSON format.
Follow these rules STRICTLY:
1. Output must be valid JSON ONLY - no markdown, text, or comments
2. Structure:
{{
  "questions": [
    {{
      "question": "Clear question text",
      "concept": "<auto-detect concept>",
      "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
      "correct": 0,
      "explanation": "Concise reasoning"
    }}
  ]
}}
3. Difficulty level: {difficulty}
4. Ensure correct answer index (0-3) matches option order
5. Transcript content: {truncated_transcript}
{previous_questions_text}
Random seed: {random_seed}..."""

        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content(prompt)
        raw_response = response.text

        # Clean and validate response
        clean_response = raw_response.strip()
        
        # Remove JSON code blocks if present
        if clean_response.startswith('```json'):
            clean_response = clean_response[7:-3].strip()
        elif clean_response.startswith('```'):
            clean_response = clean_response[3:-3].strip()

        # Validate JSON structure
        if not clean_response:
            raise ValueError("Empty response from API")
        if clean_response[0] != '{' or clean_response[-1] != '}':
            raise ValueError(f"Invalid JSON boundaries: {clean_response[:50]}...")

        quiz_data = json.loads(clean_response)
        
        # Validate quiz structure
        if 'questions' not in quiz_data:
            raise ValueError("Missing 'questions' key in response")
            
        questions = quiz_data['questions']
        if len(questions) != 5:
            raise ValueError(f"Expected 5 questions, got {len(questions)}")

        # Validate each question
        required_keys = ['question', 'options', 'correct', 'explanation']
        for idx, question in enumerate(questions):
            # Check for required fields
            if not all(key in question for key in required_keys):
                missing = [k for k in required_keys if k not in question]
                raise ValueError(f"Question {idx+1} missing keys: {missing}")
                
            # Validate options
            options = question['options']
            if len(options) != 4:
                raise ValueError(f"Question {idx+1} has {len(options)} options (need 4)")
                
            # Validate correct index
            correct_idx = question['correct']
            if not 0 <= correct_idx <= 3:
                raise ValueError(f"Question {idx+1} invalid correct index: {correct_idx}")

            # Shuffle options while preserving correct answer
            correct_answer = options[correct_idx]
            random.shuffle(options)
            new_correct = options.index(correct_answer)
            question['correct'] = new_correct

        return quiz_data

    except json.JSONDecodeError as e:
        logging.error(f"JSON PARSE ERROR: {str(e)}")
        logging.error(f"RAW RESPONSE: {raw_response}")
        logging.error(f"CLEANED RESPONSE: {clean_response}")
        raise RuntimeError("Failed to generate valid quiz format. Please try again.")
        
    except Exception as e:
        logging.error(f"QUIZ GENERATION FAILURE: {str(e)}")
        logging.error(f"PROMPT USED: {prompt}")
        raise RuntimeError(f"Quiz creation failed: {str(e)}") from e

def get_video_title(url):
    try:
        yt = YouTube(url)
        return yt.title
    except Exception as e:
        logging.error(f"Title fetch error: {str(e)}")
        return "Untitled Video"
    
@app.before_request
def initialize_research_session():
    """Essential for tracking anonymous users"""
    if 'user_id' not in session:
        # Generate anonymous ID format: anon_1234
        session['user_id'] = f"anon_{random.randint(1000, 9999)}"
    
    if 'session_start' not in session:
        session['session_start'] = datetime.now().isoformat()

def track_start_time():
    if 'start_time' not in session:
        session['start_time'] = time.time()

def track_session_start():
    if 'session_start' not in session:
        session['session_start'] = datetime.utcnow().isoformat()
        db.log_research_event('session_start', metadata={
            'path': request.path,
            'method': request.method
        })

@app.route('/track-interaction', methods=['POST'])
def track_interaction():
    data = request.json
    analyzer = LearningAnalyzer()
    
    analyzer.track_activity(data['event_type'], data.get('metadata'))
    return jsonify({'status': 'success'})
    
    
@app.route('/generate', methods=['POST'])


def handle_generation():
    try:
        url = request.form.get('url')
        action = request.form.get('action')
        difficulty = request.form.get('difficulty', 'medium')
        session['current_difficulty'] = difficulty
        video_title = get_video_title(url)
        user_id = current_user.id if current_user.is_authenticated else session['user_id']

        if not url:
            raise ValueError("URL is required")
        
        
        # Get transcript
        transcript = get_transcript(url)
        video_id = extract.video_id(url)

        valid_actions = ['quiz', 'summary', 'flashcards']
        if action not in valid_actions:
            raise ValueError(f"Invalid action. Choose from: {', '.join(valid_actions)}")

        db.db.video_progress.update_many(
            {'user_id': user_id, 'video_id': video_id},
            {'$set': {'attempts.$[elem].status': 'expired'}},
            array_filters=[{'elem.status': 'started'}]
        )

        # Common video progress tracking
        video_data = {
            'video_id': video_id,
            'video_title': video_title,
            'attempts': [{
                'action': action,
                'timestamp': datetime.utcnow(),
                'status': 'started',
                'difficulty': difficulty,
                'score': 0,
                'total_questions': 5 if action == 'quiz' else 0,
                
            }]
        }

        db.db.video_progress.update_one(
            {'user_id': user_id, 'video_id': video_id},
            {
                '$push': {'attempts': video_data['attempts'][0]},
                
                '$inc': {'aggregates.total_attempts': 1},
                
                '$set': {
                    'aggregates.last_attempt': datetime.utcnow() 
                }
            },
            upsert=True
        )

        if action == 'quiz':
            quiz_data = generate_quiz(transcript, difficulty, url)
            
            result = db.store_temp_content('quiz', {
                "questions": quiz_data['questions'],
                "transcript": transcript,
                "video_url": url,
                "difficulty": difficulty,
                "video_title": get_video_title(url)
            })
            
            session['content_id'] = str(result.inserted_id)
            session.pop('error', None)
            
            db.log_research_event('quiz_start', metadata={
                'quiz_id': str(result.inserted_id),
                'category': quiz_data.get('category', 'general')
            })

        elif action == 'summary':
            summary = generate_summary(transcript)
            result = db.store_temp_content('summary', {
                "text": summary,
                "transcript": transcript,
                "video_url": url
            })
            db.log_research_event('summary_view', metadata={
                'summary_id': str(result.inserted_id),
                'length': len(summary['overview'])
            })
            session['content_id'] = str(result.inserted_id)

        elif action == 'flashcards':
            flashcards = generate_flashcards(transcript)
            db.track_learning_activity(
                user_id=user_id,
                activity_type='flashcards_viewed',
                metadata={
                    'card_count': len(flashcards),
                    'video_id': video_id,
                    'mastered_cards': 0 
                }
            )
            
            result = db.store_temp_content('flashcards', {
                "cards": flashcards,
                "video_url": url,
                "video_title": video_title
            })
            session['content_id'] = str(result.inserted_id)
            db.log_research_event('flashcard_view', metadata={
                'card_id': str(result.inserted_id),
                'concept': flashcards[0]['front'][:50] if flashcards else ''
            })

        return redirect(url_for(f'show_{action}'))

    except PermissionError as e:
        flash(str(e))
        return redirect(url_for('auth.login'))
    
    except Exception as e:
        logging.error(f"Error in handle_generation: {str(e)}", exc_info=True)
        return render_template('index.html', 
                            error=str(e), 
                            MAX_FREE_USES=MAX_FREE_USES,
                            free_uses_remaining=MAX_FREE_USES - session.get('free_uses', 0))
    
def generate_summary(transcript):
    """Generate structured summary with robust error handling"""
    try:
        api_key = get_gemini_api_key()
        if not api_key:
            raise ValueError("Please provide a valid Gemini API key in your profile or temporary key input")
                    
        genai.configure(api_key=api_key)
        
        # Truncate transcript to prevent API overload
        truncated_transcript = transcript[:3000]  
        
        prompt = f"""Generate video summary in STRICT JSON format:
        {{
            "overview": "3-5 sentence paragraph summary",
            "key_points": [
                {{
                    "concept": "Name of concept",
                    "explanation": "Short explanation of the concept"
                }},
                ...
            ]
        }}
        Transcript: {truncated_transcript}"""


        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content(prompt)
        raw_response = response.text

        # Clean response
        clean_response = raw_response.strip()
        if clean_response.startswith('```json'):
            clean_response = clean_response[7:-3].strip()
        elif clean_response.startswith('```'):
            clean_response = clean_response[3:-3].strip()

        # Validate JSON structure
        if not clean_response:
            raise ValueError("Empty response from API")
        if clean_response[0] != '{' or clean_response[-1] != '}':
            raise ValueError(f"Invalid JSON boundaries: {clean_response[:50]}...")

        summary_data = json.loads(clean_response)
        
        # Validate required fields
        required_keys = ['overview', 'key_points']
        if not all(key in summary_data for key in required_keys):
            missing = [k for k in required_keys if k not in summary_data]
            raise ValueError(f"Missing keys in summary: {missing}")
            
        return summary_data

    except json.JSONDecodeError as e:
        logging.error(f"JSON Error: {str(e)}")
        logging.error(f"Raw Summary Response: {raw_response}")
        raise RuntimeError("Failed to generate valid summary format. Please try again.")
        
    except Exception as e:
        logging.error(f"Summary Generation Failed: {str(e)}")
        raise RuntimeError(f"Summary creation failed: {str(e)}") from e


def generate_flashcards(transcript):
    """Generate Q&A flashcards using Gemini"""
    try:
        api_key = get_gemini_api_key()
        if not api_key:
            raise ValueError("Please provide a valid Gemini API key in your profile or temporary key input")
                    
        genai.configure(api_key=api_key)
        
        prompt = """Generate 10 concise flashcards from this transcript. Each must have:
        - Front: Clear question or term (1 line)
        - Back: Direct answer (1-2 lines max)
        - Optional example (if relevant)
        
        STRICT JSON FORMAT REQUIRED! Example:
        {
            "cards": [
                {
                    "front": "What is photosynthesis?",
                    "back": "Process plants use to convert sunlight into energy",
                    "example": "Leaves turning sunlight into glucose"
                }
            ]
        }
        ONLY RETURN VALID JSON! No markdown or extra text!"""

        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content(prompt + transcript)
        raw_response = response.text
        
        logging.debug(f"Raw Gemini response: {raw_response}")
        
        # Validate response exists
        if not raw_response.strip():
            raise ValueError("Empty response from Gemini")
            
        # Remove potential markdown code fences
        clean_response = raw_response.replace('```json', '').replace('```', '').strip()
        
        cards_data = json.loads(clean_response)
        
        # Validate structure
        if 'cards' not in cards_data:
            raise ValueError("Invalid flashcard format - missing 'cards' key")
            
        if len(cards_data['cards']) < 1:
            raise ValueError("No flashcards generated")
            
        return cards_data['cards']
        
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {str(e)}")
        logging.debug(f"Failed response content: {raw_response}")
        return []
    except Exception as e:
        logging.error(f"Flashcard error: {str(e)}")
        return []

def get_retention_metrics(user_id):
    analyzer = LearningAnalyzer()
    memory_model = EnhancedMemoryModel()
    
    return {
        'estimated_retention': analyzer.calculate_retention_score(user_id),
        'optimal_review_days': analyzer.next_optimal_review_days(user_id),
        'risk_of_forgetting': analyzer.forgetting_risk(user_id),
        'curve_data': {
            'labels': ['Now', '1 Day', '3 Days', '1 Week', '1 Month'],
            'predicted': memory_model.predict_retention_curve(user_id),
            'actual': analyzer.actual_retention_curve(user_id)
        }
    }

@app.route('/')
def index():
    """Home page with input form"""
    return render_template('index.html', MAX_FREE_USES=MAX_FREE_USES,
                         free_uses_remaining=MAX_FREE_USES - session.get('free_uses', 0))

@app.route('/quiz')
def show_quiz():
    content_id = session.get('content_id')
    

    if not content_id:
        return render_template('quiz.html', error="No quiz content found.")

    quiz_data = db.get_temp_content(content_id)
    if not quiz_data or 'data' not in quiz_data or 'questions' not in quiz_data['data']:
        return render_template('quiz.html', error="Failed to load quiz data.")

    print("DEBUG: Retrieved quiz data:", quiz_data)  # Debugging statement

    return render_template('quiz.html', 
                           quiz_data=quiz_data,
                           questions=quiz_data['data']['questions'], 
                           difficulty=quiz_data['data'].get('difficulty', 'medium'),
                           heading="Test Your Knowledge: Interactive Quiz")  # Fixed heading
@app.route('/save-quiz')

@app.after_request
def track_analytics(response):
    try:
        if current_user.is_authenticated:
            user_id = current_user.id
            auth_status = 'authenticated'
        else:
            user_id = session['user_id']
            auth_status = 'anonymous'

        # Track quiz attempts
        if request.endpoint == 'submit_quiz':
            db.db.research_metrics.insert_one({
                'event_type': 'quiz_attempt',
                'user_id': user_id,
                'metadata': {
                    'score': session.get('score', 0),
                    'total': session.get('total', 0),
                    'time_spent': time.time() - session.get('start_time', 0)
                },
                'timestamp': datetime.utcnow()
            })

    except Exception as e:
        logging.error(f"Analytics tracking error: {str(e)}")
    
    return response

def track_session_end(response):
    if 'session_start' in session:
        start_time = datetime.fromisoformat(session['session_start'])
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        db.log_research_event('session_end', metadata={
            'duration': duration,
            'status_code': response.status_code,
            'path': request.path
        })
        session.pop('session_start', None)
    return response

def save_quiz():
    if 'quiz_id' not in session:
        return redirect('/')
    
    quiz_id = ObjectId(session['quiz_id'])
    db.save_quiz_permanently(quiz_id)
    return redirect(url_for('quiz_saved'))

def _calculate_difficulty_score(self, results):
    """Calculate weighted difficulty score (0-100)"""
    correct_times = [q['time_spent'] for q in results if q['is_correct']]
    incorrect_times = [q['time_spent'] for q in results if not q['is_correct']]
    
    if not correct_times or not incorrect_times:
        return 50
    
    diff_ratio = np.mean(incorrect_times) / np.mean(correct_times)
    return min(100, max(0, 50 * diff_ratio))

def _get_review_intervals(self, user_id, concepts):
    """Get time since last review for each concept"""
    intervals = []
    for concept in concepts:
        last_review = db.db.research_metrics.find_one(
            {"user_id": user_id, "metadata.concepts": concept},
            sort=[("timestamp", -1)]
        )
        if last_review:
            intervals.append((datetime.now() - last_review['timestamp']).days)
    return intervals
 
@app.route('/submit', methods=['POST'])
def submit_quiz():
    try:
        analyzer = LearningAnalyzer() 
        if 'content_id' not in session:
            return redirect('/')

        # Get quiz data
        content_id = ObjectId(session['content_id'])
        content = db.get_temp_content(content_id)
        user_id = current_user.id if current_user.is_authenticated else session['user_id']
        
        # Process answers
        answers = {}
        correct_answers_from_wrong = []
        for key in request.form:
            if key.startswith('q'):
                q_index = int(key[1:])
                answers[q_index] = int(request.form[key])

        # Calculate results
        results = []
        score = 0
        for idx, question in enumerate(content['data']['questions']):
            user_ans = answers.get(idx, -1)
            is_correct = user_ans == question['correct']
            if not is_correct:
                correct_answer = question['options'][question['correct']]
                correct_answers_from_wrong.append(correct_answer)
            
            results.append({**question, "user_choice": user_ans, "is_correct": is_correct, "concept": question.get('concept', 'general')})
            score += 1 if is_correct else 0
        session['weak_concepts'] = list(set(correct_answers_from_wrong))[:3]
        weak_concepts = session.get('weak_concepts', [])
        weak_concepts = list(set(weak_concepts))  # Remove duplicates
        weak_concepts = [c for c in weak_concepts if c]  # Remove empty strings

        if score == len(results):
            video_id = extract.video_id(content['data']['video_url'])
            # Get current user ID
            if current_user.is_authenticated:
                user_id = current_user.id
            else:
                user_id = session['user_id']
        
        total = len(results)
        time_spent = time.time() - session.get('start_time', 0)
        transcript = content['data']['transcript']
        video_id = extract.video_id(content['data']['video_url'])
        concept_analyzer = ConceptAnalyzer()
        
        db.track_learning_activity(
            user_id=user_id,
            activity_type='quiz_complete',
            metadata={
                'video_id': video_id,
                'difficulty': content['data'].get('difficulty', 'medium'),
                'question_count': total,  # From calculated results
                'score': score,           # From submission
                'total': total,           # From submission
                'percentage': int((score/total)*100)
            }
        )
        db.db.concept_analysis.insert_one({
            'user_id': user_id,
            'video_id': video_id,
            'concepts': concept_analyzer.extract_concepts(transcript[:3000]),  # Truncate if needed
            'difficulty': concept_analyzer.estimate_difficulty(transcript[:3000]),
            'user_score': score,
            'total_questions': total,
            'timestamp': datetime.now()
        })
        # Store results FIRST
        session['results_id'] = str(db.store_temp_content('results', {
            "score": score,
            "total": len(results),
            "details": results
        }).inserted_id)
        if score == total:
            db.db.user_progress.update_one(
                {'user_id': user_id},
                {'$inc': {'total_mastered': 1}},
                upsert=True
            )    
        try:
            concept_analyzer = ConceptAnalyzer()
            analyzer = LearningAnalyzer()
            transcript = content['data']['transcript']
            video_concepts = concept_analyzer.extract_concepts(transcript)
            question_concepts = concept_analyzer.extract_concepts(" ".join([q['question'] for q in results]))
            # Get weak concepts from session
            weak_concepts = session.get('weak_concepts', [])  # Initialize from session
            weak_concepts = [str(c).lower().strip() for c in weak_concepts]
            weak_concepts = list(set(weak_concepts))
            weak_concepts = [
                str(concept) for concept in 
                concept_analyzer.identify_weak_concepts(results, list(set(video_concepts + question_concepts)))
            ]            
            # After calculating weak_concepts:
            logging.debug(f"Final weak concepts: {weak_concepts}")
            logging.debug(f"Video concepts: {video_concepts}")
            logging.debug(f"Question concepts: {question_concepts}")
            concept_scores = {concept: -1 for concept in weak_concepts}
            for concept, concept_score in concept_scores.items():
                db.db.concept_mastery.update_one(
                    {"user_id": user_id, "concept": concept},
                    {
                        "$inc": {"score": concept_score},
                        "$set": {"timestamp": datetime.now()}
                    },  
                    upsert=True
                )

            # Store progress
          
            progress_data = {
                "user_id": current_user.id if current_user.is_authenticated else session['user_id'],
                "scores": {
                    "last": score,
                    "average": (score / len(results)) if len(results) > 0 else 0,
                    "highest": score  # Will be updated over time
                },
                "activity_stats": {
                    "total_quizzes": 1,
                    "total_time_spent": time.time() - session.get('start_time', 0)
                },
                "weak_concepts": weak_concepts,
                "timestamp": datetime.now()
            }
            weak_concepts = session.get('weak_concepts', [])

            db.db.user_progress.update_one(
                {'user_id': user_id},
                {'$set': {
                    'weak_concepts': weak_concepts
                }},
                upsert=True
            )
            
            session['last_score'] = score
            session['last_total'] = total
            session['weak_concepts'] = weak_concepts

            logging.debug(f"Attempting user_progress update for: {progress_data['user_id']}")
            logging.debug(f"Update data: {progress_data}")
            logging.debug(f"Identified weak concepts: {weak_concepts}")
            
            result = db.db.user_progress.update_one(
                {"user_id": progress_data["user_id"]},
                {
                    '$setOnInsert': {
                        'user_id': progress_data["user_id"],
                        'scores': {
                            'last': 0,
                            'average': 0,
                            'highest': 0
                        },
                        'activity_stats': {
                            'total_quizzes': 0,
                            'total_time_spent': 0
                        },
                        'weak_concepts': [],
                        'timestamp': datetime.now()
                    },
                    '$set': {
                        'scores.last': score,
                        'scores.average': (score / total) if total > 0 else 0,
                        'scores.highest': {'$max': ['$scores.highest', score]},
                        'weak_concepts': weak_concepts,
                       
                        'timestamp': datetime.now()
                    },
                    '$inc': {
                        'activity_stats.total_quizzes': 1,
                        'activity_stats.total_time_spent': time_spent
                    }
                },
                upsert=True
            )

            
            logging.debug(f"Update result - matched: {result.matched_count}, modified: {result.modified_count}, upserted_id: {result.upserted_id}")

            # Store research metrics
            research_entry = {
                "user_id": user_id,
                "video_id": extract.video_id(content['data']['video_url']),
                "video_title": content['data']['video_title'],
                "timestamp": datetime.now(),
                "event_type": "quiz_attempt",
                "metadata": {
                    "score": score,
                    "total": len(results),
                    "time_spent": time.time() - session.get('start_time', 0),
                    "concepts": [q['concept'] for q in results],
                    "difficulty_score": analyzer._calculate_difficulty_score(results),
                    "time_intervals": analyzer._get_review_intervals(user_id, weak_concepts),
                    "questions": [
                        {
                            "question": q['question'],
                            "is_correct": q['is_correct'],
                            "time_spent": time.time() - session.get('question_start', time.time())
                        } for q in results
                    ],
                    "concepts": weak_concepts,
                    "difficulty": content['data'].get('difficulty', 'medium')
                },
                "concept_mastery": concept_analyzer.calculate_mastery_from_results(results),
                "total_errors": sum(1 for q in results if not q['is_correct']),
                "interactions": {
                    "question_retries": session.get('retry_count', 0),
                    "hints_used": session.get('hints_used', 0)
                }
            }
            db.db.research_metrics.insert_one(research_entry)
            difficulty_score = 50 * (sum(1 for q in results if q['is_correct']))/len(results) + \
                  50 * (content['data']['difficulty'] / 3)  # Assuming difficulty 1-3

            research_entry['metadata']['difficulty_score'] = difficulty_score

            video_id = extract.video_id(content['data']['video_url'])
            # Update video_data creation to include aggregates
            video_data = {
                'video_id': video_id,
                'video_title': content['data']['video_title'],
                'attempts': [{
                    'score': score,
                    'total': len(results),
                    'timestamp': datetime.now(),
                    'time_spent': time.time() - session.get('start_time', 0),
                    'total_errors': sum(1 for q in results if not q['is_correct']),
                    'metadata': {
                        'concepts': [c[0] for c in weak_concepts],
                        'difficulty': content['data'].get('difficulty', 'medium')
                    }
                }],
                'aggregates': {
                    'mastery_level': 'mastered' if score == len(results) else 'in-progress'
                }
            }

           
            app.ab_test_manager.log_experiment_result(
            user_id=session['user_id'],
            experiment_name='difficulty_adjustment',  
            score=score,
            total=len(results))

            # Inside submit_quiz function
            mastery_level = 'mastered' if score == total else 'in-progress'

            update_result = db.db.video_progress.update_one(
            {'user_id': user_id, 'video_id': video_id},
            {
                '$set': {
                    'aggregates.mastery_level': mastery_level,
                    'aggregates.last_attempt': datetime.utcnow(),
                    'aggregates.last_score': score
                }
                
            },
            upsert=True
        )
            if update_result.modified_count == 0:
                db.db.video_progress.update_many(
                    {'user_id': user_id, 'video_id': video_id},
                    {'$set': {'attempts.$[elem].status': 'expired'}},
                    array_filters=[{'elem.status': 'started'}]
                )

                # Add new completed attempt
                db.db.video_progress.update_one(
                    {'user_id': user_id, 'video_id': video_id},
                    {
                        '$push': {
                            'attempts': {
                                'action': 'quiz',
                                'status': 'completed',
                                'timestamp': datetime.utcnow(),
                                'score': score,
                                'total_questions': total,
                                'difficulty': content['data'].get('difficulty', 'medium')
                            }
                        },
                        '$set': {
                            'aggregates.mastery_level': 'mastered' if score == total else 'in-progress',
                            'aggregates.last_attempt': datetime.utcnow(),
                            'aggregates.highest_score': score
                        },
                        
                    },
                    upsert=True
                )
            attempt_data = {
                'score': score,
                'total': total,
                'timestamp': datetime.utcnow(),
                'mastered': score == total
            }
            db.db.video_progress.update_one(
                {'user_id': user_id, 'video_id': video_id},
                {
                    '$push': {'attempt_history': attempt_data},
                    '$set': {
                        'aggregates.mastery_level': 'mastered' if score == total else 'in-progress'
                    }
                }
            )

            if current_user.is_authenticated:
                user_id = current_user.id
            else:
                user_id = session['user_id']

            db.create_video_progress(user_id, video_data)

        except Exception as analytics_error:
            logging.error(f"Analytics failed: {str(analytics_error)}")

        return redirect(url_for('show_results'))

    except Exception as main_error:
        logging.error(f"Submission error: {str(main_error)}")
        session['error'] = "Could not process results. Showing partial data."
        return redirect(url_for('show_results'))  # Always show results page

def _calculate_new_difficulty(normalized_score, weak_question_count=0):
    """Improved difficulty adjustment using normalized score (0-1)"""
    # Apply penalty only if more than 1 incorrect question
    penalty = 0.1 * weak_question_count  # 10% penalty per wrong question
    adjusted_score = normalized_score - penalty
    
    if adjusted_score < 0.4:  # <40% after adjustments
        return "easier"
    elif adjusted_score < 0.7:  # 40-70%
        return "similar"
    return "harder"  # >70%

def get_review_schedule(user_id):
    """Get optimized review schedule using multiple models"""
    model = EnhancedMemoryModel()
    concepts = db.get_weak_concepts(user_id)
    return {
        concept: model.get_forgetting_prediction(user_id, concept)
        for concept in concepts
    }

def generate_cognitive_analysis(user_id):
    """Generate textual analysis of learning patterns"""
    try:
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$sort": {"timestamp": -1}},
            {"$limit": 50},
            {"$group": {
                "_id": None,
                "avg_speed": {"$avg": "$time_spent"},
                "error_rate": {"$avg": "$total_errors"},
                "recovery_rate": {"$avg": "$interactions.hints_used"}
            }}
        ]
        
        stats = db.db.research_metrics.aggregate(pipeline).next()
        avg_speed = stats.get('avg_speed', 0) or 0
        error_rate = stats.get('error_rate', 0) or 0
        recovery_rate = stats.get('recovery_rate', 0) or 0
        
        analysis = f"""
        <h5>Cognitive Profile Analysis</h5>
        <ul>
            <li>Average processing speed: {avg_speed:.1f}s per concept</li>
            <li>Error recovery rate: {recovery_rate:.1f}x faster than average</li>
            <li>Consistency score: {(1 - error_rate)*100:.0f}%</li>
        </ul>
        """
        return analysis
        
    except Exception as e:
        logging.error(f"Analysis generation failed: {str(e)}")
        analysis = "Analysis unavailable - complete more sessions"
        return "No analysis available - complete more sessions to unlock insights"

from datetime import timezone

def calculate_streak(user_id):
    try:
        # 1. Get UTC-normalized dates
        sessions = list(db.db.research_metrics.find({
            "user_id": user_id,
            "event_type": {"$in": ["login", "quiz_attempt"]},
            "timestamp": {"$exists": True}
        }).sort("timestamp", 1))  # Sort ascending for forward calculation
        
        if not sessions:
            return 0

        # 2. Track all unique dates in UTC
        unique_dates = {
            session['timestamp'].astimezone(timezone.utc).date()
            for session in sessions
        }
        sorted_dates = sorted(unique_dates)
        
        # 3. Calculate longest consecutive sequence
        max_streak = current_streak = 1
        for i in range(1, len(sorted_dates)):
            if (sorted_dates[i] - sorted_dates[i-1]).days == 1:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1
                
        return max_streak

    except KeyError as e:
        logging.error(f"Missing timestamp in session: {str(e)}")
        return 0

@app.route('/results')
def show_results():
    if 'results_id' not in session:
        return redirect('/')
    
    try:
        results_id = ObjectId(session['results_id'])
        results_data = db.db.temp_content.find_one({"_id": results_id})
        
        if not results_data or results_data['type'] != 'results':
            return redirect('/')

        # Get user progress data
        progress = db.db.user_progress.find_one(
            {"user_id": session['user_id']},
            sort=[("timestamp", -1)]
        ) or {
            'weak_concepts': session.get('weak_concepts', []),
            'scores': {'last': 0}
        }
        questions = [q for q in results_data['data']['details']]
        correct_answers = [q['options'][q['correct']] for q in questions]

        return render_template('results.html',
            results=results_data['data']['details'],
            score=results_data['data']['score'],
            total=results_data['data']['total'],
            results_data=results_data,  # Pass full results data
           
            progress=progress  # Pass progress data
        )
        
    except Exception as e:
        app.logger.error(f"Results error: {str(e)}")
        return redirect('/')

@app.errorhandler(500)
def research_data_fallback(error):
    """Ensure research errors don't break user flow"""
    logging.error("Research system failed: " + str(error))
    return redirect(url_for('show_results'))

@app.route('/mark-reviewed/<path:concept>', methods=['POST'])
def mark_reviewed(concept):
    try:
        user_id = current_user.id if current_user.is_authenticated else session['user_id']
            
        
        return jsonify({'status': 'success'})
    except Exception as e:
        logging.error(f"Error removing concept: {str(e)}")
        return jsonify({'status': 'error'}), 500

@app.route('/summary')
def show_summary():
    
    if 'content_id' not in session:
        return redirect('/')
    
    try:
        content_id = ObjectId(session['content_id'])
        content = db.db.temp_content.find_one({"_id": content_id})
        content = db.get_temp_content(content_id)
        if content and content['type'] == 'summary':
            db.log_research_event('summary_view', metadata={
                'length': len(content['data']['text']['overview'])
            })
        
        if not content or content['type'] != 'summary':
            return redirect('/')
            
        return render_template('summary.html', 
            summary=content['data']['text'],
            transcript=content['data']['transcript'][:100] + "..."  # Excerpt
        )
    
    except Exception as e:
        app.logger.error(f"Summary error: {str(e)}")
        return redirect('/')
    
@app.route('/regenerate-quiz')
def regenerate_quiz():
    try:
        # Get previous quiz content
        content_id = ObjectId(session['content_id'])
        old_content = db.get_temp_content(content_id)
        previous_difficulty = old_content['data'].get('difficulty', 'medium')

        # Get user's previous performance data
        score = 0
        total = 5
        weak_question_count = 0  # Track number of incorrect questions
        
        # Get user ID
        user_id = current_user.id if current_user.is_authenticated else session['user_id']

        # Get results from latest attempt
        results_id = session.get('results_id')
        if results_id:
            results = db.get_temp_content(results_id)
            if results and results['type'] == 'results':
                score = results['data'].get('score', 0)
                total = results['data'].get('total', 5)
                # Count incorrect answers
                weak_question_count = sum(1 for q in results['data']['details'] if not q['is_correct'])

        # Validate score and total
        if total <= 0:
            total = 5  # Prevent division by zero
        normalized_score = score / total

        # Calculate new difficulty
        new_difficulty = _calculate_new_difficulty(
            normalized_score=normalized_score,
            weak_question_count=weak_question_count
        )

        # Get previous questions to avoid repetition
        previous_questions = old_content['data']['questions']

        # Generate updated quiz with new difficulty
        new_quiz = generate_quiz(
            transcript=old_content['data']['transcript'],
            difficulty=old_content['data']['difficulty'],
            url=old_content['data']['video_url'],
            previous_questions=previous_questions  # Pass previous questions
        )

        # Store new quiz with adaptation tracking
        result = db.store_temp_content('quiz', {
            "questions": new_quiz['questions'],
            "transcript": old_content['data']['transcript'],
            "video_url": old_content['data']['video_url'],
            "difficulty": new_difficulty,
            "adaptation_log": {
                "previous_difficulty": previous_difficulty,
                "new_difficulty": new_difficulty,
                "reason": f"Score: {score}/{total} ({int(normalized_score*100)}%), Wrong: {weak_question_count}",
            }
        })

        session['content_id'] = str(result.inserted_id)
        session['current_difficulty'] = new_difficulty

        # Update video progress with adapted attempt
        video_id = extract.video_id(old_content['data']['video_url'])
        user_id = current_user.id if current_user.is_authenticated else session['user_id']

        # Mark previous started attempts as expired
        db.db.video_progress.update_many(
            {'user_id': user_id, 'video_id': video_id},
            {'$set': {'attempts.$[elem].status': 'expired'}},
            array_filters=[{'elem.status': 'started'}]
        )

        # Add new adapted attempt
        new_attempt = {
            'action': 'quiz',
            'timestamp': datetime.utcnow(),
            'status': 'started',
            'difficulty': new_difficulty,
            'adaptation_reason': f"Previous score: {score}/{total}",
            'is_adapted': True  
        }

        db.db.video_progress.update_one(
            {'user_id': user_id, 'video_id': video_id},
            {
                '$push': {'attempts': new_attempt},
                '$inc': {'aggregates.total_attempts': 1},
                '$set': {
                    'aggregates.last_attempt': datetime.utcnow(),
                    'aggregates.current_difficulty': new_difficulty
                }
            },
            upsert=True
        )

        return redirect(url_for('show_quiz'))

    except Exception as e:
        logging.error(f"Regeneration failed: {str(e)}")
        session['error'] = "Could not generate new quiz. Please try again."
        return redirect(url_for('show_quiz'))
    
@app.after_request
def log_response(response):
    app.logger.debug(f"Session data: {dict(session)}")
    app.logger.debug(f"Response status: {response.status}")
    return response

@app.route('/flashcards')
def show_flashcards():
    content_id = session.get('content_id')
    
    if not content_id:
        return redirect('/')
    
        
    content = db.get_temp_content(content_id)
    if not content or content.get('type') != 'flashcards':
        return redirect('/')
    video_id = extract.video_id(content['data']['video_url'])
    flashcards = content['data']['cards']
    db.log_research_event('flashcard_view', metadata={
        'count': len(flashcards),
        'video_id': video_id
    })
    return render_template('flashcards.html',
                        flashcards=content['data']['cards'],
                        video_title=content['data'].get('video_title', ''))


class TranslationService:
    def translate_text(self, text, target_lang='en'):
        try:
            # Encode text properly
            encoded_text = quote(text)
            url = f"http://api.mymemory.translated.net/get?q={encoded_text}&langpair=en|{target_lang}"
            
            response = requests.get(url)
            data = response.json()
            return data['responseData']['translatedText']
            
        except Exception as e:
            logging.error(f"Translation error: {str(e)}")
            return text

translation_service = TranslationService()

@app.route('/translate', methods=['POST'])
def translate_endpoint():
    try:
        data = request.json
        translations = {}
        
        for key, text in data['content'].items():
            translated = translation_service.translate_text(
                text, 
                data['target_lang']
            )
            translations[key] = translated
            
        return jsonify(translations)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

from graph_builder import KnowledgeGraphBuilder
@app.route('/generate-graph', methods=['POST'])
def handle_graph_generation():
    try:
        url = request.form.get('url')
        if not url:
            raise ValueError("URL is required")
            
        transcript = get_transcript(url)
        video_title = get_video_title(url)

        # Store transcript
        transcript_doc = db.store_temp_content('transcript', {
            'text': transcript,
            'video_url': url,
            'video_title': video_title
        })
        
        # Generate graph
        builder = KnowledgeGraphBuilder()
        graph_data = builder.build_from_text(transcript)
        
        # Store graph
        graph_doc = db.store_temp_content('knowledge_graph', {
            'graph': graph_data,
            'original_content_id': transcript_doc.inserted_id,
            'video_url': url,
            'video_title': video_title
        })
        db.log_research_event('graph_view', metadata={
            'graph_id': str(graph_doc.inserted_id),
            'node_count': len(graph_data['nodes'])
        })
        session['graph_id'] = str(graph_doc.inserted_id)
        return redirect(url_for('show_knowledge_graph'))

    except Exception as e:
        logging.error(f"Graph generation error: {str(e)}")
        return render_template('index.html', error=f"Concept map error: {str(e)}")
    
@app.route('/knowledge-graph')
def show_knowledge_graph():
    graph_id = session.get('graph_id')
    if not graph_id:
        return redirect(url_for('index'))
    
    content = db.get_temp_content(ObjectId(graph_id))
    graph_data = content['data']['graph']
    db.log_research_event('graph_view', metadata={
        'node_count': len(graph_data['nodes'])  
    })
    
    # Handle empty graph case
    if not content or not content['data']['graph']['nodes']:
        return render_template('error.html', 
                            message="Could not generate concept map: No concepts found")
    
    return render_template('knowledge_graph.html', 
                         graph_data=content['data']['graph'])

def get_youtube_recommendations(concepts):
    """Scrape YouTube search results for concepts"""
    try:
        query = "+".join(concepts) + "+tutorial"
        url = f"https://www.youtube.com/results?search_query={query}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        videos = []
        for video in soup.select('ytd-video-renderer')[:5]:  # Get first 5 videos
            title = video.select_one('#video-title').text.strip()
            link = video.select_one('#video-title')['href']
            channel = video.select_one('.ytd-channel-name a').text.strip()
            
            videos.append({
                "title": title,
                "url": f"https://youtube.com{link}",
                "type": "video",
                "channel": channel
            })
            
        return videos
        
    except Exception as e:
        logging.error(f"Scraping failed: {str(e)}")
        return []

def default_retention_metrics():
    return {
        'estimated_retention': 0,
        'optimal_review_days': 3,
        'risk_of_forgetting': 50,
        'curve_data': {
            'labels': ['Now', '1 Day', '3 Days', '1 Week', '1 Month'],
            'predicted': [100, 80, 60, 40, 20],
            'actual': [100, 80, 60, 40, 20]
        }
    }

@app.before_request
def assign_experiment_group():
    if 'experiment_group' not in session:
        # Get user ID from session or generate anonymous ID
        user_id = session.get('user_id') or f"anon_{random.randint(1000,9999)}"
        session['experiment_group'] = app.ab_test_manager.assign_group(user_id)

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)
@app.template_filter('format_date')
def format_date(value, fmt=None):
    """Flexible date formatting filter"""
    try:
        if not value:
            return ""
            
        # Use provided format or default
        format_string = fmt or "%b %d, %Y %H:%M"
        return value.strftime(format_string)
        
    except Exception as e:
        logging.error(f"Date formatting error: {str(e)}")
        return ""

@app.template_filter('format_duration')
def format_duration(seconds):
    return f"{int(seconds // 60)}m {int(seconds % 60)}s"

@app.template_filter('extract_youtube_id')
def extract_youtube_id(url):
    return url.split('v=')[-1].split('&')[0]
@app.errorhandler(404)
@app.errorhandler(500)
def handle_errors(error):
    error_messages = {
        404: "Page not found",
        500: "Internal server error"
    }
    message = error_messages.get(error.code, "An error occurred")
    return render_template('error.html', message=message), error.code

@app.route('/progress')
def progress_dashboard():
    if not current_user.is_authenticated:
        return redirect(url_for('auth.login'))
    
    try:
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        db.db.video_progress.update_many(
            {
                'user_id': current_user.id,
                'attempts.status': 'started',
                'attempts.timestamp': {'$lt': cutoff_time}
            },
            {'$set': {'attempts.$[elem].status': 'expired'}},
            array_filters=[{'elem.status': 'started', 'elem.timestamp': {'$lt': cutoff_time}}]
        )
        current_time = datetime.utcnow() 
        video_progress = list(db.db.video_progress.find(
            {'user_id': current_user.id},
            projection={
                'video_id': 1,
                'video_title': 1,
                'aggregates.mastery_level': 1,
                'aggregates.last_attempt': 1,
                'attempts': {'$slice': -1}  # Get last attempt only
            }
        ).sort('aggregates.last_attempt', -1).limit(3))
        
       
        streak_days = calculate_streak(current_user.id)  
        user_progress = db.db.user_progress.find_one(
            {"user_id": current_user.id},
            sort=[("timestamp", -1)]
        ) or {'total_mastered': 0}

        mastered_count = user_progress.get('total_mastered', 0)
        # Fetch raw progress data with null protection
        raw_progress = db.db.user_progress.find_one(
            {"user_id": current_user.id},
            sort=[("timestamp", -1)]
        ) or {}
        concept_mastery = list(db.db.concept_mastery.find(
            {"user_id": current_user.id},
            sort=[("score", -1)]
        ))
        feature_preference = analyze_feature_preference(current_user.id)
        feature_breakdown = feature_preference['breakdown']

        # Safely structure progress data with fallbacks
        progress = {
            'processing_speed': raw_progress.get('processing_speed', 0),
            'recovery_rate': raw_progress.get('recovery_rate', 0),
            'score': max(0, raw_progress.get('score', 0)),
            'current_difficulty': raw_progress.get('current_difficulty', 'medium'),
            'weak_concepts': raw_progress.get('weak_concepts', []),
            'last_updated': raw_progress.get('timestamp', datetime.now())
        }
        pattern_analysis = generate_cognitive_analysis(current_user.id)

        # Get learning sessions history
        sessions = list(db.db.research_metrics.find(
            {"user_id": current_user.id},
            sort=[("timestamp", -1)]
        ).limit(10))  # Limit to last 10 sessions

        # Get video progress with error handling
        try:
            video_progress = db.get_video_progress(current_user.id)
        except Exception as video_error:
            logging.error(f"Video progress error: {str(video_error)}")
            video_progress = []
        
        # Generate memory model predictions
        memory_model = EnhancedMemoryModel()
        video_predictions = {}
        for vp in video_progress:
            try:
                video_predictions[vp['video_id']] = memory_model.get_video_predictions(
                    current_user.id, 
                    vp['video_id']
                )
            except KeyError:
                continue  # Skip invalid video entries

        # Process AB test data
        ab_test_manager = ABTestManager()
        raw_ab_data = ab_test_manager.analyze_results('difficulty_adjustment') or {}
        
        formatted_ab_data = format_ab_data(raw_ab_data)
        analyzer = LearningAnalyzer()
        memory_model = EnhancedMemoryModel()

        retention_metrics = {
            'estimated_retention': analyzer.calculate_retention_score(current_user.id),
            'optimal_review_days': analyzer.next_optimal_review_days(current_user.id),
            'risk_of_forgetting': analyzer.forgetting_risk(current_user.id),
            'curve_data': {
                'labels': ['Now', '1 Day', '3 Days', '1 Week', '1 Month'],
                'predicted': memory_model.predict_retention_curve(current_user.id),
                'actual': analyzer.actual_retention_curve(current_user.id)
            }
        }

        review_schedule = get_review_schedule(current_user.id)
        raw_progress = db.db.user_progress.find_one(
            {"user_id": current_user.id},
            sort=[("timestamp", -1)]
        ) or {}
        weak_concepts = raw_progress.get('weak_concepts', [])

        return render_template('progress.html',
            video_progress=video_progress,
            retention_metrics=retention_metrics,
            predictions=video_predictions,
            ab_test_data=formatted_ab_data,
            progress=progress,
            sessions=sessions,
            pattern_analysis=pattern_analysis,
            review_schedule=get_review_schedule(current_user.id),
            mastered_count=mastered_count,
            streak_days=streak_days,
            feature_preference=feature_preference,
            weak_concepts=weak_concepts,
            feature_breakdown=feature_breakdown,
            current_time=current_time
        )

    except Exception as e:
        retention_metrics = {
            'estimated_retention': 0,
            'optimal_review_days': 3,
            'risk_of_forgetting': 50,
            'curve_data': {
                'labels': ['Now', '1 Day', '3 Days', '1 Week', '1 Month'],
                'predicted': [100, 85, 70, 50, 30],
                'actual': [100, 85, 70, 50, 30]
            }
        }
        logging.error(f"Retention metrics error: {str(e)}")
        logging.error(f"Dashboard error: {str(e)}", exc_info=True)
        return render_template('error.html',
            message="Failed to load analytics data",
            support_email=os.getenv('SUPPORT_EMAIL')
        )

@app.route('/api/learning-curve/<video_id>')
def get_learning_curve(video_id):
    progress = db.db.video_progress.find_one({
        'user_id': current_user.id,
        'video_id': video_id
    })
    return jsonify({
        'scores': [a['score'] for a in progress['attempts']],
        'timestamps': [a['timestamp'].isoformat() for a in progress['attempts']]
    })

@app.route('/debug/retention/<user_id>')
def debug_retention(user_id):
    model = EnhancedMemoryModel()
    analyzer = LearningAnalyzer()
    
    return jsonify({
        "predicted": model.predict_retention_curve(user_id),
        "actual": analyzer.actual_retention_curve(user_id),
        "calculation_data": {
            "decay_rate": model._calculate_personal_decay(user_id),
            "last_activity": model._get_last_activity_date(user_id),
            "average_score": analyzer.calculate_retention_score(user_id)
        }
    })

def analyze_learning_style_from_db(user_id):
    activities = db.db.learning_activities.find({"user_id": user_id})
    return LearningAnalyzer().analyze_style(activities)

def get_review_schedule(user_id):
    """Return optimized review schedule using multiple models"""
    model = MemoryModel()
    concepts = db.get_weak_concepts(user_id)
    return {
        concept: model.schedule_review(user_id, concept)
        for concept in concepts
    }

def export_research_data():
    return db.db.research_metrics.find({}, {
        '_id': 0,
        'timestamp': 1,
        'event_type': 1,
        'metadata': 1,
        'concept_tags': 1,
        'experiment_group': 1,
        'auth_status': 1
    })
    
def anonymize_user_id(user_id):
    return hashlib.sha256(user_id.encode() + os.getenv('SALT').encode()).hexdigest()

def safe_convert(value):
    """Safely convert None to 0"""
    return float(value) if value is not None else 0.0

# In learn.py
def format_ab_data(raw_data):
    """Map groups to correct algorithm names"""
    if not raw_data.get('groups'):
        return default_zero_data()
    
    groups = raw_data['groups']
    comparisons = raw_data.get('comparisons', {})
    
    return {
        'ebbinghaus': parse_group(groups.get('control', {})),
        'act_r': parse_group(groups.get('ML_based', {})),
        'ml': parse_group(groups.get('rule_based', {})),
        'p_values': {
            'control_vs_ml': comparisons.get('control_vs_ML_based', {}).get('p_value', 0),
            'ml_vs_rule': comparisons.get('ML_based_vs_rule_based', {}).get('p_value', 0)
        }
    }

def parse_group(data):
    return {
        'size': data.get('count', 0),
        'accuracy': round(data.get('avg_score', 0) * 100, 1),
        'variance': round(data.get('variance', 0), 3)
    }

def default_zero_data():
    return {
        'ebbinghaus': {'size': 0, 'accuracy': 0, 'variance': 0},
        'act_r': {'size': 0, 'accuracy': 0, 'variance': 0},
        'ml': {'size': 0, 'accuracy': 0, 'variance': 0},
        'p_values': {'control_vs_ml': 0, 'ml_vs_rule': 0}
    }


if __name__ == '__main__':
    # Set up proper logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.StreamHandler(),  # Output to console
            logging.FileHandler('app.log')  # Output to file
        ]
    )
    app.run(debug=True, port=5000)