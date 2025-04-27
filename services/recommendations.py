# recommendations.py
from urllib.parse import quote
import logging
from typing import List

from learning_analyzer import ConceptAnalyzer

def generate_recommendations(questions: List[str], correct_answers: List[str]):
    """Generate recommendations using filtered concepts from wrong answers"""
    try:
        analyzer = ConceptAnalyzer()
        
        # Extract concepts from both questions and correct answers
        question_concepts = analyzer.extract_detailed_concepts(" ".join(questions))
        answer_concepts = analyzer.extract_detailed_concepts(" ".join(correct_answers))
        
        # Combine and filter concepts
        all_concepts = _filter_concepts(question_concepts + answer_concepts)
        
        if not all_concepts:
            return []

        # Create focused search phrases with question context
        modifiers = ['explained', 'tutorial', 'examples', 'step by step']
        base_phrases = [f"{c} {m}" for c in all_concepts for m in modifiers]
        
        # Add specific question components
        if questions:
            question_verbs = [
                q['question'].split()[0].lower()  # Get question starter (what/how/why)
                for q in questions if not q.get('is_correct', True)
            ]
            base_phrases += [f"{c} {v}" for c in all_concepts for v in set(question_verbs)]

        search_phrases = list(set(base_phrases))[:5]  # Use most relevant 5

        return [
            {
                "title": "Focused Video Tutorials",
                "url": f"https://www.youtube.com/results?search_query={quote('|'.join(search_phrases))}",
                "icon": "fab fa-youtube",
                "description": "Videos specifically addressing your mistakes"
            },
            {
                "title": "Targeted Courses",
                "url": f"https://www.coursera.org/search?query={quote(' '.join(all_concepts))}+common+mistakes",
                "icon": "fas fa-graduation-cap",
                "description": "Courses focusing on challenging areas"
            },
            {
                "title": "Practice Exercises",
                "url": f"https://www.google.com/search?q={quote(' '.join(search_phrases))}+practice+problems",
                "icon": "fas fa-pencil-alt",
                "description": "Hands-on practice for misunderstood concepts"
            }
        ]
    except Exception as e:
        logging.error(f"Recommendation error: {str(e)}")
        return []

def _filter_concepts(concepts: List[str]) -> List[str]:
    """Remove non-technical terms from both questions and answers"""
    technical_stop_words = {
        'answer', 'option', 'correct', 'incorrect', 'question', 
        'explanation', 'reason', 'example', 'examples', 'statement',
        'following', 'select', 'best', 'process', 'term', 'define'
    }
    standard_stop_words  = {'what', 'which', 'how', 'why', 'the', 'this', 'that', 'transcript', 'video', 'course', 'learn', 'about'
                  ,'introduction', 'overview', 'basic', 'beginner', 'advanced', 'intermediate', 'understand', 'understanding', 'concept', 
                  'concepts', 'explanation', 'explain', 'explaining', 'tutorial', 'guide', 'guidelines', 'summary', 'summarize', 'summarizing', 'summary',
                  'summary', 'summarize', 'summarizing', 'summary', 'summarize', 'the transcript', 'summary', 'summarize', 'summarizing','contain',  'they', 'provide'}
    combined_stop_words = standard_stop_words.union(technical_stop_words)
    
    return [
        concept.lower() for concept in concepts
        if (concept.lower() not in combined_stop_words and
            len(concept) > 3 and
            _is_technical_term(concept))
    ][:5]

def _is_technical_term(concept: str) -> bool:
    """Verify concept is actually technical"""
    technical_indicators = {
        'biology', 'physics', 'chemistry', 'math', 
        'equation', 'theory', 'hypothesis', 'experiment',
        'function', 'variable', 'constant', 'formula'
    }
    return any(indicator in concept.lower() for indicator in technical_indicators)

def _create_search_phrases(concepts: List[str]) -> str:
    """Create focused search queries"""
    modifiers = ['fundamentals', 'explained', 'tutorial']
    return "|".join([
        f"{concept} {modifier}"
        for concept in concepts[:3]
        for modifier in modifiers
    ])
