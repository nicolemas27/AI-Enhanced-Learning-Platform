# learning_analyzer.py
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import logging
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

class ConceptAnalyzer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.question_keywords = {'what', 'why', 'how', 'explain', 'difference', 'transcript', 'video', 'course', 'learn', 'about'}
        
        # Combine scikit-learn custom stopwords
        custom_stop_words = list(ENGLISH_STOP_WORDS.union({
            'transcript', 'explain', 'difference', 'video', 'course', 'learn', 'content'
        }))


        # Apply in vectorizer
        self.vectorizer = TfidfVectorizer(stop_words=custom_stop_words, ngram_range=(1, 2))


        self.logger = logging.getLogger(__name__)

    def extract_concepts(self, text):
        """Improved concept extraction with POS filtering"""
        try:
            # First pass with spaCy for NLP analysis
            doc = self.nlp(text)
            
            # Extract technical nouns and proper nouns
            noun_concepts = [
                chunk.text.lower() 
                for chunk in doc.noun_chunks
                if any(tok.pos_ in ['NOUN', 'PROPN'] for tok in chunk)
            ]
            
            # Filter out question keywords and basic terms
            filtered_concepts = [
                concept for concept in noun_concepts
                if concept not in self.question_keywords 
                and len(concept) > 3
            ]
            
            # Second pass with TF-IDF using bigrams
            tfidf_matrix = self.vectorizer.fit_transform([text])
            feature_names = self.vectorizer.get_feature_names_out()
            sorted_indices = tfidf_matrix.toarray().argsort()[0][-10:][::-1]  # Top 10 concepts
            
            # Combine approaches
            tfidf_concepts = [feature_names[i] for i in sorted_indices]
            combined = list(set(filtered_concepts + tfidf_concepts))
            
            return combined[:5]  
            
        except Exception as e:
            self.logger.error(f"Concept extraction failed: {str(e)}")
            return []

    def identify_weak_concepts(self, results, video_concepts):
        """Enhanced weak concept detection using question context"""
        try:
            weak = defaultdict(int)
            incorrect_contexts = [
                f"{q['question']} {q['explanation']} {' '.join(q['options'])}".lower()
                for q in results if not q['is_correct']
            ]
            
            # Extract concepts directly from incorrect questions
            question_concepts = []
            for text in incorrect_contexts:
                question_concepts += self.extract_detailed_concepts(text)
                
            # Combine with video concepts and score
            all_concepts = list(set(question_concepts + video_concepts))
            for concept in all_concepts:
                for context in incorrect_contexts:
                    if concept in context:
                        weak[concept] += 1

            # Normalize and filter
            max_count = max(weak.values(), default=1)
            return sorted(
                [concept for concept, count in weak.items() if count/max_count > 0.3],
                key=lambda x: -weak[x]
            )[:3]  # Return top 3 most relevant
            
        except Exception as e:
            self.logger.error(f"Weak concept analysis failed: {str(e)}")
            return []

    def extract_detailed_concepts(self, text):
        """Specialized extraction for question context"""
        doc = self.nlp(text)
        concepts = []
        
        # Get noun phrases with adjectives
        for chunk in doc.noun_chunks:
            if any(tok.pos_ in ['NOUN', 'PROPN'] for tok in chunk):
                clean_text = chunk.text.lower().strip()
                if len(clean_text) > 3 and clean_text not in self.question_keywords:
                    concepts.append(clean_text)
        
        # Add verb-based concepts for processes
        processes = [tok.lemma_ for tok in doc if tok.pos_ == 'VERB' and tok.dep_ == 'ROOT']
        concepts += processes
        
        return list(set(concepts))
        
    def estimate_difficulty(self, text):
        """Estimate content difficulty level"""
        doc = self.nlp(text)
        # Calculate technical term density
        technical_terms = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
        density = len(technical_terms) / len(list(doc))
        
        if density > 0.25:
            return "advanced"
        elif density > 0.15:
            return "intermediate"
        return "beginner"
