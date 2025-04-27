import nltk, spacy
import numpy as np
import networkx as nx
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.chunk import ne_chunk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from nltk.corpus import wordnet as wn

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
# Download NLTK resources quietly
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

class KnowledgeGraphBuilder:
    def __init__(self):
        self.min_concept_length = 4  # Minimum characters per concept
        self.max_concepts = 40       # Total concepts limit
        self.min_edge_strength = 2   # Minimum co-occurrences for edge
        self.tfidf_features = 25     # Number of TF-IDF features to extract
        self.node_size_multiplier = 35  # Visual scaling factor

    def _get_definition(self, concept):
        """Get definition from WordNet with fallback to spaCy's glossary"""
        try:
            # Try WordNet first
            synsets = wn.synsets(concept)
            if synsets:
                return synsets[0].definition()
            
            # Fallback to spaCy's glossary
            gloss = nlp.vocab[concept.lower()].text
            if gloss:
                return f"A concept related to {gloss}"
                
        except:
            pass
            
        return "Core concept in this context"
    
    def build_from_text(self, text):
        """Main entry point for graph generation"""
        try:
            # Preprocess text
            clean_text = self._preprocess_text(text)
            
            # Extract concepts
            concepts = self._extract_concepts(clean_text)
            logger.info(f"Extracted {len(concepts)} concepts")
            
            # Build and validate graph
            graph = self._build_cooccurrence_graph(clean_text, concepts)
            if len(graph.nodes) < 3:
                raise ValueError("Insufficient nodes for meaningful graph")
                
            return self._format_for_visualization(graph)
            
        except Exception as e:
            logger.error(f"Graph build failed: {str(e)}")
            raise

    def _preprocess_text(self, text):
        """Clean and normalize text input"""
        return ' '.join(text.split()).lower()

    def _extract_concepts(self, text):
        """Hybrid concept extraction with fallbacks"""
        # Get TF-IDF keywords
        tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=self.tfidf_features,
            token_pattern=fr'(?u)\b\w{{{self.min_concept_length},}}\b'
        )
        
        try:
            tfidf.fit([text])
            keywords = tfidf.get_feature_names_out().tolist()
        except ValueError:
            keywords = []
            logger.warning("TF-IDF extraction failed")

        # Get noun phrases with spaCy
        nouns = self._extract_nouns_with_spacy(text)
        nouns = [n for n in nouns if len(n) >= self.min_concept_length][:self.tfidf_features]

        # Combine and validate concepts
        concepts = list(set(keywords + nouns))[:self.max_concepts]
        if not concepts:
            raise ValueError("No valid concepts found after extraction")
            
        return concepts

    def _extract_nouns_with_spacy(self, text):
        """Extract nouns and noun phrases using spaCy"""
        try:
            doc = nlp(text)
            # Extract noun chunks (noun phrases)
            noun_phrases = [chunk.text for chunk in doc.noun_chunks]
            # Extract individual nouns (tokens with noun POS tags)
            nouns = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
            
            # Combine both and return unique nouns
            return list(set(noun_phrases + nouns))
            
        except Exception as e:
            logger.error(f"spaCy noun extraction failed: {str(e)}")
            return []

    def _build_cooccurrence_graph(self, text, concepts):
        """Build graph with validated co-occurrence matrix"""
        sentences = nltk.sent_tokenize(text)
        graph = nx.Graph()
        
        # Add nodes
        for concept in concepts:
            graph.add_node(concept, size=1)
            
        # Build edge weights
        for sent in sentences:
            present_concepts = [c for c in concepts if c in sent]
            for i in range(len(present_concepts)):
                for j in range(i+1, len(present_concepts)):
                    source = present_concepts[i]
                    target = present_concepts[j]
                    weight = graph.get_edge_data(source, target, {'weight': 0})['weight']
                    graph.add_edge(source, target, weight=weight + 1)

        # Prune weak edges
        to_remove = [(u, v) for u, v, d in graph.edges(data=True) 
                    if d['weight'] < self.min_edge_strength]
        graph.remove_edges_from(to_remove)
        
        return graph

    def _format_for_visualization(self, graph):
        """Convert to D3-compatible format with definitions"""
        try:
            pr = nx.pagerank(graph)
            max_pr = max(pr.values()) if pr else 1
            
            return {
                "nodes": [{
                    "id": n,
                    "importance": pr.get(n, 0) / max_pr,
                    "label": n,
                    "size": 10 + (pr.get(n, 0) * self.node_size_multiplier),
                    "definition": self._get_definition(n)  # Add definitions here
                } for n in graph.nodes],
                "links": [{
                    "source": u,
                    "target": v,
                    "strength": d['weight']
                } for u, v, d in graph.edges(data=True)]
            }
        except Exception as e:
            logger.error(f"Formatting failed: {str(e)}")
            return {"nodes": [], "links": []}
