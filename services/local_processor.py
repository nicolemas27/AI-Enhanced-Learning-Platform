import spacy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en_core_web_sm")

class LocalExpert:
    def __init__(self):
        self.summarizer = LsaSummarizer()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
    def process_query(self, query, transcript):
        # Try 3 local strategies before Gemini
        return (
            self._handle_definition(query, transcript) or
            self._handle_summary(query, transcript) or
            self._handle_context_search(query, transcript) or
            None
        )
    
    def _handle_definition(self, query, transcript):
        if "what is" in query.lower():
            concept = self._extract_concept(query)
            return self._explain_concept(concept, transcript)
        return None
    
    def _extract_concept(self, query):
        doc = nlp(query)
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "PRODUCT"]:
                return ent.text
        return query.split("what is ")[-1].replace("?", "").strip()
    
    def _explain_concept(self, concept, transcript):
        doc = nlp(transcript)
        concept_doc = nlp(concept)
        
        best_sentence = max(
            (sent for sent in doc.sents),
            key=lambda x: concept_doc.similarity(x)
        )
        
        return f"From the video: {best_sentence.text}"
    
    def _handle_summary(self, query, transcript):
        if "summarize" in query.lower():
            parser = PlaintextParser.from_string(transcript, Tokenizer("english"))
            summary = self.summarizer(parser.document, 3)
            return " ".join(str(s) for s in summary)
        return None
    
    def _handle_context_search(self, query, transcript):
        vectors = self.vectorizer.fit_transform([query, transcript])
        most_relevant = vectors[1:].argmax()
        return transcript.split(". ")[most_relevant]