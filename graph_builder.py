import nltk
import spacy
import numpy as np
import networkx as nx
import logging
from collections import Counter
from itertools import combinations
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.corpus import wordnet as wn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download("wordnet",                   quiet=True)
nltk.download("omw-1.4",                   quiet=True)
nltk.download("punkt",                     quiet=True)
nltk.download("punkt_tab",                 quiet=True)
nltk.download("averaged_perceptron_tagger",quiet=True)

nlp = spacy.load("en_core_web_sm")


class KnowledgeGraphBuilder:
    def __init__(self):
        self.min_concept_length    = 4     # minimum characters per concept
        self.max_concepts          = 40    # total concept cap
        self.min_edge_strength     = 1     # lowered from 2 — single co-occurrence counts
        self.top_freq_features     = 25    # high-frequency terms to keep
        self.node_size_multiplier  = 35    # D3 visual scaling factor
        self.semantic_threshold    = 0.55  # Wu-Palmer similarity cutoff for semantic edges

    # ── public entry point ─────────────────────────────────────────────────────

    def build_from_text(self, text: str) -> dict:
        """
        Return a D3-ready dict with 'nodes' and 'links'.
        Links carry a 'type' field: 'cooccurrence' or 'semantic'.
        Two concepts can share both edge types — the stronger one wins.
        """
        try:
            clean_text = self._preprocess_text(text)
            concepts   = self._extract_concepts(text, clean_text)
            logger.info(f"Extracted {len(concepts)} concepts")

            graph = self._build_graph(clean_text, concepts)
            if len(graph.nodes) < 3:
                raise ValueError("Insufficient nodes for a meaningful graph")

            return self._format_for_visualization(graph)

        except Exception as e:
            logger.error(f"Graph build failed: {e}")
            raise

    # ── preprocessing ──────────────────────────────────────────────────────────

    def _preprocess_text(self, text: str) -> str:
        """Normalise whitespace and lowercase — used only for co-occurrence matching."""
        return " ".join(text.split()).lower()

    # ── concept extraction ─────────────────────────────────────────────────────

    def _extract_concepts(self, original_text: str, clean_text: str) -> list[str]:
        """
        Hybrid extraction:
          1. Term frequency on clean_text.
             (Replaces single-doc TF-IDF: with N=1 document IDF=0 for every
              term, so all weights are identical — TF-IDF is degenerate here.)
          2. spaCy noun/noun-phrase extraction on original_text.
             Original casing is required — lowercasing before spaCy degrades
             PROPN detection significantly.
        """
        stop = ENGLISH_STOP_WORDS
        words = [
            w for w in clean_text.split()
            if len(w) >= self.min_concept_length and w not in stop
        ]
        freq_keywords = [w for w, _ in Counter(words).most_common(self.top_freq_features)]

        nouns = self._extract_nouns_with_spacy(original_text)
        nouns_clean = [
            n.lower() for n in nouns
            if len(n) >= self.min_concept_length
        ][:self.top_freq_features]

        concepts = list(set(freq_keywords + nouns_clean))[:self.max_concepts]
        if not concepts:
            raise ValueError("No valid concepts found after extraction")
        return concepts

    def _extract_nouns_with_spacy(self, text: str) -> list[str]:
        """
        Extract noun phrases and individual nouns.
        MUST receive original-cased text — lowercasing before this call
        breaks PROPN (proper noun) detection.
        """
        try:
            doc = nlp(text)
            noun_phrases = [chunk.text for chunk in doc.noun_chunks]
            nouns = [
                token.text for token in doc
                if token.pos_ in ("NOUN", "PROPN") and not token.is_stop
            ]
            return list(set(noun_phrases + nouns))
        except Exception as e:
            logger.error(f"spaCy noun extraction failed: {e}")
            return []

    # ── graph construction ─────────────────────────────────────────────────────

    def _build_graph(self, clean_text: str, concepts: list[str]) -> nx.Graph:
        """
        Build a graph with two complementary edge types:

        1. Co-occurrence edges  — concepts that appear in the same sentence.
           Captures explicit relationships stated in the text.
           Edge weight = number of sentences where both appear.

        2. Semantic edges — concepts linked by WordNet Wu-Palmer similarity.
           Captures implicit relationships (e.g. 'learning' ↔ 'memory')
           that may never share a sentence but are conceptually related.
           Wu-Palmer is preferred over path_similarity because it factors
           in the depth of the most specific common ancestor, making it
           more discriminative for closely related terms.

        If both edge types exist between the same pair, the co-occurrence
        edge takes priority (it is grounded in the actual text).
        """
        graph = nx.Graph()
        for c in concepts:
            graph.add_node(c)

        # 1. Co-occurrence edges
        sentences = nltk.sent_tokenize(clean_text)
        for sent in sentences:
            present = [c for c in concepts if c in sent]
            for a, b in combinations(present, 2):
                prev = graph.get_edge_data(a, b, {"weight": 0, "type": "cooccurrence"})
                graph.add_edge(a, b, weight=prev["weight"] + 1, type="cooccurrence")

        # Remove co-occurrence edges below threshold
        weak = [
            (u, v) for u, v, d in graph.edges(data=True)
            if d["type"] == "cooccurrence" and d["weight"] < self.min_edge_strength
        ]
        graph.remove_edges_from(weak)

        # 2. Semantic edges via WordNet Wu-Palmer similarity
        for a, b in combinations(concepts, 2):
            if graph.has_edge(a, b):
                continue   # co-occurrence edge already exists — skip
            sim = self._wordnet_similarity(a, b)
            if sim >= self.semantic_threshold:
                graph.add_edge(a, b, weight=round(sim, 3), type="semantic")

        return graph

    def _wordnet_similarity(self, word_a: str, word_b: str) -> float:
        """
        Wu-Palmer similarity between the first synsets of two words.
        Returns 0.0 if either word has no synsets.
        Considers only matching POS pairs (noun-noun, verb-verb) to avoid
        spurious links like 'learning' (noun) ↔ 'run' (verb).
        """
        try:
            syns_a = wn.synsets(word_a)
            syns_b = wn.synsets(word_b)
            if not syns_a or not syns_b:
                return 0.0

            best = 0.0
            # Compare only same-POS synsets (first 2 of each to keep it fast)
            for sa in syns_a[:2]:
                for sb in syns_b[:2]:
                    if sa.pos() != sb.pos():
                        continue
                    score = sa.wup_similarity(sb) or 0.0
                    if score > best:
                        best = score
            return best
        except Exception:
            return 0.0

    # ── formatting ─────────────────────────────────────────────────────────────

    def _format_for_visualization(self, graph: nx.Graph) -> dict:
        """
        Convert to the JSON shape expected by knowledge_graph.html:
          nodes: [{id, label, importance, size, definition}]
          links: [{source, target, strength, type}]

        'type' on links is 'cooccurrence' or 'semantic' — the HTML uses this
        to colour edges differently so learners can distinguish explicit
        (text-stated) from implicit (conceptually related) connections.
        """
        try:
            pr     = nx.pagerank(graph)
            max_pr = max(pr.values()) if pr else 1.0

            nodes = [
                {
                    "id":         n,
                    "label":      n,
                    "importance": round(pr.get(n, 0) / max_pr, 4),
                    "size":       round(10 + pr.get(n, 0) * self.node_size_multiplier, 2),
                    "definition": self._get_definition(n),
                }
                for n in graph.nodes
            ]
            links = [
                {
                    "source":   u,
                    "target":   v,
                    "strength": d["weight"],
                    "type":     d.get("type", "cooccurrence"),
                }
                for u, v, d in graph.edges(data=True)
            ]
            co  = sum(1 for l in links if l["type"] == "cooccurrence")
            sem = sum(1 for l in links if l["type"] == "semantic")
            logger.info(f"Graph: {len(nodes)} nodes, {co} co-occurrence edges, {sem} semantic edges")
            return {"nodes": nodes, "links": links}

        except Exception as e:
            logger.error(f"Formatting failed: {e}")
            return {"nodes": [], "links": []}

    # ── definition lookup ──────────────────────────────────────────────────────

    def _get_definition(self, concept: str) -> str:
        """
        WordNet definition with a plain fallback.
        nlp.vocab[word].text returns the surface form, not a gloss —
        that incorrect fallback from the original code is removed.
        """
        try:
            synsets = wn.synsets(concept)
            if synsets:
                return synsets[0].definition()
        except Exception:
            pass
        return "Core concept in this context"
