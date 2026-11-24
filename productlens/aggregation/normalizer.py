"""
Aspect Normalization and Semantic Clustering.
Groups validated aspect quads by canonical category and semantically clusters
free-text aspect phrases into unified, human-readable canonical feature names.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except Exception:
    HAVE_SENTENCE_TRANSFORMERS = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

from productlens.schemas.quads import AspectQuad


class AspectNormalizer:
    """Normalizes and clusters aspect terms within canonical taxonomy categories."""

    def __init__(
        self,
        similarity_threshold: float = 0.65,
        use_embeddings: bool = True,
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        self.similarity_threshold = similarity_threshold
        self.distance_threshold = 1.0 - similarity_threshold
        self.embedder: Optional[Any] = None

        if use_embeddings and HAVE_SENTENCE_TRANSFORMERS:
            try:
                self.embedder = SentenceTransformer(embedding_model_name)
            except Exception:
                self.embedder = None

    def _compute_similarity_matrix(self, terms: List[str]) -> np.ndarray:
        """Computes pairwise cosine similarity matrix using embeddings or TF-IDF fallback."""
        if len(terms) == 1:
            return np.array([[1.0]])

        if self.embedder is not None:
            try:
                embeddings = self.embedder.encode(terms, normalize_embeddings=True, show_progress_bar=False)
                return np.dot(embeddings, embeddings.T)
            except Exception:
                pass

        # TF-IDF fallback with character n-grams to handle compound terms and typos
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        tfidf = vectorizer.fit_transform(terms)
        return cosine_similarity(tfidf)

    @staticmethod
    def _clean_term(term: Optional[str]) -> str:
        """Standardizes aspect term text."""
        if not term:
            return ""
        t = re.sub(r"[^\w\s-]", " ", term.lower()).strip()
        t = re.sub(r"\s+", " ", t)
        return t

    @classmethod
    def format_canonical_name(cls, term: str, category: str) -> str:
        """Formats a human-readable title-cased feature name."""
        if not term or term.lower() == "none" or term.lower() == "null":
            cat_clean = category.replace("_", " ").title()
            return f"Overall {cat_clean}"
        # Capitalize acronyms like OLED, USB, RAM, SSD, TV, OS, CPU, GPU
        acronyms = {"oled", "usb", "ram", "ssd", "tv", "os", "cpu", "gpu", "wifi", "sd", "hdmi"}

        def format_subword(sub: str) -> str:
            if sub.lower() in acronyms:
                return sub.upper()
            return sub.capitalize()

        words = term.split()
        title_words = []
        for w in words:
            if "-" in w:
                title_words.append("-".join(format_subword(p) for p in w.split("-")))
            else:
                title_words.append(format_subword(w))
        return " ".join(title_words)

    def cluster_category_aspects(
        self,
        category: str,
        quad_tuples: List[Tuple[Optional[str], AspectQuad, str]],  # (aspect, quad, review_id)
    ) -> Dict[str, List[Tuple[AspectQuad, str]]]:
        """
        Clusters aspect phrases within a single canonical category.
        Returns a dict mapping canonical_feature_name -> list of (quad, review_id).
        """
        implicit_items: List[Tuple[AspectQuad, str]] = []
        explicit_items: List[Tuple[str, AspectQuad, str]] = []

        for aspect, q, rev_id in quad_tuples:
            clean = self._clean_term(aspect)
            if not clean or q.is_implicit:
                implicit_items.append((q, rev_id))
            else:
                explicit_items.append((clean, q, rev_id))

        clusters: Dict[str, List[Tuple[AspectQuad, str]]] = defaultdict(list)

        # 1. Implicit quads mapped to category-level canonical feature
        if implicit_items:
            canonical_implicit = self.format_canonical_name(None, category)
            clusters[canonical_implicit].extend(implicit_items)

        if not explicit_items:
            return dict(clusters)

        # 2. Get unique explicit terms and counts
        term_counts = Counter(item[0] for item in explicit_items)
        unique_terms = list(term_counts.keys())

        if len(unique_terms) == 1:
            best_term = unique_terms[0]
            canonical_name = self.format_canonical_name(best_term, category)
            clusters[canonical_name].extend([(item[1], item[2]) for item in explicit_items])
            return dict(clusters)

        # 3. Compute pairwise similarity and cluster
        sim_matrix = self._compute_similarity_matrix(unique_terms)
        dist_matrix = np.clip(1.0 - sim_matrix, 0.0, 1.0)
        np.fill_diagonal(dist_matrix, 0.0)

        clustering = AgglomerativeClustering(
            metric="precomputed",
            linkage="average",
            distance_threshold=self.distance_threshold,
            n_clusters=None,
        )
        labels = clustering.fit_predict(dist_matrix)

        # Map cluster_id -> representative label (most frequent term in cluster)
        cluster_to_terms: Dict[int, List[str]] = defaultdict(list)
        for term, label in zip(unique_terms, labels):
            cluster_to_terms[label].append(term)

        term_to_canonical: Dict[str, str] = {}
        for label, member_terms in cluster_to_terms.items():
            # Pick most frequent term as representative
            rep_term = max(member_terms, key=lambda t: (term_counts[t], len(t)))
            canonical_name = self.format_canonical_name(rep_term, category)
            for t in member_terms:
                term_to_canonical[t] = canonical_name

        # Assign each quad to its cluster
        for term, q, rev_id in explicit_items:
            canonical = term_to_canonical[term]
            clusters[canonical].append((q, rev_id))

        return dict(clusters)

    def group_and_normalize(
        self,
        quad_records: List[Tuple[AspectQuad, str]],  # (quad, review_id)
    ) -> Dict[str, Dict[str, List[Tuple[AspectQuad, str]]]]:
        """
        Groups all quads by category, then clusters aspects within each category.
        Returns:
            category -> {canonical_feature_name -> [(quad, review_id), ...]}
        """
        by_category: Dict[str, List[Tuple[Optional[str], AspectQuad, str]]] = defaultdict(list)
        for q, rev_id in quad_records:
            cat = q.category.upper().strip()
            by_category[cat].append((q.aspect, q, rev_id))

        result: Dict[str, Dict[str, List[Tuple[AspectQuad, str]]]] = {}
        for cat, items in by_category.items():
            result[cat] = self.cluster_category_aspects(cat, items)

        return result
