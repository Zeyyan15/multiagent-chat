# src/memory.py
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class MemoryRecord:
    id: int
    timestamp: float
    topic: str
    text: str
    source: str
    agent: str
    confidence: float
    metadata: Dict[str, Any]

class VectorMemory:
    """
    Simple in-memory vector store using TF-IDF + cosine similarity.
    Stores structured records with metadata and returns search results
    by keyword (naive substring) and vector similarity (TF-IDF).
    """
    def __init__(self):
        self.records: List[MemoryRecord] = []
        self._texts: List[str] = []
        self._vectorizer = TfidfVectorizer()
        self._matrix = None
        self._next_id = 1

    def add(self, topic: str, text: str, source: str, agent: str, confidence: float, metadata: Dict[str, Any]=None):
        metadata = metadata or {}
        rec = MemoryRecord(
            id=self._next_id,
            timestamp=time.time(),
            topic=topic,
            text=text,
            source=source,
            agent=agent,
            confidence=float(confidence),
            metadata=metadata
        )
        self._next_id += 1
        self.records.append(rec)
        self._texts.append(f"{topic} {text} {source} {agent}")
        self._rebuild_index()
        return rec

    def _rebuild_index(self):
        if len(self._texts) == 0:
            self._matrix = None
            return
        self._matrix = self._vectorizer.fit_transform(self._texts)

    def keyword_search(self, keyword: str, top_k: int=5) -> List[MemoryRecord]:
        keyword_lower = keyword.lower()
        matched = [r for r in self.records if (keyword_lower in r.topic.lower() or keyword_lower in r.text.lower())]
        return matched[:top_k]

    def vector_search(self, query: str, top_k: int=5) -> List[Tuple[MemoryRecord, float]]:
        if self._matrix is None or len(self.records) == 0:
            return []
        qv = self._vectorizer.transform([query])
        sims = cosine_similarity(qv, self._matrix)[0]  # shape (n_records,)
        idxs = np.argsort(sims)[::-1][:top_k]
        results = []
        for i in idxs:
            score = float(sims[i])
            results.append((self.records[i], score))
        return results

    def get_all(self) -> List[MemoryRecord]:
        return list(self.records)

    def as_dict(self) -> List[Dict[str, Any]]:
        return [asdict(r) for r in self.records]
