# src/agents.py
import json
from typing import Dict, Any, List, Tuple
from datetime import datetime
import random
from dataclasses import asdict

from src.memory import VectorMemory

def now_ts():
    return datetime.utcnow().isoformat() + "Z"

class ResearchAgent:
    """
    Simulates information retrieval. Uses a built-in knowledge base (simple list/dict).
    Returns results with a confidence score and provenance.
    """
    def __init__(self, kb: List[Dict[str, Any]]):
        # kb: list of {title, text, tags, source}
        self.kb = kb

    def search(self, query: str, top_k: int = 5) -> List[Dict[str,Any]]:
        # naive relevance: match count of query tokens in text/title/tags
        q_tokens = set(query.lower().split())
        scored = []
        for doc in self.kb:
            text = f"{doc.get('title','')} {doc.get('text','')} {' '.join(doc.get('tags',[]))}".lower()
            score = sum(1 for t in q_tokens if t in text)
            # small boost if appears in title
            if doc.get('title') and query.lower() in doc['title'].lower():
                score += 2
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        # if everything zero, just return some docs (simulate web-like behavior)
        results = []
        for score, doc in scored[:top_k]:
            confidence = min(0.5 + score * 0.1, 0.95) if score>0 else 0.4 + random.random()*0.15
            results.append({
                "title": doc.get("title"),
                "text": doc.get("text"),
                "tags": doc.get("tags", []),
                "source": doc.get("source", "mock_kb"),
                "confidence": round(confidence, 2),
            })
        return results

class AnalysisAgent:
    """
    Performs reasoning, comparison, simple calculations.
    Accepts research results and produces structured analysis.
    """
    def __init__(self):
        pass

    def compare(self, items: List[Dict[str,Any]], metric: str = "effectiveness") -> Dict[str,Any]:
        # Produce a simple comparison scoring items on a made-up metric derived from length & tags.
        analysis = []
        for it in items:
            base = len(it.get("text",""))
            tag_bonus = len(it.get("tags",[]))
            score = (base/200.0) + 0.2*tag_bonus
            # map to 0-1
            score = max(0.0, min(1.0, score/2.0))
            analysis.append({
                "title": it.get("title"),
                "score": round(score, 3),
                "source": it.get("source"),
                "confidence": round(it.get("confidence", 0.5) * 0.9, 2),
                "explanation": f"Length-based proxy score with tag bonus ({tag_bonus} tags)."
            })
        # sort descending
        analysis.sort(key=lambda x: x["score"], reverse=True)
        summary = {
            "metric": metric,
            "ranked": analysis,
            "summary_text": f"Top item: {analysis[0]['title']}" if analysis else "No items",
            "confidence": round(sum(a["confidence"] for a in analysis)/len(analysis), 2) if analysis else 0.0
        }
        return summary

    def synthesize(self, analyses: Dict[str,Any]) -> Dict[str,Any]:
        # produce final synthesis text
        ranked = analyses.get("ranked", [])
        if not ranked:
            return {"text":"No analysis produced.", "confidence":0.0}
        best = ranked[0]
        text = (f"Based on available data, '{best['title']}' scores highest for {analyses.get('metric')} "
                f"({best['score']}). Explanation: {best['explanation']}")
        return {"text": text, "confidence": best["confidence"]}

class MemoryAgent:
    """
    Thin wrapper around VectorMemory with structured record interface.
    """
    def __init__(self, vector_memory: VectorMemory):
        self.vm = vector_memory

    def store_finding(self, topic: str, text: str, source: str, agent: str, confidence: float, metadata: dict=None):
        rec = self.vm.add(topic=topic, text=text, source=source, agent=agent, confidence=confidence, metadata=metadata)
        return rec

    def retrieve_by_topic(self, topic: str, top_k: int=5):
        return self.vm.keyword_search(topic, top_k=top_k)

    def retrieve_similar(self, query: str, top_k: int=5):
        return self.vm.vector_search(query, top_k=top_k)

class Coordinator:
    """
    Orchestrates: receives user queries, performs simple complexity analysis and routes tasks.
    Maintains conversation context and agent state tracking.
    """
    def __init__(self, research_agent: ResearchAgent, analysis_agent: AnalysisAgent, memory_agent: MemoryAgent):
        self.research = research_agent
        self.analysis = analysis_agent
        self.memory = memory_agent
        self.context = []  # list of messages / interactions
        self.agent_state = {}  # per task logs
        self.trace = []

    def log(self, entry: Dict[str,Any]):
        entry['ts'] = now_ts()
        self.trace.append(entry)
        print(f"[TRACE] {entry['ts']} {entry.get('actor','Coordinator')}: {entry.get('action','')}")
        # optionally print payload smaller
        if 'payload' in entry:
            short = entry['payload']
            if isinstance(short, dict) and 'text' in short and len(short['text'])>200:
                short = dict(short)
                short['text'] = short['text'][:200] + "..." 
            print("       payload:", short)

    def complexity_analysis(self, query: str) -> Dict[str,Any]:
        # Very simple rules:
        q = query.lower()
        if any(w in q for w in ["compare", "compare two", "which is better", "recommend"]):
            plan = ["research", "analysis"]
        elif any(w in q for w in ["recent papers", "recent", "papers", "recently"]):
            plan = ["research", "analysis", "memory_update"]
        elif len(q.split()) < 6:
            plan = ["research"]
        else:
            plan = ["research", "analysis"]
        self.log({"actor":"Coordinator", "action":"complexity_analysis", "payload":{"query":query, "plan":plan}})
        return {"plan": plan}

    def handle_query(self, query: str) -> Dict[str,Any]:
        plan = self.complexity_analysis(query)["plan"]
        self.context.append({"role":"user", "text": query, "ts": now_ts()})
        results = {}
        # research step
        if "research" in plan:
            self.log({"actor":"Coordinator", "action":"call_research", "payload":{"query":query}})
            research_out = self.research.search(query, top_k=5)
            self.agent_state['last_research'] = research_out
            results['research'] = research_out
        # analysis step
        if "analysis" in plan:
            self.log({"actor":"Coordinator", "action":"call_analysis", "payload":{"items_count": len(results.get('research',[]))}})
            analysis_out = self.analysis.compare(results.get('research', []), metric="effectiveness")
            self.agent_state['last_analysis'] = analysis_out
            results['analysis'] = analysis_out
        # synthesis & memory update
        synthesis = None
        if "analysis" in plan:
            self.log({"actor":"Coordinator", "action":"synthesize", "payload":{}})
            synthesis = self.analysis.synthesize(results['analysis'])
            results['synthesis'] = synthesis
        if "memory_update" in plan:
            # store the synthesis + research summary to memory
            topic = query
            text = synthesis.get('text') if synthesis else "research results"
            rec = self.memory.store_finding(topic=topic, text=text, source="coordinator", agent="Coordinator", confidence=synthesis.get('confidence',0.5) if synthesis else 0.5)
            self.log({"actor":"Coordinator", "action":"memory_store", "payload": asdict(rec) if hasattr(rec,'__dict__') else rec})
            results['memory_record'] = rec
        # save context & return
        self.context.append({"role":"assistant", "text": synthesis.get('text') if synthesis else "Here are the research results.", "ts": now_ts()})
        return {"query": query, "results": results, "trace": self.trace}

    def ask_memory(self, question: str) -> Dict[str,Any]:
        # tries keyword then vector search
        self.log({"actor":"Coordinator", "action":"memory_query", "payload":{"query":question}})
        kw = self.memory.retrieve_by_topic(question, top_k=5)
        vec = self.memory.retrieve_similar(question, top_k=5)
        payload = {"keyword_results": [r.__dict__ if hasattr(r,'__dict__') else r for r in kw],
                   "vector_results": [(r.__dict__, score) for (r, score) in vec]}
        self.log({"actor":"Coordinator", "action":"memory_query_result", "payload": {"counts": {"kw": len(kw), "vec": len(vec)}}})
        # build answer
        if vec:
            rec, score = vec[0]
            answer = f"Memory found (similar): {rec.topic} - {rec.text[:300]} (score={score:.3f})"
            confidence = rec.confidence
        elif kw:
            rec = kw[0]
            answer = f"Memory found (keyword): {rec.topic} - {rec.text[:300]}"
            confidence = rec.confidence
        else:
            answer = "No relevant memory found."
            confidence = 0.0
        return {"answer": answer, "confidence": confidence, "details": payload}
