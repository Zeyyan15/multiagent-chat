# src/main.py
import os
import json
from pathlib import Path
from src.memory import VectorMemory
from src.agents import ResearchAgent, AnalysisAgent, MemoryAgent, Coordinator

# ---------- Simple mock KB ----------
MOCK_KB = [
    {"title":"Neural Networks Overview",
     "text":"Neural networks are a family of models inspired by biological neurons. Common types: feedforward, CNNs (conv nets), RNNs, LSTMs, Transformers.",
     "tags":["neural", "cnn", "rnn", "transformer"],
     "source":"mock.org/nn"},
    {"title":"Transformer Architecture Paper",
     "text":"Transformers use self-attention and are highly parallelizable. They can be computationally intensive for large models but efficient for training on modern accelerators.",
     "tags":["transformer", "attention", "efficiency"],
     "source":"mock.org/transformer"},
    {"title":"Adam Optimizer",
     "text":"Adam is an adaptive optimizer combining RMSProp and momentum. It generally converges faster than vanilla SGD in many settings.",
     "tags":["optimizers", "adam", "sgd"],
     "source":"mock.org/adam"},
    {"title":"Gradient Descent",
     "text":"Gradient Descent and its variants like SGD are fundamental optimization algorithms. SGD with momentum is simple and memory efficient.",
     "tags":["optimizers", "sgd"],
     "source":"mock.org/sgd"},
    {"title":"Reinforcement Learning Survey",
     "text":"Recent papers highlight sample efficiency, exploration, and stability as common challenges in RL.",
     "tags":["reinforcement", "rl", "papers"],
     "source":"mock.org/rl-survey"}
]

# ---------- helper to ensure outputs folder ----------
ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

def write_output(name: str, text: str):
    path = OUTPUTS / name
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[WROTE] {path}")

def run_scenarios():
    vm = VectorMemory()
    research = ResearchAgent(MOCK_KB)
    analysis = AnalysisAgent()
    memory_agent = MemoryAgent(vm)
    coord = Coordinator(research, analysis, memory_agent)

    scenarios = []

    # 1. Simple Query
    q1 = "What are the main types of neural networks?"
    res1 = coord.handle_query(q1)
    out1 = json.dumps(res1, indent=2, default=str)
    write_output("simple_query.txt", out1)
    scenarios.append(("simple_query.txt", q1, res1))

    # 2. Complex Query
    q2 = "Research transformer architectures, analyze their computational efficiency, and summarize key trade-offs."
    res2 = coord.handle_query(q2)
    write_output("complex_query.txt", json.dumps(res2, indent=2, default=str))
    scenarios.append(("complex_query.txt", q2, res2))

    # 3. Memory Test: ask about what we discussed earlier (neural networks)
    q3 = "What did we discuss about neural networks earlier?"
    # ensure we have stored something: explicitly store a memory from earlier synthesis
    if res1["results"].get("synthesis"):
        syn = res1["results"]["synthesis"]
        memory_agent.store_finding(topic="neural networks", text=syn.get("text",""), source="auto", agent="Coordinator", confidence=syn.get("confidence",0.6))
    res3 = coord.ask_memory("neural networks")
    write_output("memory_test.txt", json.dumps(res3, indent=2, default=str))
    scenarios.append(("memory_test.txt", q3, res3))

    # 4. Multi-step (recent papers on RL)
    q4 = "Find recent papers on reinforcement learning, analyze their methodologies, and identify common challenges."
    res4 = coord.handle_query(q4)
    write_output("multi_step.txt", json.dumps(res4, indent=2, default=str))
    scenarios.append(("multi_step.txt", q4, res4))

    # 5. Collaborative: compare two ML approaches
    q5 = "Compare Adam and SGD and recommend which is better for fast convergence."
    res5 = coord.handle_query(q5)
    write_output("collaborative.txt", json.dumps(res5, indent=2, default=str))
    scenarios.append(("collaborative.txt", q5, res5))

    print("\n--- SCENARIOS RUN COMPLETE ---")
    return scenarios

if __name__ == "__main__":
    run_scenarios()
