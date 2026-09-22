"""
agent_memory.py — The piece that turns FlyMemoryStore from a benchmark
component into something an actual agent can call as a tool.

Wraps: text -> sentence-transformer embedding -> FlyMemoryStore
so the agent only ever deals with plain text in and text out; embedding
and hashing are internal details.
"""

from __future__ import annotations
from sentence_transformers import SentenceTransformer

from memory_store import FlyMemoryStore


class AgentMemory:
    """
    Long-term memory for an LLM agent, backed by FlyHash.

    Usage:
        memory = AgentMemory()
        memory.remember("The user's favorite language is Python.")
        memory.remember("The user is allergic to peanuts.")
        results = memory.recall("what does the user like to code in?", k=2)
        # -> [("The user's favorite language is Python.", distance), ...]
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 hash_dim: int = 2048, sparsity: float = 0.10,
                 wta_percent: float = 0.05):
        print(f"Loading embedding model '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        embed_dim = self.model.get_sentence_embedding_dimension()
        self.store = FlyMemoryStore(
            input_dim=embed_dim, hash_dim=hash_dim,
            sparsity=sparsity, wta_percent=wta_percent, mode="scan",
        )
        # "scan" mode (exact Hamming ranking) is used here rather than
        # "bucket" mode — the benchmarks showed bucket mode isn't actually
        # sublinear at small-to-moderate scale, and correctness matters
        # more than a false speed claim for an agent's actual memory.

    def remember(self, text: str) -> None:
        """Store a new memory."""
        embedding = self.model.encode([text])[0]
        self.store.add(text, embedding)

    def remember_batch(self, texts: list[str]) -> None:
        """Store many memories at once (more efficient than one-by-one)."""
        embeddings = self.model.encode(texts, show_progress_bar=False)
        self.store.add_batch(texts, embeddings)

    def recall(self, query: str, k: int = 3):
        """
        Retrieve the k memories most relevant to `query`.
        Returns a list of (text, hamming_distance) tuples, best match first.
        """
        query_embedding = self.model.encode([query])[0]
        return self.store.query(query_embedding, k=k)

    def recall_as_context(self, query: str, k: int = 3) -> str:
        """
        Convenience method: retrieve memories and format them as a single
        string ready to drop into an LLM prompt as context.
        """
        results = self.recall(query, k=k)
        if not results:
            return "No relevant memories found."
        lines = [f"- {text}" for text, _dist in results]
        return "Relevant memories:\n" + "\n".join(lines)

    def __len__(self):
        return len(self.store.texts)


if __name__ == "__main__":
    # Quick standalone sanity check — no LLM/API key needed for this part.
    memory = AgentMemory()
    memory.remember_batch([
        "The user's favorite programming language is Python.",
        "The user is allergic to peanuts and shellfish.",
        "The user is training for a half marathon in October.",
        "The user prefers dark mode in every app.",
        "The user's dog is named Biscuit.",
        "The user works as a backend engineer at a fintech startup.",
    ])

    print(f"\nStored {len(memory)} memories.\n")

    test_queries = [
        "What food should I avoid recommending?",
        "What's a good gift for their pet?",
        "What kind of code do they write?",
    ]
    for q in test_queries:
        print(f"Query: {q}")
        print(memory.recall_as_context(q, k=2))
        print()
