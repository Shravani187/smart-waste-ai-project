"""Practical 10 — Micro and Small Language Models for a Parking Assistant.

Builds:
  - a micro language model (bigram count model with add-one smoothing),
  - a small neural SLM (two-word context, embeddings, hidden layer, softmax).

The assistant detects intent (available spaces, location, waiting time) and
generates short responses using an illustrative parking-status table.
All parking data is synthetic and must not be presented as live.
"""

from __future__ import annotations

import numpy as np

SEED = 42

CORPUS = [
    "there are five available spaces near the north gate",
    "available spaces are located near the south entrance",
    "the parking location is near the east wing",
    "you can park near the west gate",
    "the estimated waiting time is about ten minutes",
    "waiting time is approximately five minutes",
    "there are no available spaces at the moment",
    "available spaces are limited near the main gate",
    "the parking location is on level two",
    "waiting time is short near the south entrance",
]

PARKING_TABLE = {
    "north_gate": {"available": 5, "location": "North Gate", "wait": "10 minutes"},
    "south_entrance": {"available": 12, "location": "South Entrance", "wait": "5 minutes"},
    "east_wing": {"available": 0, "location": "East Wing", "wait": "15 minutes"},
    "west_gate": {"available": 8, "location": "West Gate", "wait": "7 minutes"},
}

INTENTS = ["available_spaces", "location", "waiting_time"]


def tokenize(text):
    return text.lower().split()


def build_vocab(corpus):
    vocab = {"<s>", "</s>"}
    for sent in corpus:
        vocab.update(tokenize(sent))
    return sorted(vocab)


class BigramLM:
    """Micro language model: bigram counts with add-one smoothing."""

    def __init__(self, corpus):
        self.vocab = build_vocab(corpus)
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.counts = np.zeros((len(self.vocab), len(self.vocab)))
        for sent in corpus:
            tokens = ["<s>"] + tokenize(sent) + ["</s>"]
            for a, b in zip(tokens, tokens[1:]):
                self.counts[self.word2idx[a], self.word2idx[b]] += 1
        self.probs = (self.counts + 1) / (self.counts.sum(axis=1, keepdims=True) + len(self.vocab))

    def next_word(self, prev):
        if prev not in self.word2idx:
            prev = "<s>"
        idx = self.word2idx[prev]
        return self.vocab[np.argmax(self.probs[idx])]

    def generate(self, seed_word, max_len=8):
        words = [seed_word]
        for _ in range(max_len - 1):
            nxt = self.next_word(words[-1])
            if nxt == "</s>":
                break
            words.append(nxt)
        return " ".join(words)


class NeuralSLM:
    """Small neural SLM: two-word context -> embedding -> hidden -> softmax."""

    def __init__(self, corpus, emb_dim=8, hidden_dim=16, seed=SEED):
        self.vocab = build_vocab(corpus)
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.rng = np.random.default_rng(seed)
        V, H, E = len(self.vocab), hidden_dim, emb_dim
        self.W1 = self.rng.normal(0, 0.1, (2 * E, H))
        self.b1 = np.zeros(H)
        self.W2 = self.rng.normal(0, 0.1, (H, V))
        self.b2 = np.zeros(V)
        self.emb = self.rng.normal(0, 0.1, (V, E))
        self._train(corpus)

    def _train(self, corpus, epochs=200, lr=0.1):
        for _ in range(epochs):
            for sent in corpus:
                tokens = ["<s>", "<s>"] + tokenize(sent) + ["</s>"]
                for i in range(2, len(tokens)):
                    ctx = [self.word2idx[tokens[i - 2]], self.word2idx[tokens[i - 1]]]
                    target = self.word2idx[tokens[i]]
                    x = np.concatenate([self.emb[ctx[0]], self.emb[ctx[1]]])
                    h = np.tanh(x @ self.W1 + self.b1)
                    logits = h @ self.W2 + self.b2
                    probs = np.exp(logits - logits.max())
                    probs /= probs.sum()
                    err = probs
                    err[target] -= 1
                    dW2 = np.outer(h, err)
                    db2 = err
                    dh = err @ self.W2.T * (1 - h ** 2)
                    dW1 = np.outer(x, dh)
                    db1 = dh
                    self.W2 -= lr * dW2
                    self.b2 -= lr * db2
                    self.W1 -= lr * dW1
                    self.b1 -= lr * db1

    def generate(self, seed_words, max_len=8):
        words = list(seed_words)
        for _ in range(max_len - 2):
            ctx = [self.word2idx[words[-2]], self.word2idx[words[-1]]]
            x = np.concatenate([self.emb[ctx[0]], self.emb[ctx[1]]])
            h = np.tanh(x @ self.W1 + self.b1)
            logits = h @ self.W2 + self.b2
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            nxt = self.vocab[np.argmax(probs)]
            if nxt == "</s>":
                break
            words.append(nxt)
        return " ".join(words)


def detect_intent(query):
    q = query.lower()
    if "space" in q or "available" in q:
        return "available_spaces"
    if "where" in q or "location" in q or "park" in q:
        return "location"
    if "wait" in q or "long" in q or "time" in q:
        return "waiting_time"
    return "available_spaces"


def fill_response(intent, area_key):
    info = PARKING_TABLE[area_key]
    if intent == "available_spaces":
        return f"There are {info['available']} available spaces at {info['location']}."
    if intent == "location":
        return f"The parking location is {info['location']}."
    return f"The estimated waiting time is {info['wait']}."


def run():
    bigram = BigramLM(CORPUS)
    slm = NeuralSLM(CORPUS)
    print("Micro Language Model (bigram) generations:")
    for seed in ["there", "available", "the"]:
        print(f"  seed='{seed}' -> {bigram.generate(seed)}")
    print("\nSmall Language Model (neural SLM) generations:")
    for seed in [["there", "are"], ["the", "parking"], ["waiting", "time"]]:
        print(f"  seed={seed} -> {slm.generate(seed)}")
    print("\nIntent detection and responses:")
    queries = [
        "How many spaces are available?",
        "Where can I park?",
        "How long is the waiting time?",
    ]
    for q in queries:
        intent = detect_intent(q)
        area = list(PARKING_TABLE.keys())[0]
        print(f"  Q: {q}")
        print(f"    intent={intent}, response={fill_response(intent, area)}")
    print("\nNote: all parking data above is synthetic/illustrative, not live.")


if __name__ == "__main__":
    run()
