"""Step 9: LLM generation (Anthropic API or local Ollama)."""
import requests

from . import config


def generate(system: str, user: str, max_tokens: int = 1200) -> str:
    if config.LLM_PROVIDER == "ollama":
        r = requests.post(
            f"{config.OLLAMA_URL}/api/chat",
            json={"model": config.OLLAMA_MODEL, "stream": False,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}]},
            timeout=300,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]

    import anthropic
    client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY
    resp = client.messages.create(
        model=config.ANTHROPIC_MODEL, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


QA_SYSTEM = (
    "You answer questions using ONLY the numbered context passages provided. "
    "Cite passages like [1], [2]. If the context does not contain the answer, "
    "say you could not find it in the uploaded documents. Do not use outside knowledge."
)


def answer_question(question: str, contexts: list) -> str:
    ctx = "\n\n".join(f"[{i}] ({c['source']}, {c['location']})\n{c['text']}"
                      for i, c in enumerate(contexts, 1))
    return generate(QA_SYSTEM, f"Context:\n{ctx}\n\nQuestion: {question}")
