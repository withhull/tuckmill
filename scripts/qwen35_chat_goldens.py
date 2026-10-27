"""Record Qwen3.5 chat goldens for the restricted renderer (ADR-012 step 1).

Prompts are rendered with Jinja2 configured the way Hugging Face applies
chat templates (sandboxed, trim_blocks + lstrip_blocks, loopcontrols,
`raise_exception`), from the template bytes declared by each pinned GGUF.
Token ids come from the pinned llama.cpp `llama-tokenize` tool (special
spellings parsed), not from Tuckmill's encoder, so the goldens are
independent of the code they test.

Usage (no network; run in the mlx venv, which provides jinja2):
  .deps/mlx-venv/bin/python -I scripts/qwen35_chat_goldens.py \
      artifacts-local/Qwen3.5-0.8B-Q8_0.gguf artifacts-local/Qwen3.5-2B-Q8_0.gguf \
      > crates/supervisor/tests/data/qwen35_chat_goldens.json
"""

import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZE = os.path.join(ROOT, ".deps", "llama", "llama-tokenize")


def chat_template(path):
    """The declared tokenizer.chat_template bytes of a GGUF file."""
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    with open(path, "rb") as f:
        read = f.read
        assert read(4) == b"GGUF"
        read(4)
        _, nkv = struct.unpack("<QQ", read(16))

        def string():
            (n,) = struct.unpack("<Q", read(8))
            return read(n)

        for _ in range(nkv):
            key = string().decode()
            (t,) = struct.unpack("<I", read(4))
            if t == 8:
                value = string()
                if key == "tokenizer.chat_template":
                    return value
            elif t == 9:
                (et,) = struct.unpack("<I", read(4))
                (n,) = struct.unpack("<Q", read(8))
                if et == 8:
                    for _ in range(n):
                        string()
                else:
                    f.seek(n * sizes[et], 1)
            else:
                f.seek(sizes[t], 1)
    raise SystemExit(f"{path}: no tokenizer.chat_template")


class TemplateError(Exception):
    pass


def raise_exception(message):
    raise TemplateError(message)


def render(template_bytes, messages):
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
    )
    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = lambda v, **kw: json.dumps(v, ensure_ascii=False)
    template = env.from_string(template_bytes.decode("utf-8"))
    try:
        return template.render(messages=messages, add_generation_prompt=True), None
    except TemplateError as e:
        return None, str(e)


def tokenize(gguf, text):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(text)
        name = f.name
    try:
        out = subprocess.run(
            [TOKENIZE, "-m", gguf, "-f", name, "--ids", "--log-disable"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    finally:
        os.unlink(name)
    return json.loads(out.strip().splitlines()[-1])


S, U, A = "system", "user", "assistant"
CONVERSATIONS = [
    ("single_user", [(U, "Hello, world!")]),
    ("system_then_user", [(S, "You are terse."), (U, "What is 2+2?")]),
    ("multi_turn", [(U, "Hi"), (A, "Hello! How can I help?"), (U, "Tell me a joke.")]),
    ("system_multi_turn", [(S, "Be kind."), (U, "Hi"), (A, "Hello."), (U, "Bye")]),
    ("whitespace_trimmed", [(S, "  spaced system \n"), (U, "\n\n  padded user  \t\n")]),
    ("assistant_history_with_think", [(U, "Q1"), (A, "<think>\nreason\n</think>\n\nA1"), (U, "Q2")]),
    ("trailing_assistant", [(U, "Start the story."), (A, "Once upon a time")]),
    ("trailing_assistant_with_think", [(U, "Q"), (A, "<think>\nplan it\n</think>\n\npartial")]),
    ("unicode", [(U, "Grüße, 世界! ¿Qué tal? 🙂")]),
    ("empty_user", [(U, "")]),
    ("tool_response_user_after_query", [(U, "Look it up."), (A, "Calling."), (U, "<tool_response>42</tool_response>")]),
    ("newlines_in_content", [(U, "line one\nline two\n\nline four")]),
    # Error cases the template itself rejects.
    ("error_system_only", [(S, "Only a system message.")]),
    ("error_system_not_first", [(U, "Hi"), (S, "Late system.")]),
    ("error_only_tool_response", [(U, "<tool_response>x</tool_response>")]),
]


def main():
    ggufs = sys.argv[1:]
    if not ggufs:
        raise SystemExit("usage: qwen35_chat_goldens.py <qwen35.gguf> [more.gguf ...]")
    templates = []
    for g in ggufs:
        t = chat_template(g)
        templates.append(
            {"artifact": os.path.basename(g), "bytes": len(t), "sha256": hashlib.sha256(t).hexdigest()}
        )
    conversations = []
    for cid, msgs in CONVERSATIONS:
        messages = [{"role": r, "content": c} for r, c in msgs]
        rendered = [render(chat_template(g), messages) for g in ggufs]
        # Every pinned revision must give the same result for the subset.
        assert all(r == rendered[0] for r in rendered), f"{cid}: revisions disagree"
        prompt, error = rendered[0]
        entry = {"id": cid, "messages": messages}
        if error is not None:
            entry["error"] = error
        else:
            entry["prompt"] = prompt
            entry["tokens"] = tokenize(ggufs[0], prompt)
        conversations.append(entry)
    json.dump(
        {
            "source": "pinned Qwen3.5 GGUF chat templates",
            "templates": templates,
            "method": (
                f"prompts rendered with Jinja2 {jinja2.__version__} as Hugging Face applies "
                "chat templates (ImmutableSandboxedEnvironment, trim_blocks, lstrip_blocks, "
                "loopcontrols, raise_exception), add_generation_prompt=True, enable_thinking "
                "undefined (the template default: thinking off); token ids from pinned "
                "llama.cpp b11695 llama-tokenize with special spellings parsed, tokenized "
                "against the first artifact"
            ),
            "conversations": conversations,
        },
        sys.stdout,
        ensure_ascii=False,
        indent=1,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
