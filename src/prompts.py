"""Prompt templates and message formatting following the official HuggingFaceBio/Carbon-X guidance.

Rules from Carbon-X documentation:
1. DNA sequences are enclosed in <dna>...</dna> tags.
2. The sequence and the question are placed in the same user turn.
3. Relevant context (task definition and allowed label set) is provided to prevent hallucinations.
4. Chat template uses enable_thinking=False.
"""

from typing import List, Dict, Any


def get_class_prompt(sequence: str, allowed_classes: List[str]) -> str:
    """Format prompt for repeat class prediction."""
    classes_str = ", ".join(sorted(allowed_classes))
    return (
        f"The following DNA sequence is a repetitive element from the human genome, "
        f"annotated by RepeatMasker.\n\n"
        f"Sequence: <dna>{sequence.strip()}</dna>\n\n"
        f"Which repeat class does this sequence belong to?\n"
        f"Choose exactly one from the following allowed classes:\n"
        f"{classes_str}\n\n"
        f"Answer with only the class name from the list, nothing else."
    )


def get_family_prompt(sequence: str, allowed_families: List[str]) -> str:
    """Format prompt for repeat family prediction."""
    families_str = ", ".join(sorted(allowed_families))
    return (
        f"The following DNA sequence is a repetitive element from the human genome, "
        f"annotated by RepeatMasker.\n\n"
        f"Sequence: <dna>{sequence.strip()}</dna>\n\n"
        f"Which repeat family does this sequence belong to?\n"
        f"Choose exactly one from the following allowed families:\n"
        f"{families_str}\n\n"
        f"Answer with only the family name from the list, nothing else."
    )


def build_chat_messages(user_prompt: str) -> List[Dict[str, str]]:
    """Create a single-turn user message list for the chat template."""
    return [{"role": "user", "content": user_prompt}]


def format_chat_prompt(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    """Apply the model's chat template with thinking disabled."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
