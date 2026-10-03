"""Standardize raw prompts before they reach the model.

This model is a plain causal language model, so the template is minimal.
"""

PROMPT_SUFFIX = " "  # keeps word-level prompting consistent


def format_prompt(raw_prompt: str) -> str:
    prompt = raw_prompt.strip()
    if not prompt:
        raise ValueError("prompt is empty")
    return prompt + PROMPT_SUFFIX
