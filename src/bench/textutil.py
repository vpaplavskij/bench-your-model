"""Text cleanup for scoring: servers without a separate reasoning format
stream chain-of-thought inline in delta.content."""


def strip_think(text: str) -> str:
    """Drop chain-of-thought that arrived inline in the content stream."""
    low = text.lower()
    if "</think>" in low:                       # also handles a missing opening tag
        text = text[low.rindex("</think>") + len("</think>"):]
    low = text.lower()
    if "<think>" in low:                        # unclosed => thinking was cut off, no answer
        text = text[:low.index("<think>")]
    return text.strip()
