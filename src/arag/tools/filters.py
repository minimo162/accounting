"""Chunk quality filtering and tagging for retrieval."""


def is_clearly_low_value(text: str) -> bool:
    """Hard filter: exclude chunks that are obviously not useful for answering questions.

    These are removed from search results entirely so they don't waste top_k slots.
    """
    # Table of contents
    if "目次" in text or "目 次" in text:
        return True

    # Dot leaders (TOC-style listings)
    if text.count("・・") >= 5 or text.count("...") >= 10:
        return True

    # Cover pages: very short, mostly title/date
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) < 15 and len(text) < 800:
        return True

    return False


def get_chunk_tag(text: str) -> str | None:
    """Soft tag for borderline chunks the agent should be cautious about.

    These remain in search results but are tagged so the agent can decide.
    """
    lines = [l for l in text.split("\n") if l.strip()]

    # Member lists / voting records
    if any(k in text[:200] for k in ["委員は、以下のとおり", "名全員の賛成により", "委員等名簿"]):
        return "⚠️名簿/議決"

    # Very short chunks that passed the hard filter
    if len(lines) < 20 and len(text) < 1000:
        return "⚠️短文"

    return None
