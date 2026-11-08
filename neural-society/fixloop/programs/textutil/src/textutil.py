"""Small text helpers."""


def word_count(text):
    """Map each lower-cased word to its count. Punctuation at word edges is stripped."""
    counts = {}
    for raw in text.split():
        word = raw.strip(".,!?;:\"'()").lower()
        if word:
            counts[word] = counts.get(word, 0) + 1
    return counts


def title_case(text, small_words=("a", "an", "the", "of", "and", "or", "in")):
    """Capitalise words except small words, which stay lower-case unless first."""
    words = text.split()
    out = []
    for i, w in enumerate(words):
        lw = w.lower()
        if i > 0 and lw in small_words:
            out.append(lw)
        else:
            out.append(lw[:1].upper() + lw[1:])
    return " ".join(out)


def is_palindrome(text):
    """True if the alphanumeric characters read the same both ways, ignoring case."""
    chars = [c.lower() for c in text if c.isalnum()]
    return chars == chars[::-1]


def truncate(text, limit, suffix="..."):
    """Cut text to at most `limit` characters including the suffix."""
    if len(text) <= limit:
        return text
    if limit <= len(suffix):
        return suffix[:limit]
    return text[: limit - len(suffix)] + suffix
