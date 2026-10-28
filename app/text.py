"""Text normalisation helpers shared by the metrics."""
import re
import string

STOP = set("""a an the is are was were be been being am do does did of to in on at for from by with about as into
than then so and or but if not it its this that these those there here i me my we our you your he she they them their
what which who whom when where why how can could would should will shall may might must also just only very""".split())

_PUNCT = re.compile(f"[{re.escape(string.punctuation)}]")


def normalize(s: str) -> str:
    """SQuAD-style: lowercase, drop punctuation and articles, squash spaces. Keeps digits/commas in numbers."""
    s = (s or "").lower()
    s = re.sub(r"(?<=\d),(?=\d)", "", s)          # 1,10,000 -> 110000
    s = _PUNCT.sub(" ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def tokens(s: str) -> list[str]:
    return normalize(s).split()


def stem(w: str) -> str:
    for suf in ("ing", "ies", "es", "ed", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def content_tokens(s: str) -> list[str]:
    return [stem(t) for t in tokens(s) if t not in STOP]


def strip_code_fence(s: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)```", s or "", re.S)
    return (m.group(1) if m else s or "").strip()
