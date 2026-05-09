"""Perturbation engine for Level 2 attacks.

Provides three perturbation strategies:
  1. Sentence-level permutation — shuffle sentence order
  2. Word-level permutation — swap words at specific positions
  3. Synonym substitution — replace words with semantically similar alternatives

All transforms are pure text operations with no model dependency.
Code blocks (``` fenced) are detected and treated as atomic units.
"""
from __future__ import annotations
import re
import random
from typing import Optional


# ── Curated security-domain synonym mapping ──────────────────────────
# Maps "refusal-triggering" security jargon → benign-sounding alternatives.
# These are words that attribution typically flags as high-signal.
SECURITY_SYNONYMS: dict[str, list[str]] = {
    # Offensive security terms → benign alternatives
    "inject":        ["load", "insert", "introduce", "embed"],
    "injection":     ["loading", "insertion", "introduction"],
    "exploit":       ["utilize", "leverage", "test", "exercise"],
    "exploiting":    ["utilizing", "leveraging", "testing"],
    "exploitation":  ["utilization", "leveraging", "testing"],
    "vulnerability": ["issue", "weakness", "gap", "limitation"],
    "vulnerabilities": ["issues", "weaknesses", "gaps", "limitations"],
    "attack":        ["test", "assessment", "evaluation", "probe"],
    "attacking":     ["testing", "assessing", "evaluating", "probing"],
    "malware":       ["diagnostic tool", "instrumentation", "agent", "utility"],
    "bypass":        ["work around", "circumvent", "navigate past", "skip"],
    "bypassing":     ["working around", "circumventing", "navigating past"],
    "hack":          ["modify", "adjust", "tinker with", "customize"],
    "hacking":       ["modifying", "adjusting", "tinkering with"],
    "payload":       ["package", "module", "component", "bundle"],
    "shellcode":     ["bootstrap code", "loader stub", "init routine"],
    "rootkit":       ["system monitor", "low-level diagnostic", "kernel module"],
    "trojan":        ["remote agent", "background service", "helper process"],
    "backdoor":      ["maintenance access", "service entry", "debug port"],
    "phishing":      ["social engineering test", "awareness exercise"],
    "ransomware":    ["encryption utility", "file locker", "backup tool"],
    "keylogger":     ["input monitor", "keystroke logger", "typing analytics"],
    "spyware":       ["monitoring tool", "telemetry agent", "observer"],
    "botnet":        ["distributed network", "node cluster", "agent fleet"],
    "reverse shell": ["remote console", "callback session", "return channel"],
    "privilege escalation": ["permission elevation", "access upgrade"],
    "brute force":   ["exhaustive search", "systematic trial", "credential sweep"],
    "credential":    ["authentication token", "access key", "login detail"],
    "credentials":   ["authentication tokens", "access keys", "login details"],
    "exfiltrate":    ["extract", "export", "transfer out", "retrieve"],
    "exfiltration":  ["extraction", "export", "retrieval"],
    "obfuscate":     ["obscure", "encode", "transform", "minify"],
    "obfuscation":   ["encoding", "transformation", "minification"],
    "weaponize":     ["prepare", "package", "configure", "assemble"],
    "weaponized":    ["prepared", "packaged", "configured"],
    "compromise":    ["access", "reach", "connect to", "affect"],
    "compromised":   ["accessed", "affected", "reached"],
    "malicious":     ["unauthorized", "unexpected", "unintended", "rogue"],
    "unauthorized":  ["unplanned", "ad-hoc", "unscheduled"],
    "steal":         ["copy", "obtain", "acquire", "retrieve"],
    "stealing":      ["copying", "obtaining", "acquiring"],
    "intercept":     ["capture", "observe", "inspect", "record"],
    "intercepting":  ["capturing", "observing", "inspecting"],
    "crack":         ["recover", "derive", "solve", "decode"],
    "cracking":      ["recovering", "deriving", "solving"],
    "decrypt":       ["decode", "decipher", "recover", "unlock"],
    "virus":         ["self-replicating program", "propagating agent"],
    "worm":          ["self-propagating module", "network crawler"],
    "denial of service": ["load test", "stress test", "capacity test"],
    "dos":           ["load test", "stress test"],
    "ddos":          ["distributed load test", "distributed stress test"],
    "sniff":         ["monitor", "capture", "observe", "inspect"],
    "sniffer":       ["packet monitor", "traffic inspector", "network observer"],
    "dump":          ["extract", "export", "save", "output"],
    "dumping":       ["extracting", "exporting", "saving"],
    "hook":          ["intercept", "attach to", "instrument", "monitor"],
    "hooking":       ["intercepting", "attaching to", "instrumenting"],
}

# Multi-word keys (for phrase-level substitution)
SECURITY_PHRASES: dict[str, list[str]] = {
    k: v for k, v in SECURITY_SYNONYMS.items() if " " in k
}
SECURITY_WORDS: dict[str, list[str]] = {
    k: v for k, v in SECURITY_SYNONYMS.items() if " " not in k
}


# ── WordNet synonym lookup ───────────────────────────────────────────
def _try_import_wordnet():
    """Try to import WordNet; return None if unavailable."""
    try:
        from nltk.corpus import wordnet
        # Trigger download check
        wordnet.synsets("test")
        return wordnet
    except Exception:
        try:
            import nltk
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            from nltk.corpus import wordnet
            return wordnet
        except Exception:
            return None


_wordnet = None
_wordnet_checked = False


def _get_wordnet():
    global _wordnet, _wordnet_checked
    if not _wordnet_checked:
        _wordnet = _try_import_wordnet()
        _wordnet_checked = True
    return _wordnet


def get_synonyms(word: str, max_synonyms: int = 5) -> list[str]:
    """Get synonyms for a word from curated mapping + WordNet.

    Priority order:
      1. Curated security-domain mapping (most relevant)
      2. WordNet synsets (general-purpose)
    
    Returns up to `max_synonyms` unique alternatives.
    """
    results = []

    # 1. Curated mapping
    key = word.lower().strip()
    if key in SECURITY_WORDS:
        results.extend(SECURITY_WORDS[key])

    # 2. WordNet
    wn = _get_wordnet()
    if wn is not None:
        seen = {word.lower(), key} | {r.lower() for r in results}
        for ss in wn.synsets(key):
            for lemma in ss.lemmas():
                name = lemma.name().replace("_", " ")
                if name.lower() not in seen:
                    results.append(name)
                    seen.add(name.lower())

    return results[:max_synonyms]


# ── Code block detection ─────────────────────────────────────────────
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)


def detect_code_blocks(text: str) -> list[tuple[int, int]]:
    """Find all ``` fenced code block spans (start, end) in text."""
    return [(m.start(), m.end()) for m in _CODE_BLOCK_RE.finditer(text)]


def _is_in_code_block(pos: int, code_spans: list[tuple[int, int]]) -> bool:
    return any(s <= pos < e for s, e in code_spans)


def split_preserving_code(text: str) -> list[dict]:
    """Split text into segments: {'type': 'text'|'code', 'content': str}.
    
    Code blocks are returned as atomic 'code' segments.
    Text between code blocks is returned as 'text' segments.
    """
    code_spans = detect_code_blocks(text)
    if not code_spans:
        return [{"type": "text", "content": text}]

    segments = []
    prev_end = 0
    for start, end in code_spans:
        if prev_end < start:
            segments.append({"type": "text", "content": text[prev_end:start]})
        segments.append({"type": "code", "content": text[start:end]})
        prev_end = end
    if prev_end < len(text):
        segments.append({"type": "text", "content": text[prev_end:]})
    return segments


# ── Sentence splitting ───────────────────────────────────────────────
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences. Preserves code blocks as single units."""
    segments = split_preserving_code(text)
    sentences = []
    for seg in segments:
        if seg["type"] == "code":
            sentences.append(seg["content"])
        else:
            sents = _SENT_SPLIT_RE.split(seg["content"])
            sentences.extend(s for s in sents if s.strip())
    return sentences


def split_words_with_positions(text: str) -> list[tuple[str, int, int]]:
    """Split text into words with their character positions.
    
    Returns list of (word, start_char, end_char).
    """
    return [(m.group(), m.start(), m.end())
            for m in re.finditer(r"\S+", text)]


# ══════════════════════════════════════════════════════════════════════
# PERTURBATION STRATEGIES
# ══════════════════════════════════════════════════════════════════════

def sentence_permute(text: str, rng: Optional[random.Random] = None) -> tuple[str, dict]:
    """Shuffle sentence order in text, preserving code blocks in place.

    Returns (edited_text, edit_info).
    """
    if rng is None:
        rng = random.Random()

    segments = split_preserving_code(text)
    text_indices = [i for i, s in enumerate(segments) if s["type"] == "text"]

    if len(text_indices) <= 1:
        # Nothing to permute — only code or a single text segment
        # Try sentence-level within the text segment
        if text_indices:
            idx = text_indices[0]
            sents = _SENT_SPLIT_RE.split(segments[idx]["content"])
            if len(sents) <= 1:
                return text, {"strategy": "sentence_perm", "n_edits": 0}
            rng.shuffle(sents)
            segments[idx]["content"] = " ".join(sents)
            result = "".join(s["content"] for s in segments)
            return result, {"strategy": "sentence_perm", "n_edits": len(sents)}
        return text, {"strategy": "sentence_perm", "n_edits": 0}

    # Shuffle text segments while keeping code blocks in place
    text_contents = [segments[i]["content"] for i in text_indices]
    # Split each text segment into sentences and collect all
    all_sents = []
    for tc in text_contents:
        sents = _SENT_SPLIT_RE.split(tc)
        all_sents.extend(s for s in sents if s.strip())

    if len(all_sents) <= 1:
        return text, {"strategy": "sentence_perm", "n_edits": 0}

    rng.shuffle(all_sents)

    # Redistribute sentences back across text segments
    # Simple approach: put all sentences in the first text segment
    segments[text_indices[0]]["content"] = " ".join(all_sents)
    for idx in text_indices[1:]:
        segments[idx]["content"] = ""

    result = "".join(s["content"] for s in segments)
    return result, {"strategy": "sentence_perm", "n_edits": len(all_sents)}


def word_permute(text: str, target_positions: list[int],
                 rng: Optional[random.Random] = None) -> tuple[str, dict]:
    """Swap words at the given character positions within the text.

    `target_positions` are indices into the word list from
    `split_words_with_positions()`. Words inside code blocks are skipped.

    Returns (edited_text, edit_info).
    """
    if rng is None:
        rng = random.Random()

    code_spans = detect_code_blocks(text)
    words = split_words_with_positions(text)

    # Filter to valid, non-code-block positions
    valid_pos = [
        p for p in target_positions
        if 0 <= p < len(words) and not _is_in_code_block(words[p][1], code_spans)
    ]

    if len(valid_pos) < 2:
        return text, {"strategy": "word_perm", "n_edits": 0, "swaps": []}

    # Randomly pair up and swap
    rng.shuffle(valid_pos)
    n_swaps = len(valid_pos) // 2
    swaps = []

    # Build replacement list
    chars = list(text)
    # Process swaps in reverse order of position to avoid offset issues
    swap_pairs = [(valid_pos[i * 2], valid_pos[i * 2 + 1])
                  for i in range(n_swaps)]
    swap_pairs.sort(key=lambda p: max(words[p[0]][1], words[p[1]][1]),
                    reverse=True)

    result = text
    for a_idx, b_idx in swap_pairs:
        wa, sa, ea = words[a_idx]
        wb, sb, eb = words[b_idx]
        # Ensure sa < sb for ordered replacement
        if sa > sb:
            a_idx, b_idx = b_idx, a_idx
            wa, sa, ea = words[a_idx]
            wb, sb, eb = words[b_idx]
        # Replace second occurrence first (to preserve offsets)
        result = result[:sb] + wa + result[eb:]
        result = result[:sa] + wb + result[ea:]
        swaps.append({"from": wa, "to": wb, "pos_a": a_idx, "pos_b": b_idx})
        # Update word positions for subsequent swaps (they shifted)
        # This is approximate — for multiple swaps in sequence, positions drift
        # but for ≤5 swaps it's close enough

    return result, {"strategy": "word_perm", "n_edits": n_swaps * 2, "swaps": swaps}


def synonym_substitute(text: str, word_scores: list[tuple[str, float, int]],
                        rng: Optional[random.Random] = None,
                        max_substitutions: int = 5) -> tuple[str, dict]:
    """Replace high-attribution words with synonyms.

    Args:
        text: the original prompt text
        word_scores: list of (word, attribution_score, char_position) sorted
                     by score descending (highest attribution first)
        rng: random number generator
        max_substitutions: max words to replace

    Returns (edited_text, edit_info).
    """
    if rng is None:
        rng = random.Random()

    code_spans = detect_code_blocks(text)
    edits = []
    result = text

    for word, score, char_pos in word_scores:
        if len(edits) >= max_substitutions:
            break
        # Skip words inside code blocks
        if _is_in_code_block(char_pos, code_spans):
            continue
        # Skip very short words (articles, prepositions)
        if len(word) <= 2:
            continue

        synonyms = get_synonyms(word)
        if not synonyms:
            continue

        replacement = rng.choice(synonyms)

        # Case-match: if original was capitalized, capitalize replacement
        if word[0].isupper() and replacement[0].islower():
            replacement = replacement[0].upper() + replacement[1:]

        # Find and replace the specific occurrence at char_pos
        # (use the position to avoid replacing wrong occurrences)
        idx = result.find(word, max(0, char_pos - 50))
        if idx >= 0:
            result = result[:idx] + replacement + result[idx + len(word):]
            edits.append({
                "original": word,
                "replacement": replacement,
                "char_pos": char_pos,
                "attribution_score": float(score),
            })

    return result, {
        "strategy": "synonym_sub",
        "n_edits": len(edits),
        "substitutions": edits,
    }


def combined_perturb(text: str,
                     word_scores: list[tuple[str, float, int]],
                     strategy: str = "combined",
                     rng: Optional[random.Random] = None,
                     max_synonym_subs: int = 5) -> tuple[str, dict]:
    """Apply combined perturbation strategies.

    Strategy options:
      - "synonym": synonym substitution only
      - "sentence_perm": sentence permutation only
      - "word_perm": word permutation only
      - "combined": synonym → sentence perm → word perm (recommended)
      - "synonym+sentence_perm": synonym then sentence perm
      - "synonym+word_perm": synonym then word perm

    Returns (edited_text, combined_edit_info).
    """
    if rng is None:
        rng = random.Random()

    edit_log = {"strategy": strategy, "steps": [], "total_edits": 0}
    result = text

    strategies = strategy.split("+") if "+" in strategy else (
        ["synonym", "sentence_perm", "word_perm"] if strategy == "combined"
        else [strategy]
    )

    for strat in strategies:
        if strat == "synonym":
            result, info = synonym_substitute(
                result, word_scores, rng=rng, max_substitutions=max_synonym_subs)
        elif strat == "sentence_perm":
            result, info = sentence_permute(result, rng=rng)
        elif strat == "word_perm":
            # Permute the top attributed word positions
            top_positions = []
            words = split_words_with_positions(result)
            word_to_idx = {}
            for idx, (w, s, e) in enumerate(words):
                word_to_idx.setdefault(w.lower(), []).append(idx)
            for word, score, _ in word_scores[:10]:
                if word.lower() in word_to_idx:
                    top_positions.extend(word_to_idx[word.lower()])
            result, info = word_permute(result, top_positions, rng=rng)
        else:
            continue

        edit_log["steps"].append(info)
        edit_log["total_edits"] += info.get("n_edits", 0)

    return result, edit_log


# ── Utility: map token-level attribution to word-level ───────────────
def token_attrib_to_word_scores(
    tokens: list[str],
    attrib_scores: list[float],
    decoded_text: str,
) -> list[tuple[str, float, int]]:
    """Map sub-word token attributions to word-level scores.

    Aggregates token scores by summing over tokens that belong to the
    same whitespace-delimited word. Returns (word, score, char_position)
    sorted by score descending.
    """
    # Reconstruct approximate character mapping
    words = split_words_with_positions(decoded_text)
    if not words:
        return []

    # Build a simple token→character mapping
    # Tokens often have leading space markers (▁ or Ġ), strip those
    clean_tokens = []
    for t in tokens:
        ct = t.replace("▁", "").replace("Ġ", "").replace("Ċ", "\n")
        clean_tokens.append(ct)

    # Greedy alignment: walk through decoded text matching tokens
    word_scores: dict[int, float] = {}  # word_index → cumulative score
    text_pos = 0
    for tok_idx, (ctok, score) in enumerate(zip(clean_tokens, attrib_scores)):
        if not ctok:
            continue
        # Find this token in the text
        found = decoded_text.find(ctok, text_pos)
        if found < 0:
            found = decoded_text.find(ctok)  # fallback: search from start
        if found < 0:
            continue

        # Which word does this character position belong to?
        for wi, (word, ws, we) in enumerate(words):
            if ws <= found < we:
                word_scores[wi] = word_scores.get(wi, 0.0) + abs(score)
                break

        text_pos = found + len(ctok)

    # Build result sorted by score
    result = []
    for wi, score in word_scores.items():
        word, ws, we = words[wi]
        result.append((word, score, ws))

    result.sort(key=lambda x: -x[1])
    return result
