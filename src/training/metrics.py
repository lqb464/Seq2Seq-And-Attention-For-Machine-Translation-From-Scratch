import math
from collections import Counter
from typing import List


def _ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1)))


def compute_bleu(candidate: List[str], reference: List[str], max_n: int = 4, smooth: bool = True) -> float:
    effective_max_n = min(max_n, len(candidate), len(reference))
    if effective_max_n == 0:
        return 0.0

    precisions = []
    for n in range(1, effective_max_n + 1):
        cand = _ngram_counts(candidate, n)
        ref = _ngram_counts(reference, n)
        clipped = sum(min(count, ref.get(ng, 0)) for ng, count in cand.items())
        total = max(sum(cand.values()), 1)
        precision = clipped / total
        if smooth and precision == 0.0:
            precision = 1e-9
        precisions.append(precision)

    log_avg = sum(math.log(p) for p in precisions) / effective_max_n
    bp = 1.0
    if len(candidate) < len(reference):
        bp = math.exp(1 - len(reference) / max(len(candidate), 1))
    return float(bp * math.exp(log_avg))


def _lcs_len(a: List[str], b: List[str]) -> int:
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, start=1):
            old = dp[j]
            if x == y:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = old
    return dp[-1]


def compute_rouge_l(candidate: List[str], reference: List[str]) -> float:
    if not candidate or not reference:
        return 0.0
    lcs = _lcs_len(candidate, reference)
    precision = lcs / len(candidate)
    recall = lcs / len(reference)
    if precision + recall == 0:
        return 0.0
    return float((2 * precision * recall) / (precision + recall))


def compute_chrf(candidate: List[str], reference: List[str], max_n: int = 6, beta: float = 2.0) -> float:
    cand_text = " ".join(candidate)
    ref_text = " ".join(reference)
    if not cand_text or not ref_text:
        return 0.0

    precisions, recalls = [], []
    for n in range(1, max_n + 1):
        cand = Counter(cand_text[i:i + n] for i in range(max(0, len(cand_text) - n + 1)))
        ref = Counter(ref_text[i:i + n] for i in range(max(0, len(ref_text) - n + 1)))
        overlap = sum(min(count, ref.get(ng, 0)) for ng, count in cand.items())
        precisions.append(overlap / max(sum(cand.values()), 1))
        recalls.append(overlap / max(sum(ref.values()), 1))

    precision = sum(precisions) / max(len(precisions), 1)
    recall = sum(recalls) / max(len(recalls), 1)
    if precision + recall == 0:
        return 0.0
    beta2 = beta * beta
    return float((1 + beta2) * precision * recall / (beta2 * precision + recall))
