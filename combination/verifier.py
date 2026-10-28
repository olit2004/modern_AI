"""
Infallible Linguistic Referee & Mistake Miner for Amharic LLM Distillation.
Evaluates:
- Ethiopic Unicode Script Purity (detects foreign script mixing)
- Repetition / Degenerate Loop Penalty (detects repetitive phrase loops)
- Character n-gram alignment against Teacher Reference
"""
from typing import Dict, Any

ETHIOPIC_RANGE = (0x1200, 0x137F)
ETHIOPIC_EXT_RANGE = (0x2D80, 0x2DDF)

def compute_script_purity(text: str) -> float:
    """Calculates ratio of Ethiopic characters among non-whitespace/punctuation characters."""
    if not text or not text.strip():
        return 0.0
    total_alpha = 0
    ethiopic_chars = 0
    for char in text:
        cp = ord(char)
        if char.isspace() or char in '.,!?:;"\'()[]{}0123456789':
            continue
        total_alpha += 1
        if (ETHIOPIC_RANGE[0] <= cp <= ETHIOPIC_RANGE[1]) or (ETHIOPIC_EXT_RANGE[0] <= cp <= ETHIOPIC_EXT_RANGE[1]):
            ethiopic_chars += 1
    return (ethiopic_chars / total_alpha) if total_alpha > 0 else 0.0

def detect_repetition_penalty(text: str) -> float:
    """Detects repetitive token loops and degeneration."""
    words = text.strip().split()
    if len(words) < 4:
        return 0.0
    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
    unique_ratio = len(set(bigrams)) / len(bigrams)
    return max(0.0, 1.0 - unique_ratio)

def compute_chrf_overlap(pred: str, ref: str, n: int = 4) -> float:
    """Lightweight character n-gram F-score against teacher reference."""
    def get_char_ngrams(s, n_gram):
        s_clean = ''.join(s.split())
        return [s_clean[i:i+n_gram] for i in range(len(s_clean)-n_gram+1)] if len(s_clean) >= n_gram else []
    
    pred_ngrams = get_char_ngrams(pred, n)
    ref_ngrams = get_char_ngrams(ref, n)
    if not pred_ngrams or not ref_ngrams:
        return 0.0
    common = set(pred_ngrams).intersection(set(ref_ngrams))
    precision = len(common) / len(pred_ngrams)
    recall = len(common) / len(ref_ngrams)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)

class LinguisticReferee:
    """Infallible Referee for low-resource Amharic language capability."""
    
    @staticmethod
    def evaluate(pred: str, teacher_ref: str = "") -> Dict[str, Any]:
        purity = compute_script_purity(pred)
        rep_penalty = detect_repetition_penalty(pred)
        alignment = compute_chrf_overlap(pred, teacher_ref) if teacher_ref else 0.5
        
        score = (0.50 * purity) + (0.25 * (1.0 - rep_penalty)) + (0.25 * alignment)
        
        if purity < 0.40 or rep_penalty > 0.50 or score < 0.40:
            status = "BLUNDER"
        elif score < 0.70:
            status = "SUBOPTIMAL"
        else:
            status = "OPTIMAL"
            
        return {
            "score": round(score, 4),
            "script_purity": round(purity, 4),
            "rep_penalty": round(rep_penalty, 4),
            "alignment": round(alignment, 4),
            "status": status
        }
