import re
import string
from typing import List, Dict, Any

### HELPERS
RE_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
RE_ANS = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

def _extract_answer(text:str) -> str:
    """
    Return the first <answer>...</answer> contents, or empty string.
    """
    m = RE_ANS.search(text)
    return m.group(1).strip() if m else ""

### FORMAT REWARDS
def _score_single(text: str,
                  max_think_chars: int = 4000) -> float:
    """
    Return a score in [-1, 1] based on formatting quality.
    """
    
    # extract think, ans blocks
    think = RE_THINK.findall(text)
    ans = RE_ANS.findall(text)

    # init scoring variables
    has_both = bool(think) and bool(ans) # does text have both think and answer tags?
    single_pair = (len(think) == 1 and len(ans) == 1) # is there only one think and answer tag?
    order_ok = False # is think before answer?
    non_empty = False # are tags filled out?
    min_leak = False # is anything outside the tags?
    overlong_pen = 0.0 # are there too many thinking chars?

    if single_pair:
        tspan = RE_THINK.search(text).span()
        aspan = RE_ANS.search(text).span()

        # check for order, emptiness
        order_ok = tspan[0] < aspan[0]
        non_empty = (think[0].strip() != "") and (ans[0].strip() != "")

        # check for leakage outside tags
        pre = text[:tspan[0]]
        mid = text[tspan[1]:aspan[0]]
        post = text[aspan[1]:]
        min_leak = (pre.strip() == "" and mid.strip() and post.strip() == "")

        # check for overlong
        if len(think[0]) > max_think_chars:
            overlong_pen = 0.2

    # scoring 
    score = 0.0
    score += 0.4 * float(has_both)
    score += 0.2 * float(order_ok)
    score += 0.2 * float(single_pair)
    score += 0.1 * float(non_empty)
    score += 0.1 * float(min_leak)
    score -= overlong_pen
    
    return max(-1.0, min(1.0, score))

def format_reward(completions, **kwargs):
    """
    TRL/GRPO compatible reward:
        - completions: list[list[{"content": str}]]
        - returns: list[float]
    """
    texts = [c[0]["content"] for c in completions]
    max_think_chars = int(kwargs.get("max_think_chars", 4000))
    return [_score_single(t, max_think_chars=max_think_chars) for t in texts]

### N-GRAM MATCHING
_TRIVIAL = {
    "it", "its", "they", "their",
    "that", "this", "which", "is",
    "are", "were", "be", "to",
    "a", "an", "the", "some",
    "as", "and", "also",
}
_NOPUNC = str.maketrans("", "", string.punctuation)

def _clean_and_split(s: str) -> List[str]:
    return s.lower().translate(_NOPUNC).split()

def check_accuracy(preds: List[str], targets: List[str]) -> List[float]:
    # split into unique non-trivial words
    pred_w = [set(_clean_and_split(p)) - _TRIVIAL for p in preds]
    target_w = [set(_clean_and_split(t)) - _TRIVIAL for t in targets]

    # extract words present in both preds and targets
    overlap = [p & t for p, t in zip(pred_w, target_w)]

    # compute ratio of present to total words
    acc = []
    for o, t in zip(overlap, target_w):
        acc.append((len(o) / max(1, len(t))) if t else 0.0)

    return acc 

### CORRECTNESS REWARD
def correctness_reward(completions: List[List[Dict[str, Any]]], **kwargs) -> List[float]:
    """
    Compare model <answer> block to dataset targets.
    Expects one of these columns in **kwargs**: "target", "answer".
    """

    # extract textual preds from <answer> block
    preds = [_extract_answer(c[0]["content"]) for c in completions]

    # ground truth dataset
    targets = kwargs.get("target")
    if targets is None:
        targets = kwargs.get("answer")

    # if no targets provided, give zero reward (training still runs) 
    if targets is None:
        return [0.0] * len(preds)

    return check_accuracy(preds, targets)

