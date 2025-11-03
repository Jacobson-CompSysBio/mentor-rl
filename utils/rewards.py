import re
import string
from typing import List, Dict, Any

### HELPERS
RE_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
RE_ANS = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

_TRIVIAL = {
    "it","its","they","their","that","this","which","is","are","were","be","to",
    "a","an","the","some","as","and","also",
}
_NOPUNC = str.maketrans("", "", string.punctuation)

def _extract_answer(text:str) -> str:
    """
    Return the first <answer>...</answer> contents, or empty string.
    """
    m = RE_ANS.search(text)
    return m.group(1).strip() if m else ""

def _to_text(c) -> str:
    # Accept str, dict(content=...), or list of dicts/strings; join if needed
    if isinstance(c, str):
        return c
    if isinstance(c, dict) and "content" in c:
        return str(c["content"])
    if isinstance(c, list):
        parts = []
        for x in c:
            if isinstance(x, dict) and "content" in x:
                parts.append(str(x["content"]))
            else:
                parts.append(str(x))
        return " ".join(parts)
    return str(c)


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

def format_reward(completions, max_think_chars=4000, **kwargs):
    texts = [_to_text(c) for c in completions]

    thinks, answers = [], []
    for t in texts:
        m_think = re.search(r"<think>(.*?)</think>", t, flags=re.S)
        m_ans   = re.search(r"<answer>(.*?)</answer>", t, flags=re.S)
        think   = (m_think.group(1) if m_think else "")[:max_think_chars]
        answer  = (m_ans.group(1) if m_ans else "").strip()
        thinks.append(think)
        answers.append(answer)

    # Example: simple formatting reward (you can keep your original logic here)
    # 1. has both tags
    has_tags = [1.0 if ("<think>" in t and "</think>" in t and "<answer>" in t and "</answer>" in t) else 0.0 for t in texts]
    # 2. non-empty answer
    nonempty_answer = [1.0 if a else 0.0 for a in answers]

    # combine however your original function did; here’s a simple sum
    return [0.5 * h + 0.5 * n for h, n in zip(has_tags, nonempty_answer)]

### CORRECTNESS
def _clean_and_split(s: str) -> List[str]:
    return s.lower().translate(_NOPUNC).split()

def check_accuracy(preds: List[str], targets: List[str]) -> List[float]:
    pred_w   = [set(_clean_and_split(p)) - _TRIVIAL for p in preds]
    target_w = [set(_clean_and_split(t)) - _TRIVIAL for t in targets]
    acc = []
    for pw, tw in zip(pred_w, target_w):
        acc.append((len(pw & tw) / max(1, len(tw))) if tw else 0.0)
    return acc

def correctness_reward(completions: List[Any], **kwargs) -> List[float]:
    """
    Compare the model's <answer> block to dataset targets.
    Accepts various completion shapes (strings, dicts, lists of dicts).
    Expects a target list/column in kwargs under 'target' or 'answer'.
    Returns one float per completion.
    """
    # Normalize completions to strings and extract <answer> (fallback to full text)
    texts = [_to_text(c) for c in completions]
    preds = [_extract_answer(t) or t for t in texts]

    # Pull targets from the batch (TRL passes batch columns in **kwargs)
    targets = kwargs.get("target", kwargs.get("answer"))
    if targets is None:
        # No ground truth available => neutral reward
        return [0.0] * len(preds)

    # Normalize targets to list[str]
    if isinstance(targets, list):
        targets_norm = [_to_text(t) for t in targets]
    else:
        targets_norm = [_to_text(targets)]

    # Align lengths: GRPO often has num_generations per prompt
    if len(targets_norm) != len(preds):
        if len(targets_norm) == 1:
            targets_norm = targets_norm * len(preds)
        elif len(preds) % len(targets_norm) == 0:
            k = len(preds) // len(targets_norm)
            targets_norm = [t for t in targets_norm for _ in range(k)]
        else:
            # Fallback: repeat and trim
            reps = (len(preds) + len(targets_norm) - 1) // len(targets_norm)
            targets_norm = (targets_norm * reps)[:len(preds)]

    return check_accuracy(preds, targets_norm)

