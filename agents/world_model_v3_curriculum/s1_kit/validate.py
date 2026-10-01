#!/usr/bin/env python3
"""Check a question file's worked traces against the contract in contract/.

Every violation carries the name of the check that produced it, so the coverage
table and the totals reconcile by construction. Refuses to report a pass when it
cannot do its job.
"""
import hashlib, json, re, sys, pathlib
from collections import defaultdict, Counter

try:
    import yaml
except ImportError:
    sys.exit("ERROR: pyyaml is not installed for %s\n       run: %s -m pip install pyyaml"
             % (sys.executable, sys.executable))

HERE = pathlib.Path(__file__).resolve().parent
C = HERE / "contract"
if not C.is_dir():
    sys.exit("ERROR: no contract/ directory next to %s" % __file__)
V = yaml.safe_load((C / "vocabularies.yaml").read_text())
G = yaml.safe_load((C / "layer_grammars.yaml").read_text())
T = yaml.safe_load((C / "tools.yaml").read_text())
P = yaml.safe_load((C / "provenance.yaml").read_text())
B = yaml.safe_load((C / "blocked_reasons.yaml").read_text())

R = yaml.safe_load((C / "rows.yaml").read_text())

if len(sys.argv) < 2:
    sys.exit("usage: validate.py <question-file.md | corpus.jsonl>")
qf = pathlib.Path(sys.argv[1])
if not qf.is_file():
    sys.exit("ERROR: no such question file: %s" % qf)
src = qf.read_text()

# ---------- parse ----------
# Both parsers give the same trace record, so every check reads a markdown
# trace and a generated row the same way.
def new_trace(tid, family=None):
    return {"id": tid, "family": family, "answer": [], "calls": [], "results": [],
            "context": [], "earlier": [], "user": [], "earlier_user": [],
            "harness": [], "roles": [], "meta": None, "system": None,
            "tool_defs": None, "shape": []}

def norm(l):
    l = (l.replace("“", '"').replace("”", '"')
          .replace("‘", "'").replace("’", "'"))
    return l.strip().strip("`").strip()

ROLES = ("User:", "Thinking:", "Tool call:", "Tool result:",
         "Answer:", "Final answer:", "Context (from the harness):")

def parse_markdown(src):
    """Return the traces of a question file and its number of trace titles."""
    lines = [norm(l) for l in src.split("\n")]
    traces, cur, role, family = [], None, None, None
    for raw in lines:
        l = raw.strip()
        if raw.lstrip().startswith("#"):
            fm = re.match(r"^#+\s*(S[0-9]+(?:\.[0-9]+)*)\b", raw.strip())
            family = fm.group(1) if fm else None
            cur, role = None, None; continue
        h = re.match(r"^\*\*([^*]+)\*\*", l)
        if h:
            if "verall" in h.group(1):
                cur, role = None, None; continue
            cur = new_trace(h.group(1).strip(), family)
            traces.append(cur); role = None; continue
        if cur is None or not l:
            continue
        # The corpus marks a multi-turn trace with [Earlier turn: ...] followed
        # by [This turn: trained]. The earlier turn is context, masked and not
        # trained on, so its Answer block is not this trace's answer. Reaching
        # [This turn] discards whatever the earlier turn contributed.
        if l.startswith("[This turn"):
            # The earlier turn is context, not this trace's answer, but it is
            # still evidence: its identifiers came from a registry lookup, so it
            # decides whether this turn's answer must carry provenance, and
            # which block.
            if cur["answer"]:
                cur["earlier"].append(" ".join(cur["answer"]))
            cur["earlier_user"].extend(cur["user"]); cur["user"] = []; cur["roles"] = []
            cur["context"].extend(cur["answer"]); cur["answer"] = []; role = None; continue
        if l.startswith("[Earlier turn"):
            role = None; continue
        if l in ROLES:
            role = l
            cur["roles"].append(l)        # the order of steps in the trained turn
            # A call or a result can span several lines. Each header opens a
            # new one, so the n-th call pairs with the n-th result.
            if l == "Tool call:":
                cur["calls"].append([])
            elif l == "Tool result:":
                cur["results"].append([])
            continue
        if role in ("Answer:", "Final answer:"):
            cur["answer"].append(l)
        elif role == "Tool call:":
            cur["calls"][-1].append(l)
        elif role == "Tool result:":
            cur["results"][-1].append(l)
        elif role == "Context (from the harness):":
            cur["context"].append(l)
            cur["harness"].append(l)
        elif role == "User:":
            cur["user"].append(l)
    titles = [l for l in lines
              if re.match(r"^\*\*[^*]+\*\*", l.strip()) and "verall" not in l]
    return traces, len(titles)

def as_text(content):
    return content if isinstance(content, str) else json.dumps(content)

def row_shape_problems(row):
    """Return what keeps a row from the form of rows.yaml."""
    problems = []
    missing = [k for k in R["row_keys"] if k not in row]
    if missing:
        problems.append("no %s" % ", ".join(missing))
    msgs = row.get("messages") if isinstance(row.get("messages"), list) else []
    roles = [m.get("role") if isinstance(m, dict) else None for m in msgs]
    unknown = sorted({str(x) for x in roles if x not in R["roles"]})
    if unknown:
        problems.append("unknown roles %s" % ", ".join(unknown))
    if roles[:1] != ["system"] or roles.count("system") != 1:
        problems.append("the first message, and only it, must be the system prompt")
    if not roles or roles[-1] != "assistant":
        problems.append("the row does not end with an assistant answer")
    if any(isinstance(m, dict) and m.get("role") == "assistant"
           and ("tool_call" in m) == ("content" in m) for m in msgs):
        problems.append("an assistant message needs a tool_call or content, not both")
    return problems

def parse_jsonl(src):
    """Return the traces of a JSONL corpus and its number of rows.

    A row that is not JSON gives no trace, so it counts as skipped and the run
    fails rather than passing over it."""
    traces, n_rows = [], 0
    for lineno, raw in enumerate(src.split("\n"), 1):
        if not raw.strip():
            continue
        n_rows += 1
        try:
            row = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(row, dict):
            continue
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        t = new_trace("%s: %s" % (row.get("row_id", "line %d" % lineno),
                                  meta.get("source_trace", "row")))
        t["meta"], t["tool_defs"], t["shape"] = meta, row.get("tools"), row_shape_problems(row)
        msgs = row.get("messages") if isinstance(row.get("messages"), list) else []
        users = [i for i, m in enumerate(msgs) if isinstance(m, dict) and m.get("role") == "user"]
        turn = users[-1] if users else len(msgs)   # the last user message starts the trained turn
        for i, m in enumerate(msgs):
            if not isinstance(m, dict):
                continue
            role, text = m.get("role"), as_text(m.get("content"))
            if role == "system":
                t["system"] = text
            elif role == "harness":
                t["context"].append(text)
                t["harness"].append(text)
            elif role == "user":
                (t["user"] if i == turn else t["earlier_user"]).append(text)
            elif role == "assistant" and i < turn:
                # An earlier turn is rendered as at inference: its answer only.
                t["earlier"].append(text)
                t["context"].append(text)
            elif role == "assistant":
                if m.get("thinking"):
                    t["roles"].append("Thinking:")
                call = m.get("tool_call")
                if isinstance(call, dict):
                    t["roles"].append("Tool call:")
                    t["calls"].append(["%s(%s)" % (call.get("name"), json.dumps(call.get("arguments")))])
                else:
                    t["roles"].append("Answer:")
                    t["answer"].append(text)
            elif role == "tool":
                t["roles"].append("Tool result:")
                t["results"].append([text])
        traces.append(t)
    return traces, n_rows

JSONL = qf.suffix == ".jsonl"
traces, n_items = parse_jsonl(src) if JSONL else parse_markdown(src)
skipped = n_items - len(traces)

# ---------- contract helpers ----------
def parse_tag(tag):
    parts = tag.split(G["sep"])
    for fam, g in G["families"].items():
        if parts[0] != g["match_prefix"]:
            continue
        if len(parts) != g["arity"]:
            return "arity %d, %s grammar expects %d" % (len(parts), fam, g["arity"])
        f = dict(zip(g["fields"], parts))
        # bulkPEN, scPEN and TFs repeat the family in field 2. A tag whose
        # family part disagrees with its prefix, such as
        # bulkPEN:tf_target:brain:amygdala:v1, names two families at once.
        if "family_part" in g and f["family_part"] != g["family_part"]:
            return "family part %r, %s grammar expects %r" % (
                f["family_part"], fam, g["family_part"])
        if not re.fullmatch(G["version_pattern"], f["version"]):
            return "version %r does not match %s" % (f["version"], G["version_pattern"])
        return None
    return "no family grammar matches first field %r" % parts[0]

TAG_PREFIXES = [g["match_prefix"] for g in G["families"].values()]
TAG_RE = re.compile(r"\b(?:%s)[A-Za-z0-9_:.\-]*[A-Za-z0-9]"
                    % "|".join(re.escape(p) for p in TAG_PREFIXES))

# Each blocked_reason template becomes an anchored pattern with one group per
# slot. A sentence must match exactly one template, and filling that template
# from the answer must give the sentence back.
TEMPLATES = []
for code, ts in B["codes"].items():
    for t in ts:
        pat = re.escape(t["template"])
        for slot in t["slots"]:
            pat = pat.replace(re.escape("{%s}" % slot), "(?P<%s>.+?)" % slot)
        TEMPLATES.append((code, t, re.compile(pat)))

def at_path(obj, path, side):
    node = obj
    for part in path.replace("<side>", side or "").split("."):
        if isinstance(node, list) and part.isdigit():
            node = node[int(part)] if int(part) < len(node) else None
        elif isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node

def refill(t, m, a, meta=None):
    """Fill template t from answer a. Return the sentence, or a problem."""
    r = a.get("result") if isinstance(a.get("result"), dict) else {}
    # In S1.4 the slot reads the side that blocks the answer, the one whose
    # gene_id is null.
    side = next((s for s in ("first", "second")
                 if isinstance(r.get(s), dict) and r[s].get("gene_id") is None), None)
    out = t["template"]
    for slot, how in t["slots"].items():
        got, rendered = None, False
        for path in how["from"]:
            if path.startswith("metadata."):
                # A generated row carries its hidden metadata, so the slot is
                # filled from it. A question file has none, so the captured
                # word must at least be one of the labels for the slot.
                if meta is None:
                    got = m.group(slot) if m.group(slot) in B["labels"][slot].values() else None
                else:
                    got = B["labels"][slot].get(meta.get(path.split(".", 1)[1]))
                rendered = True
                break
            got = at_path(a, path, side)
            if got is not None:
                break
        if got is None:
            return None, "slot {%s} has no value at %s" % (slot, ", ".join(how["from"]))
        if how["kind"] == "count":
            got = len(got) if isinstance(got, list) else None
            if got is None:
                return None, "slot {%s} is a count but %s is not a list" % (slot, how["from"])
        elif how["kind"] == "label" and not rendered:
            got = B["labels"][slot].get(got)
            if got is None:
                return None, "slot {%s} has no label" % slot
        out = out.replace("{%s}" % slot, str(got))
    return out, None

def match_reason(sentence, a, meta=None):
    """Return (code, template, problem) for one blocked_reason sentence.

    A generated row names its code in the hidden metadata, so only the
    templates of that code may match. A question file has no code, so any
    template may."""
    hits = [(code, t, m) for code, t, rx in TEMPLATES for m in [rx.fullmatch(sentence)] if m]
    if meta is not None:
        code = meta.get("blocked_reason_code")
        if not code:
            return None, None, "the row has no blocked_reason_code"
        others = sorted({c for c, _, _ in hits if c != code})
        hits = [h for h in hits if h[0] == code]
        if not hits and others:
            return None, None, "the sentence is a %s template, but the row's code is %s" % (
                others[0], code)
    if len(hits) != 1:
        return None, None, "%d templates match %r" % (len(hits), sentence[:70])
    code, t, m = hits[0]
    filled, problem = refill(t, m, a, meta)
    if problem:
        return code, t, "%s: %s" % (t["id"], problem)
    if filled != sentence:
        return code, t, "%s filled from the answer gives %r" % (t["id"], filled[:70])
    return code, t, None

def provenance_shape(prov):
    """Return a problem with a provenance block, or None."""
    if not isinstance(prov, dict):
        return "rests on registry evidence but carries no provenance block"
    if list(prov) != P["required_at_s1"]:
        return "provenance has fields %s, expects %s" % (list(prov), P["required_at_s1"])
    if not re.fullmatch(P["registry_block"]["evidence_id_pattern"], str(prov["evidence_id"])):
        return "evidence_id %r does not match %s" % (
            prov["evidence_id"], P["registry_block"]["evidence_id_pattern"])
    return None

def tool_block_problem(producer, prov, status):
    """Return a problem with the block that a tool or the harness produced."""
    # A failed call read no record, so its block is null. A successful call
    # carries the block of its producer: at S1 every producer gives the
    # registry block, and the evidence_id prefix names the producer.
    if status == "error":
        return None if prov is None else "an error result carries %s" % json.dumps(prov)
    problem = provenance_shape(prov)
    if problem:
        return problem
    want = P["registry_block"]
    if (prov["registry"], prov["release"]) != (want["registry"], want["release"]):
        return "the block names %s release %s, a registry block is %s release %s" % (
            prov["registry"], prov["release"], want["registry"], want["release"])
    prefix = want["evidence_id_prefix"].get(producer)
    if prefix and not prov["evidence_id"].startswith(prefix):
        return "evidence_id %r, but %s gives %s IDs" % (prov["evidence_id"], producer, prefix)
    return None

def strings_in(obj):
    """Return every string value and every key in obj."""
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(k); out |= strings_in(v)
    elif isinstance(obj, list):
        for v in obj:
            out |= strings_in(v)
    elif isinstance(obj, str):
        out.add(obj)
    return out

REG_HEADER = P["harness_registry_block"]["header"]
ANCHOR_RE = re.compile(r"<([A-Za-z_]+):([^>|]+)(?:\|([^>]+))?>")
EVIDENCE_ID_RE = re.compile(r'"evidence_id":\s*"([^"]*)"')

# ---------- check ----------
CHECKS = ["status vocabulary", "continuation vocabulary", "defer needs blocked_reason",
          "blocked_reason matches template", "blocked_reason fits status",
          "blocked_reason null unless defer", "tool in inventory", "layer tag parses",
          "rejected input not parsed", "no by-reference placeholder",
          "provenance present", "provenance copied", "no invented provenance",
          "context answer carries provenance",
          "trace label matches its family", "claim sets partition the vocabulary", "tool failure not reported as a negative",
          "layer_family vocabulary", "claim vocabulary", "claim order",
          "no duplicate prompt", "thinking after tool result", "tool result shape",
          "provenance key present", "tool provenance fits tool",
          "evidence_id unique across rows", "anchor vocabulary", "summary anchors in result",
          "summary asks only on defer", "defer summary asks for input", "failure keys",
          "row shape", "no hidden field in visible text", "one prompt for every row"]
seen = defaultdict(int)
issues = []                       # (trace_id, check_name, detail)
def flag(tid, check, detail): issues.append((tid, check, detail))
evidence_owner = {}               # evidence_id -> the first trace that has it
prompt_owner = {}                 # visible input and call outcomes -> the first trace

for t in traces:
    tid = t["id"]
    if t["meta"] is not None:
        # A generated row has the form of rows.yaml, so that each check reads
        # the step it expects.
        seen["row shape"] += 1
        if t["shape"]:
            flag(tid, "row shape", "; ".join(t["shape"]))
        # The model must not see a hidden field. A field name, or the book
        # mode, step, family, or source trace of the row in the prompt, tells
        # the model the form or the answer before it acts.
        visible = " ".join([t["system"] or ""] + t["harness"] + t["earlier_user"] + t["user"])
        leaks = [n for n in R["names_never_visible"] if n in visible]
        leaks += ["%s %r" % (k, t["meta"][k]) for k in R["values_never_visible"]
                  if isinstance(t["meta"].get(k), str) and t["meta"][k]
                  and re.search(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(t["meta"][k]), visible)]
        seen["no hidden field in visible text"] += 1
        if leaks:
            flag(tid, "no hidden field in visible text", "visible: %s" % ", ".join(leaks))
    a = r = None                  # never let a previous trace's parse leak in
    pairs = []                    # (tool name, parsed result or None)
    for i, call in enumerate(t["calls"]):
        m = re.match(r"^([a-z_]+)\(", " ".join(call))
        name = m.group(1) if m else None
        if m:
            seen["tool in inventory"] += 1
            if name not in T["s1_callable"]:
                flag(tid, "tool in inventory", name)
        res = None
        if i < len(t["results"]):
            try:
                res = json.loads(" ".join(t["results"][i]))
            except Exception as e:
                flag(tid, "tool result JSON parses", str(e)[:60])
        pairs.append((name, res))
    # A failed call is not evidence. It read no record, so there is nothing to
    # cite. Only a tool result that is not an error counts.
    good = [(n, res) for n, res in pairs
            if isinstance(res, dict) and res.get("status") != "error"]
    lookups = [res for n, res in good if n in P["lookup_tools"]]

    block = None
    ctx = " ".join(t["context"])
    if REG_HEADER in ctx:
        try:
            block, _ = json.JSONDecoder().raw_decode(
                ctx[ctx.index(REG_HEADER) + len(REG_HEADER):].lstrip())
        except ValueError as e:
            flag(tid, "harness registry block parses", str(e)[:60])
    earlier = []
    for prior in t["earlier"]:
        try:
            earlier.append(json.loads(prior))
        except ValueError:
            pass                  # the context-answer check reads the raw text
    # The evidence an answer rests on, in order: this turn's first successful
    # lookup, an earlier turn's answer, the harness registry block, and last a
    # successful membership result. A near-miss row has a block and a lookup,
    # and its answer copies the lookup. An answer that rests on a lookup and a
    # membership check copies the lookup.
    evidence = None
    if lookups:
        evidence = lookups[0].get("provenance")
    elif any(isinstance(e.get("provenance"), dict) for e in earlier):
        evidence = [e for e in earlier if isinstance(e.get("provenance"), dict)][-1]["provenance"]
    elif isinstance(block, dict) and isinstance(block.get("provenance"), dict):
        evidence = block["provenance"]
    elif good:
        evidence = good[0][1].get("provenance")

    # A Thinking step follows each tool result, before the next call or the
    # answer. The S0 renderer once dropped it, so a model trained on those rows
    # answered straight from a result it never reasoned about.
    n_result = 0
    for i, step in enumerate(t["roles"]):
        if step != "Tool result:":
            continue
        n_result += 1
        seen["thinking after tool result"] += 1
        nxt = t["roles"][i + 1] if i + 1 < len(t["roles"]) else "the end of the trace"
        if nxt != "Thinking:":
            flag(tid, "thinking after tool result", "tool result %d is followed by %s" % (n_result, nxt))

    # The tool writes its result, so a result the tool cannot produce teaches
    # the model to read a shape it will never see. status is first, provenance
    # second, and the other keys follow tools.yaml for that status.
    for n, res in pairs:
        if not isinstance(res, dict) or n not in T["s1_callable"]:
            continue
        seen["tool result shape"] += 1
        st = res.get("status")
        want = T["error_result"]["keys"] if st == "error" else T["tools"][n]["returns"].get(st)
        if list(res) != want:
            detail = "%s %s result has keys %s, expects %s" % (n, st, list(res), want)
            flag(tid, "tool result shape", detail)

    # The block a result carries must be the block its producer writes: null
    # after a failed call, the registry block otherwise, with the evidence_id
    # prefix of that producer. The harness registry block is a producer too.
    blocks = [(n, "provenance" in res, res.get("provenance"), res.get("status"))
              for n, res in pairs if isinstance(res, dict) and n in T["s1_callable"]]
    if isinstance(block, dict):
        blocks.append(("harness_registry_block", "provenance" in block,
                       block.get("provenance"), "resolved"))
    for producer, has_key, prov, st in blocks:
        if not has_key:
            continue              # the provenance key check reports it
        seen["tool provenance fits tool"] += 1
        problem = tool_block_problem(producer, prov, st)
        if problem:
            flag(tid, "tool provenance fits tool", "%s: %s" % (producer, problem))

    # The runtime makes a new evidence_id for every call, so one ID in two rows
    # means a row was copied, and a copied ID can be memorized.
    own_text = " ".join([l for res in t["results"] for l in res] + t["context"] + t["answer"])
    for ev in sorted(set(EVIDENCE_ID_RE.findall(own_text))):
        seen["evidence_id unique across rows"] += 1
        owner = evidence_owner.setdefault(ev, tid)
        if owner != tid:
            flag(tid, "evidence_id unique across rows", "%s also appears in %s" % (ev, owner))

    # S1.8.6 once repeated S1.8.5 word for word. Two rows may share user text
    # on purpose: a failure row repeats its success row (S1.1.1 and S1.1.8),
    # and a near-miss row repeats its prompt under another harness block
    # (S1.3.1 and S1.3.4). So the key is everything the model sees before it
    # acts, plus the outcome of each call. evidence_ids are left out of it,
    # because they differ in every row and would hide a copy.
    if t["user"]:
        visible = [" ".join(t["harness"]), " ".join(t["earlier_user"]),
                   " | ".join(t["earlier"]), " ".join(t["user"])]
        visible = [EVIDENCE_ID_RE.sub('"evidence_id": ""', v) for v in visible]
        outcomes = tuple((n, res.get("status") if isinstance(res, dict) else None)
                         for n, res in pairs)
        seen["no duplicate prompt"] += 1
        owner = prompt_owner.setdefault((tuple(visible), outcomes), tid)
        if owner != tid:
            flag(tid, "no duplicate prompt", "the same input and tool outcomes as %s" % owner)

    if not t["answer"]:
        continue
    blob = " ".join(t["answer"])
    # Seven blocks in this corpus once read "(the same answer as trace 1)". A
    # generator cannot resolve that, and an answer that is only a placeholder
    # used to be excused here as a category of its own, which made it invisible
    # to every other check. It is a defect, so it is flagged.
    phs = re.findall(r"\((?:the same|the answer)[^)]*\)|\[Earlier turn: the same[^\]]*\]", blob)
    for ph in phs:
        seen["no by-reference placeholder"] += 1
        flag(tid, "no by-reference placeholder", ph)
    if phs:
        continue
    try:
        a = json.loads(blob)
        if not isinstance(a, dict):
            raise ValueError("the answer is not one JSON object")
    except Exception as e:
        flag(tid, "answer JSON parses", str(e)[:60])
        continue
    r = a.get("result") if isinstance(a.get("result"), dict) else {}

    seen["status vocabulary"] += 1
    if a.get("status") not in V["status"]:
        flag(tid, "status vocabulary", repr(a.get("status")))
    seen["continuation vocabulary"] += 1
    if a.get("continuation_decision") not in V["continuation_decision"]:
        flag(tid, "continuation vocabulary", repr(a.get("continuation_decision")))

    # The provenance key is always present, on the answer, on every tool
    # result, and on the harness registry block. The model decides its value,
    # not whether to write it, so null and a missing key are different.
    missing = ["%s result" % n for n, res in pairs if isinstance(res, dict) and "provenance" not in res]
    if isinstance(block, dict) and "provenance" not in block:
        missing.append("harness registry block")
    if "provenance" not in a:
        missing.append("answer")
    seen["provenance key present"] += 1
    if missing:
        flag(tid, "provenance key present", "no provenance key in: %s" % ", ".join(missing))

    # The summary is readable text with anchors. An anchor type outside the
    # list is a type the model invented. An anchored ID must be in the result,
    # and a symbol label must come from the result, a tool result, or an item,
    # so the summary claims nothing that the result does not.
    summary = a.get("summary") if isinstance(a.get("summary"), str) else ""
    anchors = ANCHOR_RE.findall(summary)
    for kind, value, label in anchors:
        seen["anchor vocabulary"] += 1
        if kind not in V["anchor_type"]:
            flag(tid, "anchor vocabulary", "<%s:%s>" % (kind, value))
    in_result = strings_in(r)
    given = " ".join([l for res in t["results"] for l in res] + t["context"] + t["user"])
    for kind, value, label in anchors:
        if kind not in V["anchor_type"]:
            continue
        seen["summary anchors in result"] += 1
        problem = None
        if value not in in_result:
            problem = "<%s:%s> is not in the result" % (kind, value)
        elif label and label not in in_result and label not in given:
            problem = "the label %s of %s comes from no result, tool result, or item" % (label, value)
        if problem:
            flag(tid, "summary anchors in result", problem)

    # The summary asks the user for input only when the answer defers, and a
    # deferring summary always says what input is needed. The phrases that
    # ask are listed in vocabularies.yaml.
    asks = [p for p in V["input_request_phrases"] if p in summary]
    if a.get("continuation_decision") == "defer":
        seen["defer summary asks for input"] += 1
        if not asks:
            flag(tid, "defer summary asks for input", "the summary of a defer answer asks for nothing: %r" % summary[:60])
    else:
        seen["summary asks only on defer"] += 1
        if asks:
            flag(tid, "summary asks only on defer", "%r in a %s answer" % (asks[0], a.get("continuation_decision")))

    # A failure result names the failed tool. When every call failed, the
    # answer rests on the failure alone and the key is "tool". When a lookup
    # succeeded first, the result keeps the lookup and the key is
    # "failed_tool". A result that names a failed tool when no call failed
    # reports a failure that did not happen.
    errors = [(n, res) for n, res in pairs if isinstance(res, dict) and res.get("status") == "error"]
    if errors or "tool" in r or "failed_tool" in r:
        seen["failure keys"] += 1
        n_err, res_err = errors[0] if errors else (None, {})
        if errors and not good:
            want, banned = {"tool": n_err, "error_code": res_err.get("error_code")}, "failed_tool"
        else:
            want, banned = {"failed_tool": n_err, "error_code": res_err.get("error_code")}, "tool"
        wrong = [k for k, v in want.items() if r.get(k) != v]
        problem = None
        if not errors:
            problem = "the result names a failed tool, but no call failed"
        elif banned in r:
            problem = "the result has %r, but this failure uses %r" % (banned, list(want)[0])
        elif wrong:
            problem = "result %s does not match the failed call" % ", ".join(wrong)
        if problem:
            flag(tid, "failure keys", problem)

    # blocked_reason. A defer answer tells the user in one sentence what blocks
    # the task. The code stays in the backend, so the sentence is checked
    # against the templates of blocked_reasons.yaml: exactly one template
    # matches, filling it from the answer gives the sentence back, and the
    # template fits the answer's status. Every other answer has null.
    reason = a.get("blocked_reason")
    reason_code = None
    if a.get("continuation_decision") == "defer":
        seen["defer needs blocked_reason"] += 1
        if not isinstance(reason, str) or not reason:
            detail = "field absent" if "blocked_reason" not in a else repr(reason)
            flag(tid, "defer needs blocked_reason", detail)
        else:
            seen["blocked_reason matches template"] += 1
            reason_code, tmpl, problem = match_reason(reason, a, t["meta"])
            if problem:
                flag(tid, "blocked_reason matches template", problem)
            else:
                # "lookup" is the status of the successful lookup in a partial
                # failure. A planned template has no trace yet and fits nothing.
                seen["blocked_reason fits status"] += 1
                want = tmpl["status"]
                if want == "lookup":
                    want = lookups[0].get("status") if lookups else "a successful lookup"
                if want != a.get("status"):
                    detail = "%s fits %s, the answer has %r" % (tmpl["id"], want, a.get("status"))
                    flag(tid, "blocked_reason fits status", detail)
    else:
        seen["blocked_reason null unless defer"] += 1
        code = (t["meta"] or {}).get("blocked_reason_code")
        if "blocked_reason" not in a or reason is not None or code:
            detail = ("field absent" if "blocked_reason" not in a else
                      repr(reason)[:70] if reason is not None else
                      "the row has blocked_reason_code %r" % code)
            flag(tid, "blocked_reason null unless defer", detail)

    # Provenance. An answer resting on registry evidence copies the block of
    # that evidence, every field included, because the field exists to prove a
    # copy rather than to decorate. The evidence_id changes with every call,
    # so a copied block and a memorized block differ. An answer that rests on
    # no registry evidence has null: inventing a block for a convention answer
    # would teach the model to fabricate one.
    prov = a.get("provenance")
    if evidence is not None:
        seen["provenance present"] += 1
        problem = provenance_shape(prov)
        if problem:
            flag(tid, "provenance present", problem)
        else:
            seen["provenance copied"] += 1
            if prov != evidence:
                detail = "the answer has %s but the evidence has %s" % (
                    json.dumps(prov), json.dumps(evidence))
                flag(tid, "provenance copied", detail)
    else:
        seen["no invented provenance"] += 1
        if prov is not None:
            detail = "the answer rests on no registry evidence but carries %s" % json.dumps(prov)[:70]
            flag(tid, "no invented provenance", detail)

    # Six traces under the S1.8 Direction Conventions heading were labelled
    # S1.9.n, one family off, while the real S1.9 heading sat empty. A label
    # that disagrees with its heading misroutes the trace for anyone reading
    # by number, including a generator.
    fam = t.get("family")
    if fam:
        seen["trace label matches its family"] += 1
        head = tid.split(":")[0].strip()
        if not head.startswith(fam + "."):
            detail = "labelled %s but sits under %s" % (head, fam)
            flag(tid, "trace label matches its family", detail)

    # An answer inlined into a trace's context, such as an earlier turn, is a
    # copy of an answer that lives elsewhere and drifts when the original
    # changes. It did: the two earlier turns inlined into S1.3.8 and S1.3.9
    # kept the pre-provenance form of S1.3.2's answer for one commit. If it
    # reports a registry identifier it carries provenance, same as any answer.
    for obj in re.findall(r'\{"status":[^\n]*?"result":.*?(?=$|\{"status")', ctx):
        if not re.search(r'"(gene_id|candidate_gene_ids)"', obj):
            continue
        seen["context answer carries provenance"] += 1
        if '"provenance": {' not in obj:
            detail = "an answer in this trace's context reports a registry identifier without provenance"
            flag(tid, "context answer carries provenance", detail)

    # The failure this family exists to prevent. When every tool call failed,
    # the model learned nothing, so reporting not_found asserts an absence it
    # has no evidence for. A backend outage must not read as a missing gene.
    if t["results"] and not good:
        seen["tool failure not reported as a negative"] += 1
        if a.get("status") == "not_found" or reason_code == "not_found":
            detail = "every tool call failed but the answer reports not_found"
            flag(tid, "tool failure not reported as a negative", detail)

    # A rejected string lives in result.input by construction, so parsing it is
    # a category error: the trace's whole point is that it does not parse. Skip
    # those, but only when the answer actually declares the input invalid, so a
    # trace that hides a bad tag behind status resolved is still caught.
    rejected = set()
    if isinstance(r, dict) and a.get("status") == "invalid_input":
        v = r.get("input")
        if isinstance(v, str):
            rejected.add(v)
    for tag in sorted(set(TAG_RE.findall(blob))):
        if tag in rejected:
            seen["rejected input not parsed"] += 1
            continue
        seen["layer tag parses"] += 1
        err = parse_tag(tag)
        if err:
            flag(tid, "layer tag parses", "%s: %s" % (tag, err))
    if r.get("layer_family"):
        seen["layer_family vocabulary"] += 1
        if r["layer_family"] not in V["layer_family"]:
            flag(tid, "layer_family vocabulary", repr(r["layer_family"]))
    for k in ("supported_claims", "unsupported_claims"):
        for c in r.get(k) or []:
            seen["claim vocabulary"] += 1
            if c not in V["layer_claims"]:
                flag(tid, "claim vocabulary", repr(c))
    # Each claim list keeps the vocabulary order. S1.7.3 and S1.7.4 once listed
    # the same claims in two orders, which teaches the model that order is
    # free and makes exact-match scoring fail on a correct answer.
    for k in ("supported_claims", "unsupported_claims"):
        listed = [c for c in (r.get(k) or []) if c in V["layer_claims"]]
        if listed:
            seen["claim order"] += 1
            if listed != sorted(listed, key=V["layer_claims"].index):
                flag(tid, "claim order", "%s is not in vocabulary order" % k)
    # The spec says the two sets partition the whole vocabulary, so the model
    # states every negative rather than leaving it to inference. Only check a
    # trace that offers both sets; a single-claim question carries one.
    if "supported_claims" in r and "unsupported_claims" in r:
        sup = list(r["supported_claims"] or [])
        uns = list(r["unsupported_claims"] or [])
        seen["claim sets partition the vocabulary"] += 1
        both = set(sup) & set(uns)
        if both:
            detail = "claims in both sets: %s" % ", ".join(sorted(both))
            flag(tid, "claim sets partition the vocabulary", detail)
        missing = [c for c in V["layer_claims"] if c not in sup and c not in uns]
        if missing:
            detail = "claims in neither set: %s" % ", ".join(missing)
            flag(tid, "claim sets partition the vocabulary", detail)

# Every S1 row has the same system prompt and the same tool list, closed-book
# rows included. A prompt or a tool list that changes with the book mode, the
# step, or the family tells the model whether to call before it reads the
# task. The first row is the reference, and system_prompt_sha256 must hash the
# prompt that the row shows.
rows = [t for t in traces if t["meta"] is not None]
for t in rows:
    tid = t["id"]
    seen["one prompt for every row"] += 1
    problems = []
    if t["system"] != rows[0]["system"]:
        problems.append("its system prompt differs from %s" % rows[0]["id"])
    if json.dumps(t["tool_defs"]) != json.dumps(rows[0]["tool_defs"]):
        problems.append("its tool list differs from %s" % rows[0]["id"])
    sha = t["meta"].get("system_prompt_sha256")
    if sha is not None and sha != hashlib.sha256((t["system"] or "").encode("utf-8")).hexdigest():
        problems.append("system_prompt_sha256 does not hash its system prompt")
    if problems:
        flag(tid, "one prompt for every row", "; ".join(problems))

# ---------- report ----------
print("file: %s" % qf.name)
print("%-28s: %d" % ("rows in file" if JSONL else "trace titles in file", n_items))
print("traces parsed               : %d" % len(traces))
print("silently skipped            : %d" % skipped)
print("answers checked             : %d" % seen["status vocabulary"])
print("violations                  : %d\n" % len(issues))

byk = Counter(c for _, c, _ in issues)
print("%-42s %14s %11s" % ("check", "items examined", "violations"))
for k in CHECKS:
    n = seen.get(k, 0)
    print("  %-40s %12d %11d%s" % (k, n, byk.get(k, 0), "   <-- never ran" if n == 0 else ""))
extra = sorted(set(byk) - set(CHECKS))
for k in extra:
    print("  %-40s %12s %11d" % (k, "n/a", byk[k]))

total_by_check = sum(byk.values())
if total_by_check != len(issues):
    print("\nFAIL: per-check violations sum to %d but total is %d. Report is inconsistent."
          % (total_by_check, len(issues))); sys.exit(2)
if len(traces) == 0:
    print("\nFAIL: parsed no traces. Refusing to report a pass."); sys.exit(2)
if skipped != 0:
    print("\nFAIL: %d trace(s) found but not parsed." % skipped); sys.exit(2)
if seen["status vocabulary"] == 0:
    print("\nFAIL: no answers checked."); sys.exit(2)

if issues:
    print("\ndetail")
    for tid, check, detail in issues:
        print("  %-46s %-32s %s" % (tid[:46], check, detail))
sys.exit(1 if issues else 0)
