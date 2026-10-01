#!/bin/sh
# Verifies the validator against known-bad and known-good input.
# Asserts WHICH check catches each known-bad fixture, not merely that one did.
# Runs both input forms: the markdown traces (F fixtures) and the JSONL rows
# of rows.yaml (J fixtures).
# validate.py exit codes: 0 = clean, 1 = violations found, 2+ = could not run.
cd "$(dirname "$0")" || exit 2
PY=${PYTHON:-python3}
command -v "$PY" >/dev/null 2>&1 || { echo "ERROR: $PY not found. Set PYTHON=/path/to/python3"; exit 2; }
"$PY" - "$PY" <<'PYEOF'
import subprocess,re,sys,pathlib,json
PY=sys.argv[1]
def run(f):
    r=subprocess.run([PY,"validate.py",f],capture_output=True,text=True)
    if r.returncode>=2 or "violations" not in r.stdout:
        print("ERROR: validate.py could not run on %s (exit %d)"%(f,r.returncode))
        if r.stderr.strip(): print("--- stderr ---\n"+r.stderr.strip())
        if r.stdout.strip(): print("--- stdout ---\n"+r.stdout.strip())
        sys.exit(2)
    return r.stdout
want={}
for ln in pathlib.Path("fixtures/expected.tsv").read_text().split("\n"):
    if ln.strip() and "\t" in ln:
        fid,chk=ln.split("\t",1); want[fid.strip()]=chk.strip()
declared=set(re.findall(r'^\*\*(F\d+):',pathlib.Path("fixtures/must_fail.md").read_text(),re.M))
for ln in pathlib.Path("fixtures/must_fail.jsonl").read_text().split("\n"):
    if ln.strip():
        rid=str(json.loads(ln).get("row_id",""))
        if re.fullmatch(r'J\d+',rid): declared.add(rid)
key=lambda s:(s[0],int(s[1:]))
undeclared=sorted(declared-set(want),key=key)
orphans=sorted(set(want)-declared,key=key)
out=run("fixtures/must_fail.md")+run("fixtures/must_fail.jsonl")
wrong=[]
for fid,chk in sorted(want.items(),key=lambda kv:key(kv[0])):
    if fid not in declared:
        continue
    if not re.search(r'^\s{2}%s:.*?\s%s\s'%(re.escape(fid),re.escape(chk)),out,re.M):
        got=re.findall(r'^\s{2}%s:\s+\S.*?\s\s+(\S[^\n]*?)\s\s'%re.escape(fid),out,re.M)
        wrong.append((fid,chk,got[0] if got else "nothing fired"))
fp=sum(int(re.search(r'violations\s*:\s*(\d+)',run(f)).group(1))
       for f in ("fixtures/must_pass.md","fixtures/must_pass.jsonl"))
n=len(declared&set(want))
ok = not wrong and not undeclared and not orphans and fp==0
print("selftest:", "PASS" if ok else "FAIL")
print("  known-bad fixtures with the right check: %d of %d"%(n-len(wrong),n))
print("  false positives on clean input         : %d"%fp)
if undeclared: print("  fixtures with no declared check        :",undeclared)
if orphans: print("  declared checks with no fixture        :",orphans)
for fid,chk,got in wrong: print("    %s expected %r but got %r"%(fid,chk,got))
sys.exit(0 if ok else 1)
PYEOF
