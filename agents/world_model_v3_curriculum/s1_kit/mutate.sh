#!/bin/sh
# Tests the fixtures, not the validator. Breaks each violation site in
# validate.py and checks that selftest.sh notices. A survivor is a fixture gap.
cd "$(dirname "$0")" || exit 2
PY=${PYTHON:-python3}
# A broken baseline kills every mutant and reports a clean sweep, which is
# the same silent false negative the fixtures exist to prevent.
if ! ./selftest.sh >/dev/null 2>&1; then
  echo "mutate: selftest already fails on the unmutated validator. Fix that first."
  exit 2
fi
"$PY" - <<'PYEOF'
import pathlib,re,subprocess,shutil
orig=pathlib.Path("validate.py").read_text()
pts=[(i,l) for i,l in enumerate(orig.split("\n")) if re.search(r'^\s+flag\(tid,',l)]
shutil.copy("validate.py","validate.py.bak"); survived=[]
try:
    for idx,line in pts:
        ls=orig.split("\n"); ind=len(ls[idx])-len(ls[idx].lstrip())
        ls[idx]=" "*ind+"pass  # MUTANT"
        pathlib.Path("validate.py").write_text("\n".join(ls))
        killed=subprocess.run(["./selftest.sh"],capture_output=True,text=True).returncode!=0
        lbl=re.sub(r'\s+',' ',line.strip())[:70]
        print(("  killed   " if killed else "SURVIVED   ")+lbl)
        if not killed: survived.append(lbl)
finally:
    shutil.copy("validate.py.bak","validate.py"); pathlib.Path("validate.py.bak").unlink()
print("\nmutants %d  killed %d  survived %d"%(len(pts),len(pts)-len(survived),len(survived)))
raise SystemExit(1 if survived else 0)
PYEOF
