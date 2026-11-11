"""Exploratory profiling of the patent JSON files. Usage: python3 -I explore.py <data_dir>"""
import json, re, sys, glob, os, statistics as st
from collections import Counter, defaultdict

data_dir = sys.argv[1]
files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
print(f"## FILES: {len(files)}")

records, load_errors, top_types = [], [], Counter()
per_file = {}
for f in files:
    name = os.path.basename(f)
    try:
        d = json.load(open(f, encoding="utf-8"))
    except Exception as e:
        load_errors.append((name, repr(e))); continue
    top_types[type(d).__name__] += 1
    if not isinstance(d, list):
        load_errors.append((name, f"top-level is {type(d).__name__}")); continue
    per_file[name] = len(d)
    for i, p in enumerate(d):
        records.append((name, i, p))
print("load errors:", load_errors or "none")
print("top-level types:", dict(top_types))
n = len(records)
print(f"total records: {n}")
non_dict = [(f, i, type(p).__name__) for f, i, p in records if not isinstance(p, dict)]
print("non-dict records:", len(non_dict))
records = [(f, i, p) for f, i, p in records if isinstance(p, dict)]
n = len(records)

c = list(per_file.values())
print(f"records/file: min={min(c)} median={st.median(c)} mean={st.mean(c):.1f} max={max(c)}")
print("empty files:", [k for k, v in per_file.items() if v == 0] or "none")

# ---------- key presence ----------
print("\n## KEY PRESENCE (key exists in dict)")
keys = Counter(k for _, _, p in records for k in p)
for k, v in keys.most_common():
    print(f"  {k:22s} present {v:6d}/{n}  absent {n - v:5d} ({100*(n-v)/n:.2f}%)")
shapes = Counter(tuple(sorted(p)) for _, _, p in records)
print(f"distinct key-sets: {len(shapes)}")
for s, v in shapes.most_common(5):
    print(f"  {v:6d}  {s}")

# ---------- types ----------
print("\n## VALUE TYPES per key")
tp = defaultdict(Counter)
for _, _, p in records:
    for k, v in p.items():
        tp[k][type(v).__name__] += 1
for k in tp:
    print(f"  {k:22s} {dict(tp[k])}")

# ---------- emptiness ----------
def is_empty(v):
    if v is None: return "None"
    if isinstance(v, str):
        return "empty_str" if v == "" else ("whitespace_only" if not v.strip() else None)
    if isinstance(v, (list, tuple)):
        if len(v) == 0: return "empty_list"
        if all(isinstance(x, str) and not x.strip() for x in v): return "list_of_blank"
        return None
    if isinstance(v, dict): return "empty_dict" if not v else None
    return None

print("\n## MISSING / EMPTY SUMMARY (absent key + null + empty + blank)")
all_keys = list(keys)
hdr = f"  {'field':22s} {'absent':>7s} {'None':>6s} {'empty':>6s} {'blank':>6s} {'TOTAL':>7s} {'%':>7s}"
print(hdr)
missing_by_rec = defaultdict(list)
for k in all_keys:
    absent = sum(1 for _, _, p in records if k not in p)
    kinds = Counter()
    for fi, (f, i, p) in enumerate(records):
        if k in p:
            e = is_empty(p[k])
            if e: kinds[e] += 1; missing_by_rec[fi].append(k)
        else:
            missing_by_rec[fi].append(k)
    emp = kinds["empty_str"] + kinds["empty_list"] + kinds["empty_dict"]
    blank = kinds["whitespace_only"] + kinds["list_of_blank"]
    tot = absent + kinds["None"] + emp + blank
    print(f"  {k:22s} {absent:7d} {kinds['None']:6d} {emp:6d} {blank:6d} {tot:7d} {100*tot/n:6.2f}%")
nmiss = Counter(len(set(v)) for v in missing_by_rec.values())
complete = n - len(missing_by_rec)
print(f"\nrecords fully complete (no missing field): {complete} ({100*complete/n:.2f}%)")
print(f"records with >=1 missing field: {len(missing_by_rec)}")
print("number of missing fields per record:", dict(sorted(nmiss.items())))
combo = Counter(tuple(sorted(set(v))) for v in missing_by_rec.values())
print("most common missing-field combos:")
for cb, v in combo.most_common(8): print(f"  {v:5d}  {cb}")

# ---------- identity / duplicates ----------
print("\n## IDENTITY")
dn = [p.get("doc_number") for _, _, p in records]
dnc = Counter(x for x in dn if x)
dups = {k: v for k, v in dnc.items() if v > 1}
print(f"unique doc_number: {len(dnc)}  duplicated ids: {len(dups)}  extra copies: {sum(v-1 for v in dups.values())}")
fmt = Counter(re.sub(r"\d", "9", x) if isinstance(x, str) else repr(x) for x in dn)
print("doc_number formats:", dict(fmt.most_common(5)))
# are duplicate ids exact duplicates?
byid = defaultdict(list)
for f, i, p in records:
    if p.get("doc_number") in dups: byid[p["doc_number"]].append((f, p))
identical = sum(1 for v in byid.values() if all(json.dumps(x[1], sort_keys=True) == json.dumps(v[0][1], sort_keys=True) for x in v))
print(f"  of duplicated ids, byte-identical across copies: {identical}/{len(byid)}")
if byid:
    ex = next(iter(byid.items()))
    print("  example:", ex[0], [x[0] for x in ex[1]])
fnc = Counter(p.get("filename") for _, _, p in records if p.get("filename"))
print(f"unique filename: {len(fnc)}  duplicated: {sum(1 for v in fnc.values() if v>1)}")
tc = Counter(p.get("title", "").strip().lower() for _, _, p in records if isinstance(p.get("title"), str) and p.get("title").strip())
print(f"unique titles: {len(tc)}  titles shared by >1 patent: {sum(1 for v in tc.values() if v>1)}")
print("  most common titles:", tc.most_common(8))

# ---------- filename / date consistency ----------
print("\n## FILENAME / DATE CONSISTENCY")
fre = re.compile(r"^US(\d{11})A1-(\d{8})\.XML$")
bad_fn, mism_num, mism_week, kinds_fn = [], 0, 0, Counter()
for f, i, p in records:
    fn = p.get("filename")
    if not isinstance(fn, str): continue
    m = fre.match(fn)
    if not m: bad_fn.append(fn); continue
    if p.get("doc_number") != m.group(1): mism_num += 1
    filedate = re.search(r"ipa(\d{6})", f).group(1)
    if m.group(2)[2:] != filedate: mism_week += 1
print(f"filenames not matching expected pattern: {len(bad_fn)}  {bad_fn[:3]}")
print(f"filename-number != doc_number: {mism_num}")
print(f"publication date in filename != date in json file name: {mism_week}")
yrs = Counter()
for f, i, p in records:
    m = fre.match(p.get("filename", "") or "")
    if m: yrs[m.group(2)[:4]] += 1
print("year counts (from filename):", dict(yrs))

# ---------- bibtex ----------
print("\n## BIBTEX")
bre = re.compile(r"@patent\{(\d+),\s*title = \{(.*?)\},\s*number = \{(\d+)\},\s*year = \{(\d+)\},\s*note = \{File: (.*?)\}\s*\}", re.S)
bnomatch = bid = btitle = bfile = 0
bex = []
for f, i, p in records:
    b = p.get("bibtex")
    if not isinstance(b, str) or not b.strip(): continue
    m = bre.match(b.strip())
    if not m: bnomatch += 1; bex.append(b[:120]); continue
    if m.group(1) != p.get("doc_number") or m.group(3) != p.get("doc_number"): bid += 1
    if m.group(2) != p.get("title"): btitle += 1
    if m.group(5) != p.get("filename"): bfile += 1
print(f"unparseable by template: {bnomatch}  id mismatch: {bid}  title mismatch: {btitle}  filename mismatch: {bfile}")
for e in bex[:3]: print("   ", repr(e))

# ---------- classification ----------
print("\n## CLASSIFICATION")
cls = [p.get("classification") for _, _, p in records]
cv = [c for c in cls if isinstance(c, str) and c.strip()]
print(f"non-empty: {len(cv)}  distinct: {len(set(cv))}")
shape = Counter(re.sub(r"[A-Z]", "A", re.sub(r"\d", "9", c)) for c in cv)
print("code shapes:", dict(shape.most_common(8)))
print("length dist:", dict(sorted(Counter(len(c) for c in cv).items())))
print("whitespace/lowercase anomalies:", sum(1 for c in cv if c != c.strip() or c != c.upper() or " " in c))
print("top 15 full codes:", Counter(cv).most_common(15))
for L, name in [(1, "section"), (3, "class (e.g. B60)"), (4, "subclass (e.g. B60B)")]:
    cc = Counter(c[:L] for c in cv)
    print(f"{name}: {len(cc)} distinct; top:", cc.most_common(10))
print("starts with B60B:", sum(1 for c in cv if c.startswith("B60B")))
print("does NOT start with B60:", sum(1 for c in cv if not c.startswith("B60")), "-> sample:", [c for c in cv if not c.startswith("B60")][:8])

# ---------- text field stats ----------
print("\n## TEXT STATS")
def stats(name, xs):
    if not xs: print(f"  {name}: no data"); return
    xs = sorted(xs)
    q = lambda f: xs[min(len(xs)-1, int(f*len(xs)))]
    print(f"  {name:32s} n={len(xs):5d} min={xs[0]:>7} p25={q(.25):>7} med={q(.5):>7} p75={q(.75):>7} p95={q(.95):>7} max={xs[-1]:>8} mean={st.mean(xs):.0f}")
L = lambda k, fn: [fn(p[k]) for _, _, p in records if isinstance(p.get(k), (str, list)) and not is_empty(p[k])]
stats("title chars", L("title", len))
stats("title words", L("title", lambda s: len(s.split())))
stats("abstract chars", L("abstract", len))
stats("abstract words", L("abstract", lambda s: len(s.split())))
stats("claims: count / patent", L("claims", len))
stats("claim chars (each)", [len(c) for _, _, p in records if isinstance(p.get("claims"), list) for c in p["claims"] if isinstance(c, str)])
stats("detailed_desc: paragraphs / patent", L("detailed_description", len))
stats("detailed_desc: total chars", L("detailed_description", lambda l: sum(len(x) for x in l if isinstance(x, str))))
stats("detailed_desc: chars / paragraph", [len(x) for _, _, p in records if isinstance(p.get("detailed_description"), list) for x in p["detailed_description"] if isinstance(x, str)])

# element-level problems inside lists
print("\n## LIST ELEMENT QUALITY")
for k in ("claims", "detailed_description"):
    nonstr = blank = tot = 0
    for _, _, p in records:
        v = p.get(k)
        if isinstance(v, list):
            for x in v:
                tot += 1
                if not isinstance(x, str): nonstr += 1
                elif not x.strip(): blank += 1
    print(f"  {k}: elements={tot} non-str={nonstr} blank={blank}")

# claims structure
print("\n## CLAIMS STRUCTURE")
numbered = dep = tot = 0
dep_re = re.compile(r"\b(of|to|in|with) (any (one )?of )?claims? \d+", re.I)
lead = Counter()
for _, _, p in records:
    if isinstance(p.get("claims"), list):
        for c in p["claims"]:
            if not isinstance(c, str): continue
            tot += 1
            if re.match(r"^\s*\d+\s*[\.\)]", c): numbered += 1
            if dep_re.search(c[:200]): dep += 1
            lead[" ".join(c.split()[:2]).lower()] += 1
print(f"  claims total={tot}  start with a number prefix={numbered}  look dependent={dep} ({100*dep/max(tot,1):.1f}%)")
print("  most common claim openers:", lead.most_common(8))
ind = [sum(1 for c in p["claims"] if isinstance(c, str) and not dep_re.search(c[:200])) for _, _, p in records if isinstance(p.get("claims"), list) and p["claims"]]
stats("independent claims / patent", ind)

# text hygiene
print("\n## TEXT HYGIENE")
def scan(label, texts):
    texts = list(texts)
    print(f"  {label}: n={len(texts)} leading/trailing ws={sum(1 for t in texts if t != t.strip())} "
          f"html-entity={sum(1 for t in texts if re.search(r'&(amp|lt|gt|quot|#\\d+);', t))} "
          f"html-tag={sum(1 for t in texts if re.search(r'</?[a-z][^>]*>', t, re.I))} "
          f"non-ascii={sum(1 for t in texts if any(ord(ch) > 127 for ch in t))} "
          f"newline={sum(1 for t in texts if chr(10) in t)}")
scan("title", (p["title"] for _, _, p in records if isinstance(p.get("title"), str)))
scan("abstract", (p["abstract"] for _, _, p in records if isinstance(p.get("abstract"), str)))
scan("claims", (c for _, _, p in records if isinstance(p.get("claims"), list) for c in p["claims"] if isinstance(c, str)))
scan("desc paragraphs", (c for _, _, p in records if isinstance(p.get("detailed_description"), list) for c in p["detailed_description"] if isinstance(c, str)))
print("  all-uppercase titles:", sum(1 for _, _, p in records if isinstance(p.get("title"), str) and p["title"].isupper()), "of", n)
print("  placeholder-like values (n/a, null, none, tbd):",
      sum(1 for _, _, p in records for k, v in p.items() if isinstance(v, str) and v.strip().lower() in ("n/a", "na", "null", "none", "tbd", "-", "unknown")))

# near-duplicate content across different ids
print("\n## CONTENT DUPLICATION (different ids)")
ab = Counter(p["abstract"].strip() for _, _, p in records if isinstance(p.get("abstract"), str) and p["abstract"].strip())
print("  abstracts shared by >1 record:", sum(1 for v in ab.values() if v > 1))
cl = Counter(json.dumps(p["claims"]) for _, _, p in records if isinstance(p.get("claims"), list) and p["claims"])
print("  identical claim sets shared by >1 record:", sum(1 for v in cl.values() if v > 1))

# ---------- time ----------
print("\n## RECORDS PER WEEK")
wk = Counter(re.search(r"ipa(\d{6})", f).group(1) for f, _, _ in records)
print("  weeks:", len(wk), " first:", min(wk), " last:", max(wk))
print("  ", dict(sorted(wk.items())))
