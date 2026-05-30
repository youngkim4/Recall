#!/usr/bin/env python3
"""Analyze + consolidate app/src/App.css.

App.css is flat (no @media, no nesting, single @keyframes). It accreted a dark
theme on top of a light/blue theme by re-declaring the same selectors. This tool:

  1. Parses the file into top-level blocks (comment / :root / @keyframes / rule).
  2. For each selector defined more than once, merges declarations last-wins per
     property (respecting !important), keeping the merged block at the selector's
     LAST (winning) source position. This is a fidelity-preserving collapse: the
     final computed value of every property is unchanged.
  3. Routes color literals to :root design tokens ONLY when the literal exactly
     equals a token value (value-identical => zero visual change).
  4. Emits scripts/out/consolidated.css plus analysis reports.

Nothing here changes rendering by construction; screenshots are the final check.
"""
from __future__ import annotations
import os, re, sys
from collections import OrderedDict, defaultdict

ROOT = "/Users/youngkim/Projects/imessage-memories"
CSS = os.path.join(ROOT, "app/src/App.css")
OUT = os.path.join(ROOT, "scripts/out")
os.makedirs(OUT, exist_ok=True)

src = open(CSS).read()
N = len(src)


def line_at(idx: int) -> int:
    return src.count("\n", 0, idx) + 1


# ---------------------------------------------------------------- block scanner
def scan_blocks():
    """Return list of blocks. Each: dict(kind, prelude, body, sl, el).

    Comment-, string-, and brace-depth aware. The only nested construct is
    @keyframes, captured opaquely (depth-balanced)."""
    blocks = []
    i = 0
    prelude_start = 0
    while i < N:
        c = src[i]
        if c == "/" and i + 1 < N and src[i + 1] == "*":
            j = src.find("*/", i + 2)
            j = N if j == -1 else j + 2
            # standalone comment between rules -> record so we can keep section markers
            if src[prelude_start:i].strip() == "":
                blocks.append(dict(kind="comment", prelude="", body=src[i:j],
                                   sl=line_at(i), el=line_at(j - 1)))
                prelude_start = j
            i = j
            continue
        if c in "\"'":
            q = c
            i += 1
            while i < N and src[i] != q:
                if src[i] == "\\":
                    i += 1
                i += 1
            i += 1
            continue
        if c == "{":
            prelude = src[prelude_start:i].strip()
            depth = 1
            j = i + 1
            while j < N and depth > 0:
                cj = src[j]
                if cj == "/" and j + 1 < N and src[j + 1] == "*":
                    k = src.find("*/", j + 2)
                    j = N if k == -1 else k + 2
                    continue
                if cj in "\"'":
                    q = cj
                    j += 1
                    while j < N and src[j] != q:
                        if src[j] == "\\":
                            j += 1
                        j += 1
                    j += 1
                    continue
                if cj == "{":
                    depth += 1
                elif cj == "}":
                    depth -= 1
                j += 1
            body = src[i + 1:j - 1]
            if prelude.startswith("@keyframes"):
                kind = "keyframes"
            elif prelude.startswith("@"):
                kind = "atrule"
            elif prelude == ":root":
                kind = "root"
            else:
                kind = "rule"
            sl = line_at(prelude_start + (len(src[prelude_start:i]) - len(src[prelude_start:i].lstrip())))
            blocks.append(dict(kind=kind, prelude=prelude, body=body,
                               sl=sl, el=line_at(j - 1)))
            i = j
            prelude_start = j
            continue
        i += 1
    return blocks


# ----------------------------------------------------------- declaration parser
def split_decls(body: str):
    """Split a declaration block into [(prop, value, important)] preserving order.

    Respects parens (gradients, url, rgb) and strings; strips in-body comments."""
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    decls = []
    depth = 0
    buf = []
    i = 0
    while i < len(body):
        c = body[i]
        if c in "\"'":
            q = c
            buf.append(c)
            i += 1
            while i < len(body) and body[i] != q:
                buf.append(body[i])
                if body[i] == "\\":
                    i += 1
                    if i < len(body):
                        buf.append(body[i])
                i += 1
            if i < len(body):
                buf.append(body[i])
            i += 1
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        if c == ";" and depth == 0:
            decls.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(c)
        i += 1
    if "".join(buf).strip():
        decls.append("".join(buf))

    out = []
    for d in decls:
        d = d.strip()
        if not d:
            continue
        # split prop:value at first top-level colon
        depth = 0
        ci = -1
        for k, ch in enumerate(d):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == ":" and depth == 0:
                ci = k
                break
        if ci == -1:
            continue  # malformed; skip
        prop = d[:ci].strip()
        val = d[ci + 1:].strip()
        important = False
        m = re.search(r"!\s*important\s*$", val, re.I)
        if m:
            important = True
            val = val[:m.start()].strip()
        out.append((prop, val, important))
    return out


# ------------------------------------------------------------- color normalize
NAMED = {"white": "#ffffff", "black": "#000000"}


def norm_color(tok: str):
    t = tok.strip().lower()
    if t in NAMED:
        t = NAMED[t]
    m = re.fullmatch(r"#([0-9a-f]{3,8})", t)
    if m:
        h = m.group(1)
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        elif len(h) == 4:
            h = "".join(ch * 2 for ch in h)
        return f"#{h}"
    m = re.fullmatch(r"rgba?\(([^)]*)\)", t)
    if m:
        parts = [p.strip() for p in re.split(r"[,/]", m.group(1)) if p.strip() != ""]
        if len(parts) in (3, 4):
            try:
                nums = []
                for k, p in enumerate(parts):
                    if k < 3:
                        nums.append(str(int(round(float(p.rstrip("%"))))))
                    else:
                        nums.append(f"{float(p):.3f}")
                if len(nums) == 3:
                    nums.append("1.000")
                return "rgba(" + ",".join(nums) + ")"
            except ValueError:
                return None
    return None


COLOR_RE = re.compile(r"#[0-9a-fA-F]{3,8}\b|rgba?\([^)]*\)", re.I)


# ------------------------------------------------------------------------- main
blocks = scan_blocks()
rules = [b for b in blocks if b["kind"] == "rule"]
root = next((b for b in blocks if b["kind"] == "root"), None)

# token map: normalized color value -> token name (single-literal color tokens)
token_val = OrderedDict()  # name -> raw value
val_to_token = {}
if root:
    for prop, val, _ in split_decls(root["body"]):
        if prop.startswith("--"):
            token_val[prop] = val
            if COLOR_RE.fullmatch(val.strip()):
                nv = norm_color(val.strip())
                if nv and nv not in val_to_token:
                    val_to_token[nv] = prop

# selector groups (exact normalized selector text)
def norm_sel(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

groups = OrderedDict()  # sel -> list of rule-block dicts (in source order)
for b in rules:
    groups.setdefault(norm_sel(b["prelude"]), []).append(b)

dups = {s: bs for s, bs in groups.items() if len(bs) > 1}


def merge_group(bs):
    """Merge declarations across blocks (source order). Return list of
    (prop, value, important) with each property kept once at its WINNING
    position. Also return hazards list."""
    seq = []  # (prop, val, important, block_idx)
    for bi, b in enumerate(bs):
        for prop, val, imp in split_decls(b["body"]):
            seq.append([prop, val, imp, bi])
    # winner index per property
    winner = {}
    for idx, (prop, val, imp, bi) in enumerate(seq):
        if prop not in winner:
            winner[prop] = idx
        else:
            w = seq[winner[prop]]
            # important beats non-important; otherwise later wins
            if imp and not w[2]:
                winner[prop] = idx
            elif imp == w[2]:
                winner[prop] = idx  # later (current) wins on tie
    keep_idx = set(winner.values())
    merged = [(seq[i][0], seq[i][1], seq[i][2]) for i in range(len(seq)) if i in keep_idx]
    # hazard: same prop appears 2+ times with a shorthand/longhand relative between
    SHORT = {"background": "background-", "border": "border-", "margin": "margin-",
             "padding": "padding-", "font": "font-", "inset": "top|right|bottom|left",
             "border-radius": "border-.*-radius", "transition": "transition-",
             "animation": "animation-", "flex": "flex-", "grid": "grid-",
             "overflow": "overflow-", "gap": "(row|column)-gap"}
    hazards = []
    occ = defaultdict(list)
    for idx, (prop, *_ ) in enumerate(seq):
        occ[prop].append(idx)
    for prop, idxs in occ.items():
        if len(idxs) < 2:
            continue
        lo, hi = idxs[0], idxs[-1]
        between = [seq[k][0] for k in range(lo + 1, hi)]
        related = []
        for b in between:
            for sh, pat in SHORT.items():
                if prop == sh and re.match(pat, b):
                    related.append(b)
                if b == sh and re.match(pat, prop):
                    related.append(b)
        if related:
            hazards.append((prop, related))
    return merged, hazards


def route_value(val: str) -> str:
    def repl(m):
        nv = norm_color(m.group(0))
        if nv and nv in val_to_token:
            return f"var({val_to_token[nv]})"
        return m.group(0)
    return COLOR_RE.sub(repl, val)


def fmt_block(prelude, decls, route=True):
    lines = [prelude + " {"]
    for prop, val, imp in decls:
        v = route_value(val) if (route and not prop.startswith("--")) else val
        imps = " !important" if imp else ""
        lines.append(f"  {prop}: {v}{imps};")
    lines.append("}")
    return "\n".join(lines)


# ----------------------------------------------------------------- emit reports
# duplicates report
with open(os.path.join(OUT, "duplicates.txt"), "w") as f:
    f.write(f"# Duplicate selectors: {len(dups)} of {len(groups)} unique\n\n")
    for s, bs in sorted(dups.items(), key=lambda kv: -len(kv[1])):
        ranges = ", ".join(f"L{b['sl']}-{b['el']}" for b in bs)
        f.write(f"{len(bs)}x  {s}\n     {ranges}\n")

# hazards report
all_hazards = []
for s, bs in dups.items():
    _, hz = merge_group(bs)
    if hz:
        all_hazards.append((s, hz))
with open(os.path.join(OUT, "hazards.txt"), "w") as f:
    f.write(f"# Shorthand/longhand merge hazards: {len(all_hazards)} selectors\n")
    f.write("# (same property repeated with a related shorthand/longhand between -> review)\n\n")
    for s, hz in all_hazards:
        f.write(f"{s}\n")
        for prop, rel in hz:
            f.write(f"    {prop}  <-> {rel}\n")

# color literal inventory (outside :root) + token suggestions + dead survivors
DEAD = {norm_color(x) for x in ["#fff", "#ffffff", "#fbfbfc", "#2457da",
                                "#f2f5ff", "#cfd9ff", "rgba(47,98,232,1)"]}
DEAD_BLUE = {norm_color(x) for x in ["#2457da", "#f2f5ff", "#cfd9ff"]}
DEAD_BLUE.add("rgba(47,98,232")  # prefix flag handled below
color_use = defaultdict(int)        # normalized literal -> count (live, post-merge)
color_untokened = defaultdict(int)  # literal with no exact token, post-merge
dead_blue_survivors = []
# build merged live set
merged_by_sel = OrderedDict()
for s, bs in groups.items():
    if len(bs) > 1:
        merged, _ = merge_group(bs)
    else:
        merged = split_decls(bs[0]["body"])
    merged_by_sel[s] = (bs, merged)
    for prop, val, imp in merged:
        for m in COLOR_RE.finditer(val):
            nv = norm_color(m.group(0))
            if not nv:
                continue
            color_use[nv] += 1
            if nv not in val_to_token:
                color_untokened[nv] += 1
            if nv in {norm_color(x) for x in ["#2457da", "#f2f5ff", "#cfd9ff"]} or \
               (nv and nv.startswith("rgba(47,98,232")):
                dead_blue_survivors.append((s, prop, m.group(0)))

with open(os.path.join(OUT, "colors.txt"), "w") as f:
    f.write("# Live color literals AFTER merge (normalized -> count, token?)\n\n")
    for nv, cnt in sorted(color_use.items(), key=lambda kv: -kv[1]):
        tok = val_to_token.get(nv, "-- no token --")
        f.write(f"{cnt:4d}  {nv:24s} {tok}\n")
    f.write("\n# Untokened live colors (candidates for a token or intentional one-off)\n\n")
    for nv, cnt in sorted(color_untokened.items(), key=lambda kv: -kv[1]):
        f.write(f"{cnt:4d}  {nv}\n")
    f.write("\n# Dead-BLUE survivors after merge (should be ZERO; else still rendered)\n\n")
    for s, prop, lit in dead_blue_survivors:
        f.write(f"{lit}  in  {prop}  of  {s}\n")

# orphan candidates: class selectors whose class token never appears in any tsx
tsx = []
for dirpath, _, files in os.walk(os.path.join(ROOT, "app/src")):
    for fn in files:
        if fn.endswith((".tsx", ".ts")):
            tsx.append(open(os.path.join(dirpath, fn)).read())
tsx_all = "\n".join(tsx)
classes_in_tsx = set(re.findall(r"[A-Za-z0-9_-]+", tsx_all))
def sel_classes(sel):
    return set(re.findall(r"\.([A-Za-z0-9_-]+)", sel))
orphans = []
for s in groups:
    cls = sel_classes(s)
    if cls and not (cls & classes_in_tsx):
        orphans.append(s)
with open(os.path.join(OUT, "orphans.txt"), "w") as f:
    f.write(f"# Orphan candidates: {len(orphans)} selectors whose classes appear in NO .tsx\n")
    f.write("# (CAUTION: verify against dynamic/template classNames before removing)\n\n")
    for s in orphans:
        f.write(s + "\n")

# ------------------------------------------------------------- emit consolidated
# Emit blocks in source order. For a duplicated selector emit ONLY at its last
# occurrence (merged). Keep comments / :root / keyframes / atrules as-is.
last_pos = {}
for s, bs in groups.items():
    last_pos[s] = id(bs[-1])

orphan_set = set(orphans)  # confirmed dead: classes appear in NO .tsx/.ts
removed_orphans = []
emitted_parts = []
for b in blocks:
    if b["kind"] == "comment":
        emitted_parts.append(b["body"])
    elif b["kind"] == "root":
        # keep tokens verbatim (do not route the token definitions)
        emitted_parts.append(b["prelude"] + " {" + b["body"].rstrip() + "\n}")
    elif b["kind"] in ("keyframes", "atrule"):
        emitted_parts.append(b["prelude"] + " {" + b["body"].rstrip() + "\n}")
    elif b["kind"] == "rule":
        s = norm_sel(b["prelude"])
        if id(b) != last_pos[s]:
            continue  # skip non-final occurrence of a duplicated selector
        if s in orphan_set:
            removed_orphans.append(s)
            continue  # drop confirmed-dead selector (no element matches)
        _, merged = merged_by_sel[s]
        emitted_parts.append(fmt_block(b["prelude"], merged, route=True))
open(os.path.join(OUT, "removed_orphans.txt"), "w").write("\n".join(removed_orphans) + "\n")

consolidated = "\n\n".join(p for p in emitted_parts if p.strip()) + "\n"
open(os.path.join(OUT, "consolidated.css"), "w").write(consolidated)

# ------------------------------------------------------------------- summary
print(f"blocks: {len(blocks)}  rules: {len(rules)}  unique selectors: {len(groups)}")
print(f"duplicate selectors: {len(dups)}  (collapsed)")
print(f"merge hazards: {len(all_hazards)} selectors  -> scripts/out/hazards.txt")
print(f"dead-blue survivors: {len(dead_blue_survivors)} (want 0) -> scripts/out/colors.txt")
print(f"orphan candidates: {len(orphans)} -> scripts/out/orphans.txt")
print(f"color tokens in :root: {len(val_to_token)} distinct values")
print(f"untokened live colors: {len(color_untokened)}")
print(f"consolidated.css lines: {consolidated.count(chr(10))+1}  (was {N and src.count(chr(10))+1})")
