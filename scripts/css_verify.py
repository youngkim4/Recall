#!/usr/bin/env python3
"""Prove two stylesheets are render-equivalent at the per-selector level.

Usage: python3 scripts/css_verify.py OLD.css NEW.css

For each file: group rules by selector, merge declarations last-wins (respecting
!important) into a {prop: value} map, resolve var(--x) to :root token values, and
canonicalize colors + whitespace. Then diff:

  - selectors only in OLD  -> expected to equal the removed-orphan set (printed)
  - selectors only in NEW  -> must be empty
  - common selectors       -> every prop's normalized value must match

Exit 0 only when the sole differences are dropped orphan selectors and ZERO
property values changed. This is the rigorous identical-render gate.
"""
from __future__ import annotations
import os, re, sys
from collections import OrderedDict

OUT = os.path.join(os.path.dirname(__file__), "out")


def scan_blocks(src: str):
    blocks, i, n, ps = [], 0, len(src), 0
    while i < n:
        c = src[i]
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            j = src.find("*/", i + 2); i = n if j == -1 else j + 2; ps = i; continue
        if c in "\"'":
            q = c; i += 1
            while i < n and src[i] != q:
                i += 2 if src[i] == "\\" else 1
            i += 1; continue
        if c == "{":
            prelude = src[ps:i].strip()
            depth, j = 1, i + 1
            while j < n and depth:
                cj = src[j]
                if cj == "/" and j + 1 < n and src[j + 1] == "*":
                    k = src.find("*/", j + 2); j = n if k == -1 else k + 2; continue
                if cj in "\"'":
                    q = cj; j += 1
                    while j < n and src[j] != q:
                        j += 2 if src[j] == "\\" else 1
                    j += 1; continue
                depth += 1 if cj == "{" else -1 if cj == "}" else 0
                j += 1
            kind = ("keyframes" if prelude.startswith("@keyframes")
                    else "atrule" if prelude.startswith("@")
                    else "root" if prelude == ":root" else "rule")
            blocks.append((kind, prelude, src[i + 1:j - 1]))
            i = j; ps = j; continue
        i += 1
    return blocks


def split_decls(body: str):
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    parts, depth, buf, i = [], 0, [], 0
    while i < len(body):
        c = body[i]
        if c in "\"'":
            q = c; buf.append(c); i += 1
            while i < len(body) and body[i] != q:
                buf.append(body[i]); i += 1
            if i < len(body): buf.append(body[i])
            i += 1; continue
        if c == "(": depth += 1
        elif c == ")": depth -= 1
        if c == ";" and depth == 0:
            parts.append("".join(buf)); buf = []; i += 1; continue
        buf.append(c); i += 1
    if "".join(buf).strip(): parts.append("".join(buf))
    out = []
    for d in parts:
        d = d.strip()
        if not d: continue
        depth, ci = 0, -1
        for k, ch in enumerate(d):
            if ch == "(": depth += 1
            elif ch == ")": depth -= 1
            elif ch == ":" and depth == 0: ci = k; break
        if ci == -1: continue
        prop = d[:ci].strip().lower()
        val = d[ci + 1:].strip()
        imp = False
        m = re.search(r"!\s*important\s*$", val, re.I)
        if m: imp = True; val = val[:m.start()].strip()
        out.append((prop, val, imp))
    return out


NAMED = {"white": "#ffffff", "black": "#000000", "transparent": "rgba(0,0,0,0.000)"}
COLOR_RE = re.compile(r"#[0-9a-fA-F]{3,8}\b|rgba?\([^)]*\)", re.I)


def norm_color(tok: str):
    t = tok.strip().lower()
    t = NAMED.get(t, t)
    m = re.fullmatch(r"#([0-9a-f]{3,8})", t)
    if m:
        h = m.group(1)
        if len(h) in (3, 4): h = "".join(ch * 2 for ch in h)
        return f"#{h}"
    m = re.fullmatch(r"rgba?\(([^)]*)\)", t)
    if m:
        ps = [p.strip() for p in re.split(r"[,/]", m.group(1)) if p.strip()]
        if len(ps) in (3, 4):
            try:
                nums = [str(int(round(float(p.rstrip("%"))))) if k < 3 else f"{float(p):.3f}"
                        for k, p in enumerate(ps)]
                if len(nums) == 3: nums.append("1.000")
                return "rgba(" + ",".join(nums) + ")"
            except ValueError:
                return tok.strip().lower()
    return tok.strip().lower()


def token_map(blocks):
    tm = {}
    for kind, prelude, body in blocks:
        if kind == "root":
            for p, v, _ in split_decls(body):
                if p.startswith("--"): tm[p] = v
    return tm


def resolve_vars(value: str, tm, depth=0):
    if depth > 10 or "var(" not in value:
        return value
    def repl(m):
        inner = m.group(1)
        # split name, fallback at top-level comma
        d, ci = 0, -1
        for k, ch in enumerate(inner):
            if ch == "(": d += 1
            elif ch == ")": d -= 1
            elif ch == "," and d == 0: ci = k; break
        name = (inner if ci == -1 else inner[:ci]).strip()
        fb = "" if ci == -1 else inner[ci + 1:].strip()
        return tm.get(name, fb)
    prev = None
    out = value
    while prev != out and "var(" in out:
        prev = out
        out = re.sub(r"var\(([^()]*(?:\([^()]*\)[^()]*)*)\)", repl, out)
    return resolve_vars(out, tm, depth + 1) if "var(" in out and out != value else out


def norm_value(value: str, tm) -> str:
    v = resolve_vars(value, tm)
    v = COLOR_RE.sub(lambda m: norm_color(m.group(0)), v)
    v = re.sub(r"\s+", " ", v).strip().lower().rstrip(";").strip()
    return v


def merged_dict(blocks):
    """selector(normalized) -> {prop: (value, important)} last-wins."""
    groups = OrderedDict()
    for kind, prelude, body in blocks:
        if kind != "rule": continue
        sel = re.sub(r"\s+", " ", prelude).strip()
        groups.setdefault(sel, []).append(body)
    out = {}
    for sel, bodies in groups.items():
        cur = {}
        for body in bodies:
            for prop, val, imp in split_decls(body):
                if prop in cur and cur[prop][1] and not imp:
                    continue  # don't let a non-important override an !important
                cur[prop] = (val, imp)
        out[sel] = cur
    return out


def main():
    old_p, new_p = sys.argv[1], sys.argv[2]
    old = scan_blocks(open(old_p).read())
    new = scan_blocks(open(new_p).read())
    tm_old, tm_new = token_map(old), token_map(new)
    mo, mn = merged_dict(old), merged_dict(new)

    only_old = sorted(set(mo) - set(mn))
    only_new = sorted(set(mn) - set(mo))
    common = sorted(set(mo) & set(mn))

    expected_removed = set()
    rp = os.path.join(OUT, "removed_orphans.txt")
    if os.path.exists(rp):
        expected_removed = {l.strip() for l in open(rp) if l.strip()}

    unexpected_removed = [s for s in only_old if s not in expected_removed]

    value_diffs = []
    for sel in common:
        po, pn = mo[sel], mn[sel]
        props = set(po) | set(pn)
        for p in sorted(props):
            if p not in po:
                value_diffs.append((sel, p, "<absent-old>", norm_value(pn[p][0], tm_new)))
                continue
            if p not in pn:
                value_diffs.append((sel, p, norm_value(po[p][0], tm_old), "<absent-new>"))
                continue
            vo = norm_value(po[p][0], tm_old)
            vn = norm_value(pn[p][0], tm_new)
            if vo != vn:
                value_diffs.append((sel, p, vo, vn))

    print(f"OLD selectors: {len(mo)}   NEW selectors: {len(mn)}")
    print(f"only in OLD (removed): {len(only_old)}  | expected orphan removals: {len(expected_removed)}")
    print(f"only in NEW (added):   {len(only_new)}")
    print(f"UNEXPECTED removals:   {len(unexpected_removed)}")
    print(f"VALUE DIFFS:           {len(value_diffs)}")
    if unexpected_removed:
        print("\n-- UNEXPECTED removed selectors --")
        for s in unexpected_removed: print("   ", s)
    if only_new:
        print("\n-- unexpected NEW selectors --")
        for s in only_new: print("   ", s)
    if value_diffs:
        print("\n-- value diffs (selector | prop | old -> new) --")
        for sel, p, vo, vn in value_diffs[:200]:
            print(f"   {sel}\n       {p}: {vo!r}  ->  {vn!r}")

    ok = not unexpected_removed and not only_new and not value_diffs
    print("\nRESULT:", "PASS — render-equivalent (only orphans dropped)" if ok else "FAIL — investigate above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
