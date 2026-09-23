# -*- coding: utf-8 -*-
"""QC re-extraction of Blinded_Manuscript_R1.docx."""
import re, docx
from docx.text.paragraph import Paragraph
from docx.table import Table
from docx.oxml.ns import qn

d = docx.Document("/workspace/jdim/Blinded_Manuscript_R1.docx")

# full text in body order
lines = []
n_img = 0
for child in d.element.body.iterchildren():
    if child.tag.endswith("}p"):
        p = Paragraph(child, d)
        if child.findall(".//" + qn("a:blip")):
            n_img += 1
            lines.append(f"[IMAGE #{n_img}]")
        if p.text.strip():
            lines.append(f"[{p.style.name}] {p.text}")
    elif child.tag.endswith("}tbl"):
        t = Table(child, d)
        lines.append(f"[TABLE {len(t.rows)}x{len(t.columns)}]")
        for row in t.rows:
            seen, cells = set(), []
            for c in row.cells:
                if id(c._tc) in seen:
                    continue
                seen.add(id(c._tc))
                cells.append(c.text.replace("\n", " / "))
            lines.append("  | " + " | ".join(cells))
open("/workspace/jdim/revised_text.txt", "w").write("\n".join(lines))
print("paragraph+table dump -> revised_text.txt ;", len(lines), "lines;", n_img, "images")

full = "\n".join(lines)

# 1. leftover key markers / unconverted numeric citations
print("\n-- leftover [{key}]:", re.findall(r"\[\{[^]]+\}\]", full)[:5])
bad_num = [m for m in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", full)]
print("-- numeric citation groups found (valid, renumbered):", len(bad_num))

# 2. abstract word count
abs_m = re.search(r"\[Abstract\] (.*?)\n", full)
abs_text = abs_m.group(1)
print("-- Abstract words:", len(abs_text.split()))

# 3. figure/table cross-references in text vs captions
fig_refs = sorted(set(re.findall(r"Fig(?:ure)?\.?\s*(\d+)", full)))
tab_refs = sorted(set(re.findall(r"Table\s*(\d+)", full)))
print("-- Figure numbers referenced:", fig_refs)
print("-- Table numbers referenced:", tab_refs)
fig_caps = re.findall(r"Fig\.\s*(\d+)\s", full)
tab_caps = re.findall(r"Table\s*(\d+)\s+[A-Z]", full)
print("-- Figure captions present:", sorted(set(fig_caps), key=int))
print("-- Table captions present:", sorted(set(tab_caps), key=int))

# 4. section headings
print("\n-- Headings:")
for ln in lines:
    if ln.startswith("[Heading 1]") or ln.startswith("[Heading 2]"):
        print("  ", ln)

# 5. references count and first/last
refs = re.findall(r"\n\[Body Text\] (\d+)\. ", full)
print("\n-- References:", len(refs), "entries; first:", refs[:3], "last:", refs[-3:])

# 6. key numbers spot-check
for pat in ["0.774", "0.663", "6.2×10⁻²²", "0.0018", "0.829", "0.741", "647 spectra, 319 patients",
            "152 lesion-level", "0.077", "0.038", "20/20", "REPOSITORY-DOI"]:
    print(f"-- '{pat}':", full.count(pat), "occurrences")
