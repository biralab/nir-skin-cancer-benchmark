# -*- coding: utf-8 -*-
"""JDIM-D-26-02857 R1: in-place revision engine for Blinded_Manuscript.docx.
Applies revtext.py replacement text, updates Tables 1-8, inserts Tables 9-11 and
Fig. 6, replaces/deletes figures per Editor request, renumbers all citations by
first appearance and rebuilds the reference list from refs_final.json.
Output: /workspace/jdim/Blinded_Manuscript_R1.docx
"""
import re, json, sys
sys.path.insert(0, "/workspace/benchmark")
import docx
from docx.text.paragraph import Paragraph
from docx.table import Table
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches
import revtext as RT

SRC = "/workspace/jdim/Blinded_Manuscript.docx"
DST = "/workspace/jdim/Blinded_Manuscript_R1.docx"
FIG = "/workspace/benchmark/figures/"
REFS = json.load(open("/workspace/benchmark/refs_final.json"))

ALIAS = {  # revtext symbolic key -> refs_final key
 "vaswani2017":"k15","choi2023":"k16","rayarao2025":"k17","bratchenko2022":"k18",
 "zhao2024":"k19","mishra2021":"k20","zhang2020":"k21","yan2025":"k22","wu2021":"k23",
 "guo2021":"k24","wang2024":"k25","qiu2022":"k26","loss2024":"k27","courtenay2024":"k28",
 "dragka2026":"k29","varga2025":"k30","haenssle2018":"k10","araujo2021":"k61",
 "bratchenko2021":"k13","alba2017":"k39","vancalster2019":"k40","nicora2022":"k65",
 "demler2012":"k45","kantidakis2023":"k63","navarro2022":"k64","rinnan2009":"k14",
 "jernelv2020":"k46","malek2018":"k47","mosquera2024":"k66","collins2024tripod":"k67",
 "collins2021protocol":"k68","cohen2024tripod":"k69","chuduc2025":"k62","hu2021":"k75",
 "khan2025":"k76","demsar2006":"new_demsar","vickers2006":"new_vickers",
 "msystems2025":"new_msystems","zhu2025":"k42","bi2016":"k70","weber2025":"k71",
 "bian2022":"k72","babatunde2025":"k73","seoni2026":"k74","gerretzen2017":"k38",
}

def sub_keys(text):
    return re.sub(r"\{([a-z0-9_]+)\}", lambda m: "{" + ALIAS[m.group(1)] + "}", text)

doc = docx.Document(SRC)

# ---------------------------------------------------------------- helpers
def _norm(s):
    return s.replace("\xa0", " ").strip()

def find_para(anchor):
    for p in doc.paragraphs:
        if _norm(p.text).startswith(anchor):
            return p
    raise KeyError("anchor not found: " + anchor[:60])

_KEEPTAGS = {qn("w:pPr"), qn("w:bookmarkStart"), qn("w:bookmarkEnd")}
def set_text(p, text):
    # remove ALL content children (runs, hyperlinks, fields) but keep bookmarks
    for child in list(p._p):
        if child.tag not in _KEEPTAGS:
            p._p.remove(child)
    p.add_run(text)

def insert_para_after(p, text="", style=None):
    el = OxmlElement("w:p")
    p._p.addnext(el)
    np = Paragraph(el, p._parent)
    if style:
        np.style = doc.styles[style] if isinstance(style, str) else style
    if text:
        np.add_run(text)
    return np

def para_before(p, text="", style=None):
    el = OxmlElement("w:p")
    p._p.addprevious(el)
    np = Paragraph(el, p._parent)
    if style:
        np.style = doc.styles[style] if isinstance(style, str) else style
    if text:
        np.add_run(text)
    return np

def delete_para(p):
    p._p.getparent().remove(p._p)

def png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")

def replace_image(caption_anchor, png_path, width_in=6.0):
    cap = find_para(caption_anchor)
    prev = cap._p.getprevious()
    assert prev is not None and prev.findall(".//" + qn("a:blip")), "no image before " + caption_anchor
    blip = prev.findall(".//" + qn("a:blip"))[0]
    part = doc.part.related_parts[blip.get(qn("r:embed"))]
    assert part.content_type == "image/png", part.content_type
    with open(png_path, "rb") as f:
        part._blob = f.read()
    w, h = png_size(png_path)
    cx = int(width_in * 914400); cy = int(cx * h / w)
    for ext in prev.findall(".//" + qn("wp:extent")) + prev.findall(".//" + qn("a:ext")):
        ext.set("cx", str(cx)); ext.set("cy", str(cy))

def set_cell(cell, text, bold=False):
    cell.text = text
    if bold:
        for pr in cell.paragraphs:
            for r in pr.runs:
                r.bold = True

def fill_table(tbl, rows, bold_header=True):
    for ri, row in enumerate(tbl.rows):
        cells = row.cells
        if len(rows[ri]) == 1 and cells[0]._tc is cells[-1]._tc:  # merged section row
            set_cell(cells[0], rows[ri][0])
            continue
        assert len(rows[ri]) == len(cells), f"row {ri}: {len(rows[ri])} vs {len(cells)}"
        for ci, val in enumerate(rows[ri]):
            set_cell(cells[ci], val, bold=(bold_header and ri == 0))

def add_table_after(anchor_para, rows, style_from):
    tbl = doc.add_table(rows=len(rows), cols=len(rows[0]))
    tbl.style = style_from.style
    fill_table(tbl, rows)
    anchor_para._p.addnext(tbl._tbl)
    return tbl

T1, T2, T3, T4, T5, T6, T7, T8 = list(doc.tables)  # capture originals

# ---------------------------------------------------------------- 1. paragraph replacements
REPL = [
    ("Background. Optical vibrational spectroscopy", RT.ABSTRACT),
    ("In parallel, the ML literature has drifted", RT.P8),
    ("This article makes four contributions.", RT.P9),
    ("Table 2 synthesises twelve optical skin-cancer studies", RT.P51),
    ("The development cohort, NIR-SC-UFES", RT.P57),
    ("Four classical models (logistic regression", RT.P60),
    ("On NIR-SC-UFES the best classical model", RT.P65),
    ("Trained on NIR-SC-UFES and applied without re-fitting", RT.P69),
    ("A single split can mislead.", RT.P74),
    ("Discrimination is necessary but not sufficient", RT.P80),
    ("Binary malignant/benign framing hides", RT.P87),
    ("The recurrent failure modes in this literature", RT.P91),
    ("Moving optical skin-cancer ML from proof-of-concept", RT.P100),
    ("Our reproducible benchmark uses two NIR cohorts", RT.P102),
    ("Across the published optical skin-cancer literature", RT.P104),
    ("All benchmark numbers reproduce from released artifacts", RT.P121),
]
for anchor, newtext in REPL:
    set_text(find_para(anchor), sub_keys(newtext))
print("1. paragraph replacements:", len(REPL))

# ---------------------------------------------------------------- 2. heading renumbering
for old, new in [("9. Methodological Pitfalls", "10. Methodological Pitfalls and a Six-Point Checklist"),
                 ("10. A Translation Roadmap", "11. A Translation Roadmap"),
                 ("11. Limitations", "12. Limitations"),
                 ("12. Conclusion", "13. Conclusion")]:
    set_text(find_para(old), new)
print("2. headings renumbered")

# ---------------------------------------------------------------- 3. deep-spec block + Table 9 after P60
p60 = find_para("Four classical models (logistic regression")
h = insert_para_after(p60, RT.H4_DEEP, style="Heading 4")
p = insert_para_after(h, sub_keys(RT.P_DEEP1), style="First Paragraph")
cap9 = insert_para_after(p, RT.CAP_T9, style="Table Caption")
T9_ROWS = [
 ["Component", "1D-CNN", "Hybrid CNN–transformer", "Pure transformer"],
 ["Feature stem",
  "Three Conv1d blocks (1→32, 32→64, 64→128; kernels 9/7/5, stride 2), each with batch normalisation + ReLU",
  "Conv1d stem (1→32, k=9, s=2; 32→64, k=5, s=2), linear projection to d_model=64 + learned positional embedding",
  "Patch embedding: 25 non-overlapping 5-channel patches → d_model=64; CLS token + learned positional embedding"],
 ["Sequence encoder", "—",
  "Transformer encoder, 2 layers (4 heads, feed-forward 128, GELU, dropout 0.1)",
  "Transformer encoder, 2 layers (4 heads, feed-forward 128, GELU, dropout 0.1)"],
 ["Classifier head", "Global average pooling + dropout 0.3 + linear",
  "Mean pooling + dropout + linear", "CLS token + dropout + linear"],
 ["Parameters (125 channels)", "≈83,000", "≈60,000", "≈50,000"],
 ["Training (shared)",
  "AdamW (learning rate 10⁻³, weight decay 10⁻⁴), batch size 32, cross-entropy with inverse-frequency class weights; early stopping (patience 20, min Δ 10⁻⁴) on an inner grouped 80/20 patient-level split of each training fold; ≤200 epochs; per-epoch train/validation loss and AUC logged",
  "", ""],
]
t9 = add_table_after(cap9, T9_ROWS, T3)
t9.rows[5].cells[1].merge(t9.rows[5].cells[2]).merge(t9.rows[5].cells[3])
print("3. deep-spec block + Table 9 inserted")

# ---------------------------------------------------------------- 4. new Section 5.5 + Fig. 6 before "6. Robustness"
h6 = find_para("6. Robustness: Repeated and Nested Cross-Validation")
cap6p = para_before(h6, sub_keys(RT.CAP_FIG6), style="Image Caption")
img_para = para_before(cap6p, "", style="Captioned Figure")
img_para.add_run().add_picture(FIG + "Fig6_learning_curves.png", width=Inches(6.0))
p55 = para_before(img_para, sub_keys(RT.P55_1), style="First Paragraph")
para_before(p55, RT.H2_55, style="Heading 2")
print("4. Section 5.5 + Fig. 6 inserted")

# ---------------------------------------------------------------- 5. Discussion + Table 10 before "10. Methodological Pitfalls"
h10 = find_para("10. Methodological Pitfalls")
pclin = para_before(h10, sub_keys(RT.P_CLIN), style="First Paragraph")
hclin = para_before(pclin, RT.H4_CLIN, style="Heading 4")
pwhy = para_before(hclin, sub_keys(RT.P_WHY), style="First Paragraph")
hwhy = para_before(pwhy, RT.H4_WHY, style="Heading 4")
cap10 = para_before(hwhy, RT.CAP_T10, style="Table Caption")
T10_ROWS = [
 ["Study", "Setting", "Headline finding", "Gap relative to the present work"],
 ["[{bratchenko2022}]", "617 in-vivo Raman lesions (615 patients), portable device",
  "1D-CNN AUC 0.96, significantly above PLS-DA (p<0.01)",
  "Spectrum-level splits; no calibration analysis; no external validation"],
 ["[{zhao2024}]", "731 in-vivo Raman lesions",
  "CNN gain over PLS-DA small (AUC 0.886 vs 0.870; 0.909 vs 0.899 with augmentation)",
  "No external validation; no calibration; splitting level unclear"],
 ["[{mishra2021}]", "NIR spectroscopy, non-medical (fruit quality)",
  "Chemometrics–deep-learning synergy improves prediction",
  "Non-clinical domain; no leakage audit or calibration"],
 ["[{loss2024}]", "971 NIR spectra (MicroNIR, in vivo)",
  "Classical gradient boosting best (balanced accuracy 0.839); PLS-DA ≤0.790",
  "Spectrum-level evaluation; no calibration; no external cohort"],
 ["This work", "647 spectra / 319 patients (NIR-SC-UFES) + 152 external (Deteccthia)",
  "Classical ≥ deep under patient-level grouped CV (0.774 vs 0.663); deep models miscalibrated",
  "Patient-level splits, 20 seeds, nested CV, calibration, external stress test and DCA in one protocol"],
]
t10 = doc.add_table(rows=len(T10_ROWS), cols=4)
t10.style = T3.style
fill_table(t10, [[sub_keys(c) for c in r] for r in T10_ROWS])
cap10._p.addnext(t10._tbl)
ppos = para_before(cap10, sub_keys(RT.P_POS), style="First Paragraph")
hpos = para_before(ppos, RT.H4_POS, style="Heading 4")
para_before(hpos, RT.H1_DISC, style="Heading 1")
print("5. Discussion + Table 10 inserted")

# ---------------------------------------------------------------- 6. Table 11 after P102
p102 = find_para("Several limitations qualify our findings")
cap11 = insert_para_after(p102, RT.CAP_T11, style="Table Caption")
T11_ROWS = [
 ["Preprocessing chain", "Logistic Reg.", "1D-CNN", "Pure Transformer"],
 ["raw (no preprocessing)", "0.742", "0.639", "0.633"],
 ["L2", "0.756", "0.654", "0.617"],
 ["L2 + SNV", "0.775", "0.630", "0.589"],
 ["L2 + SNV + SG", "0.772", "0.667", "0.598"],
]
t11 = add_table_after(cap11, T11_ROWS, T3)
print("6. Table 11 inserted")

# ---------------------------------------------------------------- 7. delete checklist item paragraphs (moved to Supplement)
for anchor in ["Report a matched classical baseline.", "Isolate preprocessing to the training fold.",
               "Use lesion-level, not spectrum-level, splits.", "Report calibration, not only discrimination.",
               "Test ranking stability.", "State the modality honestly."]:
    delete_para(find_para(anchor))
print("7. checklist item paragraphs deleted")

# ---------------------------------------------------------------- 8. replace four figure images
replace_image("Fig. 2 Reported/benchmarked", FIG + "Fig2_landscape.png")
replace_image("Fig. 4 Pairwise DeLong", FIG + "Fig3_delong_matrix.png")
replace_image("Fig. 8 Decision-curve", FIG + "Fig4_dca.png")
replace_image("Fig. 9 Three-class", FIG + "Fig5_threeclass.png")
print("8. figure images replaced")

# ---------------------------------------------------------------- 9. delete old Figs 3, 5, 6, 7
for cap_anchor in ["Fig. 3 Receiver-operating", "Fig. 5 Cross-cohort",
                   "Fig. 6 Distribution of pooled", "Fig. 7 Brier score"]:
    cap = find_para(cap_anchor)
    img = cap._p.getprevious()
    if img is not None and img.findall(".//" + qn("a:blip")):
        img.getparent().remove(img)
    delete_para(cap)
print("9. old Figs 3/5/6/7 deleted")

# ---------------------------------------------------------------- 10. caption updates
CAP_REPL = [
    ("Fig. 2 Reported/benchmarked", RT.CAP_FIG2),
    ("Fig. 4 Pairwise DeLong", RT.CAP_FIG3),
    ("Fig. 8 Decision-curve", RT.CAP_FIG4),
    ("Fig. 9 Three-class", RT.CAP_FIG5),
    ("Table 2 Composition", RT.CAP_T2),
    ("Table 3 Full performance", RT.CAP_T3),
    ("Table 4 External validation", RT.CAP_T4),
    ("Table 5 Stability of ranking", RT.CAP_T5),
    ("Table 6 Nested cross-validation", RT.CAP_T6),
    ("Table 7 Effect of post-hoc Platt", RT.CAP_T7),
    ("Table 8 Six-point reproducibility", RT.CAP_T8),
]
for anchor, newtext in CAP_REPL:
    set_text(find_para(anchor), sub_keys(newtext))
print("10. captions updated")

# ---------------------------------------------------------------- 11. Table 1 (verified literature)
T1_ROWS = [
 ["Study", "Year", "Modality", "Cohort", "Task", "Model finding", "Headline result"],
 ["[{araujo2021}]", "2021", "Raman (ex vivo)", "436 spectra (168 nevus / 268 melanoma)",
  "melanoma vs nevus", "Classical gradient boosting (LightGBM, reduced fingerprint)",
  "AUC 0.98 (95% CI 0.97–0.99); 0.97 with a miniaturized range"],
 ["[{qiu2022}]", "2023", "SERS (single cell, in vitro)", "20,000 single-cell spectra",
  "melanoma vs melanocyte", "Deep (CNN on SERS)",
  ">98% cell-level accuracy; no classical baseline reported"],
 ["[{bratchenko2022}]", "2022", "Raman (in vivo, portable)", "617 lesions (615 patients)",
  "malignant vs benign; melanoma subtasks", "1D-CNN vs PLS-DA",
  "CNN AUC 0.96 (0.94–0.97), significantly above PLS-DA (p<0.01)"],
 ["[{zhao2024}]", "2024", "Raman (in vivo)", "731 lesions (340 cancer/pre-cancer)",
  "cancer/pre-cancer vs benign", "1D-CNN vs PLS-DA/PC-LDA/SVM/LR",
  "CNN AUC 0.886 vs PLS-DA 0.870; augmentation lifts both (0.909 vs 0.899)"],
 ["[{wang2024}]", "2024", "Raman + autofluorescence (ex vivo)", "149 spectra (63 normal / 50 SCC / 36 BCC)",
  "three-class tissue classification", "Transfer contrastive deep learning vs LR/SVM/RF/kNN",
  "TCLP accuracy 0.833 (AUC 0.925); best classical (LR) 0.766"],
 ["[{loss2024}]", "2024", "NIR (in vivo, MicroNIR)", "971 spectra (586 benign / 385 malignant)",
  "cancer vs non-cancer", "Classical gradient boosting (LightGBM) best",
  "Balanced accuracy 0.839; PLS-DA ≤0.790, SVM ≤0.813"],
 ["[{courtenay2024}]", "2024", "NIR-HSI (in vivo, Deteccthia)", "125 patients; 8,250 pixel spectra",
  "BCC/SCC/AK vs healthy", "Robust statistics (no classifier trained)",
  "Optimal NIR spectral windows reported; no classification metric"],
 ["[{bratchenko2021}]", "2021", "Raman (in vivo, portable)", "617 lesions (615 patients)",
  "malignant vs benign; melanoma subtasks", "Classical (PLS-DA)",
  "AUC 0.75 (0.71–0.79) malignant vs benign"],
 ["[{dragka2026}]", "2026", "NIR ± impedance (primary care)", "80 participants / 109 lesions",
  "any skin cancer", "LR / linear & RBF SVM / RF / NN compared",
  "NIR-alone AUC 0.826; NIR+impedance 0.776; impedance adds no clear benefit"],
 ["[{chuduc2025}]", "2025", "Raman (this group)", "1,200 spectra (150 MM / 350 BCC / 350 SCC / 350 benign)",
  "melanoma vs non-melanoma vs benign", "Hybrid 1D-CNN + transformer",
  "Accuracy 96.8%, AUC 0.987; deep baseline only"],
 ["[{varga2025}]", "2025", "Optical imaging (meta-analysis)", "138 studies",
  "melanoma detection", "RCM, dermoscopy + AI",
  "Pooled sensitivity 0.93 for RCM and dermoscopy + AI"],
 ["[{haenssle2018}]", "2018", "Dermoscopy (images)", "100-image test set; 58 dermatologists",
  "melanoma vs benign", "Deep CNN (Inception v4)",
  "CNN AUC 0.86 vs dermatologists 0.79 (p<0.01); specificity 82.5%"],
]
fill_table(T1, [[sub_keys(c) for c in r] for r in T1_ROWS])
print("11. Table 1 updated")

# ---------------------------------------------------------------- 12. Table 2
T2_ROWS = [
 ["Cohort / class", "Category", "n spectra (n patients)"],
 ["NIR-SC-UFES (development, MicroNIR, 125 bands, 908–1676 nm)"],
 ["ACK (actinic keratosis)", "pre-malignant", "296 (107)"],
 ["BCC (basal cell carcinoma)", "malignant", "159 (117)"],
 ["SCC (squamous cell carcinoma)", "malignant", "38 (32)"],
 ["MEL (melanoma)", "malignant", "6 (6)"],
 ["NEV (nevus)", "benign", "41 (27)"],
 ["SEK (seborrheic keratosis)", "benign", "107 (63)"],
 ["Other (ambiguous label)", "excluded", "67 (51)"],
 ["Binary task total", "malignant/pre-malignant 499 / benign 148", "647 (319)"],
 ["Three-class task", "benign 148 / pre-malignant (AK) 296 / carcinoma (BCC/SCC) 197", "641 (313)"],
 ["Deteccthia (external, pushbroom NIR-HSI, 150 bands, 900–1606 nm)"],
 ["Malignant (BCC+SCC+AK)", "malignant", "115"],
 ["Healthy skin", "negative class", "37"],
 ["External total", "lesion-level mean spectra (per Photo_ID)", "152"],
]
fill_table(T2, T2_ROWS)
print("12. Table 2 updated")

# ---------------------------------------------------------------- 13. Tables 3-7
T3_ROWS = [
 ["Model", "", "AUC", "95% CI", "PR-AUC", "Brier", "ECE", "Sens.", "Spec.", "F1", "p†"],
 ["Logistic Reg.", "C", "0.774", "0.709–0.838", "0.907", "0.152", "0.077", "0.896", "0.405", "0.865", "—"],
 ["PLS-DA+SVM", "C", "0.751", "0.680–0.816", "0.889", "0.143", "0.044", "0.948", "0.378", "0.889", "0.297"],
 ["PCA-LDA", "C", "0.736", "0.673–0.799", "0.893", "0.158", "0.066", "0.926", "0.338", "0.873", "0.060"],
 ["Random Forest", "C", "0.703", "0.629–0.771", "0.875", "0.160", "0.046", "0.934", "0.236", "0.865", "0.106"],
 ["1D-CNN", "D", "0.663", "0.596–0.727", "0.853", "0.219", "0.198", "0.713", "0.561", "0.774", "0.040"],
 ["Hybrid CNN-Tr.", "D", "0.629", "0.549–0.707", "0.835", "0.259", "0.273", "0.555", "0.676", "0.672", "0.010"],
 ["Pure Transformer", "D", "0.611", "0.540–0.676", "0.837", "0.252", "0.259", "0.505", "0.655", "0.628", "<0.001"],
]
fill_table(T3, T3_ROWS)
T4_ROWS = [
 ["Model", "", "AUC (95% CI)", "Brier"],
 ["Logistic Reg.", "C", "0.467 (0.357–0.587)", "0.757"],
 ["PLS-DA+SVM", "C", "0.500 (degenerate)", "0.192"],
 ["PCA-LDA", "C", "0.417 (0.304–0.535)", "0.757"],
 ["Random Forest", "C", "0.560 (0.463–0.650)", "0.202"],
 ["1D-CNN", "D", "0.461 (0.358–0.571)", "0.242"],
 ["Hybrid CNN-Tr.", "D", "0.439 (0.335–0.554)", "0.186"],
 ["Pure Transformer", "D", "0.481 (0.371–0.588)", "0.246"],
]
fill_table(T4, T4_ROWS)
T5_ROWS = [
 ["Model", "", "Mean AUC", "SD", "Range"],
 ["Logistic Reg.", "C", "0.765", "0.011", "0.747–0.784"],
 ["PLS-DA+SVM", "C", "0.741", "0.015", "0.713–0.764"],
 ["PCA-LDA", "C", "0.753", "0.019", "0.702–0.776"],
 ["Random Forest", "C", "0.702", "0.015", "0.674–0.731"],
 ["1D-CNN", "D", "0.620", "0.036", "0.561–0.694"],
 ["Hybrid CNN-Tr.", "D", "0.559", "0.032", "0.500–0.629"],
 ["Pure Transformer", "D", "0.564", "0.038", "0.496–0.622"],
]
fill_table(T5, T5_ROWS)
T6_ROWS = [
 ["Model", "Nested AUC", "Single-split", "|Δ|", "Chosen HP (per fold)"],
 ["Logistic Reg.", "0.752 ± 0.103", "0.774", "0.021", "1, 1, 1, 10, 1"],
 ["PLS-DA+SVM", "0.759 ± 0.044", "0.751", "0.008", "10, 25, 15, 10, 25"],
 ["PCA-LDA", "0.735 ± 0.052", "0.736", "0.001", "20, 15, 20, 20, 15"],
 ["Random Forest", "0.708 ± 0.043", "0.703", "0.005", "8, 4, None, 4, 8"],
]
fill_table(T6, T6_ROWS)
T7_ROWS = [
 ["Setting", "AUC", "Brier", "ECE"],
 ["Raw probabilities", "0.774", "0.152", "0.077"],
 ["After Platt scaling", "0.757", "0.150", "0.038"],
]
fill_table(T7, T7_ROWS)
print("13. Tables 3-7 updated")

# ---------------------------------------------------------------- 14. Table 8: item 1 -> patient level; keep supporting citations in cells
set_cell(T8.rows[1].cells[0], "1. Patient-level (not spectrum-level) splitting")
set_cell(T8.rows[1].cells[1], "Optimistic leakage when replicate spectra from one patient straddle the train/test boundary")
set_cell(T8.rows[2].cells[1], sub_keys("Information leakage from test-set statistics into normalisation, SNV parameters or feature selection [{rinnan2009}, {yan2025}, {gerretzen2017}, {bi2016}, {weber2025}, {bian2022}, {babatunde2025}]"))
set_cell(T8.rows[3].cells[1], sub_keys("Miscalibrated probabilities that mislead clinical thresholding [{alba2017}, {vancalster2019}, {zhu2025}, {mosquera2024}, {seoni2026}]"))
set_cell(T8.rows[5].cells[1], sub_keys("Ranking artifacts from a single lucky split or tuning optimism [{kantidakis2023}, {navarro2022}]"))
print("14. Table 8 updated")

# ---------------------------------------------------------------- 15. numeric citations -> keys (skip References)
CIT = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
def num_to_keys(m):
    nums = [int(x) for x in re.split(r"\s*,\s*", m.group(1))]
    return "[" + ", ".join("{k%d}" % n for n in nums) + "]"

in_refs = False
n_conv = 0
for p in doc.paragraphs:
    if p.text.strip() == "References":
        in_refs = True
    if in_refs:
        continue
    t = p.text
    if CIT.search(t):
        set_text(p, CIT.sub(num_to_keys, t))
        n_conv += 1
print("15. numeric citations converted in", n_conv, "paragraphs")

# ---------------------------------------------------------------- 16. renumber + rebuild references
KEYGROUP = re.compile(r"\[((?:\{[a-z0-9_]+\})(?:\s*,\s*\{[a-z0-9_]+\})*)\]")
order, num_of = [], {}
def assign(key):
    if key not in num_of:
        order.append(key)
        num_of[key] = len(order)
    return num_of[key]

def compress(nums):
    nums = sorted(set(nums)); out = []; i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        if j - i >= 2:
            out.append(f"{nums[i]}\u2013{nums[j]}")
        else:
            out.extend(str(n) for n in nums[i:j + 1])
        i = j + 1
    return "[" + ", ".join(out) + "]"

def renumber_text(t):
    def rep(m):
        keys = re.findall(r"\{([a-z0-9_]+)\}", m.group(1))
        return compress([assign(k) for k in keys])
    return KEYGROUP.sub(rep, t)

in_refs = False
for child in doc.element.body.iterchildren():
    if child.tag.endswith("}p"):
        p = Paragraph(child, doc)
        if p.text.strip() == "References":
            in_refs = True
            continue
        if in_refs:
            continue
        if KEYGROUP.search(p.text):
            set_text(p, renumber_text(p.text))
    elif child.tag.endswith("}tbl"):
        tbl = Table(child, doc)
        for row in tbl.rows:
            seen = set()
            for cell in row.cells:
                if id(cell._tc) in seen:
                    continue
                seen.add(id(cell._tc))
                for p in cell.paragraphs:
                    if KEYGROUP.search(p.text):
                        set_text(p, renumber_text(p.text))
print("16. renumbering done:", len(order), "distinct references cited")

ref_paras, in_refs = [], False
for p in doc.paragraphs:
    if p.text.strip() == "References":
        in_refs = True
        continue
    if in_refs:
        ref_paras.append(p)
ref_style = ref_paras[0].style
for p in ref_paras:
    delete_para(p)
anchor = find_para("References")
for i, key in enumerate(order, 1):
    anchor = insert_para_after(anchor, f"{i}. {REFS[key]['entry']}", style=ref_style)
uncited = set(REFS) - set(order)
print("    reference list rebuilt:", len(order), "entries; uncited dropped:", sorted(uncited))

# ---------------------------------------------------------------- 17. fixed layout + explicit column widths
def set_col_widths(tbl, widths):
    tblPr = tbl._tbl.tblPr
    tblW = tblPr.find(qn("w:tblW"))
    if tblW is None:
        tblW = OxmlElement("w:tblW"); tblPr.append(tblW)
    tblW.set(qn("w:type"), "dxa"); tblW.set(qn("w:w"), str(sum(widths)))
    layout = tblPr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout"); tblPr.append(layout)
    layout.set(qn("w:type"), "fixed")
    # tight cell margins so 9pt numbers fit
    mar = tblPr.find(qn("w:tblCellMar"))
    if mar is None:
        mar = OxmlElement("w:tblCellMar"); tblPr.append(mar)
    for side in ("left", "right"):
        el = mar.find(qn("w:" + side))
        if el is None:
            el = OxmlElement("w:" + side); mar.append(el)
        el.set(qn("w:w"), "60"); el.set(qn("w:type"), "dxa")
    grid = tbl._tbl.find(qn("w:tblGrid"))
    for gc, w in zip(grid.findall(qn("w:gridCol")), widths):
        gc.set(qn("w:w"), str(w))
    for row in tbl.rows:
        seen = set()
        for cell, w in zip(row.cells, widths):
            if id(cell._tc) in seen:
                continue
            seen.add(id(cell._tc))
            tcW = cell._tc.get_or_add_tcPr().find(qn("w:tcW"))
            if tcW is None:
                tcW = OxmlElement("w:tcW"); cell._tc.get_or_add_tcPr().append(tcW)
            tcW.set(qn("w:type"), "dxa"); tcW.set(qn("w:w"), str(w))

from docx.shared import Pt
def set_table_font(tbl, size_pt=9):
    for row in tbl.rows:
        seen = set()
        for cell in row.cells:
            if id(cell._tc) in seen:
                continue
            seen.add(id(cell._tc))
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(size_pt)

WIDTHS = [
 (T1, [650, 550, 1200, 1600, 1300, 1800, 1800]),
 (T2, [3500, 3300, 2500]),
 (t9, [1500, 2600, 2600, 2600]),
 (T3, [1260, 360, 560, 1000, 640, 560, 540, 600, 620, 560, 740]),
 (T4, [2400, 600, 3600, 1400]),
 (T5, [2200, 600, 1400, 1000, 1600]),
 (T6, [1800, 1800, 1300, 900, 2600]),
 (T7, [3000, 1500, 1500, 1500]),
 (T8, [3800, 5500]),
 (t10, [900, 2400, 2900, 3100]),
 (t11, [2800, 2100, 2100, 2100]),
]
for t, w in WIDTHS:
    set_col_widths(t, w)
    set_table_font(t, 9)
print("17. column widths + 9pt font set for", len(WIDTHS), "tables")

# Repeat-header only makes sense for the two long tables (T1, T2) that span
# pages; on single-page tables it can paint an orphan header row when the table
# ends exactly at a page boundary (observed for Table 4).
def drop_repeat_header(tbl):
    trPr = tbl.rows[0]._tr.find(qn("w:trPr"))
    if trPr is None:
        return
    th = trPr.find(qn("w:tblHeader"))
    if th is not None:
        trPr.remove(th)

for t in (t9, T3, T4, T5, T6, T7, T8, t10, t11):
    drop_repeat_header(t)
print("17b. repeat-header kept only for Tables 1-2")

doc.save(DST)
print("SAVED", DST)
