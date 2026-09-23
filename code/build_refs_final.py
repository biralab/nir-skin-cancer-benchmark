# -*- coding: utf-8 -*-
"""Build the final JDIM-formatted reference list (refs_final.json).
- Re-queries CrossRef for every DOI (fixes missing article numbers, author lists, HTML tags).
- Manual overrides for refs whose CrossRef title-match failed or that have no DOI.
- Adds three new references cited in the revision (Demšar 2006, Vickers 2006, Bratchenko 2025 mSystems).
Output: refs_final.json = {key: {"old_n": int|None, "entry": str}}  (entry WITHOUT leading number)
"""
import json, re, time, urllib.request, os

CACHE = "/workspace/benchmark/crossref_cache.json"
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

def crossref(doi):
    if doi in cache:
        return cache[doi]
    req = urllib.request.Request(
        "https://api.crossref.org/works/" + urllib.parse.quote(doi),
        headers={"User-Agent": "jdim-revision/1.0 (mailto:anonymous@example.com)"})
    for attempt in range(5):
        try:
            m = json.load(urllib.request.urlopen(req, timeout=30))["message"]
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 4:
                time.sleep(2 * (attempt + 1))
            else:
                raise
    cache[doi] = m
    time.sleep(0.3)
    return m

def initials(given):
    # "Matheus B." -> "MB"; "Ben" -> "B"; "F. P." -> "FP"
    parts = re.split(r"[\s.\-]+", given.strip())
    return "".join(p[0].upper() for p in parts if p and p[0].isalpha())

def fmt_authors(auths):
    names = []
    for a in auths:
        fam = a.get("family", "").strip()
        giv = a.get("given", "").strip()
        nm = (fam + " " + initials(giv)).strip()
        if nm:  # skip group/consortium placeholder names with no family/given
            names.append(nm)
    if len(names) > 6:
        names = names[:3] + ["et al."]
    return ", ".join(names)

def clean(t):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", t or "")).strip()

JOURNAL_MAP = {  # CrossRef short-title gaps -> ISO abbreviation
    "Experimental Dermatology": "Exp Dermatol",
    "The Lancet Oncology": "Lancet Oncol",
    "Annals of Oncology": "Ann Oncol",
    "Journal of the American Academy of Dermatology": "J Am Acad Dermatol",
    "American Journal of Clinical Oncology": "Am J Clin Oncol",
    "Computer Methods and Programs in Biomedicine": "Comput Methods Programs Biomed",
    "Chemometrics and Intelligent Laboratory Systems": "Chemom Intell Lab Syst",
    "Analytica Chimica Acta": "Anal Chim Acta",
    "Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy": "Spectrochim Acta A Mol Biomol Spectrosc",
    "Int J Dermatology": "Int J Dermatol",
    "Journal of Biophotonics": "J Biophotonics",
    "Computers in Biology and Medicine": "Comput Biol Med",
    "Journal of Clinical Epidemiology": "J Clin Epidemiol",
    "Statistics in Medicine": "Stat Med",
    "Journal of Chemometrics": "J Chemom",
    "Applied Soft Computing": "Appl Soft Comput",
    "Artificial Intelligence in Medicine": "Artif Intell Med",
    "Journal of Biomedical Informatics": "J Biomed Inform",
    "Skin Research and Technology": "Skin Res Technol",
    "JCM": "J Clin Med",
    "IJECE": "Int J Electr Comput Eng",
}

def first_nonempty(m, *fields):
    for f in fields:
        v = m.get(f) or []
        if v and clean(v[0]):
            return clean(v[0])
    return ""

def fmt_entry(m, doi):
    auths = fmt_authors(m.get("author", []))
    title = clean((m.get("title") or [""])[0])
    cont = first_nonempty(m, "short-container-title", "container-title")
    cont = JOURNAL_MAP.get(cont, cont).replace(".", "")
    year = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
    vol = m.get("volume") or ""
    page = m.get("page") or (m.get("article-number") or "")
    typ = m.get("type", "")
    sep = " " if auths.endswith(".") else ". "
    if typ == "posted-content":
        s = f"{auths}{sep}{title}. {cont} preprint, {year}."
    elif typ in ("book-chapter", "book-part"):
        s = f"{auths}{sep}{title}. In: {cont}"
        if page: s += f", pp {page}"
        s += f", {year}."
    elif typ == "proceedings-article":
        s = f"{auths}{sep}{title}. In: {cont}"
        if page: s += f", pp {page}"
        s += f", {year}."
    else:
        s = f"{auths}{sep}{title}. {cont}"
        if vol:
            s += f" {vol}"
            if page: s += f":{page}"
        elif page:
            s += f" {page}"
        s += f", {year}."
    if doi:
        s += f" https://doi.org/{doi.lower()}"
    return s.replace("\u212b", "\u00c5")  # ANGSTROM SIGN -> LATIN A WITH RING

MANUAL = {
 9:  "Harrison K. The accuracy of skin cancer detection rates with the implementation of dermoscopy among dermatology clinicians: a scoping review. J Clin Aesthet Dermatol 17(9-10 Suppl 1):S18-S27, 2024.",
 15: "Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need. In: Advances in Neural Information Processing Systems 30 (NeurIPS 2017), pp 5998-6008, 2017.",
 27: "Rocha MB, Loss FP, da Cunha PH, et al. Skin cancer diagnosis using NIR spectroscopy data of skin lesions in vivo using machine learning algorithms. Biocybern Biomed Eng 44:824-835, 2024. https://doi.org/10.1016/j.bbe.2024.10.001",
 34: "Flosdorf C, Engelker J, Keller I. Skin cancer detection utilizing deep learning: classification of skin lesion images using a vision transformer. arXiv preprint arXiv:2407.18554, 2024. https://doi.org/10.48550/arxiv.2407.18554",
 38: "Gerretzen J. Chemical data preprocessing: from art to science. PhD thesis, Radboud University Nijmegen, 2017.",
 46: "Jernelv IL, Hjelme D, Matsuura Y. Convolutional neural networks for classification and regression analysis of one-dimensional spectral data. arXiv preprint arXiv:2005.07530, 2020. https://doi.org/10.48550/arxiv.2005.07530",
 54: "Gu A, Dao T. Mamba: linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023. https://doi.org/10.48550/arxiv.2312.00752",
 59: "Sadi AA, Chowdhury L, Jahan N, Rafi MNS, Chowdhury R, Khan FA. LMFLOSS: a hybrid loss for imbalanced medical image classification. arXiv preprint arXiv:2212.12741, 2022. https://doi.org/10.48550/arxiv.2212.12741",
 14: "Rinnan \u00c5, van den Berg F, Engelsen SB. Review of the most common pre-processing techniques for near-infrared spectra. TrAC Trends Anal Chem 28:1201-1222, 2009. https://doi.org/10.1016/j.trac.2009.07.007",
 69: "Cohen JF, Bossuyt PMM. TRIPOD+AI: an updated reporting guideline for clinical prediction models. BMJ 384:q824, 2024. https://doi.org/10.1136/bmj.q824",
 58: "Kempanna SR, Rangappa AA, Maheshappa S, et al. Revolutionizing brain tumor diagnoses: a ResNet18 and focal loss approach to magnetic resonance imaging-based classification in neuro-oncology. Int J Electr Comput Eng 14:6551-6559, 2024. https://doi.org/10.11591/ijece.v14i6.pp6551-6559",
 71: "Weber AR. Preprocessing of spectral data. In: Practical Quantitative Vibrational and Electronic Spectroscopy, pp 123-153, 2025. https://doi.org/10.1002/9781394227259.ch6",
 17: "Rayarao SR, Donikena N. Transformers in deep learning: a comprehensive technical review. Authorea preprint, 2025. https://doi.org/10.22541/au.176297289.99757855/v1",
 56: "Gopi V. Lightweight transformer models for biomedical signal processing: trends, challenges, and future directions. Research Square preprint, 2025. https://doi.org/10.21203/rs.3.rs-7620509/v1",
}
NEW = {
 "new_demsar":   "Demšar J. Statistical comparisons of classifiers over multiple data sets. J Mach Learn Res 7:1-30, 2006.",
 "new_vickers":  "Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. Med Decis Making 26:565-574, 2006. https://doi.org/10.1177/0272989x06295361",
 "new_msystems": "Bratchenko IA, Bratchenko LA. Overestimation of the classification model for Raman spectroscopy data of biological samples. mSystems 10:e00063-25, 2025. https://doi.org/10.1128/msystems.00063-25",
}

rj = json.load(open("/workspace/benchmark/refs_jdim.json"))
out = {}
fails = []
for r in rj:
    n = r["n"]
    key = f"k{n}"
    if n in MANUAL:
        out[key] = {"old_n": n, "entry": MANUAL[n]}
        continue
    doi = r.get("doi")
    if not doi:
        fails.append((n, "no doi, no manual"))
        continue
    try:
        m = crossref(doi)
        out[key] = {"old_n": n, "entry": fmt_entry(m, doi)}
    except Exception as e:
        fails.append((n, f"{doi}: {e}"))
for k, v in NEW.items():
    out[k] = {"old_n": None, "entry": v}

json.dump(cache, open(CACHE, "w"))
json.dump(out, open("/workspace/benchmark/refs_final.json", "w"), ensure_ascii=False, indent=1)
print("refs written:", len(out), "| failures:", fails)
for k in ["k9", "k13", "k14", "k27", "k40", "k69", "k5", "k19", "k23", "k31", "k62", "k76"]:
    print(k, "->", out[k]["entry"][:170])
