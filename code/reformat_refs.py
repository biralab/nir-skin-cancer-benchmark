"""Reformat the 76 references to JDIM (Springer Vancouver) style using CrossRef.

Target style (observed in JDIM/J Digit Imaging papers):
  N. Surname AB, Surname CD, Surname EF, et al. Article title (sentence case).
  Journal Abbrev vol:pages, year. https://doi.org/...

Rules: <=6 authors -> all; >6 -> first 3 + et al. Journal names use NLM/ISO
abbreviation (CrossRef short-container-title, fallback manual map). Preprints
(arXiv/medRxiv/Authorea/Research Square/SSRN) formatted as:
  Surname AB. Title. arXiv. Published online year. https://doi.org/...
Every DOI is verified to resolve and its CrossRef title is checked against the
manuscript title (integrity audit — mismatches are flagged, never silently fixed).
"""
import json, re, time, sys
import requests

REFS = json.load(open("/workspace/benchmark/refs_parsed.json"))
HDR = {"User-Agent": "JDIM-revision-reference-audit (mailto:hoangcd@vnu.edu.vn)"}

# Manual NLM abbreviations for journals where CrossRef short title is missing
J_ABBREV = {
    "JAMA Dermatology": "JAMA Dermatol",
    "American Journal of Clinical Dermatology": "Am J Clin Dermatol",
    "Scientific Reports": "Sci Rep",
    "Journal of the American Academy of Dermatology": "J Am Acad Dermatol",
    "Frontiers in oncology": "Front Oncol",
    "American Journal of Clinical Oncology": "Am J Clin Oncol",
    "The Lancet. Oncology": "Lancet Oncol",
    "Dermatology Practical & Conceptual": "Dermatol Pract Concept",
    "The Journal of Clinical and Aesthetic Dermatology": "J Clin Aesthet Dermatol",
    "Annals of Oncology": "Ann Oncol",
    "IEEE Transactions on Biomedical Engineering": "IEEE Trans Biomed Eng",
    "Journal of Clinical Medicine": "J Clin Med",
    "Experimental Dermatology": "Exp Dermatol",
    "TrAC Trends in Analytical Chemistry": "TrAC Trends Anal Chem",
    "Advances in Neural Information Processing Systems (NeurIPS)": "Adv Neural Inf Process Syst",
    "Biology": "Biology (Basel)",
    "Authorea preprint": "Authorea",
    "Computer methods and programs in biomedicine": "Comput Methods Programs Biomed",
    "Frontiers in Oncology": "Front Oncol",
    "Chemometrics and Intelligent Laboratory Systems": "Chemom Intell Lab Syst",
    "Analytica chimica acta": "Anal Chim Acta",
    "iScience": "iScience",
    "Nature Protocols": "Nat Protoc",
    "IEEE Journal of Biomedical and Health Informatics": "IEEE J Biomed Health Inform",
    "Spectrochimica acta. Part A, Molecular and biomolecular spectroscopy": "Spectrochim Acta A Mol Biomol Spectrosc",
    "arXiv preprint": "arXiv",
    "PLOS ONE": "PLoS One",
    "Skin Research and Technology": "Skin Res Technol",
    "International Journal of Dermatology": "Int J Dermatol",
    "2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)": "Proc IEEE/RSJ Int Conf Intell Robots Syst (IROS)",
    "Journal of Biophotonics": "J Biophotonics",
    "Computers in biology and medicine": "Comput Biol Med",
    "JAMA": "JAMA",
    "BMC Medicine": "BMC Med",
    "Journal of clinical epidemiology": "J Clin Epidemiol",
    "Statistics in Biosciences": "Stat Biosci",
    "Korean Journal of Radiology": "Korean J Radiol",
    "Statistics in medicine": "Stat Med",
    "Statistics in Medicine": "Stat Med",
    "Journal of Chemometrics": "J Chemom",
    "Systems and Control Transactions": "Syst Control Trans",
    "ACS Omega": "ACS Omega",
    "IEEE Transactions on Medical Imaging": "IEEE Trans Med Imaging",
    "Appl. Soft Comput.": "Appl Soft Comput",
    "IEEE Access": "IEEE Access",
    "Micromachines": "Micromachines (Basel)",
    "Research Square preprint": "Research Square",
    "The European Physical Journal Special Topics": "Eur Phys J Spec Top",
    "International Journal of Electrical and Computer Engineering (IJECE)": "Int J Electr Comput Eng",
    "npj Digital Medicine": "NPJ Digit Med",
    "Artificial intelligence in medicine": "Artif Intell Med",
    "2025 International Conference on Advanced Technologies for Communications (ATC)": "Proc Int Conf Adv Technol Commun (ATC)",
    "BMC Medical Research Methodology": "BMC Med Res Methodol",
    "Journal of Biomedical Informatics": "J Biomed Inform",
    "European Radiology": "Eur Radiol",
    "The BMJ": "BMJ",
    "BMJ Open": "BMJ Open",
    "BMJ": "BMJ",
    "Analytica Chimica Acta": "Anal Chim Acta",
    "Practical Quantitative Vibrational and Electronic Spectroscopy": "Pract Quant Vib Electron Spectrosc",
    "Chemometric Methods in Analytical Spectroscopy Technology": "Chemom Methods Anal Spectrosc Technol",
    "Foods": "Foods",
    "Computer Methods and Programs in Biomedicine": "Comput Methods Programs Biomed",
    "Sensors": "Sensors (Basel)",
    "Advanced Biomedical and Clinical Diagnostic and Surgical Guidance Systems XXIII": "Proc SPIE",
    "Ph.D. dissertation, Radboud University Nijmegen": "PhD dissertation, Radboud University Nijmegen",
}

PREPRINT_HOSTS = {"arXiv", "Authorea", "Research Square", "medRxiv", "SSRN"}


def crossref_by_doi(doi):
    r = requests.get(f"https://api.crossref.org/works/{doi}", headers=HDR, timeout=30)
    if r.status_code != 200:
        return None
    return r.json()["message"]


def crossref_by_title(title, year):
    r = requests.get("https://api.crossref.org/works",
                     params={"query.bibliographic": title, "rows": 3},
                     headers=HDR, timeout=30)
    if r.status_code != 200:
        return None
    for it in r.json()["message"]["items"]:
        t = (it.get("title") or [""])[0].lower()
        if t and t.rstrip(".") == title.lower().rstrip("."):
            return it
    items = r.json()["message"]["items"]
    return items[0] if items else None


def fmt_authors(cr_authors):
    if not cr_authors:
        return ""
    names = []
    for a in cr_authors:
        fam = a.get("family", "").strip()
        giv = a.get("given", "").strip()
        initials = "".join(p[0] for p in re.split(r"[\s\-]+", giv) if p)
        names.append(f"{fam} {initials}".strip())
    if len(names) > 6:
        return ", ".join(names[:3]) + ", et al."
    return ", ".join(names)


def sentence_case(t):
    return t.rstrip(".")


def fmt_entry(n, cr, fallback):
    """Build the JDIM-style string from a CrossRef record (or fallback parse)."""
    if cr is None:
        return None
    authors = fmt_authors(cr.get("author"))
    title = sentence_case((cr.get("title") or [fallback["title"]])[0])
    container = (cr.get("container-title") or [""])[0]
    short = (cr.get("short-container-title") or [""])[0]
    journal = J_ABBREV.get(container) or J_ABBREV.get(fallback["journal"]) or short or container
    year = None
    for k in ("published-print", "published-online", "issued", "created"):
        if cr.get(k, {}).get("date-parts"):
            year = cr[k]["date-parts"][0][0]
            break
    vol = cr.get("volume", "")
    page = cr.get("page", "")
    art = cr.get("article-number", "")
    doi = cr.get("DOI", fallback.get("doi") or "")
    typ = cr.get("type", "")
    if typ in ("posted-content",) or journal in PREPRINT_HOSTS or "arXiv" in container:
        host = journal if journal else container
        return f"{n}. {authors}. {title}. {host}. Published online {year}. https://doi.org/{doi}"
    if typ in ("book-chapter", "book", "monograph", "dissertation") or not container:
        pub = cr.get("publisher", "")
        return f"{n}. {authors}. {title}. {container or pub}, {year}. https://doi.org/{doi}"
    vp = ""
    if vol and page:
        vp = f" {vol}:{page}"
    elif vol:
        vp = f" {vol}"
    elif art:
        vp = f" {art}"
    return f"{n}. {authors}. {title}. {journal}{vp}, {year}. https://doi.org/{doi}"


def title_match(a, b):
    na = re.sub(r"[^a-z0-9]+", " ", a.lower()).strip()
    nb = re.sub(r"[^a-z0-9]+", " ", b.lower()).strip()
    return na == nb or na in nb or nb in na


if __name__ == "__main__":
    out, flags = [], []
    for ref in REFS:
        cr = None
        if ref["doi"]:
            cr = crossref_by_doi(ref["doi"])
            time.sleep(0.15)
            if cr is None:
                flags.append(f"#{ref['n']}: DOI {ref['doi']} did NOT resolve on CrossRef")
        if cr is None:
            cr = crossref_by_title(ref["title"], ref["year"])
            time.sleep(0.15)
            if cr:
                flags.append(f"#{ref['n']}: no DOI in ms; matched by title -> {cr.get('DOI')}")
        if cr and not title_match((cr.get("title") or [""])[0], ref["title"]):
            flags.append(f"#{ref['n']}: TITLE MISMATCH ms='{ref['title'][:60]}' cr='{(cr.get('title') or [''])[0][:60]}'")
        entry = fmt_entry(ref["n"], cr, ref)
        if entry is None:
            flags.append(f"#{ref['n']}: NO CrossRef record at all")
            entry = f"{ref['n']}. {ref['authors']}. {ref['title']}. {ref['journal']}, {ref['year']}."
        out.append({"n": ref["n"], "entry": entry,
                    "doi": (cr or {}).get("DOI", ref.get("doi")),
                    "type": (cr or {}).get("type", "unknown")})
        print(entry, flush=True)
    json.dump(out, open("/workspace/benchmark/refs_jdim.json", "w"), indent=1)
    print("\n=== FLAGS ===")
    for f in flags:
        print(f)
    json.dump(flags, open("/workspace/benchmark/refs_flags.json", "w"), indent=1)
