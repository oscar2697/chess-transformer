"""Academic retrieval node: canonical arXiv IDs -> JSON + BibTeX + LaTeX patch."""
import urllib.request, xml.etree.ElementTree as ET, json, pathlib, re

# Canonical IDs as requested
CANONICAL_IDS = [
    "1706.03762",  # Attention Is All You Need
    "1712.01815",  # AlphaZero
    "2106.01345",  # Decision Transformer
    "2402.04494",  # Searchless Chess (ChessFormer/ Ruoss)
]

ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}

def _fetch_by_ids(ids):
    id_list = ",".join(ids)
    url = f"http://export.arxiv.org/api/query?id_list={id_list}"
    data = urllib.request.urlopen(url, timeout=20).read()
    root = ET.fromstring(data)
    out=[]
    for e in root.findall("atom:entry", ARXIV_NS):
        title = e.findtext("atom:title", default="", namespaces=ARXIV_NS).strip().replace("\n"," ")
        summary = e.findtext("atom:summary", default="", namespaces=ARXIV_NS).strip().replace("\n"," ")
        arxiv_id = e.findtext("atom:id", default="", namespaces=ARXIV_NS).split("/")[-1].split("v")[0]
        # full versioned id
        full_id = e.findtext("atom:id", default="", namespaces=ARXIV_NS).split("/")[-1]
        authors = [a.findtext("atom:name", namespaces=ARXIV_NS) for a in e.findall("atom:author", ARXIV_NS)]
        published = e.findtext("atom:published", default="", namespaces=ARXIV_NS)[:10]
        out.append({"id": arxiv_id, "full_id": full_id, "title": title, "authors": authors, "summary": summary, "published": published})
    return out

def _bibtex_key(entry):
    first = entry["authors"][0].split()[-1] if entry["authors"] else "Anon"
    year = entry["published"][:4] if entry["published"] else "2024"
    return f"{re.sub(r'[^A-Za-z]','',first)}{year}_{entry['id'].replace('.','_')}"

def _to_bibtex(entry, key):
    authors = " and ".join(entry["authors"])
    return f"@article{{{key},\n  title={{{entry['title']}}},\n  author={{{authors}}},\n  journal={{arXiv preprint arXiv:{entry['id']}}},\n  year={{{entry['published'][:4]}}}\n}}\n"

def run_retrieval(output_dir="paper"):
    results = _fetch_by_ids(CANONICAL_IDS)
    bib_entries=[]
    for r in results:
        k=_bibtex_key(r); r["bibkey"]=k
        bib_entries.append(_to_bibtex(r,k))
    base = pathlib.Path(__file__).resolve().parents[3] / output_dir
    base.mkdir(parents=True, exist_ok=True)
    (base / "retrieval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (base / "references.bib").write_text("\n".join(bib_entries), encoding="utf-8")
    # Patch main.tex Related Work
    tex_path = base / "main.tex"
    if tex_path.exists():
        tex = tex_path.read_text(encoding="utf-8")
        related = "\\section{Related Work}\n"
        for r in results:
            related += f"\\cite{{{r['bibkey']}}} {r['title']} -- {r['summary'][:220]}...\n\n"
        if "\\section{Related Work}" in tex:
            tex = re.sub(r"\\section\{Related Work\}.*?(?=\\section|\\bibliography)", lambda m: related, tex, flags=re.DOTALL)
            tex_path.write_text(tex, encoding="utf-8")
    print(f"Retrieved {len(results)} canonical papers -> {base/'retrieval.json'}")
    return results

if __name__ == "__main__":
    run_retrieval()
