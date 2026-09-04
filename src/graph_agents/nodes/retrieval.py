"""Academic retrieval node: canonical arXiv IDs -> JSON + BibTeX + LaTeX patch."""
import urllib.request, urllib.error, xml.etree.ElementTree as ET, json, pathlib, re, time

# Canonical IDs as requested
CANONICAL_IDS = [
    "1706.03762",  # Attention Is All You Need
    "1712.01815",  # AlphaZero
    "2106.01345",  # Decision Transformer
    "2402.04494",  # Searchless Chess (ChessFormer/ Ruoss)
]

ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}

def _fetch_by_ids(ids, retries=3):
    id_list = ",".join(ids)
    url = f"http://export.arxiv.org/api/query?id_list={id_list}"
    headers = {"User-Agent": "Mozilla/5.0 (chess-transformer retrieval; contact: research@example.com)"}
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(retries):
        try:
            data = urllib.request.urlopen(req, timeout=20).read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                time.sleep(2 ** attempt * 5)
                continue
            raise
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
    base = pathlib.Path(__file__).resolve().parents[3] / output_dir
    base.mkdir(parents=True, exist_ok=True)
    cached = base / "retrieval.json"
    # Use cache if valid schema (requires authors+id) to avoid 429; rejects Colab fallback stubs
    def _valid_cache(data):
        if not isinstance(data, list) or len(data) < len(CANONICAL_IDS):
            return False
        return all(isinstance(r, dict) and r.get("authors") and r.get("id") for r in data)
    if cached.exists() and cached.stat().st_size > 10:
        try:
            cached_data = json.loads(cached.read_text(encoding="utf-8"))
            if _valid_cache(cached_data):
                print(f"Using cached {cached} ({len(cached_data)} papers) - skipping arXiv request")
                return cached_data
            print(f"Cache {cached} invalid schema, refetching")
        except json.JSONDecodeError as e:
            print(f"Cache decode error: {e}, refetching")
    try:
        results = _fetch_by_ids(CANONICAL_IDS)
    except urllib.error.HTTPError as e:
        if e.code == 429 and cached.exists() and cached.stat().st_size > 10:
            print(f"HTTP 429 - falling back to cached {cached}")
            return json.loads(cached.read_text(encoding="utf-8"))
        raise
    bib_entries=[]
    for r in results:
        k=_bibtex_key(r); r["bibkey"]=k
        bib_entries.append(_to_bibtex(r,k))
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
