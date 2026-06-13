#!/usr/bin/env python3
"""
Fetch latest papers from Journal of Dentistry (Elsevier) via PubMed.
ISSN 0300-5712 = Journal of Dentistry. No API key, no human verification.
"""
import urllib.request
import urllib.parse
import json
import xml.etree.ElementTree as ET
import sys

JOD_ISSN = "0300-5712"
BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def esearch(retmax=20):
    """Search Journal of Dentistry by ISSN, return PMID list."""
    params = {
        "db": "pubmed",
        "term": f"{JOD_ISSN}[issn]",
        "retmax": retmax,
        "retmode": "json",
        "sort": "date",  # newest first
    }
    url = f"{BASE}/esearch.fcgi?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=15) as r:
        data = json.load(r)
    return data.get("esearchresult", {}).get("idlist", [])


def efetch(pmids):
    """Fetch article details for given PMIDs."""
    if not pmids:
        return []
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    url = f"{BASE}/efetch.fcgi?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=30) as r:
        xml_str = r.read().decode()
    return parse_efetch(xml_str)


def _strip_ns(elem):
    """Strip XML namespace for simpler parsing."""
    if elem.tag.startswith("{"):
        elem.tag = elem.tag.split("}")[1]
    for child in elem:
        _strip_ns(child)


def parse_efetch(xml_str):
    """Parse efetch XML into list of paper dicts."""
    root = ET.fromstring(xml_str)
    _strip_ns(root)
    articles = root.findall(".//PubmedArticle")

    papers = []
    for art in articles:
        med = art.find("MedlineCitation") or art
        art_elem = med.find("Article") if med is not None else None
        if art_elem is None:
            continue

        def txt(path, default=""):
            elem = art_elem.find(path) if art_elem is not None else None
            if elem is None:
                elem = med.find(path) if med is not None else None
            return (elem.text or "").strip() if elem is not None else default

        def alltxt(path):
            elems = art_elem.findall(path) if art_elem is not None else []
            if not elems:
                elems = med.findall(path) if med is not None else []
            return [e.text or "" for e in elems if e.text]

        pmid_elem = med.find("PMID") if med is not None else art.find(".//PMID")
        pmid = pmid_elem.text if pmid_elem is not None else ""

        journal = art_elem.find("Journal")
        jtitle = ""
        if journal is not None:
            jt = journal.find("Title")
            jtitle = (jt.text or "").strip() if jt is not None else ""

        title = txt("ArticleTitle")
        abstract_parts = []
        ab = art_elem.find("Abstract")
        if ab is not None:
            for at in ab.findall("AbstractText"):
                abstract_parts.append(at.text or "")
        abstract = " ".join(abstract_parts).strip()

        authors = []
        alist = art_elem.find("AuthorList")
        if alist is not None:
            for a in alist.findall("Author"):
                ln = a.find("LastName")
                fn = a.find("ForeName")
                ln_t = (ln.text or "").strip() if ln is not None else ""
                fn_t = (fn.text or "").strip() if fn is not None else ""
                if ln_t or fn_t:
                    authors.append(f"{fn_t} {ln_t}".strip())

        kw = alltxt(".//Keyword")
        if not kw:
            mesh = med.find("MeshHeadingList")
            if mesh is not None:
                for m in mesh.findall(".//DescriptorName"):
                    if m.text:
                        kw.append(m.text)

        doi = ""
        for eid in art_elem.findall("ELocationID") or []:
            if eid.get("EIdType") == "doi":
                doi = (eid.text or "").strip()
                break
        if not doi:
            pubdata = art.find("PubmedData")
            if pubdata is not None:
                for aid in pubdata.findall(".//ArticleId"):
                    if aid.get("IdType") == "doi":
                        doi = (aid.text or "").strip()
                        break

        pubdate = ""
        ji = journal.find("JournalIssue") if journal is not None else None
        pd = ji.find("PubDate") if ji is not None else None
        if pd is not None:
            y = pd.find("Year")
            m = pd.find("Month")
            d = pd.find("Day")
            pubdate = f"{(y.text or '')}-{(m.text or '')}-{(d.text or '')}".strip("-")

        papers.append({
            "pmid": pmid,
            "title": title,
            "authors": authors[:5],
            "abstract": abstract[:1500] + ("..." if len(abstract) > 1500 else ""),
            "keywords": kw[:10],
            "doi": doi,
            "pub_date": pubdate,
            "journal": jtitle or "Journal of Dentistry",
        })
    return papers


def main():
    n = 15
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            pass

    print("Fetching latest papers from Journal of Dentistry (ISSN 0300-5712) via PubMed...", file=sys.stderr)
    pmids = esearch(retmax=n)
    if not pmids:
        print(json.dumps({"success": False, "error": "No papers found"}), file=sys.stderr)
        sys.exit(1)

    papers = efetch(pmids)
    out = {"success": True, "source": "pubmed", "issn": JOD_ISSN, "count": len(papers), "papers": papers}
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
