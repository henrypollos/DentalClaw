---
name: journal-of-dentistry-survey
description: Fetch latest papers from Journal of Dentistry (Elsevier) and produce research survey reports. Uses PubMed E-utilities (no API key, no human verification). Use when monitoring Journal of Dentistry, dental research literature, or doing small literature surveys.
keywords:
  - journal of dentistry
  - dental research
  - literature survey
  - pubmed
  - sciencedirect
  - dentistry
license: MIT
---

# Journal of Dentistry Survey

Fetch latest papers from **Journal of Dentistry** (Elsevier, ISSN 0300-5712) and produce structured research surveys. Uses **PubMed** as data source — same articles as ScienceDirect, without human verification or API keys.

## Why PubMed Instead of ScienceDirect

- ScienceDirect may require human verification (CAPTCHA), blocking automation
- Journal of Dentistry articles are indexed in PubMed
- PubMed E-utilities: free, no API key, no rate limit concerns for reasonable use

## Requirements

- Python 3.6+ (stdlib only)
- Network access to `eutils.ncbi.nlm.nih.gov`

## Script Path (Research Agent)

From the research agent workspace (`agents/research`):

```bash
skills/literature/journal-of-dentistry-survey/scripts/fetch_latest.py
```

## Usage

### Fetch Latest Papers

```bash
python skills/literature/journal-of-dentistry-survey/scripts/fetch_latest.py [N]
```

- `N`: number of papers (default 15)
- Output: JSON to stdout

### Example

```bash
# From agents/research directory
python skills/literature/journal-of-dentistry-survey/scripts/fetch_latest.py 10
```

## Output Format

```json
{
  "success": true,
  "source": "pubmed",
  "issn": "0300-5712",
  "count": 10,
  "papers": [
    {
      "pmid": "41839249",
      "title": "Advances in Artificial Intelligence Enhanced Robotics...",
      "authors": ["Eslam Abdelwahab Dawood", "Rand Abumaylih", ...],
      "abstract": "To map current evidence on AI-enhanced robotic systems...",
      "keywords": ["Artificial intelligence", "Dental Robot", ...],
      "doi": "10.1016/j.jdent.2026.106628",
      "pub_date": "2026-Mar-14",
      "journal": "Journal of dentistry"
    }
  ]
}
```

## Survey Workflow

1. **Run fetch**: `python skills/literature/journal-of-dentistry-survey/scripts/fetch_latest.py 15`
2. **Parse output**: pipe to `jq` or read JSON
3. **Summarize**:
   - Group by topic (keywords/themes)
   - List highlights (novel methods, clinical relevance)
   - Note trends (e.g., AI, digital dentistry, implants)
4. **Write report** to `reports/journal-of-dentistry-YYYY-MM-DD.md`

## Report Template

Use this structure for survey output:

```markdown
# Journal of Dentistry — 最新论文调研

**日期**: YYYY-MM-DD  
**数据源**: PubMed (ISSN 0300-5712)  
**论文数**: N

## 摘要

1–2 段概括本期重点方向与趋势。

## 论文列表

| 标题 | 作者 | 关键词 | DOI |
|------|------|--------|-----|
| ... | ... | ... | ... |

## 主题分布

- **主题 A**: N 篇 — 简短描述
- **主题 B**: M 篇 — 简短描述

## 重点论文

1. **论文标题** — 主要贡献、临床意义
2. ...

## 趋势与展望

...
```

## Error Handling

- Network unreachable: check connectivity to NCBI
- Empty results: verify ISSN 0300-5712 in PubMed
- Parse errors: PubMed XML structure may change; check `parse_efetch`
