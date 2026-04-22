import requests
import feedparser
import re
import time
import csv
import json
from tqdm import tqdm

# -----------------------------
# Konfiguration
# -----------------------------

ARXIV_QUERY = (
    "CBCT OR cone beam computed tomography "
    "OR dental CT OR maxillofacial imaging"
)

MAX_RESULTS = 50

DATASET_DOMAINS = [
    "zenodo.org",
    "figshare.com",
    "osf.io",
    "github.com",
    "openneuro.org"
]

OUTPUT_JSON = "cbct_dataset_index.json"
OUTPUT_CSV = "cbct_dataset_index.csv"


# -----------------------------
# arXiv API Abfrage
# -----------------------------

def query_arxiv(query, max_results):
    url = (
        "http://export.arxiv.org/api/query?"
        f"search_query=all:{query}"
        f"&start=0&max_results={max_results}"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    return feedparser.parse(response.text)


# -----------------------------
# Datensatz-Links extrahieren
# -----------------------------

def extract_dataset_links(text):
    urls = re.findall(r"https?://[^\s<>\"']+", text)

    filtered = []
    for u in urls:
        if any(domain in u for domain in DATASET_DOMAINS):
            filtered.append(u)

    return list(set(filtered))


# -----------------------------
# Pipeline
# -----------------------------

def run_pipeline():
    print("🚀 Starte CBCT Dataset Pipeline (arXiv → Repositories)\n")

    feed = query_arxiv(ARXIV_QUERY, MAX_RESULTS)
    results = []

    print(f"📄 {len(feed.entries)} arXiv-Papers gefunden\n")

    for entry in tqdm(feed.entries):
        title = entry.title
        summary = entry.summary.replace("\n", " ")
        paper_url = entry.link

        dataset_links = extract_dataset_links(summary)

        if dataset_links:
            for link in dataset_links:
                results.append({
                    "title": title,
                    "paper_url": paper_url,
                    "dataset_url": link
                })

        time.sleep(0.5)  # arXiv Rate-Limit

    return results


# -----------------------------
# Speichern
# -----------------------------

def save_results(results):
    # JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["title", "paper_url", "dataset_url"]
        )
        writer.writeheader()
        writer.writerows(results)


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    results = run_pipeline()

    print(f"\n📦 {len(results)} Datensatz-Links gefunden")

    if results:
        save_results(results)
        print(f"💾 Gespeichert in:")
        print(f"   - {OUTPUT_JSON}")
        print(f"   - {OUTPUT_CSV}")

        print("\n🔗 Beispiel-Links:")
        for r in results[:5]:
            print("  ", r["dataset_url"])
    else:
        print("⚠️ Keine Datensatz-Links gefunden.")
