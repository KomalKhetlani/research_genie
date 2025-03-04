import feedparser
import os
import requests
import time
import urllib.parse
from dotenv import load_dotenv

load_dotenv()
RESEARCH_PAPER_PDF_FOLDER_PATH = os.getenv('RESEARCH_PAPER_PDF_FOLDER_PATH')

def fetch_arxiv_papers(query, start=0, max_results=100):
    base_url = "http://export.arxiv.org/api/query?"
    encoded_query = urllib.parse.quote(query)
    query_url = (f"{base_url}search_query={encoded_query}&start={start}&max_results={max_results}"
                 f"&sortBy=submittedDate&sortOrder=descending")
    return feedparser.parse(query_url)


def extract_pdf_url(entry):
    for link in entry.links:
        if link.rel == "related" and "pdf" in link.href:
            return link.href
    return entry.link.replace('abs', 'pdf')  # Fallback method


def download_paper(pdf_url, title, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    pdf_filename = f"{save_dir}/{title.replace(' ', '_').replace('/', '_')}.pdf"

    print(f"Downloading: {title}")
    try:
        response = requests.get(pdf_url, timeout=10)
        response.raise_for_status()
        with open(pdf_filename, 'wb') as f:
            f.write(response.content)
        print(f"Saved: {pdf_filename}")
    except requests.RequestException as e:
        print(f"Failed to download {title}: {e}")


def main(topics, save_dir, total_papers=500, batch_size=100, delay=3):
    query = " OR ".join(topics)
    downloaded = 0

    while downloaded < total_papers:
        feed = fetch_arxiv_papers(query, start=downloaded, max_results=batch_size)
        if not feed.entries:
            print("No more papers found.")
            break

        for entry in feed.entries:
            title = entry.title.replace('\n', ' ').strip()
            pdf_url = extract_pdf_url(entry)
            download_paper(pdf_url, title, save_dir)
            downloaded += 1
            if downloaded >= total_papers:
                break

        print(f"Downloaded {downloaded}/{total_papers} papers. Waiting {delay} seconds to avoid rate limits...")
        time.sleep(delay)  # Prevent hitting the API too quickly

    print("Download complete!")


if __name__ == "__main__":
    topics = ["cat:cs.CL"]
    save_path = RESEARCH_PAPER_PDF_FOLDER_PATH
    main(topics, save_path, total_papers=500, batch_size=100, delay=3)