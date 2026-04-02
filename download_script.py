import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
import time

# Configuration
BASE_URL = "https://itlararen.se/"
HUB_URL = urljoin(BASE_URL, "videos.html")
INPUT_DIR = "input"

def download_videos_flat():
    # 1. Create the input directory if it doesn't exist
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"Created directory: {INPUT_DIR}")

    print(f"Connecting to {HUB_URL}...")
    try:
        response = requests.get(HUB_URL)
        soup = BeautifulSoup(response.text, 'html.parser')
    except Exception as e:
        print(f"Could not connect to site: {e}")
        return

    # 2. Collect all unique video page links
    video_page_urls = set()
    for a in soup.find_all('a', href=True):
        href = a['href']
        if 'videos/' in href and href.endswith('.html'):
            video_page_urls.add(urljoin(BASE_URL, href))

    print(f"Found {len(video_page_urls)} potential video pages. Starting downloads...")

    # 3. Process each page
    for page_url in video_page_urls:
        try:
            res = requests.get(page_url)
            page_soup = BeautifulSoup(res.text, 'html.parser')
            
            video_tag = page_soup.find('video')
            source_tag = video_tag.find('source') if video_tag else None
            
            if source_tag and source_tag.get('src'):
                video_rel_src = source_tag['src']
                video_url = urljoin(page_url, video_rel_src)
                
                # Extract filename and determine destination path
                filename = video_url.split('/')[-1]
                file_dest = os.path.join(INPUT_DIR, filename)

                # Skip if it's already there
                if os.path.exists(file_dest):
                    print(f"  [Skipping] {filename} (Already exists)")
                    continue

                print(f"  [Downloading] {filename}...")
                
                with requests.get(video_url, stream=True) as r:
                    r.raise_for_status()
                    with open(file_dest, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=1024*1024): # 1MB chunks
                            if chunk:
                                f.write(chunk)
                
                # Small sleep to prevent being throttled
                time.sleep(0.3)

        except Exception as e:
            print(f"  [Error] Failed page {page_url}: {e}")

if __name__ == "__main__":
    download_videos_flat()
    print(f"\nFinished. All files are in the '{INPUT_DIR}' directory.")