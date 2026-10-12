import os
import sys
import time
import socket
import concurrent.futures
import requests

# IPv4 monkey patch
orig_getaddrinfo = socket.getaddrinfo
def ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
    return orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = ipv4_only

url = "https://huggingface.co/ggml-org/Kev-0.8B-GGUF/resolve/main/Kev-0.8B-BF16.gguf"
target_file = "data/models/Kev-0.8B-BF16.gguf"
temp_file = target_file + ".part"

head_resp = requests.head(url, allow_redirects=True, timeout=15)
total_bytes = int(head_resp.headers.get("content-length", 0))
print(f"Total size: {total_bytes / (1024*1024):.1f} MB")

os.makedirs("data/models", exist_ok=True)
with open(temp_file, "wb") as f:
    f.seek(total_bytes - 1)
    f.write(b"\0")

num_workers = 8
chunk_size = total_bytes // num_workers

def download_range(idx, start, end):
    headers = {"Range": f"bytes={start}-{end}"}
    for attempt in range(5):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=30) as r:
                if r.status_code in (200, 206):
                    with open(temp_file, "r+b") as f:
                        f.seek(start)
                        for chunk in r.iter_content(chunk_size=1024*1024):
                            if chunk:
                                f.write(chunk)
                    return idx, True
        except Exception as e:
            time.sleep(1)
    return idx, False

ranges = []
for i in range(num_workers):
    start = i * chunk_size
    end = total_bytes - 1 if i == num_workers - 1 else (i + 1) * chunk_size - 1
    ranges.append((i, start, end))

t0 = time.time()
print(f"[+] Starting multi-threaded download with {num_workers} parallel connections...")
with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = [executor.submit(download_range, *r) for r in ranges]
    for fut in concurrent.futures.as_completed(futures):
        idx, ok = fut.result()
        print(f"  [-] Chunk {idx+1}/{num_workers} complete ({ok})")

if os.path.exists(target_file):
    os.remove(target_file)
os.replace(temp_file, target_file)
print(f"[+] Successfully downloaded {target_file} in {time.time() - t0:.1f}s!")
