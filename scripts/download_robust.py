import os
import sys
import time
import requests

def download_file_with_resume(url, filename):
    print(f"Starting download of {filename} from {url}")
    
    # Get total file size
    try:
        head_resp = requests.head(url, allow_redirects=True)
        total_size = int(head_resp.headers.get('content-length', 0))
    except Exception as e:
        print(f"Failed to get file size: {e}")
        total_size = 0
        
    downloaded_size = 0
    if os.path.exists(filename):
        downloaded_size = os.path.getsize(filename)
        print(f"Found existing file with {downloaded_size} bytes downloaded.")

    if total_size > 0 and downloaded_size >= total_size:
        print("File is already fully downloaded.")
        return

    # Download in chunks with retries
    max_retries = 20
    retries = 0
    
    while retries < max_retries:
        try:
            headers = {}
            if downloaded_size > 0:
                headers['Range'] = f'bytes={downloaded_size}-'
                
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            
            # If server doesn't support range and we asked for it, we might get a 200 instead of 206
            if response.status_code == 200 and downloaded_size > 0:
                print("Server ignored Range header, restarting download from 0")
                downloaded_size = 0
                mode = 'wb'
            elif response.status_code == 206:
                mode = 'ab'
            elif response.status_code == 416: # Range not satisfiable (usually means we have the whole file)
                print("Server returned 416, assuming file is fully downloaded.")
                return
            else:
                mode = 'wb'
                if downloaded_size == 0:
                    response.raise_for_status()
                
            with open(filename, mode) as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        if total_size > 0:
                            done = int(50 * downloaded_size / total_size)
                            sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded_size/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB")
                            sys.stdout.flush()
                            
            if total_size == 0 or downloaded_size >= total_size:
                print(f"\nDownload complete! Total size: {downloaded_size} bytes.")
                return
                
        except Exception as e:
            retries += 1
            print(f"\nConnection dropped. Retrying {retries}/{max_retries} in 2 seconds... Error: {e}")
            time.sleep(2)
            
            # update downloaded size for next resume attempt
            if os.path.exists(filename):
                downloaded_size = os.path.getsize(filename)
                
    print("\nMax retries reached. Download failed.")

if __name__ == "__main__":
    download_file_with_resume("https://files.pythonhosted.org/packages/ec/6c/fab8113424af5049f85717e8e527ca3773299a3c6b02506e66436e19874f/opencv_python-4.10.0.84-cp37-abi3-win_amd64.whl", "opencv_python-4.10.0.84-cp37-abi3-win_amd64.whl")
