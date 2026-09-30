from __future__ import annotations

import argparse
from pathlib import Path

import requests
import torch
from tqdm import tqdm


BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0"


def is_valid_checkpoint(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1_000_000:
        return False
    try:
        torch.load(path, map_location="cpu")
        return True
    except Exception:
        return False


def download(url: str, output: Path) -> None:
    tmp = output.with_suffix(output.suffix + ".part")
    existing = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0)) + existing
        mode = "ab" if existing and response.status_code == 206 else "wb"
        if mode == "wb":
            existing = 0
        with tmp.open(mode) as f, tqdm(
            total=total,
            initial=existing,
            unit="B",
            unit_scale=True,
            desc=output.name,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    tmp.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("weight", help="Example: yolo26m.pt")
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()

    output = Path(args.weight)
    url = f"{BASE_URL}/{output.name}"
    if is_valid_checkpoint(output):
        print(f"OK: {output}")
        return
    if output.exists():
        output.unlink()

    last_error: Exception | None = None
    for attempt in range(1, args.retries + 1):
        try:
            print(f"Download attempt {attempt}/{args.retries}: {url}")
            download(url, output)
            if is_valid_checkpoint(output):
                print(f"OK: {output}")
                return
            raise RuntimeError(f"Downloaded file is not a valid PyTorch checkpoint: {output}")
        except Exception as exc:
            last_error = exc
            print(f"Attempt failed: {exc}")
    raise SystemExit(f"Failed to download valid checkpoint after {args.retries} tries: {last_error}")


if __name__ == "__main__":
    main()
