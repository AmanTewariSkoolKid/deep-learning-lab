import os
import tarfile
import urllib.request
import json
from pathlib import Path

CONFIG_PATH = 'config_transfer.json'

def load_config():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def download_and_extract(url: str, target_dir: str):
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    archive_name = 'dataset_download.tgz'
    archive_path = Path(target_dir).parent / archive_name

    if any(Path(target_dir).glob('*/*')):
        print(f"[INFO] Dataset appears to exist at {target_dir}, skipping download.")
        return

    print(f"[INFO] Downloading dataset from {url} ...")
    urllib.request.urlretrieve(url, archive_path)
    print(f"[INFO] Downloaded to {archive_path}")

    print("[INFO] Extracting...")
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(Path(target_dir).parent)

    # The TensorFlow flowers dataset extracts to 'flower_photos'; ensure it matches expected path
    extracted_root = Path(target_dir).parent / 'flower_photos'
    if extracted_root.exists() and extracted_root.is_dir() and extracted_root.name != Path(target_dir).name.rstrip('/'):
        # Rename to match config path 'flowers'
        expected = Path(target_dir)
        if not expected.exists():
            extracted_root.rename(expected)
            print(f"[INFO] Renamed {extracted_root} to {expected}")
    print("[INFO] Extraction complete.")

    # Clean archive
    try:
        archive_path.unlink()
    except OSError:
        pass


def main():
    cfg = load_config()
    ds_cfg = cfg['dataset']
    url = ds_cfg.get('download_url')
    if not url:
        print('[INFO] No download_url specified; skipping download.')
        return
    download_and_extract(url, ds_cfg['path'])

if __name__ == '__main__':
    main()
