import os
import socket
import sys
from huggingface_hub import snapshot_download, HfApi


def check_internet(host="hf-mirror.com", port=443, timeout=5):
    """Check if the Hugging Face mirror is accessible"""
    try:
        socket.create_connection((host, port), timeout=timeout)
        return True
    except OSError:
        return False


def main():
    print("=" * 70)
    print("🚀 Starting automatic download script for U-Bench dataset")
    print("=" * 70)

    # 1️⃣ Current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"📂 Current script directory: {current_dir}")

    # 2️⃣ Target directory for download (U-Bench/data)
    target_dir = os.path.join(current_dir, "data")
    os.makedirs(target_dir, exist_ok=True)
    print(f"📁 Target download directory: {target_dir}")

    # 3️⃣ Check network connectivity
    print("🌐 Checking network connection to Hugging Face ...", end=" ")
    if not check_internet():
        print("❌ Connection failed! Please check your network or proxy settings.")
        sys.exit(1)
    print("✅ Network is available.")

    # 4️⃣ Verify Hugging Face dataset availability
    try:
        api = HfApi(
            endpoint="https://hf-mirror.com"
        )
        repo_id = "FengheTan9/U-Bench"
        ds_info = api.dataset_info(repo_id=repo_id)
        file_count = len(ds_info.siblings) if hasattr(ds_info, "siblings") else "Unknown"
        print(f"📊 Dataset: {ds_info.id}")
        print(f"📦 Approximate number of files: {file_count}")
        print(f"🕓 Last modified: {ds_info.lastModified}")
    except Exception as e:
        print("⚠️ Failed to retrieve dataset info from Hugging Face. Please verify the repository name.")
        print("Error details:", e)
        sys.exit(1)

    # 5️⃣ Start download
    print("⬇️ Downloading dataset (skipping if already cached)...")
    try:
        data_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",  # ✅ Download dataset
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            endpoint="https://hf-mirror.com",  # ✅ Use mirror for faster access
        )
    except Exception as e:
        print("❌ Download failed:", e)
        sys.exit(1)

    print(f"✅ Dataset successfully downloaded to: {data_dir}")

    # 6️⃣ Print directory structure (first level only)
    print("\n📦 Dataset directory preview:")
    for root, dirs, files in os.walk(data_dir):
        print(f"📁 {root}  —  contains {len(files)} files, {len(dirs)} subdirectories")
        for d in dirs[:5]:
            print(f"    ├── {d}/")
        for f in files[:5]:
            print(f"    ├── {f}")
        break

    print("\n✅ Download complete. Files available at:")
    print(f"   {data_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
