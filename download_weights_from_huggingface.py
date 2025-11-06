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
    print("🚀 Starting automatic download script for U-Bench model weights")
    print("=" * 70)

    # 1️⃣ Current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"📂 Current script directory: {current_dir}")

    # 2️⃣ Target download directory (U-Bench/weights)
    target_dir = os.path.join(current_dir, "weights")
    os.makedirs(target_dir, exist_ok=True)
    print(f"📁 Target download directory: {target_dir}")

    # 3️⃣ Check network connectivity
    print("🌐 Checking network connection to Hugging Face ...", end=" ")
    if not check_internet():
        print("❌ Connection failed! Please check your network or proxy settings.")
        sys.exit(1)
    print("✅ Network is available.")

    # 4️⃣ Retrieve Hugging Face repository information
    try:
        api = HfApi(
            endpoint="https://hf-mirror.com"
        )
        repo_id = "FengheTan9/U-Bench"
        repo_info = api.repo_info(repo_id=repo_id, repo_type="model")
        print(f"📊 Model weights repository: {repo_info.id}")
        print(f"📦 Number of files: {len(repo_info.siblings)}")
    except Exception as e:
        print("⚠️ Failed to retrieve model weights repository information from Hugging Face.")
        print("Please verify the repository name or type.")
        print("Error details:", e)
        sys.exit(1)

    # 5️⃣ Start download
    print("⬇️ Downloading model weights (skipping if already cached)...")
    try:
        model_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="model",  # ✅ Download model weights
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            endpoint="https://hf-mirror.com"  # ✅ Use mirror for faster access
        )
    except Exception as e:
        print("❌ Download failed:", e)
        sys.exit(1)

    print(f"✅ Model weights successfully downloaded to: {model_dir}")

    # 6️⃣ Display directory structure (first level only)
    print("\n📦 Model weights directory preview:")
    for root, dirs, files in os.walk(model_dir):
        print(f"📁 {root}  —  contains {len(files)} files, {len(dirs)} subdirectories")
        for d in dirs[:5]:
            print(f"    ├── {d}/")
        for f in files[:5]:
            print(f"    ├── {f}")
        break

    print("\n✅ Download complete. Files available at:")
    print(f"   {model_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
