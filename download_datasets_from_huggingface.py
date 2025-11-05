import os
import socket
import sys
from huggingface_hub import snapshot_download, HfApi


def check_internet(host="hf-mirror.com", port=443, timeout=5):
    """检查是否能访问 Hugging Face"""
    try:
        socket.create_connection((host, port), timeout=timeout)
        return True
    except OSError:
        return False


def main():
    print("=" * 70)
    print("🚀 开始执行 U-Bench 数据集自动下载脚本")
    print("=" * 70)

    # 1️⃣ 当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"📂 当前脚本路径: {current_dir}")

    # 2️⃣ 目标下载路径（U-Bench/data）
    target_dir = os.path.join(current_dir, "data")
    os.makedirs(target_dir, exist_ok=True)
    print(f"📁 目标下载目录: {target_dir}")

    # 3️⃣ 检查网络连接
    print("🌐 检查网络连接至 Hugging Face ...", end=" ")
    if not check_internet():
        print("❌ 无法连接！请检查网络或代理设置。")
        sys.exit(1)
    print("✅ 网络正常。")

    # 4️⃣ 检查 Hugging Face 数据集可用性
    try:
        api = HfApi(
            endpoint="https://hf-mirror.com"
        )
        repo_id = "FengheTan9/U-Bench"
        ds_info = api.dataset_info(repo_id=repo_id)
        file_count = len(ds_info.siblings) if hasattr(ds_info, "siblings") else "未知"
        print(f"📊 数据集：{ds_info.id}")
        print(f"📦 文件数量约：{file_count}")
        print(f"🕓 最近更新时间：{ds_info.lastModified}")
    except Exception as e:
        print("⚠️ 无法从 Hugging Face 获取数据集信息，请确认仓库名称。")
        print("错误信息：", e)
        sys.exit(1)

    # 5️⃣ 开始下载
    print("⬇️ 正在下载数据集（如已缓存则跳过）...")
    try:
        data_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",  # ✅ 下载数据集
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            endpoint="https://hf-mirror.com",  # ✅ 使用国内镜像加速
        )
    except Exception as e:
        print("❌ 下载失败：", e)
        sys.exit(1)

    print(f"✅ 数据集已下载至：{data_dir}")

    # 6️⃣ 打印目录结构（仅第一层）
    print("\n📦 数据集目录结构预览：")
    for root, dirs, files in os.walk(data_dir):
        print(f"📁 {root}  —  包含 {len(files)} 个文件, {len(dirs)} 个子目录")
        for d in dirs[:5]:
            print(f"    ├── {d}/")
        for f in files[:5]:
            print(f"    ├── {f}")
        break

    print("\n✅ 下载完成，可在以下路径查看：")
    print(f"   {data_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
