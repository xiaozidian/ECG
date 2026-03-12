import argparse
import os
import time
import sys
from huggingface_hub import snapshot_download, hf_hub_download, constants
from requests.exceptions import RequestException

def download_repo(repo_id, local_dir=None, repo_type='model', token=None, max_retries=5, revision=None, allow_patterns=None, ignore_patterns=None):
    """
    Downloads a Hugging Face repository with retry logic and resume support.
    """
    print(f"Preparing to download: {repo_id} (Type: {repo_type})")
    if local_dir:
        print(f"Target directory: {local_dir}")
    else:
        # Show the default cache directory
        print(f"Target directory: Default Hugging Face Cache ({constants.HUGGINGFACE_HUB_CACHE})")

    for attempt in range(max_retries):
        try:
            print(f"\n[Attempt {attempt + 1}/{max_retries}] Starting download...")
            
            # Using snapshot_download to download the entire repository
            # It inherently supports resuming downloads (checks existing files)
            path = snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                repo_type=repo_type,
                token=token,
                revision=revision,
                resume_download=True,  # Explicitly enable resume
                local_dir_use_symlinks=False if local_dir else "auto", # If local_dir is set, we usually want real files, not symlinks
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                tqdm_class=None # Use default tqdm
            )
            
            print(f"\nSuccessfully downloaded to: {path}")
            return path

        except (RequestException, Exception) as e:
            print(f"\nError during download (Attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("\nMax retries reached. Download failed.")
                raise e

def main():
    parser = argparse.ArgumentParser(description="Generic Hugging Face Download Script with Resume and Retry support.")
    
    parser.add_argument("repo_id", type=str, help="The ID of the repository (e.g., 'meta-llama/Llama-2-7b-hf')")
    parser.add_argument("--dir", type=str, default=None, help="Local directory to save the files. If not provided, uses the default cache.")
    parser.add_argument("--type", type=str, default="model", choices=["model", "dataset", "space"], help="Type of repository (default: model)")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face User Access Token (required for private/gated repos)")
    parser.add_argument("--retries", type=int, default=10, help="Maximum number of retries (default: 10)")
    parser.add_argument("--revision", type=str, default=None, help="Branch, tag, or commit hash to download (default: main/master)")
    parser.add_argument("--allow", type=str, nargs="+", default=None, help="Patterns of files to download (e.g. '*.json' '*.bin')")
    parser.add_argument("--ignore", type=str, nargs="+", default=None, help="Patterns of files to ignore")

    args = parser.parse_args()

    # -------- 基础环境设置 (保持与 Bash 脚本一致) --------
    # 禁用遥测
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    # 强制关闭 hf_transfer（更稳）
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    # Check for token in env if not provided
    token = args.token or os.environ.get("HF_TOKEN")

    try:
        path = download_repo(
            repo_id=args.repo_id,
            local_dir=args.dir,
            repo_type=args.type,
            token=token,
            max_retries=args.retries,
            revision=args.revision,
            allow_patterns=args.allow,
            ignore_patterns=args.ignore
        )
        
        # -------- 兼容性提示 --------
        print("\n" + "="*40)
        print("✅ Download Complete!")
        print("To load this in Python:")
        print("-" * 20)
        if args.dir:
            print(f'from transformers import AutoModel, AutoTokenizer')
            print(f'# Because you downloaded to a specific directory:')
            print(f'model = AutoModel.from_pretrained("{os.path.abspath(path)}")')
            print(f'tokenizer = AutoTokenizer.from_pretrained("{os.path.abspath(path)}")')
        else:
            print(f'from transformers import AutoModel, AutoTokenizer')
            print(f'# Because you downloaded to the default cache:')
            print(f'model = AutoModel.from_pretrained("{args.repo_id}")')
            print(f'tokenizer = AutoTokenizer.from_pretrained("{args.repo_id}")')
        print("="*40 + "\n")

    except KeyboardInterrupt:
        print("\nDownload interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
