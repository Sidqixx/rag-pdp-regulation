"""Upload aplikasi ke Hugging Face Space (Docker), termasuk setting & API key dari `.env`.

Sekali saja: `pip install huggingface_hub`, lalu `hf auth login` (token "Write" dari huggingface.co/settings/tokens).
Jalankan  : python -m deploy.deploy <username>/<nama-space>
"""
import os
import sys

from huggingface_hub import HfApi

from ragpdp.retrieve import ROOT, Index  # juga memuat .env

repo = sys.argv[1]
Index.load()  # pastikan cache embedding sudah ada, supaya Space tidak meng-embed 301 pasal saat start
api = HfApi()
api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True)
for key in ("CHAT_BASE_URL", "CHAT_MODEL", "CHAT_REASONING_EFFORT", "EMBED_BASE_URL", "EMBED_MODEL"):
    api.add_space_variable(repo, key, os.environ[key])
for key in ("CHAT_API_KEY", "EMBED_API_KEY"):  # secret: tidak terlihat publik
    api.add_space_secret(repo, key, os.environ[key])
api.upload_folder(
    repo_id=repo,
    repo_type="space",
    folder_path=ROOT,
    allow_patterns=["Dockerfile", "requirements.txt", "ragpdp/*.py", "web/*", "data/chunks.jsonl", "data/emb-*.npy"],
)
api.upload_file(path_or_fileobj=ROOT / "deploy/README.md", path_in_repo="README.md", repo_id=repo, repo_type="space")
print(f"Selesai. Build butuh beberapa menit: https://huggingface.co/spaces/{repo}")
