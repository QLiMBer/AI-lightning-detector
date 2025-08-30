Session Restart Checklist

Use this to quickly resume work in a fresh session and/or after creating the GitHub repo.

1) Create the GitHub repo (SSH)

- Create an empty repo on GitHub (no README/license at creation time to avoid merge). Example SSH URL:
  - `git@github.com:<your-user-or-org>/lightning-detector.git`

2) Initialize Git locally and push

- From the project root:
  - `git init`
  - `git add .`
  - `git commit -m "chore: initialize docs and uv project"`
  - `git branch -M main`
  - `git remote add origin git@github.com:<your-user-or-org>/lightning-detector.git`
  - `git push -u origin main`

Notes:
- Use SSH remotes (not HTTPS) per your preference.
- If you want a repo-specific identity, set it via:
  - `git config user.name "Your Name"`
  - `git config user.email "you@example.com"`
  Otherwise Git will use your global config.

3) Recreate the Python environment with uv

- Install uv (once per machine):
  - `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Ensure `~/.local/bin` is on your PATH.
- Sync base dependencies from the lockfile into a venv:
  - `uv lock`
  - `uv sync`
- Install PyTorch with CUDA using uv’s backend selector (CUDA 12.4 recommended for your 12.8 driver):
  - `uv pip install --prefix .venv --torch-backend cu124 torch torchvision torchaudio`
- Activate and verify GPU:
  - `source .venv/bin/activate`
  - `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`
  - Expected: `True 12.4`

4) Quick sanity checks

- Inspect videos:
  - `python scripts/inspect_videos.py`
  - Confirms fps/resolution/duration are read correctly.
- Optional: pull a quick MiniCPM‑V 4.5 image/video test from `docs/minicpm-v/usage-video.md` once we implement the CLI.

5) Folder expectations

- Input videos: `videos/`
- Outputs (once implemented): `reports/` (JSON + text summaries)

6) How to start the next session

- Tell me when the repo is created and pushed (SSH remote), and that uv sync + torch install are done. Example message:
  - “Repo is up, SSH remote configured, env synced, torch CUDA 12.4 installed. Proceed to implement the CLI.”
- I’ll then scaffold the CLI and implement the baseline lightning detection pipeline per `docs/plan.md`.

