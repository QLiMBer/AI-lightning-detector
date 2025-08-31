#!/usr/bin/env python3
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI_DOC = ROOT / "docs" / "cli.md"

BEGIN = "<!-- BEGIN: GENERATED SCAN HELP -->"
END = "<!-- END: GENERATED SCAN HELP -->"

def get_scan_help() -> str:
    env = os.environ.copy()
    # Prefer running module from the repo so it works without activation/install
    cmd = ["python", "-m", "lightning_detector.cli", "scan", "--help"]
    try:
        out = subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.STDOUT)
        return out.strip()
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"Failed to get scan help. Output:\n{e.output}")

def update_cli_doc(help_text: str) -> None:
    content = CLI_DOC.read_text(encoding="utf-8")
    block = f"{BEGIN}\n\n```\n{help_text}\n```\n\n{END}"
    if BEGIN in content and END in content:
        pattern = re.compile(re.escape(BEGIN) + r"[\s\S]*?" + re.escape(END), re.MULTILINE)
        new_content = pattern.sub(block, content)
    else:
        # Append a new section at the end
        new_content = content.rstrip() + "\n\n## Generated Help (sync via scripts/sync_cli_help.py)\n\n" + block + "\n"
    CLI_DOC.write_text(new_content, encoding="utf-8")

def main():
    help_text = get_scan_help()
    update_cli_doc(help_text)
    print(f"Updated {CLI_DOC}")

if __name__ == "__main__":
    main()

