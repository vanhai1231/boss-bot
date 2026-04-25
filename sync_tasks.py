#!/usr/bin/env python3
"""
Quét thư mục /Users/havanhai/shipd/ và tạo tasks.json chứa metadata
của tất cả task (description + drive link).

Chạy script này mỗi khi thêm task mới, rồi push lên GitHub.
"""

import json
import os
from pathlib import Path

SHIPD_DIR = Path("/Users/havanhai/shipd")
OUTPUT = Path(__file__).parent / "tasks.json"

# Các folder không phải task
SKIP_FOLDERS = {"bot", "_workflow", "bom logic", "Olympus", ".git"}

# Tên file description có thể dùng
DESC_NAMES = [
    "challenge_description.md",
    "challenge_description.txt",
    "description.md",
    "description.txt",
    "dataset_description.md",
]

# Tên file chứa link Drive
LINK_NAMES = ["link.txt"]

# File đề bài nhỏ có thể gửi kèm (< 500KB)
EXTRA_FILE_EXTENSIONS = {".md", ".txt", ".py"}
MAX_EXTRA_SIZE = 500 * 1024  # 500KB


def scan_tasks() -> dict:
    tasks = {}

    for folder in sorted(SHIPD_DIR.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name in SKIP_FOLDERS or folder.name.startswith("."):
            continue

        task_data = {
            "name": folder.name,
            "description": "",
            "drive_link": "",
            "extra_files": {},  # filename -> content (text nhỏ)
        }

        # Tìm description
        for dname in DESC_NAMES:
            desc_path = folder / dname
            if desc_path.exists():
                try:
                    task_data["description"] = desc_path.read_text("utf-8").strip()
                except Exception:
                    pass
                break

        # Tìm link.txt
        for lname in LINK_NAMES:
            link_path = folder / lname
            if link_path.exists():
                try:
                    task_data["drive_link"] = link_path.read_text("utf-8").strip()
                except Exception:
                    pass
                break

        # Thu thập các file text nhỏ có thể gửi kèm
        for item in sorted(folder.iterdir()):
            if item.is_dir():
                continue
            if item.suffix.lower() not in EXTRA_FILE_EXTENSIONS:
                continue
            if item.stat().st_size > MAX_EXTRA_SIZE:
                continue
            if item.name.startswith("."):
                continue
            # Bỏ qua file đã đọc ở trên
            if item.name.lower() in [d.lower() for d in DESC_NAMES + LINK_NAMES]:
                continue
            try:
                content = item.read_text("utf-8").strip()
                if content:
                    task_data["extra_files"][item.name] = content
            except Exception:
                pass

        # Chỉ thêm task có ít nhất description hoặc link
        if task_data["description"] or task_data["drive_link"]:
            tasks[folder.name.lower()] = task_data

    return tasks


def main():
    print(f"Scanning {SHIPD_DIR}...")
    tasks = scan_tasks()
    print(f"Found {len(tasks)} tasks")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"Saved to {OUTPUT}")

    # Hiện danh sách
    for key, t in tasks.items():
        link = "✅" if t["drive_link"] else "❌"
        desc = "✅" if t["description"] else "❌"
        extras = len(t["extra_files"])
        print(f"  {t['name']:60s} desc={desc}  link={link}  extras={extras}")


if __name__ == "__main__":
    main()
