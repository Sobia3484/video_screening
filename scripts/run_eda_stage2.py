#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vidscreen.config import load_settings
from vidscreen.eda.runner import run_eda
from vidscreen.logging_setup import setup_logging

SECTIONS = ["quality", "features", "overview", "duration", "views", "engagement", "upload_trends", "language_country", "creators", "subtopics", "query_analysis", "text_analysis"]

def main():
    s = load_settings()
    setup_logging(s.log_dir)
    done = run_eda(s, ("main", "confident"), SECTIONS)
    for k, v in done.items(): print(f"{k}: {v}")

if __name__ == "__main__": main()
