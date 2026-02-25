#!/usr/bin/env python3
# run.py
"""
FT_Bot — Job Search Pipeline Entry Point

Usage:
    python run.py                          # search + open tracker
    python run.py --role "Data Scientist"  # override role
    python run.py --no-search              # open tracker only (no scrape)
    python run.py --headless               # run pipeline, no UI
    python run.py --help                   # show all options

Examples:
    python run.py --role "ML Engineer" --location "San Francisco"
    python run.py --role "Backend Engineer" --min-score 0.6
    python run.py --no-search              # just browse saved jobs
"""

import argparse
import logging
import sys
from pathlib import Path

# ------------------------------------------------------------------
# Logging — info level to stdout, suppress noisy third-party libs
# ------------------------------------------------------------------
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt = "%H:%M:%S",
)
logging.getLogger("selenium").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger("run")


# ------------------------------------------------------------------
# CLI ARGUMENTS
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog        = "run.py",
        description = "FT_Bot — AI-powered job search pipeline",
    )

    p.add_argument(
        "--role", "-r",
        type    = str,
        default = "ML Engineer",       # ← always has a value
        help    = "Job title to search for (default: ML Engineer)",
    )
    p.add_argument(
        "--location", "-l",
        type    = str,
        default = None,
        help    = "Location string (default: Remote)",
    )
    p.add_argument(
        "--resume", "-R",
        type    = str,
        default = "data/resumes/my_resume.pdf",
        help    = "Path to your resume PDF",
    )
    p.add_argument(
        "--min-score",
        type    = float,
        default = 0.5,
        help    = "Minimum match score to save a job (0.0–1.0, default: 0.5)",
    )
    p.add_argument(
        "--db",
        type    = str,
        default = "data/jobs.db",
        help    = "Path to SQLite database (default: data/jobs.db)",
    )
    p.add_argument(
        "--no-search",
        action  = "store_true",
        help    = "Skip scraping — open tracker with existing jobs only",
    )
    p.add_argument(
        "--headless",
        action  = "store_true",
        help    = "Run pipeline only — no interactive UI",
    )
    p.add_argument(
        "--results",
        type    = int,
        default = 20,
        help    = "Number of results to fetch per site (default: 20)",
    )
    p.add_argument(
        "--sites",
        nargs   = "+",
        default = ["linkedin", "indeed"],
        help    = "Job sites to scrape (default: linkedin indeed)",
    )

    return p.parse_args()


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    # -- Import here so CLI --help is instant (no heavy imports) ---
    from src.modules.pipeline_config  import PipelineConfig
    from src.modules.pipeline_runner  import PipelineRunner
    from src.modules.job_store        import JobStore
    from src.modules.tracker_ui       import TrackerApp

    # -- Build config ----------------------------------------------
    config_kwargs = dict(
        search_role     = args.role,           # always included
        resume_path     = args.resume,
        min_score       = args.min_score,
        db_path         = args.db,
        results_wanted  = args.results,
        job_sites       = args.sites,
    )
    if args.location:
        config_kwargs["location"] = args.location

    try:
        config = PipelineConfig(**config_kwargs)
    except (ValueError, FileNotFoundError) as e:
        print(f"\n✗ Config error: {e}\n")
        return 1

    print(f"\n{'='*60}")
    print(f"  FT_Bot Job Search Pipeline")
    print(f"{'='*60}")
    print(f"  {config.summary()}")
    print(f"  Database: {args.db}")
    print(f"{'='*60}\n")

    # -- Build runner (keeps store open) ---------------------------
    runner = PipelineRunner(config)

    # -- Stage 1: run pipeline (unless --no-search) ----------------
    if not args.no_search:
        print("🔍 Searching for jobs...\n")
        result = runner.run()

        if result.error:
            print(f"\n✗ Pipeline error: {result.error}\n")
            if not args.headless:
                print("  Opening tracker with existing jobs...\n")
        else:
            print(f"\n✓ {result.summary()}\n")

    # -- Stage 2: open tracker (unless --headless) -----------------
    if args.headless:
        stats = runner.store.get_stats()
        print(f"\nDatabase stats:")
        print(f"  Total jobs:    {stats['total']}")
        print(f"  Not applied:   {stats['not_applied']}")
        print(f"  Applied:       {stats['applied']}")
        print(f"  Interviewing:  {stats['interviewing']}")
        print(f"  Offers:        {stats['offer_received']}")
        print(f"  Best score:    {stats['best_score']:.3f}")
        runner.close()
        return 0

    # -- Launch TrackerApp -----------------------------------------
    def run_search_from_ui():
        """Called when user presses [s] inside the tracker."""
        return runner.run()

    app = TrackerApp(
        store        = runner.store,
        config       = config,
        run_pipeline = run_search_from_ui,
    )

    print("🖥️  Opening tracker UI... (press [q] to quit)\n")

    try:
        app.run()
    finally:
        runner.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
