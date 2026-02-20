from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime, timezone
import hashlib
import json
import threading

import numpy as np

import requests
from bs4 import BeautifulSoup

# Selenium is optional; we import it, but we only start a driver if/when needed.
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


@dataclass
class JobPosting:
    """Structured representation of a single job posting scraped from a site."""

    title: str
    company: str
    location: str
    job_type: str  # e.g. "Full-time", "Internship", "Contract"
    description: str  # Full job description text
    requirements: List[str] = field(default_factory=list)
    url: str = ""  # Direct link to the job posting
    source: str = ""  # "indeed", "linkedin", "glassdoor", etc.

    # Optional / computed fields
    embedding: Optional[np.ndarray] = None  # Later: job description embedding
    hash_id: Optional[str] = None  # For deduplication
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))  # When scraped


class JobScraper:
    def __init__(self, enable_selenium: bool = True) -> None:
        """Initialize HTTP session + dedupe structures.

        Selenium is expensive and sometimes fails on machines without a working
        Chrome/Chromedriver setup. So we lazy-init the driver only when needed.
        """

        # Reusable HTTP session for sites that do not require JS.
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0 Safari/537.36"
                )
            }
        )

        # Deduplication set (stores hash_ids of already-seen jobs).
        self.seen_job_hashes: set[str] = set()

        # Selenium-related state.
        self._enable_selenium = enable_selenium
        self._driver: Optional[webdriver.Chrome] = None
        self._driver_lock = threading.Lock()

        print("✓ JobScraper initialized (session + dedupe set)")

    # -------------------------
    # Internal helpers
    # -------------------------

    def _get_driver(self) -> webdriver.Chrome:
        """Lazy-create and return a Selenium Chrome driver."""
        if not self._enable_selenium:
            raise RuntimeError(
                "Selenium is disabled (enable_selenium=False). "
                "You called a Selenium-based scraper method."
            )

        if self._driver is None:
            chrome_options = Options()
            chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            self._driver = webdriver.Chrome(options=chrome_options)
            print("✓ Selenium driver initialized")

        return self._driver

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join((text or "").split()).strip().lower()

    def _compute_hash_id(self, source: str, url: str, title: str, company: str, location: str) -> str:
        """Compute a stable hash for deduplication.

        We prefer URL, but fall back to other fields if URL is missing.
        """
        base = "|".join(
            [
                self._normalize_text(source),
                self._normalize_text(url),
                self._normalize_text(title),
                self._normalize_text(company),
                self._normalize_text(location),
            ]
        )
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

    def _dedupe(self, job: JobPosting) -> bool:
        """Return True if job is new, False if it was already seen."""
        if not job.hash_id:
            job.hash_id = self._compute_hash_id(job.source, job.url, job.title, job.company, job.location)

        if job.hash_id in self.seen_job_hashes:
            return False

        self.seen_job_hashes.add(job.hash_id)
        return True

    def close(self) -> None:
        """Close Selenium driver if it was created."""
        if self._driver is not None:
            try:
                self._driver.quit()
            finally:
                self._driver = None

    # -------------------------
    # Scrapers
    # -------------------------

    def scrape_indeed(self, query: str, location: str = "Remote", pages: int = 5) -> List[JobPosting]:
        """Scrape Indeed search results with requests + BeautifulSoup.

        Note: Indeed markup changes frequently. This implementation is intentionally
        conservative: it tries to extract a basic list of job cards, and if it can’t,
        it will safely return an empty list (but without crashing your pipeline).
        """

        jobs: List[JobPosting] = []

        for page in range(pages):
            # Indeed typically uses a start offset for pagination.
            start = page * 10

            # Basic search URL pattern.
            url = (
                "https://www.indeed.com/jobs?"
                f"q={requests.utils.quote(query)}&"
                f"l={requests.utils.quote(location)}&"
                f"start={start}"
            )

            try:
                resp = self.session.get(url, timeout=20)
                resp.raise_for_status()
            except Exception:
                # Don’t crash the whole run if a request fails.
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            # Indeed job cards have varied. We try a few common selectors.
            cards = soup.select("a.tapItem")
            if not cards:
                cards = soup.select("div.job_seen_beacon")

            for card in cards:
                # Attempt to extract a link.
                link_tag = card if card.name == "a" else card.select_one("a")
                href = link_tag.get("href") if link_tag else None
                job_url = ""
                if href:
                    job_url = href if href.startswith("http") else f"https://www.indeed.com{href}"

                # Attempt to extract title/company/location.
                title = ""
                title_tag = card.select_one("h2.jobTitle") or card.select_one("span[title]")
                if title_tag:
                    title = title_tag.get_text(strip=True)

                company = ""
                company_tag = card.select_one("span.companyName")
                if company_tag:
                    company = company_tag.get_text(strip=True)

                loc = location
                loc_tag = card.select_one("div.companyLocation")
                if loc_tag:
                    loc = loc_tag.get_text(strip=True)

                if not title and not job_url:
                    continue

                job = JobPosting(
                    title=title or "(unknown title)",
                    company=company or "(unknown company)",
                    location=loc or "(unknown location)",
                    job_type="",  # You can fill this later
                    description="",  # You can fetch detail pages later
                    requirements=[],
                    url=job_url,
                    source="indeed",
                )

                job.hash_id = self._compute_hash_id(job.source, job.url, job.title, job.company, job.location)
                if self._dedupe(job):
                    jobs.append(job)

        return jobs

    def scrape_linkedin(self, query: str, location: str = "Remote", pages: int = 3) -> List[JobPosting]:
        """Scrape LinkedIn jobs (placeholder).

        LinkedIn blocks aggressive scraping. Keep this as a stub until you implement
        a compliant approach (or use an approved integration).
        """

        # Placeholder: no-op, but valid method.
        _ = (query, location, pages)
        return []

    def scrape_glassdoor(self, query: str, location: str = "Remote", pages: int = 3) -> List[JobPosting]:
        """Scrape Glassdoor jobs (placeholder)."""

        # Placeholder: no-op, but valid method.
        _ = (query, location, pages)
        return []

    def scrape_all_sources_parallel(self, queries: List[str], location: str = "Remote") -> List[JobPosting]:
        """Scrape multiple sources and multiple queries.

        Important note:
        - requests-based scrapers are safe to run in threads.
        - selenium-based scrapers are NOT thread-safe with a shared driver.

        This implementation parallelizes Indeed (requests) and runs Selenium scrapers
        sequentially (currently they are placeholders anyway).
        """

        all_jobs: List[JobPosting] = []

        # 1) Parallelize requests-based scraping (Indeed)
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _run(fn: Callable[..., List[JobPosting]], q: str) -> List[JobPosting]:
            return fn(q, location=location)

        with ThreadPoolExecutor(max_workers=min(8, max(1, len(queries)))) as ex:
            futures = [ex.submit(_run, self.scrape_indeed, q) for q in queries]
            for fut in as_completed(futures):
                try:
                    all_jobs.extend(fut.result())
                except Exception:
                    # Keep going even if one task fails.
                    continue

        # 2) Selenium scrapers (sequential, guarded) - placeholders for now
        # If you later implement them, either create one driver per thread or keep sequential.
        for q in queries:
            try:
                # Guard any future selenium usage with a lock.
                with self._driver_lock:
                    all_jobs.extend(self.scrape_linkedin(q, location=location))
                    all_jobs.extend(self.scrape_glassdoor(q, location=location))
            except Exception:
                continue

        return all_jobs

    def save_jobs_to_json(self, jobs: List[JobPosting], output_file: str) -> None:
        """Serialize a list of JobPosting objects to JSON.

        Handles numpy arrays and datetimes safely.
        """

        def to_jsonable(job: JobPosting) -> Dict[str, Any]:
            d: Dict[str, Any] = {
                "title": job.title,
                "company": job.company,
                "location": job.location,
                "job_type": job.job_type,
                "description": job.description,
                "requirements": list(job.requirements or []),
                "url": job.url,
                "source": job.source,
                "hash_id": job.hash_id,
                "created_at": job.created_at.isoformat() if isinstance(job.created_at, datetime) else None,
            }

            if job.embedding is not None:
                # Convert numpy array to list for JSON.
                d["embedding"] = job.embedding.tolist()
            else:
                d["embedding"] = None

            return d

        payload = [to_jsonable(j) for j in jobs]
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


# Optional convenience for local testing
if __name__ == "__main__":
    scraper = JobScraper(enable_selenium=False)
    try:
        q = ["data engineer", "qa automation"]
        jobs = scraper.scrape_all_sources_parallel(q, location="Remote")
        print(f"Scraped {len(jobs)} jobs")
        scraper.save_jobs_to_json(jobs, "jobs.json")
        print("Saved to jobs.json")
    finally:
        scraper.close()
