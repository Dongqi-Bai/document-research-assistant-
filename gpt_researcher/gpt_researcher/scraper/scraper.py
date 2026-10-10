"""Web scraper module for GPT Researcher.

This module provides the Scraper class that extracts content from URLs
using the BeautifulSoup scraping backend.
"""

import asyncio
import logging
from urllib.parse import urlparse

import requests

from gpt_researcher.utils.workers import WorkerPool
from gpt_researcher.utils.url_security import UnsafeURLError, validate_url

from . import BeautifulSoupScraper

# Known anti-bot/challenge-page markers, case-insensitive substring match.
# These pages return HTTP 200 with real (often large) HTML bodies, so
# neither an exception nor the short-content check below catches them --
# without this, a block/challenge page is ingested as if it were the
# article's real content.
_BLOCK_PAGE_MARKERS = (
    "anubis uses a proof-of-work scheme",  # HAL and other Anubis-fronted sites
    "making sure you're not a bot",
    "checking your browser before accessing",
    "enable javascript and cookies to continue",
    "researchgate - temporarily unavailable",  # ResearchGate's specific wording
    "please verify you are a human",
    "attention required! | cloudflare",
    "sorry, you have been blocked",
)

# Block pages are the entire response body -- a "please wait" message, not
# a real article -- so they're always short and always appear at the very
# start of the content.
_BLOCK_PAGE_CHECK_PREFIX_LEN = 5_000

# Word-list/vocab dumps (plain lists of unrelated words, no prose) scrape
# cleanly and can dominate a report's context. They have essentially zero
# sentence-ending punctuation relative to their size. Includes CJK fullwidth
# sentence terminators (。！？) alongside the Latin ones.
_MIN_LENGTH_FOR_WORDLIST_CHECK = 200_000
_MAX_SENTENCE_DENSITY = 1 / 5000  # at most 1 sentence-ending mark per 5000 chars
_SENTENCE_ENDING_CHARS = frozenset(".!?。！？")


def _looks_like_block_page(text: str) -> bool:
    prefix = text[:_BLOCK_PAGE_CHECK_PREFIX_LEN].lower()
    return any(marker in prefix for marker in _BLOCK_PAGE_MARKERS)


def _looks_like_word_list(text: str) -> bool:
    if len(text) < _MIN_LENGTH_FOR_WORDLIST_CHECK:
        return False
    sentence_endings = sum(1 for ch in text if ch in _SENTENCE_ENDING_CHARS)
    return (sentence_endings / len(text)) < _MAX_SENTENCE_DENSITY


class Scraper:
    """
    Scraper class to extract the content from the links
    """

    def __init__(self, urls, user_agent, scraper, worker_pool: WorkerPool):
        """
        Initialize the Scraper class.
        Args:
            urls: List of URLs to scrape (duplicates will be removed)
        """
        # Optimization: Remove duplicate URLs to avoid redundant scraping
        unique_urls = list(dict.fromkeys(urls))  # Preserves order while removing duplicates
        duplicates_removed = len(urls) - len(unique_urls)

        self.urls = unique_urls
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.scraper = scraper
        self.logger = logging.getLogger(__name__)
        self.worker_pool = worker_pool

        # Log deduplication results if duplicates were found
        if duplicates_removed > 0:
            self.logger.info(
                f"Removed {duplicates_removed} duplicate URL(s). "
                f"Scraping {len(unique_urls)} unique URLs instead of {len(urls)}."
            )

    async def run(self):
        """
        Extracts the content from the links
        """
        contents = await asyncio.gather(
            *(self.extract_data_from_url(url, self.session) for url in self.urls)
        )

        # extract_data_from_url is expected to return a dict, but a buggy backend
        # or cancelled worker can still yield None / non-dict. Indexing those with
        # content["raw_content"] crashes the whole gather result and drops good rows.
        res = [
            content
            for content in contents
            if isinstance(content, dict) and content.get("raw_content") is not None
        ]
        return res

    async def extract_data_from_url(self, link, session):
        """
        Extracts the data from the link with logging
        """
        async with self.worker_pool.throttle():
            try:
                # Reject SSRF / local-file targets (internal hosts, cloud metadata
                # endpoints, file:// paths, etc.) before any request is made.
                try:
                    validate_url(link)
                except UnsafeURLError as e:
                    self.logger.warning(f"Skipping unsafe URL {link}: {e}")
                    return {
                        "url": link,
                        "raw_content": None,
                        "image_urls": [],
                        "title": "",
                    }

                ScraperClass = self.get_scraper(link)
                scraper = ScraperClass(link, session)

                # Get scraper name
                scraper_name = scraper.__class__.__name__
                self.logger.info(f"\n=== Using {scraper_name} ===")

                # Get content
                if hasattr(scraper, "scrape_async"):
                    content, image_urls, title = await scraper.scrape_async()
                else:
                    (
                        content,
                        image_urls,
                        title,
                    ) = await asyncio.get_running_loop().run_in_executor(
                        self.worker_pool.executor, scraper.scrape
                    )

                if not content or len(content) < 100:
                    return self._reject(link, title, "Content too short or empty")

                # Log results
                self.logger.info(f"\nTitle: {title}")
                self.logger.info(f"Content length: {len(content)} characters")
                self.logger.info(f"Number of images: {len(image_urls)}")
                self.logger.info(f"URL: {link}")
                self.logger.info("=" * 50)

                if _looks_like_block_page(content):
                    return self._reject(link, title, "Anti-bot/challenge page detected")

                if _looks_like_word_list(content):
                    return self._reject(link, title, "Word-list-like content detected")

                return {
                    "url": link,
                    "raw_content": content,
                    "image_urls": image_urls,
                    "title": title,
                }

            except Exception as e:
                self.logger.error(f"Error processing {link}: {str(e)}")
                return {"url": link, "raw_content": None, "image_urls": [], "title": ""}

    def _reject(self, link, title, reason):
        """Treat a fetched-but-unusable page as a scrape failure, in the same
        shape a raised exception or too-short content already returns --
        downstream code (Scraper.run) drops any raw_content: None entry."""
        self.logger.warning(f"{reason} for {link}, treating as fetch failure")
        return {"url": link, "raw_content": None, "image_urls": [], "title": title}

    def get_scraper(self, link):
        """
        The function `get_scraper` determines the appropriate scraper class based on the provided link
        or a default scraper if none matches.
        """
        # 本学习版裁剪为仅 BeautifulSoup 后端
        SCRAPER_CLASSES = {
            "bs": BeautifulSoupScraper,
        }

        path = urlparse(link).path
        if path.lower().endswith(".pdf"):
            # 无 PDF 专用 scraper，PDF 链接也走 BeautifulSoup（可能抓不到，但避免崩溃）
            scraper_key = "bs"
        else:
            scraper_key = self.scraper

        scraper_class = SCRAPER_CLASSES.get(scraper_key)
        if scraper_class is None:
            raise Exception("Scraper not found.")
        return scraper_class
