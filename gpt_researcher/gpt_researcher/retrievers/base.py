"""Shared contract for retrievers.

Retrievers fall into two kinds, and the difference matters:

* most return **URLs that still need scraping** — the search API gives back a
  link plus a short snippet, and the real page text is fetched later;
* a few return **content they already fetched** — PubMed Central hands back
  full article text, and a custom retriever's documented contract is
  ``list[{url, raw_content}]``.

``requires_scraping`` replaces the old length-based guess with a declaration.
It defaults to ``True``, which is the correct answer for the large majority of
retrievers.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseRetriever(ABC):
    """Optional base class documenting the retriever contract.

    Subclassing is not required; ``_search_relevant_source_urls`` only looks
    for a ``requires_scraping`` attribute. Inheriting simply makes the
    declaration explicit and gives new retrievers a place to read the contract.
    """

    #: Whether results are URLs that still need fetching (``True``, the
    #: default) or content the retriever has already retrieved (``False``).
    requires_scraping: bool = True

    @abstractmethod
    def search(self, max_results: int = 7) -> List[Dict[str, Any]]:
        """Return search results.

        With ``requires_scraping = True`` each item should carry a URL under
        ``url`` or ``href``; any ``body``/``snippet`` is treated as a preview
        and the page is scraped for its real content.

        With ``requires_scraping = False`` each item should carry a URL and the
        already-retrieved text under ``raw_content``; no scraping is performed.
        """
        raise NotImplementedError
