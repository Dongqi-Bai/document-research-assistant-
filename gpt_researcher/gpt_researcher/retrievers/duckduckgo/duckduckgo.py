from itertools import islice
from ..utils import check_pkg

# gpt_researcher.skills.researcher._search_relevant_source_urls() treats
# any search result whose raw_content/body exceeds 100 characters as
# already-fetched full text -- a heuristic meant for retrievers (e.g.
# PubMed Central) that genuinely return full article text inline. ddgs
# populates "body" with an ordinary search-result snippet, which routinely
# exceeds 100 characters, so without this cap every result is wrongly
# treated as already-fetched and the real page is never actually scraped.
_MAX_PREFETCHED_LEN = 100


class Duckduckgo:
    """
    Duckduckgo API Retriever
    """

    # ddgs returns a snippet in "body"; the real page text still has to be
    # fetched.
    requires_scraping = True
    def __init__(self, query, query_domains=None):
        check_pkg('ddgs')
        from ddgs import DDGS
        self.ddg = DDGS()
        self.query = query
        self.query_domains = query_domains or None

    def search(self, max_results=5):
        """
        Performs the search and normalizes ddgs payloads to the shared
        ``{href, body, title?}`` contract used by other retrievers.
        """
        # TODO: Add support for query domains
        try:
            search_response = self.ddg.text(self.query, region='wt-wt', max_results=max_results)
        except Exception as e:
            print(f"Error: {e}. Failed fetching sources. Resulting in empty response.")
            search_response = []

        if not search_response:
            return []

        normalized = []
        for result in search_response:
            if not isinstance(result, dict):
                continue
            href = (
                result.get("href")
                or result.get("link")
                or result.get("url")
                or ""
            )
            if not href:
                continue
            body = (
                result.get("body")
                or result.get("snippet")
                or result.get("description")
                or ""
            )
            item = {"href": href, "body": body[:_MAX_PREFETCHED_LEN]}
            title = result.get("title")
            if title:
                item["title"] = title
            normalized.append(item)

        # Preserve caller-requested max_results even if the client ignored it.
        return list(islice(normalized, max_results))
