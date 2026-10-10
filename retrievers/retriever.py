from typing import Dict, List
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from base import BaseRetriever

_MAX_CONTENT_CHARS = 50000


class SearchAPIRetriever(BaseRetriever):
    """Search API retriever."""
    pages: List[Dict] = []

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        docs = [
            Document(
                # ``raw_content`` may be explicitly None (the scraper sets it to
                # None for pages that failed to scrape), and slicing None raises
                # TypeError. Coerce to a string before truncating.
                page_content=(page.get("raw_content") or "")[:_MAX_CONTENT_CHARS],
                metadata={
                    "title": page.get("title", ""),
                    "source": page.get("url", ""),
                },
            )
            for page in self.pages
        ]

        return docs