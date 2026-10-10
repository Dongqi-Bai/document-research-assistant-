from typing import Any
from colorama import Fore, Style

from gpt_researcher.utils.workers import WorkerPool
from ..scraper import Scraper
from ..config import Config
from ..utils.logger import get_formatted_logger

logger = get_formatted_logger()


async def scrape_urls(
    urls, cfg: Config, worker_pool: WorkerPool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Scrapes the urls
    Args:
        urls: List of urls
        cfg: Config (optional)

    Returns:
        tuple[list[dict[str, Any]], list[dict[str, Any]]]: tuple containing scraped content and images

    """
    scraped_data = []
    images = []
    user_agent = (
        cfg.user_agent
        if cfg
        else "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    )

    scraper = None
    try:
        scraper = Scraper(urls, user_agent, cfg.scraper, worker_pool=worker_pool)
        scraped_data = await scraper.run()
        for item in scraped_data:
            if not isinstance(item, dict):
                continue
            image_urls = item.get("image_urls")
            if image_urls:
                images.extend(image_urls)
    except Exception as e:
        print(f"{Fore.RED}Error in scrape_urls: {e}{Style.RESET_ALL}")
    finally:
        # Close the requests.Session so its underlying connection pool (and the
        # sockets it keeps alive) is released
        if scraper is not None and getattr(scraper, "session", None) is not None:
            scraper.session.close()

    return scraped_data, images
