"""GPT Researcher agent module.

This module provides the main GPTResearcher class that orchestrates
autonomous research and report generation using LLMs and web search.

(学习版: 以原版 gpt_researcher/agent.py 为蓝本, __init__ 签名对齐,
 裁剪了 deep research / 图片生成 / vector store 分支。)
"""

import asyncio
import json
import os
from typing import Any, Optional

from .actions import choose_agent
from .config import Config
from .embeddings import Memory
from .get_retrievers import get_retrievers
from .prompts import get_prompt_family
from .skills.context_manager import ContextManager
from .skills.curator import SourceCurator
from .skills.browser import BrowserManager
from .skills.researcher import ResearchConductor
from .skills.writer import ReportGenerator
from .utils.enum import ReportSource, ReportType, Tone


class GPTResearcher:
    """Main GPT Researcher agent class.

    This class orchestrates the entire research process including
    web searching, content scraping, context management, and
    report generation using LLMs.
    """

    def __init__(
        self,
        query: str,
        report_type: str = ReportType.ResearchReport.value,
        report_format: str = "markdown",
        report_source: str = ReportSource.Web.value,
        tone: Tone = Tone.Objective,
        source_urls: list[str] | None = None,
        document_urls: list[str] | None = None,
        complement_source_urls: bool = False,
        query_domains: list[str] | None = None,
        documents=None,
        vector_store=None,
        vector_store_filter=None,
        config_path=None,
        websocket=None,
        agent=None,
        role=None,
        parent_query: str = "",
        subtopics: list | None = None,
        visited_urls: set | None = None,
        verbose: bool = True,
        context=None,
        headers: dict | None = None,
        max_subtopics: int = 5,
        log_handler=None,
        prompt_family: str | None = None,
        mcp_configs: list[dict] | None = None,
        mcp_max_iterations: int | None = None,
        mcp_strategy: str | None = None,
        **kwargs
    ):
        """
        Initialize a GPT Researcher instance.
        """
        self.kwargs = kwargs
        self.query = query
        self.report_type = report_type
        self.cfg = Config(config_path)
        self.cfg.set_verbose(verbose)
        self.report_source = report_source if report_source else getattr(self.cfg, 'report_source', None)
        self.report_format = report_format
        self.max_subtopics = max_subtopics
        self.tone = tone if isinstance(tone, Tone) else Tone.Objective
        self.source_urls = source_urls
        self.document_urls = document_urls
        self.complement_source_urls = complement_source_urls
        self.query_domains = query_domains or []
        self.research_sources = []  # The list of scraped sources including title, content and images
        self.research_images = []  # The list of selected research images
        self.documents = documents
        self.vector_store = None
        self.vector_store_filter = vector_store_filter
        self.websocket = websocket
        self.agent = agent
        self.role = role
        self.parent_query = parent_query
        self.subtopics = subtopics or []
        self.visited_urls = visited_urls or set()
        self.verbose = verbose
        self.context = context or []
        self.headers = headers or {}
        self.research_costs = 0.0
        self.step_costs: dict[str, float] = {}
        self._current_step: str = "general"
        self.log_handler = log_handler
        self.prompt_family = get_prompt_family(prompt_family or self.cfg.prompt_family, self.cfg)

        # Process MCP configurations if provided
        self.mcp_configs = mcp_configs
        if mcp_configs:
            self._process_mcp_configs(mcp_configs)

        self.retrievers = get_retrievers(self.headers, self.cfg)
        self.memory = Memory(
            self.cfg.embedding_provider, self.cfg.embedding_model, **self.cfg.embedding_kwargs
        )

        # Set default encoding to utf-8
        self.encoding = kwargs.get('encoding', 'utf-8')
        self.kwargs.pop('encoding', None)  # Remove encoding from kwargs to avoid passing it to LLM calls

        # Initialize components
        self.research_conductor: ResearchConductor = ResearchConductor(self)
        self.report_generator: ReportGenerator = ReportGenerator(self)
        self.context_manager: ContextManager = ContextManager(self)
        self.scraper_manager: BrowserManager = BrowserManager(self)
        self.source_curator: SourceCurator = SourceCurator(self)

        self.available_images: list = []  # Pre-generated images ready for embedding
        self._research_id: str = ""  # Unique ID for this research session

        # Handle MCP strategy configuration with backwards compatibility
        self.mcp_strategy = self._resolve_mcp_strategy(mcp_strategy, mcp_max_iterations)

    def _resolve_mcp_strategy(self, mcp_strategy: str | None, mcp_max_iterations: int | None) -> str:
        """
        Resolve MCP strategy from various sources with backwards compatibility.

        Priority:
        1. Parameter mcp_strategy (new approach)
        2. Parameter mcp_max_iterations (backwards compatibility)
        3. Config MCP_STRATEGY
        4. Default "fast"
        """
        if mcp_strategy is not None:
            if mcp_strategy in ["fast", "deep", "disabled"]:
                return mcp_strategy
            elif mcp_strategy == "optimized":
                return "fast"
            elif mcp_strategy == "comprehensive":
                return "deep"
            else:
                return "fast"

        if mcp_max_iterations is not None:
            if mcp_max_iterations == 0:
                return "disabled"
            elif mcp_max_iterations == 1:
                return "fast"
            elif mcp_max_iterations == -1:
                return "deep"
            else:
                return "fast"

        if hasattr(self.cfg, 'mcp_strategy'):
            config_strategy = self.cfg.mcp_strategy
            if config_strategy in ["fast", "deep", "disabled"]:
                return config_strategy
            elif config_strategy == "optimized":
                return "fast"
            elif config_strategy == "comprehensive":
                return "deep"

        return "fast"

    def _process_mcp_configs(self, mcp_configs: list[dict]) -> None:
        """
        Process MCP configurations from a list of configuration dictionaries.

        Adds the MCP retriever to the active retriever list by modifying
        self.cfg.retrievers directly.
        """
        if hasattr(self.cfg, 'retrievers') and self.cfg.retrievers:
            current_retrievers = (
                list(self.cfg.retrievers)
                if isinstance(self.cfg.retrievers, list)
                else [r.strip() for r in str(self.cfg.retrievers).split(",") if r.strip()]
            )
            if "mcp" not in current_retrievers:
                current_retrievers.append("mcp")
                self.cfg.retrievers = current_retrievers
        else:
            self.cfg.retrievers = ["mcp"]

        # Store the mcp_configs for use by the MCP retriever
        self.mcp_configs = mcp_configs

    async def conduct_research(self, on_progress=None):
        """Conduct the research process.

        This method orchestrates the main research workflow including
        agent selection, web searching, and context gathering.

        Returns:
            The accumulated research context.
        """
        if not (self.agent and self.role):
            self._current_step = "agent_selection"
            self.agent, self.role = await choose_agent(
                query=self.query,
                cfg=self.cfg,
                parent_query=self.parent_query,
                cost_callback=self.add_costs,
                headers=self.headers,
                prompt_family=self.prompt_family,
                **self.kwargs,
            )

        self._current_step = "research"
        self.context = await self.research_conductor.conduct_research()

        return self.context

    async def write_report(
        self,
        existing_headers: list = [],
        relevant_written_contents: list = [],
        ext_context=None,
        custom_prompt="",
    ) -> str:
        """Write the research report.

        Returns:
            The generated report as a string.
        """
        self._current_step = "report_writing"

        report = await self.report_generator.write_report(
            existing_headers=existing_headers,
            relevant_written_contents=relevant_written_contents,
            ext_context=ext_context or self.context,
            custom_prompt=custom_prompt,
            available_images=self.available_images,  # Pass pre-generated images
        )

        return report

    # Utility methods
    def get_research_images(self, top_k: int = 10) -> list[dict[str, Any]]:
        """Get the top research images collected during research."""
        return self.research_images[:top_k]

    def add_research_images(self, images: list[dict[str, Any]]) -> None:
        """Add images to the research image collection."""
        self.research_images.extend(images)

    def get_research_sources(self) -> list[dict[str, Any]]:
        """Get all research sources collected during research."""
        return self.research_sources

    def add_research_sources(self, sources: list[dict[str, Any]]) -> None:
        """Add sources to the research source collection."""
        self.research_sources.extend(sources)

    def get_source_urls(self) -> list:
        """Get all visited source URLs."""
        return list(self.visited_urls)

    def get_research_context(self) -> list:
        """Get the accumulated research context."""
        return self.context

    def get_costs(self) -> float:
        """Get the total accumulated API costs."""
        return self.research_costs

    def get_step_costs(self) -> dict[str, float]:
        """Get a breakdown of API costs per research step."""
        return dict(self.step_costs)

    def set_verbose(self, verbose: bool) -> None:
        """Set the verbose output mode."""
        self.verbose = verbose

    def add_costs(self, cost: float) -> None:
        """Add to the accumulated API costs.

        The cost is attributed to the current step set via ``_current_step``.
        """
        if not isinstance(cost, (float, int)):
            raise ValueError("Cost must be an integer or float")
        self.research_costs += cost
        step = self._current_step
        self.step_costs[step] = self.step_costs.get(step, 0.0) + cost
