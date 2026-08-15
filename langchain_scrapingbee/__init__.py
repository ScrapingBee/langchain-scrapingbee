from importlib import metadata

from langchain_scrapingbee.tools import (
    ScrapeUrlTool,
    GoogleSearchTool,
    FastSearchTool,
    CheckUsageTool,
    AmazonSearchTool,
    AmazonProductTool,
    AmazonPricingTool,
    WalmartSearchTool,
    WalmartProductTool,
    ChatGPTTool,
    GeminiTool,
    YouTubeMetadataTool,
    YouTubeSearchTool,
    YouTubeSubtitlesTool,
    YouTubeTranscriptTool,
)


try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ScrapeUrlTool",
    "GoogleSearchTool",
    "FastSearchTool",
    "CheckUsageTool",
    "AmazonSearchTool",
    "AmazonProductTool",
    "AmazonPricingTool",
    "WalmartSearchTool",
    "WalmartProductTool",
    "ChatGPTTool",
    "GeminiTool",
    "YouTubeMetadataTool",
    "YouTubeSearchTool",
    "YouTubeSubtitlesTool",
    # Deprecated alias kept for backward compatibility
    "YouTubeTranscriptTool",
]
