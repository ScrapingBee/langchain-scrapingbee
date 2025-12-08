# 🐝 langchain-scrapingbee
*The Best Web Scraping API to Avoid Getting Blocked*

## Overview
The ScrapingBee web scraping API handles headless browsers, rotates proxies for you, and offers AI-powered data extraction.

This package contains the LangChain integration with Scrapingbee.

## Installation

```bash
pip install -U langchain-scrapingbee
```

And you should configure credentials by setting the following environment variables:

* `SCRAPINGBEE_API_KEY`

## Tools

ScrapingBee Integration provides you access to the following tools:

* **ScrapeUrlTool** - Scrape the contents of any public website. You can also use this to extract data, capture screenshots, interact with the page before scraping and capture the internal requests sent by the webpage.
* **GoogleSearchTool** - Search Google to obtain the following types of information regular search (classic), news, maps, and images.
* **CheckUsageTool** — Monitor your ScrapingBee credit or concurrency usage using this tool.
* **AmazonSearchTool** - Perform a product search on Amazon with options for localization, pagination, and advanced filtering.
* **AmazonProductTool** - Retrieve detailed information, including reviews, for a specific product on Amazon using its ASIN.
* **WalmartSearchTool** - Search for products on Walmart with parameters for sorting and price filtering.
* **WalmartProductTool** - Get specific details and reviews for a Walmart product by its ID.
* **ChatGPTTool** - Send your prompt to ChatGPT with an option to enhance its responses with live web search results.
* **YouTubeMetadataTool** - Retrieve comprehensive metadata for a YouTube video including title, description, view count, likes, channel info, publish date, duration, thumbnails, and tags.
* **YouTubeSearchTool** - Search YouTube with extensive filtering options for video quality (HD, 4K, HDR), duration, upload date, content type (video, channel, playlist), live streams, and more.
* **YouTubeTrainabilityTool** - Check whether a YouTube video's content can be used for AI/ML training purposes based on the video's settings and permissions.
* **YouTubeTranscriptTool** - Retrieve transcripts/captions for a YouTube video with support for multiple languages and choice between auto-generated or uploader-provided transcripts.

## Example

```python
import os
import getpass
from langchain_scrapingbee import (
    ScrapeUrlTool, 
    GoogleSearchTool, 
    CheckUsageTool,
    AmazonSearchTool,
    AmazonProductTool,
    WalmartSearchTool,
    WalmartProductTool,
    ChatGPTTool,
    YouTubeMetadataTool,
    YouTubeSearchTool,
    YouTubeTrainabilityTool,
    YouTubeTranscriptTool,
)

api_key = os.environ.get("SCRAPINGBEE_API_KEY")
if not api_key:
    print("SCRAPINGBEE_API_KEY environment variable is not set. Please enter the API Key here:")
    os.environ["SCRAPINGBEE_API_KEY"] = getpass.getpass()

api_key = os.environ.get("SCRAPINGBEE_API_KEY")

scrape_tool = ScrapeUrlTool(api_key=api_key)
google_search_tool = GoogleSearchTool(api_key=api_key)
usage_tool = CheckUsageTool(api_key=api_key)
amazon_search_tool = AmazonSearchTool(api_key=api_key)
amazon_product_tool = AmazonProductTool(api_key=api_key)
walmart_search_tool = WalmartSearchTool(api_key=api_key)
walmart_product_tool = WalmartProductTool(api_key=api_key)
chatgpt_tool = ChatGPTTool(api_key=api_key)
youtube_metadata_tool = YouTubeMetadataTool(api_key=api_key)
youtube_search_tool = YouTubeSearchTool(api_key=api_key)
youtube_trainability_tool = YouTubeTrainabilityTool(api_key=api_key)
youtube_transcript_tool = YouTubeTranscriptTool(api_key=api_key)


# --- Test Case 1: Scrape a standard HTML page ---
print("--- 1. Testing ScrapeUrlTool (HTML) ---")
html_result = scrape_tool.invoke({
    'url': 'http://httpbin.org/html'
})
print(html_result)

# --- Test Case 2: Scrape a PDF file ---
print("\n--- 2. Testing ScrapeUrlTool (PDF) ---")
pdf_result = scrape_tool.invoke({
    'url': 'https://treaties.un.org/doc/publication/ctc/uncharter.pdf',
    'params': {'render_js': False} 
})
print(pdf_result)

# --- Test Case 3: Google Search ---
print("\n--- 3. Testing GoogleSearchTool ---")
search_result = google_search_tool.invoke({
    'search': 'What is LangChain?'
})
print(search_result)

# --- Test Case 4: Check Usage ---
print("\n--- 4. Testing CheckUsageTool ---")
usage_result = usage_tool.invoke({}) # No arguments needed
print(usage_result)

# --- Test Case 5: Amazon Search ---
print("\n--- 5. Testing AmazonSearchTool ---")
amazon_search_result = amazon_search_tool.invoke({
    'query': 'iphone 16'
})
print(amazon_search_result)

# --- Test Case 6: Amazon Product ---
print("\n--- 6. Testing AmazonProductTool ---")
amazon_product_result = amazon_product_tool.invoke({
    'query': 'B0DPDRNSXV'
})
print(amazon_product_result)

# --- Test Case 7: Walmart Search ---
print("\n--- 7. Testing WalmartSearchTool ---")
walmart_search_result = walmart_search_tool.invoke({
    'query': 'iphone'
})
print(walmart_search_result)

# --- Test Case 8: Walmart Product ---
print("\n--- 8. Testing WalmartProductTool ---")
walmart_product_result = walmart_product_tool.invoke({
    'product_id': '454408250'
})
print(walmart_product_result)

# --- Test Case 9: ChatGPT ---
print("\n--- 9. Testing ChatGPTTool ---")
chatgpt_result = chatgpt_tool.invoke({
    'prompt': 'Explain the benefits of renewable energy in 100 words'
})
print(chatgpt_result)

# --- Test Case 10: YouTube Metadata ---
print("\n--- 10. Testing YouTubeMetadataTool ---")
youtube_metadata_result = youtube_metadata_tool.invoke({
    'video_id': 'dQw4w9WgXcQ'
})
print(youtube_metadata_result)

# --- Test Case 11: YouTube Search ---
print("\n--- 11. Testing YouTubeSearchTool ---")
youtube_search_result = youtube_search_tool.invoke({
    'search': 'python programming tutorial',
    'params': {'hd': True, 'sort_by': 'view_count'}
})
print(youtube_search_result)

# --- Test Case 12: YouTube Trainability ---
print("\n--- 12. Testing YouTubeTrainabilityTool ---")
youtube_trainability_result = youtube_trainability_tool.invoke({
    'video_id': 'dQw4w9WgXcQ'
})
print(youtube_trainability_result)

# --- Test Case 13: YouTube Transcript ---
print("\n--- 13. Testing YouTubeTranscriptTool ---")
youtube_transcript_result = youtube_transcript_tool.invoke({
    'video_id': 'dQw4w9WgXcQ',
    'params': {'language': 'en'}
})
print(youtube_transcript_result)
```

## Example Using Agent

```python
import os
from langchain_scrapingbee import (
    ScrapeUrlTool,
    GoogleSearchTool,
    CheckUsageTool,
    AmazonSearchTool,
    AmazonProductTool,
    WalmartSearchTool,
    WalmartProductTool,
    ChatGPTTool,
    YouTubeMetadataTool,
    YouTubeSearchTool,
    YouTubeTrainabilityTool,
    YouTubeTranscriptTool,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

if not os.environ.get("GOOGLE_API_KEY") or not os.environ.get("SCRAPINGBEE_API_KEY"):
    raise ValueError("Google and ScrapingBee API keys must be set in environment variables.")

llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-2.5-flash")
scrapingbee_api_key = os.environ.get("SCRAPINGBEE_API_KEY")

tools = [
    ScrapeUrlTool(api_key=scrapingbee_api_key),
    GoogleSearchTool(api_key=scrapingbee_api_key),
    CheckUsageTool(api_key=scrapingbee_api_key),
    AmazonSearchTool(api_key=scrapingbee_api_key),
    AmazonProductTool(api_key=scrapingbee_api_key),
    WalmartSearchTool(api_key=scrapingbee_api_key),
    WalmartProductTool(api_key=scrapingbee_api_key),
    ChatGPTTool(api_key=scrapingbee_api_key),
    YouTubeMetadataTool(api_key=scrapingbee_api_key),
    YouTubeSearchTool(api_key=scrapingbee_api_key),
    YouTubeTrainabilityTool(api_key=scrapingbee_api_key),
    YouTubeTranscriptTool(api_key=scrapingbee_api_key),
]

agent = create_react_agent(llm, tools)

user_input = """
If I have enough API Credits, perform the following tasks:

1. Search for "movie trailers" on YouTube

2. Get metadata of the video with the shortest duration.

3. Get the transcript of the video.

4. Check if the video is trainable for AI/ML purposes.

5. Search for "harry potter" book on Amazon.

6. Get the product details for the top Python book result from Amazon (use the ASIN).

7. Search for "harry potter" book on Walmart.

8. Get the product details for the top "harry potter" book result from Walmart (use the product ID).

9. Ask ChatGPT to summarize the story of harry potter series in 1000 words.

10. Use Google to search for "harry potter book" pdf files and download the first PDF you find.
"""

# Stream the agent's output step-by-step
for step in agent.stream(
    {"messages": user_input},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
```

## Documentation
* [HTML API](https://www.scrapingbee.com/documentation/)
* [Google Search API](https://www.scrapingbee.com/documentation/google/)
* [Data Extraction](https://www.scrapingbee.com/documentation/data-extraction/)
* [JavaScript Scenario](https://www.scrapingbee.com/documentation/js-scenario/)
* [Amazon Search/Product API](https://www.scrapingbee.com/documentation/amazon/)
* [Walmart Search/Product API](https://www.scrapingbee.com/documentation/walmart/)
* [ChatGPT API](https://www.scrapingbee.com/documentation/chatgpt/)
* [YouTube APIs](https://www.scrapingbee.com/documentation/youtube/)