# 🐝 langchain-scrapingbee
*The Best Web Scraping API to Avoid Getting Blocked*

## Overview
The ScrapingBee web scraping API handles headless browsers, rotates proxies for you, and offers AI-powered data extraction.

This package contains the LangChain integration with Scrapingbee

## Installation

```bash
pip install -U langchain-scrapingbee
```

And you should configure credentials by setting the following environment variables:

* SCRAPINGBEE_API_KEY

## Tools

ScrapingBee Integration provides you acceess to the following tools:

* ScrapeUrlTool - Scrape the contents of any public website.
* GoogleSearchTool - Search Google to obtain the following types of information regular search (classic), news, maps, and images.
* CheckUsageTool — Monitor your ScrapingBee credit or concurrency usage using this tool.

## Example

```python
import os
import getpass
from langchain_scrapingbee import (
    ScrapeUrlTool, 
    GoogleSearchTool, 
    CheckUsageTool,
)

api_key = os.environ.get("SCRAPINGBEE_API_KEY")
if not api_key:
    print("SCRAPINGBEE_API_KEY environment variable is not set. Please enter the API Key here:")
    os.environ["SCRAPINGBEE_API_KEY"] = getpass.getpass()

scrape_tool = ScrapeUrlTool(api_key=os.environ.get("SCRAPINGBEE_API_KEY"))
search_tool = GoogleSearchTool(api_key=os.environ.get("SCRAPINGBEE_API_KEY"))
usage_tool = CheckUsageTool(api_key=os.environ.get("SCRAPINGBEE_API_KEY"))

# --- Test Case 1: Scrape a standard HTML page ---
print("--- 1. Testing ScrapeUrlTool (HTML) ---")
html_result = scrape_tool.invoke({
    'url': 'http://httpbin.org/html'
})
print(html_result)


# --- Test Case 2: Scrape a PDF file ---
print("--- 2. Testing ScrapeUrlTool (PDF) ---")
pdf_result = scrape_tool.invoke({
    'url': 'https://treaties.un.org/doc/publication/ctc/uncharter.pdf',
    'params': {'render_js': False} 
})
print(pdf_result)


# --- Test Case 3: Google Search ---
print("--- 3. Testing GoogleSearchTool ---")
search_result = search_tool.invoke({
    'search': 'What is LangChain?'
})
print(search_result)


# --- Test Case 4: Check Usage ---
print("--- 4. Testing CheckUsageTool ---")
usage_result = usage_tool.invoke({}) # No arguments needed
print(usage_result)
```

## Example Using Agent

```python
import os
from langchain_scrapingbee import (
    ScrapeUrlTool, 
    GoogleSearchTool, 
    CheckUsageTool,
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
]

agent = create_react_agent(llm, tools)

user_input = "If I have enough API Credits, search for pdfs about langchain and save 3 pdfs."

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