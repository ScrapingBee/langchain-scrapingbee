import os
import requests
import json
import base64
from typing import Optional, Type, Dict, Any
import datetime
import re

from langchain.tools import BaseTool
from pydantic import BaseModel, Field, field_validator

scraping_prompt = (
        "Scrapes web content, takes screenshots, or downloads files from URLs."
        "For screenshots/binary files, returns JSON with 'reference_id' that MUST be passed to write_file tool immediately to save the file. "
        "if file name not available, use random name"
        "For text content, returns the scraped HTML/text directly. "
        "Use params for screenshots: {'screenshot_full_page': 'true'} or data extraction: {'extract_rules': '{...}'}. "
        "Supports JavaScript rendering, mobile simulation, proxy geolocation, and AI-powered extraction."
        "params should be a valid dictionary"
        "For non-text files, use 'render_js=false'. Non-HTML downloads are limited to 2 MB per request."
        "Before running ai_query and ai_extract_rules, scrapingbee converts the HTML content to markdown. So the ai model only have access to markdown not the html"
        "EXTREMELY IMPORTANT: Must use valid parameters and instructions, do not make up non-existent parameters and instructions"
        "For Google SERPs, prefer the dedicated google_search or fast_search tools."
        """
        SUPPORTED PARAMS:
        - "ai_extract_rules": "JSON string" - AI-based extraction with structured schema (+5 credits)
            [
                - "ai_query": Natural language, e.g., "Extract product names, prices, and availability"
                - "ai_extract_rules": Structured schema with types:
                    * "type": "string|list|number|boolean|item"
                    * "description": "What to extract"
                    * "enum": ["option1", "option2"] - Allowed values for lists
                    * "output": {...} - Nested structure for "item" type
                - "ai_selector": Focus extraction on specific CSS selector area
                - Cost: +5 credits per request
            ]
        - "ai_query": "What information to extract" - Natural language extraction (+5 credits)
        - "ai_selector": "css-selector" - Focus AI extraction on specific area
        - "block_ads": true - Block advertisements
        - "block_resources": true - Block images/CSS for faster text extraction (default: true)
        - "cookies": "name=value,domain=example.com;name2=value2" - Custom cookies with attributes
        - "country_code": "us"/"gb"/"de" etc. - Premium proxy location (lower-case ISO 3166-1)
        - "custom_google": true - Required when scraping Google domains directly (requests without it return 400); costs 15 credits. For structured SERPs, prefer the google_search or fast_search tools
        - "device": "desktop"/"mobile" - Device simulation
        - "extract_rules": CSS/XPath extraction rules, example: '{"page_title": "title", "first_heading": "h1", "links": {"selector": "a", "type": "list", "output": "@href"}}' (use the "title" selector for the page <title> tag, "h1" for the visible heading)
            [
                - use this feature only if CSS or XPath selectors are known or if the request requires clean text or markdown
                - type only accepts item or list
                - Basic syntax: {"field": "selector"} or {"field": "selector@attribute"}
                - Advanced syntax: {"field": {"selector": "css-or-xpath", "type": "item|list", "output": "text|text_relevant|markdown_relevant|html|@attribute|table_json|table_array"}}
                - "selector_type": "auto|css|xpath" - Force selector type (XPath must start with / or specify type)
                - "output": "text" (default), "html", "@attribute", "table_array", "table_json", "text_relevant", "markdown_relevant"
                - "type": "item" (first match) or "list" (all matches)
                - "clean": true/false - Clean whitespace (default: true)
                - Nested extraction: Use "output" object with child selectors for complex structures
                - Table extraction: "table_json" (objects with headers), "table_array" (raw arrays)
                - Providing any other options inside data extraction would result in an error.
            ]
        - "forward_headers": true - Forward your headers + ScrapingBee headers
        - "forward_headers_pure": true - Forward only your specified headers (only useful with render_js=false)
        - "js_scenario": JavaScript execution instructions, example: '{"instructions": [{"click": "#button"}, {"wait": 1000}]}'
            [
                - Structure: {"strict": true/false, "instructions": [{"instruction1":"value1"},{"instruction2":"value2"}]}
                - "strict": false - Continue on errors (default: true stops on errors)
                - SUPPORTED INSTRUCTIONS:
                    * {"click": "selector"} - Click element
                    * {"wait": 1000} - Wait milliseconds
                    * {"wait_for": "selector"} - Wait for element to appear
                    * {"wait_for_and_click": "selector"} - Wait then click
                    * {"scroll_x": 1000} - Horizontal scroll pixels
                    * {"scroll_y": 1000} - Vertical scroll pixels
                    * {"fill": ["selector", "value"]} - Fill input field
                    * {"evaluate": "javascript_code"} - Execute custom Java Script code (results in evaluate_results).
                    * {"infinite_scroll": {"max_count": 0, "delay": 1000, "end_click": {"selector": "#more"}}} - Auto-scroll, adding a minimum delay of 1000ms between each scroll is recommended, setting max_count to 0 means infinite scroll
                - SPECIAL NOTE: When a JavaScript instruction can cause a change in url, then add a wait of 5 seconds after the instruction to wait for the page to load.
                - MOST IMPORTANT: Ensure the instuctions structure is valid. For example: {"instructions": [{"click": "#button"}, {"wait": 1000}]}
                - Selectors: CSS or XPath (XPath must start with /)
                - Timeout: 40 seconds maximum
                - Stealth proxy limitations: No infinite_scroll, timeout, custom headers/cookies, or evaluate_results in a JSON response
            ]
        - "json_response": true - Wrap response in JSON format with metadata. This can also be used to find internal xhr requests
        - "max_cost": 25 - Only valid with mode=auto; caps the most expensive tier auto mode is allowed to try (e.g., 25 prevents escalation to stealth)
        - "mode": "auto" - Automatic escalation from cheapest to most expensive configuration (rotating 1/5 credits -> premium 10/25 -> stealth 75) until one succeeds; only the successful tier is billed, a total failure costs 0. GET-only. Do NOT combine with render_js, premium_proxy, stealth_proxy, or transparent_status_code (returns 400). It never adds js_scenario, waits, headers, or cookies - pass those yourself
        - "own_proxy": "protocol://user:pass@host:port" - Use your own proxy (port defaults to 1080 if omitted)
        - "premium_proxy": true - Use premium/residential proxy pool (10 credits without JS rendering, 25 with)
        - "render_js": true/false - Enable JS rendering (default: true)
        - "return_page_markdown": true - Return the page content in markdown format
        - "return_page_source": true - Return original HTML before JS rendering
        - "return_page_text": true - Return the text content of the page you want to scrape
        - "scraping_config": config_name (Must only use if provided by user) - Use a pre-saved request configuration on your request
        - "screenshot": true - Screenshot of viewport (requires JS rendering; forces block_resources=false)
        - "screenshot_full_page": true - Full page screenshot
        - "screenshot_selector": "css-selector" - Screenshot specific element
        - "session_id": 123 - Reuse the same IP for 5 minutes (random integer, 0-10000000)
        - "stealth_proxy": true - Use stealth proxies for difficult sites or when the previous request fails (75 credits, requires JS rendering). Not supported: infinite_scroll, custom headers/cookies, timeout, evaluate_results
        - "tag": "my-label" - Arbitrary label returned in response headers; does not affect scraping
        - "timeout": 140000 - Request timeout in milliseconds (1000-140000, default: 140000). Lowering it can reduce success rate
        - "transparent_status_code": true - Return original HTTP status codes (every request is billed and target 500s are not retried)
        - "wait": 3000 - Wait milliseconds before capture (0-35000, runs after wait_for)
        - "wait_browser": "domcontentloaded"/"load"/"networkidle0"/"networkidle2" - Browser wait condition
        - "wait_for": "css-selector" - Wait for element to appear
        - "window_height": 1080 - Viewport height in pixels
        - "window_width": 1920 - Viewport width in pixels

        UNSUPPORTED PARAMS:
        - Anything that is not mentioned in the above list

        SCRAPING STRATEGY:
        - Research Well using Google Search API: When you lack sufficient information, use Google Search API to find it first before attempting to scrape pages. You can use it multiple times before and in-between scrapes
        - Prefer AI Extraction Rules over Data Extraction rules: Use extract_rules if selector is known, otherwise use extract_rules to get the body in html to find the selector, do not use ai_query or ai_extract_rules for finding selector and do not guess selector
        - Proxy escalation: Start without a proxy. If a target is blocked, retry with {"premium_proxy": true}, then {"stealth_proxy": true}. Alternatively use {"mode": "auto"} (optionally with "max_cost") to let ScrapingBee escalate automatically and bill only the tier that succeeds
        """
    )

# ======================================================================================
# Result Saver Utility Functions
# ======================================================================================

def create_results_folder(base_folder: str = "scraping_results") -> str:
    """Creates a timestamped folder for saving results."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = os.path.join(base_folder, timestamp)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def sanitize_filename(url: str, max_length: int = 100) -> str:
    """Creates a safe filename from a URL."""
    # Remove protocol and clean up
    clean_name = re.sub(r'^https?://', '', url)
    clean_name = re.sub(r'[^\w\s.-]', '_', clean_name)
    clean_name = re.sub(r'[-\s]+', '_', clean_name)
    return clean_name[:max_length]

def save_scraping_metadata(folder_path: str, url: str, params: Dict, result_type: str,
                          filename: str = None, reference_id: str = None) -> str:
    """Saves metadata about the scraping operation."""
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "url": url,
        "params": params,
        "result_type": result_type,
        "filename": filename,
        "reference_id": reference_id
    }

    metadata_file = os.path.join(folder_path, "scraping_metadata.jsonl")
    with open(metadata_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(metadata) + "\n")

    return metadata_file

def stringify_nested_objects(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Iterates through a dictionary of parameters and converts any nested
    dict or list values into JSON strings. This is required for certain APIs
    that expect complex objects to be passed as a string.

    Args:
        params: The dictionary of parameters.

    Returns:
        A new dictionary with nested objects stringified.
    """
    processed_params = {}
    for key, value in params.items():
        # Check if the value is a dictionary or a list
        if isinstance(value, (dict, list)):
            processed_params[key] = json.dumps(value)
        else:
            # Otherwise, keep the value as is
            processed_params[key] = value
    return processed_params


def str_to_dict_validator(v: Any) -> Any:
    if v == '':
        return {}
    if isinstance(v, str):
        # First try to parse as JSON
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            print(v)

        # Try to parse as Python dictionary literal (e.g., "{'key': True}")
        try:
            if v.strip().startswith('{') and v.strip().endswith('}'):
                # Use ast.literal_eval to safely evaluate Python literals
                import ast
                return ast.literal_eval(v)
        except (ValueError, SyntaxError, TypeError):
            print(v)

        # Try to parse as URL parameters (key=value&key2=value2)
        try:
            if '=' in v:
                # Handle URL parameter format like "screenshot_full_page=True&wait=3000"
                params = {}
                pairs = v.split('&')
                for pair in pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        # Convert common boolean and numeric values
                        if value.lower() == 'true':
                            params[key] = True
                        elif value.lower() == 'false':
                            params[key] = False
                        elif value.isdigit():
                            params[key] = int(value)
                        else:
                            params[key] = value
                return params
        except Exception as e:
            print(e)
            print(print(v))

        # If all else fails, let Pydantic handle it
    return v


# ======================================================================================
# Tool 1: The Pure URL Scraper
# ======================================================================================

class ScrapeUrlInput(BaseModel):
    """Input model for the URL Scraper tool."""
    url: str = Field(
        description="The fully qualified URL to scrape (must include http:// or https://)"
    )
    params: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="""Optional parameters dictionary for ScrapingBee API. Must be a proper dictionary/object.

        Examples:
        {"screenshot_full_page": true, "wait": 2000}
        {"extract_rules": '{"title": "h1", "price": ".price"}'}
        {"country_code": "gb", "device": "mobile"}"""
    )
    headers: Optional[Dict[str, str]] = Field(  # ADD THIS
        default_factory=dict,
        description="Custom headers to forward to the target website. Will be prefixed with 'Spb-' automatically."
    )
    results_folder: Optional[str] = Field(
        default="scraping_results",
        description="Base folder for results (timestamped subfolder will be created)"
    )
    custom_filename: Optional[str] = Field(
        default=None,
        description="Custom filename (with extension)"
    )
    return_content: Optional[bool] = Field(
        default=False,
        description="Whether to return the actual content in response. If False, only returns file info to save tokens. Must be set to True if the agent needs to read the contents."
    )

    @field_validator('params', mode='before')
    @classmethod
    def validate_params(cls, v: Any) -> Any:
        return str_to_dict_validator(v)

    @field_validator('headers', mode='before')
    @classmethod
    def validate_headers(cls, v: Any) -> Any:
        return str_to_dict_validator(v)

class ScrapeUrlTool(BaseTool):
    """
    Comprehensive web scraping tool using ScrapingBee API. Handles text extraction, screenshots, file downloads, and data extraction.

    KEY WORKFLOW FOR BINARY CONTENT:
    When taking screenshots or downloading files, returns JSON: {"status": "binary_content_staged", "reference_id": "ref_..."}
    MUST immediately call write_file tool with the reference_id to save the file.

    CAPABILITIES:
    - Text scraping with JavaScript rendering
    - Full page and element screenshots
    - Structured data extraction with CSS selectors
    - AI-powered content extraction
    - File downloads (PDFs, images, etc.)
    - Mobile/desktop device simulation
    - Geo-located proxy access
    - Session management for multi-page scraping
    """
    args_schema: Type[BaseModel] = ScrapeUrlInput
    api_key: str
    name: str = "scrape_url"
    description: str = scraping_prompt

    def _get_extension_from_content_type(self, content_type: str) -> str:
        """Determines file extension from content type."""
        if "image/png" in content_type: return "png"
        elif "image/jpeg" in content_type or "image/jpg" in content_type: return "jpg"
        elif "application/pdf" in content_type: return "pdf"
        elif "image/webp" in content_type: return "webp"
        elif "image/gif" in content_type: return "gif"
        else: return "bin"

    def _run(self, url: str, params: Optional[Dict[str, Any]] = None,
             headers: Optional[Dict[str, str]] = None,
             results_folder: str = "scraping_results", custom_filename: str = None,
             return_content: bool = False) -> str:
        if params is None:
            params = {}

        processed_params = stringify_nested_objects(params)

        if headers is None:
            headers = {}

        final_headers = {}
        if headers:
            for key, value in headers.items():
                spb_key = f"Spb-{key}"
                final_headers[spb_key] = value

            processed_params['forward_headers'] = True

        final_headers['User-Agent'] = 'LangChain'
        final_headers['Authorization'] = f'Bearer {self.api_key}'

        api_url = "https://app.scrapingbee.com/api/v1/"
        request_params = {'url': url, **processed_params}

        try:
            response = requests.get(api_url, params=request_params, headers=final_headers, timeout=180)
            response.raise_for_status()

            content_type = response.headers.get('Content-Type', '')
            # Check if binary content (screenshots, PDFs, images)
            is_binary = (
                any(sub in content_type for sub in ['image/', 'application/pdf', 'octet-stream']) or
                params.get('screenshot') or
                params.get('screenshot_full_page') or
                params.get('screenshot_selector')
            )

            # Always save content first
            folder_path = create_results_folder(results_folder)

            if is_binary:
                # Save binary content
                if custom_filename:
                    filename = custom_filename
                else:
                    base_name = sanitize_filename(url)
                    ext = self._get_extension_from_content_type(content_type)
                    filename = f"{base_name}.{ext}"

                file_path = os.path.join(folder_path, filename)

                with open(file_path, 'wb') as f:
                    f.write(response.content)

                save_scraping_metadata(folder_path, url, params, "binary", filename=filename)

                if return_content:
                    # For binary files, we can't return content directly, so return file info + note
                    return f"""Binary content saved and processed:
                            File: {file_path}
                            Size: {len(response.content):,} bytes
                            Content-Type: {content_type}
                            URL: {url}

                            Note: Binary content cannot be displayed in text. File is saved and ready for use."""
                else:
                    return f"""Binary content saved successfully:
                            File: {file_path}
                            Size: {len(response.content):,} bytes
                            Content-Type: {content_type}
                            URL: {url}"""
            else:
                # Save text content
                if custom_filename:
                    filename = custom_filename
                else:
                    filename = f"{sanitize_filename(url)}.html"

                file_path = os.path.join(folder_path, filename)

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)

                save_scraping_metadata(folder_path, url, params, "text", filename=filename)

                if return_content:
                    return f"""Text content saved and loaded:
                            File: {file_path}
                            Size: {len(response.text):,} characters
                            Content-Type: {content_type}
                            URL: {url}

                            CONTENT:
                            {response.text}"""
                else:
                    return f"""Text content saved successfully:
                            File: {file_path}
                            Size: {len(response.text):,} characters
                            Content-Type: {content_type}
                            URL: {url}"""

        except requests.exceptions.RequestException as e:
            error_detail = (getattr(e.response, 'text', str(e)) if hasattr(e, 'response') else str(e))[:1000]
            return f"Error: Request failed. Details: {error_detail}"



# ======================================================================================
# Tool 2: Google Searcher with INTEGRATED Image Saving Feature
# ======================================================================================

class GoogleSearchInput(BaseModel):
    """Input model for the Google Search tool."""
    search: str = Field(description="The search query text to send to Google")
    params: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="""Optional parameters dictionary for Google Search API. Must be a proper dictionary/object.
Examples:
{"search_type": "news", "country_code": "gb"}
{"pages": 2, "language": "es"}
{"search_type": "images", "device": "mobile"}"""
    )
    results_folder: Optional[str] = Field(
        default="scraping_results",
        description="Base folder path to save results (timestamped subfolder will be created)"
    )
    return_content: Optional[bool] = Field(
        default=False,
        description="Whether to return the actual search results in response. If False, only returns file info to save tokens. Must be set to True if the agent needs to read the contents."
    )

    @field_validator('params', mode='before')
    @classmethod
    def validate_params(cls, v: Any) -> Any:
        return str_to_dict_validator(v)

class GoogleSearchTool(BaseTool):
    """
    Comprehensive Google Search tool supporting web, news, images, and maps search with geolocation and language options.
    All results are saved to disk automatically. Use return_content=True to return the search result in response for AI analysis when needed.
    """
    name: str = "google_search"
    description: str = (
        "Performs Google searches across different search types (classic web, news, maps, lens, shopping, images, ai_mode, ads). "
        "All results are automatically saved to disk to conserve AI tokens. "
        "For image searches: Downloads base64 images and saves image URLs to image_links.txt. "
        "For other searches: Saves JSON results to file. "
        "Use return_content=True to return the search result in response for AI analysis when needed."
        "Use params for specific searches: {'search_type': 'news'} or location: {'country_code': 'gb'}. "
        "Supports pagination, language preferences, and result filtering. "
        "params should be a valid dictionary. "
        """
        SUPPORTED PARAMS:
        - "add_html": true - Include full HTML of result pages in response (default: false)
        - "country_code": "us"/"gb"/"de" etc. - Country for localized results (default: "us")
        - "date_range": "past_hour"/"past_day"/"past_week"/"past_month"/"past_year" - Time filter (classic, news, and images only)
        - "device": "desktop"/"mobile" - Device type for search (default: "desktop"; news is unavailable on mobile)
        - "extra_params": "safe=active&filter=0" - Additional URL-encoded Google URL parameters (encode & as %26 for multiple values)
        - "language": "en"/"es"/"fr" etc. - Language for results (default: "en")
        - "latitude"/"longitude": decimal degrees - Geographic point for localized results (must be supplied together)
        - "light_request": true/false - Fast non-browser request (default: true). Set false for a full browser request or when AI Overviews are needed
        - "max_price"/"min_price": number - Price filters (shopping search only, native marketplace currency)
        - "nfpr": true - Exclude auto-corrected spelling results
        - "page": 2 - Page number for pagination (default: 1)
        - "pages": 3 - Number of consecutive Google pages to fetch and combine in one response (max: 10, 3 or fewer recommended)
        - "radius": meters - Search radius (takes effect only with latitude and longitude)
        - "search_type": "classic"/"news"/"maps"/"lens"/"shopping"/"images"/"ai_mode"/"ads" - Type of Google search (default: "classic")
        - "sort_by": "relevance"/"reviews"/"price_asc"/"price_desc" - Sorting (shopping search only)
        - "tag": "my-label" - Arbitrary label returned in response headers; does not affect scraping

        IGNORED PARAMS (accepted but have no effect - do not use):
        - "nb_results": silently ignored by the API; use "page"/"pages" for pagination instead

        NOTES:
        - Cost: 10 credits per light request, 15 with light_request=false
        - search_type=lens requires an image URL in the search input; search_type=ai_mode accepts at most 400 input characters
        - AI Overviews are a response field (ai_overviews), not a search_type, and require light_request=false
        """
    )
    args_schema: Type[BaseModel] = GoogleSearchInput
    api_key: str

    def _sanitize_filename(self, name: str) -> str:
        """Cleans a string to be a valid filename."""
        name = re.sub(r'[^\w\s-]', '', name).strip()
        name = re.sub(r'[-\s]+', '_', name)
        return name[:100]

    def _is_base64_image(self, image_data: str) -> bool:
        """Checks if the image data is base64 encoded content."""
        if not image_data:
            return False

        # Check for data URI format
        if image_data.startswith('data:image/') and 'base64,' in image_data:
            return True

        # Check if it looks like base64 (not a URL)
        if image_data.startswith(('http://', 'https://', '//', '/')):
            return False

        # Try to decode as base64 to verify
        try:
            clean_data = image_data
            if 'base64,' in image_data:
                clean_data = image_data.split('base64,', 1)[1]

            clean_data = re.sub(r'\s+', '', clean_data)

            # Add padding if missing
            missing_padding = len(clean_data) % 4
            if missing_padding != 0:
                clean_data += '=' * (4 - missing_padding)

            base64.b64decode(clean_data)
            return True
        except Exception:
            return False

    def _save_base64_image(self, image_data: str, folder_path: str, filename_prefix: str, title: str) -> str:
        """Saves a base64 image to disk."""
        try:
            # Clean the data: remove URI prefix and whitespace
            if 'base64,' in image_data:
                clean_b64_data = image_data.split('base64,', 1)[1]
            else:
                clean_b64_data = image_data
            clean_b64_data = re.sub(r'\s+', '', clean_b64_data)

            # Add padding if missing
            missing_padding = len(clean_b64_data) % 4
            if missing_padding != 0:
                clean_b64_data += '=' * (4 - missing_padding)

            # Decode the base64 string
            image_bytes = base64.b64decode(clean_b64_data)

            # Detect format from the decoded bytes
            image_format = 'jpg'  # Default
            if image_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
                image_format = 'png'
            elif image_bytes.startswith(b'GIF8'):
                image_format = 'gif'
            elif image_bytes.startswith(b'RIFF') and b'WEBP' in image_bytes:
                image_format = 'webp'
            elif image_bytes.startswith(b'\xff\xd8\xff'):
                image_format = 'jpg'

            # Create filename and save
            sanitized_title = self._sanitize_filename(title)
            filename = f"{filename_prefix}_{sanitized_title}.{image_format}"

            os.makedirs(folder_path, exist_ok=True)
            file_path = os.path.join(folder_path, filename)

            with open(file_path, 'wb') as f:
                f.write(image_bytes)

            return f"Saved: {file_path}"
        except Exception as e:
            return f"Failed to save '{title}': {str(e)}"

    def _save_image_links(self, image_links: list, folder_path: str) -> str:
        """Saves image URLs to a text file."""
        if not image_links:
            return "No image links to save"

        os.makedirs(folder_path, exist_ok=True)
        links_file = os.path.join(folder_path, "image_links.txt")

        try:
            with open(links_file, 'w', encoding='utf-8') as f:
                f.write("# Image Links from Google Search\n")
                f.write(f"# Generated: {datetime.datetime.now().isoformat()}\n\n")
                for i, link_info in enumerate(image_links, 1):
                    f.write(f"{i}. Title: {link_info['title']}\n")
                    f.write(f"   URL: {link_info['url']}\n")
                    f.write(f"   Position: {link_info['position']}\n\n")

            return f"Saved {len(image_links)} image links to: {links_file}"
        except Exception as e:
            return f"Failed to save image links: {str(e)}"

    def _run(self, search: str, params: Optional[Dict[str, Any]] = None,
             results_folder: str = "scraping_results", return_content: bool = False) -> str:
        params = params or {}
        api_url = "https://app.scrapingbee.com/api/v1/store/google"
        request_params = {'search': search, **params}

        try:
            response = requests.get(api_url, params=request_params, headers={'User-Agent': 'LangChain', 'Authorization': f'Bearer {self.api_key}'}, timeout=120)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return f"Error during Google Search API call: {getattr(e.response, 'text', str(e))}"

        # Handle different search types
        if params.get("search_type") == "images":
            return self._handle_image_search(response, search, params, results_folder, return_content)
        else:
            return self._handle_regular_search(response, search, params, results_folder, return_content)

    def _handle_image_search(self, response, search: str, params: dict, results_folder: str, return_content: bool) -> str:
        """Handles image search results with base64 vs URL separation."""
        try:
            results = response.json()
            image_results = results.get("images", [])

            # Always save results
            folder_path = create_results_folder(results_folder)

            if not image_results:
                # Still save the empty results for reference
                filename = f"image_search_{sanitize_filename(search)}.json"
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2)

                if return_content:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    return f"""Image search complete but no results found:
                            File: {file_path}

                            CONTENT:
                            {content}"""
                else:
                    return f"Image search complete but no results found. Empty results saved to: {file_path}"

            # Separate base64 images from URL links
            base64_images = []
            image_links = []

            for item in image_results:
                image_data = item.get("image", "")
                title = item.get("title", "untitled")
                position = item.get("position", 0)

                if self._is_base64_image(image_data):
                    base64_images.append({
                        "data": image_data,
                        "title": title,
                        "position": position
                    })
                else:
                    image_links.append({
                        "url": image_data,
                        "title": title,
                        "position": position
                    })

            # Save base64 images
            saved_images = []
            for item in base64_images:
                result = self._save_base64_image(
                    item["data"],
                    folder_path,
                    f"{item['position']:02d}",
                    item["title"]
                )
                saved_images.append(result)

            # Save image links
            links_result = self._save_image_links(image_links, folder_path)

            # Save full JSON results
            filename = f"image_search_{sanitize_filename(search)}.json"
            json_file_path = os.path.join(folder_path, filename)
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)

            # Save metadata
            save_scraping_metadata(folder_path, f"google_image_search:{search}", params, "image_search")

            success_count = sum(1 for r in saved_images if r.startswith("Saved:"))

            base_response = f"""Image search complete:
                                - Saved {success_count} base64 images
                                - {links_result}
                                - Full results saved to: {json_file_path}
                                - Results folder: {folder_path}"""

            if return_content:
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return f"""{base_response}

                        CONTENT:
                        {content}"""
            else:
                return f"""{base_response}"""

        except json.JSONDecodeError:
            return "Error: Failed to parse the image search response as JSON."
        except Exception as e:
            return f"An unexpected error occurred during image search: {e}"

    def _handle_regular_search(self, response, search: str, params: dict, results_folder: str, return_content: bool) -> str:
        """Handles regular search results (web, news, maps)."""
        response_text = response.text

        # Always save results
        folder_path = create_results_folder(results_folder)

        # Determine filename based on search type
        search_type = params.get("search_type", "web")
        filename = f"{search_type}_search_{sanitize_filename(search)}.json"
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response_text)

        save_scraping_metadata(folder_path, f"google_{search_type}_search:{search}", params, "search_results", filename=filename)

        # Count results for summary
        try:
            results = json.loads(response_text)
            result_count = 0
            result_count = max([len(results.get("organic_results", [])), len(results.get("news_results", [])), len(results.get("map_results", results.get("maps_results", [])))])
        except:
            result_count = "unknown"

        base_response = f"""Search complete:
                        Query: "{search}"
                        Type: {search_type}
                        Results: {result_count}
                        Saved to: {file_path}"""

        if return_content:
            return f"""{base_response}

                    CONTENT:
                    {response_text}"""
        else:
            return f"""{base_response}"""


# ======================================================================================
# Tool 3: The Usage Checker
# ======================================================================================

class CheckUsageTool(BaseTool):
    """Checks ScrapingBee API usage, remaining credits, and account limits. No parameters required."""
    name: str = "check_scrapingbee_usage"
    description: str = (
        "Checks current ScrapingBee API usage statistics. Returns max_api_credit, "
        "used_api_credit, max_concurrency, current_concurrency, and renewal_subscription_date. "
        "Takes no parameters. Rate-limited to 6 calls per minute and does not consume concurrency."
    )
    api_key: str

    def _run(self) -> str:
        api_url = "https://app.scrapingbee.com/api/v1/usage"

        try:
            response = requests.get(api_url, headers={'Authorization': f'Bearer {self.api_key}'}, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            error_detail = getattr(e.response, 'text', str(e)) if hasattr(e, 'response') else str(e)
            return f"Error checking usage: {error_detail}"

# ======================================================================================
# Tool 4: Amazon Search API
# ======================================================================================

class AmazonSearchInput(BaseModel):
    """Input model for the Amazon Search tool."""
    query: str = Field(description="The search query for Amazon")
    params: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="""
            Optional parameters for the Amazon Search APIO. Must be a proper dictionary/object.
            Examples:
            {"add_html": true, "device": "mobile"}
            {"light_request": false, "language": "es"}
            {"zip_code": "90210", "device": "mobile"}
            """
    )
    results_folder: Optional[str] = Field(
        default="scraping_results",
        description="Base folder to save results"
    )
    return_content: Optional[bool] = Field(
        default=False,
        description="Whether to return the search results in the response"
    )

    @field_validator('params', mode='before')
    @classmethod
    def validate_params(cls, v: Any) -> Any:
        return str_to_dict_validator(v)

class AmazonSearchTool(BaseTool):
    """
    Performs a product search on Amazon. All results are automatically saved to a JSON file.
    Use return_content=True to get the results back for analysis.
    """
    name: str = "amazon_search"
    description: str = (
        "Performs a product search on Amazon. All results are automatically saved to a JSON file. "
        "Use return_content=True to get the results back for analysis."
        """
        SUPPORTED PARAMS:
        - "add_html": true/false - Include the full HTML of the search result page in the JSON response (default: false).
        - "autoselect_variant": true/false - If a product page is reached, automatically select the default/most popular variant (default: false).
        - "category_id": "123" - Restrict results to a specific Amazon category.
        - "country": "us" / "gb" / "de" etc. - Two-letter country code for geolocation. Do NOT set it to the same country as the selected domain (e.g., country=fr with domain=fr returns 400); use zip_code instead.
        - "currency": "USD" / "GBP" / "EUR" etc. - Three-letter currency code (ISO 4217) to display prices in (conversion may be unavailable for some domains/products).
        - "device": "desktop" / "mobile" / "tablet" - Device type to simulate for the request (default: desktop).
        - "domain": "com" / "co.uk" / "de" etc. - The Amazon top-level domain to use for the search (default: com).
        - "language": "en-US" / "fr-FR" etc. - Language code for the request to get results in a specific language.
        - "light_request": true/false - Perform a light, faster request. Set to false to force a full JavaScript render which may yield more data (default: true).
        - "merchant_id": "A2XXXXXXXX" - Restrict results to a specific Amazon seller.
        - "pages": 3 - Number of consecutive result pages to fetch starting at start_page (default: 1; cost is per fetched page).
        - "screenshot": true - Force a browser screenshot (15 credits, ignores light_request, returns a base64-encoded image).
        - "sort_by": "most_recent" / "price_low_to_high" / "price_high_to_low" / "average_review" / "bestsellers" / "featured" - Sorting order (availability depends on category).
        - "start_page": 2 - First result page to fetch (default: 1).
        - "tag": "my-label" - Arbitrary label returned in response headers; does not affect scraping.
        - "zip_code": "90210" - Postal or ZIP code for geolocation to see local delivery information and availability.

        Cost: 5 credits per light-request page, 15 per rendered page.
        """
    )
    args_schema: Type[BaseModel] = AmazonSearchInput
    api_key: str

    def _run(self, query: str, params: Optional[Dict[str, Any]] = None,
             results_folder: str = "scraping_results", return_content: bool = False) -> str:
        params = params or {}
        api_url = "https://app.scrapingbee.com/api/v1/amazon/search"
        request_params = {'query': query, **params}

        try:
            response = requests.get(api_url, params=request_params, headers={'User-Agent': 'LangChain', 'Authorization': f'Bearer {self.api_key}'}, timeout=120)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return f"Error during Amazon Search API call: {getattr(e.response, 'text', str(e))}"

        folder_path = create_results_folder(results_folder)
        filename = f"amazon_search_{sanitize_filename(query)}.json"
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

        save_scraping_metadata(folder_path, f"amazon_search:{query}", params, "search_results", filename=filename)

        try:
            results = json.loads(response.text)
            result_count = results.get("products_count", len(results.get("products", [])))
        except json.JSONDecodeError:
            result_count = "unknown (invalid JSON)"

        base_response = (f"Amazon search complete:\n"
                         f"Query: \"{query}\"\n"
                         f"Results: {result_count}\n"
                         f"Saved to: {file_path}")

        if return_content:
            return f"{base_response}\n\nCONTENT:\n{response.text}"
        else:
            return base_response

# ======================================================================================
# Tool 5: Amazon Product API
# ======================================================================================

class AmazonProductInput(BaseModel):
    """Input model for the Amazon Product tool."""
    query: str = Field(description="The ASIN (Amazon Standard Identification Number) of the product")
    params: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional parameters for the Amazon Product API"
    )
    results_folder: Optional[str] = Field(
        default="scraping_results",
        description="Base folder to save results"
    )
    return_content: Optional[bool] = Field(
        default=False,
        description="Whether to return the product data in the response"
    )

    @field_validator('params', mode='before')
    @classmethod
    def validate_params(cls, v: Any) -> Any:
        return str_to_dict_validator(v)

class AmazonProductTool(BaseTool):
    """
    Retrieves detailed information for a specific Amazon product using its ASIN.
    All results are automatically saved to a JSON file.
    """
    name: str = "amazon_product"
    description: str = (
        "Retrieves detailed information for a specific Amazon product using its ASIN "
        "(the 10-character identifier in an Amazon /dp/ASIN URL). "
        "All results are automatically saved to a JSON file."
        """
        SUPPORTED PARAMS:
        - "add_html": true/false - Include the full HTML of the product page in the JSON response (default: false).
        - "autoselect_variant": true/false - If the main variant is unavailable, automatically select an available one (default: false). Not listed in the current API docs for this endpoint, but accepted by the API.
        - "country": "us" / "gb" / "de" etc. - Two-letter country code for geolocation. Do NOT set it to the same country as the selected domain (e.g., country=fr with domain=fr returns 400); use zip_code instead.
        - "currency": "USD" / "GBP" / "EUR" etc. - Three-letter currency code (ISO 4217) to display prices in (conversion may be unavailable for some domains/products).
        - "device": "desktop" / "mobile" / "tablet" - Device type to simulate for the request (default: desktop).
        - "domain": "com" / "co.uk" / "de" etc. - The Amazon top-level domain to use for the request (default: com).
        - "language": "en" / "es" / "fr" / "de" / "it" / "ja" etc. - Language code for the request to get results in a specific language.
        - "light_request": true/false - Perform a light, faster request. Set to false to force a full JavaScript render which may yield more data, such as reviews (default: true).
        - "screenshot": true - Force a browser screenshot (15 credits, ignores light_request, returns a base64-encoded image).
        - "tag": "my-label" - Arbitrary label returned in response headers; does not affect scraping.
        - "zip_code": "90210" - Postal or ZIP code for geolocation to see local delivery information and availability.

        Cost: 5 credits for a light request, 15 for a rendered request.
        """
    )
    args_schema: Type[BaseModel] = AmazonProductInput
    api_key: str

    def _run(self, query: str, params: Optional[Dict[str, Any]] = None,
             results_folder: str = "scraping_results", return_content: bool = False) -> str:
        params = params or {}
        api_url = "https://app.scrapingbee.com/api/v1/amazon/product"
        request_params = {'query': query, **params}

        try:
            response = requests.get(api_url, params=request_params, headers={'User-Agent': 'LangChain', 'Authorization': f'Bearer {self.api_key}'}, timeout=120)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return f"Error during Amazon Product API call: {getattr(e.response, 'text', str(e))}"

        folder_path = create_results_folder(results_folder)
        filename = f"amazon_product_{sanitize_filename(query)}.json"
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

        save_scraping_metadata(folder_path, f"amazon_product:{query}", params, "product_data", filename=filename)

        base_response = (f"Amazon product data retrieved:\n"
                         f"ASIN: \"{query}\"\n"
                         f"Saved to: {file_path}")

        if return_content:
            return f"{base_response}\n\nCONTENT:\n{response.text}"
        else:
            return base_response

# ======================================================================================
# Tool 6: Walmart Search API
# ======================================================================================

class WalmartSearchInput(BaseModel):
    """Input model for the Walmart Search tool."""
    query: str = Field(description="The search query for Walmart")
    params: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional parameters for the Walmart Search API"
    )
    results_folder: Optional[str] = Field(
        default="scraping_results",
        description="Base folder to save results"
    )
    return_content: Optional[bool] = Field(
        default=False,
        description="Whether to return the search results in the response"
    )

    @field_validator('params', mode='before')
    @classmethod
    def validate_params(cls, v: Any) -> Any:
        return str_to_dict_validator(v)

class WalmartSearchTool(BaseTool):
    """
    Performs a product search on Walmart. All results are automatically saved to a JSON file.
    """
    name: str = "walmart_search"
    description: str = (
        "Performs a product search on Walmart. All results are automatically saved to a JSON file."
        """
        SUPPORTED PARAMS:
        - "add_html": true/false - Include the full HTML of the search result page in the JSON response (default: false).
        - "delivery_zip": "72716" - ZIP code to check for local delivery options and availability.
        - "device": "desktop" / "mobile" / "tablet" - Device type to simulate for the request (default: desktop).
        - "domain": "com" - Optional Walmart domain for localization.
        - "fulfillment_speed": "today" / "tomorrow" / "2_days" / "anytime" - Filter results by delivery speed.
        - "fulfillment_type": "in_store" - Filter results to show only items available for in-store pickup.
        - "light_request": true/false - Perform a light, faster request. Set to false to force a full JavaScript render which may yield more data (default: true).
        - "max_price": integer - Filter results by a maximum price.
        - "min_price": integer - Filter results by a minimum price.
        - "screenshot": true - Force a browser screenshot (15 credits, ignores light_request, returns a base64-encoded image).
        - "sort_by": "best_match" / "price_low" / "price_high" / "best_seller" - Sorting order for results (default: best_match).
        - "start_page": 2 - Result page to fetch (default: 1; one page per request, no multi-page aggregation).
        - "store_id": "1234" - A specific Walmart store ID to check for inventory and pricing (store IDs: https://www.scrapingbee.com/download/walmart_stores.json).
        - "tag": "my-label" - Arbitrary label returned in response headers; does not affect scraping.

        Cost: 10 credits for a light request, 15 for a regular request.
        """
    )
    args_schema: Type[BaseModel] = WalmartSearchInput
    api_key: str

    def _run(self, query: str, params: Optional[Dict[str, Any]] = None,
             results_folder: str = "scraping_results", return_content: bool = False) -> str:
        params = params or {}
        api_url = "https://app.scrapingbee.com/api/v1/walmart/search"
        request_params = {'query': query, **params}

        try:
            response = requests.get(api_url, params=request_params, headers={'User-Agent': 'LangChain', 'Authorization': f'Bearer {self.api_key}'}, timeout=120)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return f"Error during Walmart Search API call: {getattr(e.response, 'text', str(e))}"

        folder_path = create_results_folder(results_folder)
        filename = f"walmart_search_{sanitize_filename(query)}.json"
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

        save_scraping_metadata(folder_path, f"walmart_search:{query}", params, "search_results", filename=filename)

        try:
            results = json.loads(response.text)
            result_count = len(results.get("products", []))
        except json.JSONDecodeError:
            result_count = "unknown (invalid JSON)"

        base_response = (f"Walmart search complete:\n"
                         f"Query: \"{query}\"\n"
                         f"Results: {result_count}\n"
                         f"Saved to: {file_path}")

        if return_content:
            return f"{base_response}\n\nCONTENT:\n{response.text}"
        else:
            return base_response

# ======================================================================================
# Tool 7: Walmart Product API
# ======================================================================================

class WalmartProductInput(BaseModel):
    """Input model for the Walmart Product tool."""
    product_id: str = Field(description="The ID of the Walmart product")
    params: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional parameters for the Walmart Product API"
    )
    results_folder: Optional[str] = Field(
        default="scraping_results",
        description="Base folder to save results"
    )
    return_content: Optional[bool] = Field(
        default=False,
        description="Whether to return the product data in the response"
    )

    @field_validator('params', mode='before')
    @classmethod
    def validate_params(cls, v: Any) -> Any:
        return str_to_dict_validator(v)

class WalmartProductTool(BaseTool):
    """
    Retrieves detailed information for a specific Walmart product using its ID.
    All results are automatically saved to a JSON file.
    """
    name: str = "walmart_product"
    description: str = (
        "Retrieves detailed information for a specific Walmart product using its numeric ID "
        "(the number in a Walmart /ip/ product URL, e.g., /ip/name/5491199371). "
        "All results are automatically saved to a JSON file."
        """
        SUPPORTED PARAMS:
        - "add_html": true/false - Include the full HTML of the product page in the JSON response (default: false).
        - "delivery_zip": "72716" - ZIP code to check for local delivery options and product availability. If the request errors with this param, retry with light_request=false.
        - "device": "desktop" / "mobile" / "tablet" - Device type to simulate for the request (default: desktop).
        - "domain": "com" - Optional Walmart domain for localization.
        - "light_request": true/false - Perform a light, faster request. Set to false to force a full JavaScript render which may yield more data (default: true).
        - "screenshot": true - Force a browser screenshot (15 credits, ignores light_request, returns a base64-encoded image).
        - "store_id": "1234" - A specific Walmart store ID to check for local inventory and pricing (store IDs: https://www.scrapingbee.com/download/walmart_stores.json).
        - "tag": "my-label" - Arbitrary label returned in response headers; does not affect scraping.

        Cost: 10 credits for a light request, 15 for a regular request.
        """
    )
    args_schema: Type[BaseModel] = WalmartProductInput
    api_key: str

    def _run(self, product_id: str, params: Optional[Dict[str, Any]] = None,
             results_folder: str = "scraping_results", return_content: bool = False) -> str:
        params = params or {}
        api_url = "https://app.scrapingbee.com/api/v1/walmart/product"
        request_params = {'product_id': product_id, **params}

        try:
            response = requests.get(api_url, params=request_params, headers={'User-Agent': 'LangChain', 'Authorization': f'Bearer {self.api_key}'}, timeout=120)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return f"Error during Walmart Product API call: {getattr(e.response, 'text', str(e))}"

        folder_path = create_results_folder(results_folder)
        filename = f"walmart_product_{sanitize_filename(product_id)}.json"
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

        save_scraping_metadata(folder_path, f"walmart_product:{product_id}", params, "product_data", filename=filename)

        base_response = (f"Walmart product data retrieved:\n"
                         f"Product ID: \"{product_id}\"\n"
                         f"Saved to: {file_path}")

        if return_content:
            return f"{base_response}\n\nCONTENT:\n{response.text}"
        else:
            return base_response

# ======================================================================================
# Tool 8: ChatGPT API
# ======================================================================================

class ChatGPTInput(BaseModel):
    """Input model for the ChatGPT tool."""
    prompt: str = Field(description="The prompt you want to send to the GPT model")
    params: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional parameters for the ChatGPT API"
    )
    results_folder: Optional[str] = Field(
        default="scraping_results",
        description="Base folder to save results"
    )
    return_content: Optional[bool] = Field(
        default=False,
        description="Whether to return the API response in the output"
    )

    @field_validator('params', mode='before')
    @classmethod
    def validate_params(cls, v: Any) -> Any:
        return str_to_dict_validator(v)

class ChatGPTTool(BaseTool):
    """
    Interacts with the ChatGPT API for conversational AI tasks. The response is saved to a JSON file.
    """
    name: str = "chat_gpt"
    description: str = (
        "Interacts with the ChatGPT API for conversational AI tasks, optionally with web-enhanced search. "
        "The response is saved to a JSON file and contains llm_model, prompt, results_markdown, "
        "results_text, results_json, and optional full_html."
        """
        SUPPORTED PARAMS:
        - "add_html": true/false - Include the full HTML (full_html) in the results (default: false).
        - "country_code": "us" / "gb" / "de" etc. - Two-letter country code for the web search request to get localized search results (used with search=true).
        - "search": true/false - Enable web search to provide the model with up-to-date information to answer the query (default: false).
        - "tag": "my-label" - Arbitrary label returned in response headers; does not affect the model behavior.

        Cost: 15 credits per successful request. Citations are not guaranteed; when returned they appear at the end of results_markdown.
        """
    )
    args_schema: Type[BaseModel] = ChatGPTInput
    api_key: str

    def _run(self, prompt: str, params: Optional[Dict[str, Any]] = None,
             results_folder: str = "scraping_results", return_content: bool = False) -> str:
        params = params or {}
        api_url = "https://app.scrapingbee.com/api/v1/chatgpt"
        headers = {'User-Agent': 'LangChain', 'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'}
        request_params = {'prompt': prompt, **params}

        try:
            response = requests.get(api_url, headers=headers, params=request_params, timeout=180)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return f"Error during ChatGPT API call: {getattr(e.response, 'text', str(e))}"

        folder_path = create_results_folder(results_folder)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chatgpt_response_{timestamp}.json"
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

        # For metadata, we omit the potentially very large message payload for cleaner logs
        logged_params = {**params, "prompt": "omitted for brevity"}
        save_scraping_metadata(folder_path, "chat_gpt_api", logged_params, "api_response", filename=filename)

        base_response = (f"ChatGPT API call successful:\n"
                         f"Saved to: {file_path}")

        if return_content:
            return f"{base_response}\n\nCONTENT:\n{response.text}"
        else:
            return base_response

# ======================================================================================
# Tool 9: YouTube Metadata API
# ======================================================================================

class YouTubeMetadataInput(BaseModel):
    """Input model for the YouTube Metadata tool."""
    video_id: str = Field(description="YouTube video identifier (e.g., 'dQw4w9WgXcQ')")
    results_folder: Optional[str] = Field(
        default="scraping_results",
        description="Base folder to save results"
    )
    return_content: Optional[bool] = Field(
        default=False,
        description="Whether to return the metadata in the response"
    )


class YouTubeMetadataTool(BaseTool):
    """
    Retrieves metadata for a YouTube video including title, description, view count,
    likes, channel info, publish date, and more. All results are automatically saved to a JSON file.
    """
    name: str = "youtube_metadata"
    description: str = (
        "Retrieves comprehensive metadata for a YouTube video using its video ID. "
        "Returns information such as title, description, view count, likes, channel info, "
        "publish date, duration, thumbnails, tags, and more. "
        "All results are automatically saved to a JSON file. "
        "Use return_content=True to get the metadata in the response for analysis. "
        """
        REQUIRED PARAMS:
        - "video_id": The YouTube video identifier (the part after 'v=' in a YouTube URL, e.g., 'dQw4w9WgXcQ')

        NOTES:
        - Cost: 5 credits per request

        EXAMPLE USAGE:
        - To get metadata for https://www.youtube.com/watch?v=dQw4w9WgXcQ, use video_id="dQw4w9WgXcQ"
        """
    )
    args_schema: Type[BaseModel] = YouTubeMetadataInput
    api_key: str

    def _run(self, video_id: str, results_folder: str = "scraping_results",
             return_content: bool = False) -> str:
        api_url = "https://app.scrapingbee.com/api/v1/youtube/metadata"
        request_params = {'video_id': video_id}

        try:
            response = requests.get(api_url, params=request_params,
                                    headers={'User-Agent': 'LangChain', 'Authorization': f'Bearer {self.api_key}'}, timeout=120)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return f"Error during YouTube Metadata API call: {getattr(e.response, 'text', str(e))}"

        folder_path = create_results_folder(results_folder)
        filename = f"youtube_metadata_{sanitize_filename(video_id)}.json"
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

        save_scraping_metadata(folder_path, f"youtube_metadata:{video_id}",
                               {"video_id": video_id}, "video_metadata", filename=filename)

        # Try to extract key info for summary
        try:
            data = json.loads(response.text)
            title = data.get("title", "Unknown")
            channel = data.get("channel", {}).get("name", "Unknown")
            views = data.get("view_count", "Unknown")
            summary = f"Title: \"{title}\"\nChannel: {channel}\nViews: {views}"
        except json.JSONDecodeError:
            summary = "Video ID: " + video_id

        base_response = (f"YouTube metadata retrieved:\n"
                         f"{summary}\n"
                         f"Saved to: {file_path}")

        if return_content:
            return f"{base_response}\n\nCONTENT:\n{response.text}"
        else:
            return base_response


# ======================================================================================
# Tool 10: YouTube Search API
# ======================================================================================

class YouTubeSearchInput(BaseModel):
    """Input model for the YouTube Search tool."""
    search: str = Field(description="The search terms for YouTube search")
    params: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="""Optional parameters for the YouTube Search API. Must be a proper dictionary/object.
            Examples:
            {"sort_by": "view_count", "hd": true}
            {"type": "video", "duration": "4-20", "upload_date": "this_week"}
            {"live": true, "subtitles": true}"""
    )
    results_folder: Optional[str] = Field(
        default="scraping_results",
        description="Base folder to save results"
    )
    return_content: Optional[bool] = Field(
        default=False,
        description="Whether to return the search results in the response"
    )

    @field_validator('params', mode='before')
    @classmethod
    def validate_params(cls, v: Any) -> Any:
        return str_to_dict_validator(v)


class YouTubeSearchTool(BaseTool):
    """
    Performs a search on YouTube and retrieves video, channel, or playlist results.
    Supports extensive filtering options for duration, quality, upload date, and more.
    All results are automatically saved to a JSON file.
    """
    name: str = "youtube_search"
    description: str = (
        "Performs a search on YouTube and retrieves video, channel, or playlist results. "
        "Supports extensive filtering options including video quality, duration, upload date, "
        "and content type. All results are automatically saved to a JSON file. "
        "Use return_content=True to get the results in the response for analysis. "
        """
        SUPPORTED PARAMS:
        - "360": true/false - Return only 360-degree videos (default: false)
        - "3d": true/false - Return only 3D videos (default: false)
        - "4k": true/false - Return only 4K videos (default: false)
        - "creative_commons": true/false - Return only videos with Creative Commons license (default: false)
        - "duration": "<4" / "4-20" / ">20" - Filter by video duration in minutes
        - "hd": true/false - Return only HD videos (default: false)
        - "hdr": true/false - Return only HDR videos (default: false)
        - "live": true/false - Return only live streams (default: false)
        - "location": true/false - Return only videos with location metadata (default: false)
        - "purchased": true/false - Return only purchased content (default: false)
        - "sort_by": "rating" / "relevance" / "view_count" / "upload_date" - Sorting method (default: "relevance")
        - "subtitles": true/false - Return only videos with subtitles/closed captions (default: false)
        - "type": "video" / "channel" / "playlist" / "movie" - Result type to return
        - "upload_date": "today" / "last_hour" / "this_week" / "this_month" / "this_year" - Filter by upload date
        - "vr180": true/false - Return only VR180 videos (default: false)

        NOTES:
        - Cost: 5 credits per successful request
        - Results contain raw YouTube renderer data; treat the structure as YouTube-controlled and variable
        - Page, country, and language controls are not supported for this endpoint

        EXAMPLE USAGE:
        - Search for recent HD cooking tutorials: {"search": "cooking tutorial", "params": {"hd": true, "upload_date": "this_month", "type": "video"}}
        - Find live gaming streams: {"search": "gaming", "params": {"live": true, "sort_by": "view_count"}}
        """
    )
    args_schema: Type[BaseModel] = YouTubeSearchInput
    api_key: str

    def _run(self, search: str, params: Optional[Dict[str, Any]] = None,
             results_folder: str = "scraping_results", return_content: bool = False) -> str:
        params = params or {}
        api_url = "https://app.scrapingbee.com/api/v1/youtube/search"
        request_params = {'search': search, **params}

        try:
            response = requests.get(api_url, params=request_params,
                                    headers={'User-Agent': 'LangChain', 'Authorization': f'Bearer {self.api_key}'}, timeout=120)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return f"Error during YouTube Search API call: {getattr(e.response, 'text', str(e))}"

        folder_path = create_results_folder(results_folder)
        filename = f"youtube_search_{sanitize_filename(search)}.json"
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

        save_scraping_metadata(folder_path, f"youtube_search:{search}", params,
                               "search_results", filename=filename)

        # Try to count results
        try:
            results = json.loads(response.text)
            # Handle different possible response structures
            if isinstance(results, list):
                result_count = len(results)
            elif isinstance(results, dict):
                result_count = len(results.get("results", results.get("videos", results.get("items", []))))
            else:
                result_count = "unknown"
        except json.JSONDecodeError:
            result_count = "unknown (invalid JSON)"

        base_response = (f"YouTube search complete:\n"
                         f"Query: \"{search}\"\n"
                         f"Results: {result_count}\n"
                         f"Filters: {params if params else 'None'}\n"
                         f"Saved to: {file_path}")

        if return_content:
            return f"{base_response}\n\nCONTENT:\n{response.text}"
        else:
            return base_response


# ======================================================================================
# Tool 12: YouTube Subtitles API
# ======================================================================================

class YouTubeSubtitlesInput(BaseModel):
    """Input model for the YouTube Subtitles tool."""
    video_id: str = Field(description="YouTube video identifier (e.g., 'dQw4w9WgXcQ')")
    params: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="""Optional parameters for the YouTube Subtitles API. Must be a proper dictionary/object.
            Examples:
            {"language": "es"}
            {"subtitle_origin": "uploader_provided"}
            {"language": "fr", "subtitle_origin": "auto_generated"}"""
    )
    results_folder: Optional[str] = Field(
        default="scraping_results",
        description="Base folder to save results"
    )
    return_content: Optional[bool] = Field(
        default=False,
        description="Whether to return the subtitles in the response"
    )

    @field_validator('params', mode='before')
    @classmethod
    def validate_params(cls, v: Any) -> Any:
        return str_to_dict_validator(v)


class YouTubeSubtitlesTool(BaseTool):
    """
    Retrieves the subtitles/captions for a YouTube video. Supports multiple languages
    and can fetch either auto-generated or uploader-provided subtitles.
    All results are automatically saved to a JSON file.
    """
    name: str = "youtube_subtitles"
    description: str = (
        "Retrieves the subtitles/captions/transcript for a YouTube video. Supports multiple languages "
        "and can fetch either auto-generated or uploader-provided subtitles. "
        "Useful for content analysis, accessibility, translation, and search indexing. "
        "All results are automatically saved to a JSON file. "
        "Use return_content=True to get the subtitles in the response for analysis. "
        """
        SUPPORTED PARAMS:
        - "language": ISO language code for the subtitles
            Examples: "en" (English), "es" (Spanish), "fr" (French), "de" (German),
                      "ja" (Japanese), "ko" (Korean), "zh" (Chinese), "pt" (Portuguese)
        - "subtitle_origin": "auto_generated" / "uploader_provided" - Choose the source of the subtitles
            - "auto_generated": YouTube's automatic speech recognition captions
            - "uploader_provided": Captions uploaded by the video creator (usually more accurate)

        NOTES:
        - Cost: 5 credits per request
        - A requested language with no matching subtitles returns a 404 error
        - The response separates subtitles.auto_generated and subtitles.uploader_provided, keyed by
          language, with timestamped text runs (start_ms, d_duration_ms, snippet.runs[].text)

        EXAMPLE USAGE:
        - Get subtitles: {"video_id": "dQw4w9WgXcQ"}
        - Get Spanish uploader-provided subtitles: {"video_id": "dQw4w9WgXcQ", "params": {"language": "es", "subtitle_origin": "uploader_provided"}}
        """
    )
    args_schema: Type[BaseModel] = YouTubeSubtitlesInput
    api_key: str

    def _run(self, video_id: str, params: Optional[Dict[str, Any]] = None,
             results_folder: str = "scraping_results", return_content: bool = False) -> str:
        params = params or {}
        # Backward compatibility with the old transcript parameter name
        if "transcript_origin" in params and "subtitle_origin" not in params:
            params["subtitle_origin"] = params.pop("transcript_origin")

        api_url = "https://app.scrapingbee.com/api/v1/youtube/subtitles"
        request_params = {'video_id': video_id, **params}

        try:
            response = requests.get(api_url, params=request_params,
                                    headers={'User-Agent': 'LangChain', 'Authorization': f'Bearer {self.api_key}'}, timeout=120)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return f"Error during YouTube Subtitles API call: {getattr(e.response, 'text', str(e))}"

        folder_path = create_results_folder(results_folder)
        language = params.get("language", "default")
        origin = params.get("subtitle_origin", "any")
        filename = f"youtube_subtitles_{sanitize_filename(video_id)}_{language}_{origin}.json"
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

        save_scraping_metadata(folder_path, f"youtube_subtitles:{video_id}",
                               {"video_id": video_id, **params}, "subtitles", filename=filename)

        # Try to get subtitle info for summary
        try:
            data = json.loads(response.text)
            if isinstance(data, dict) and isinstance(data.get("subtitles"), dict):
                subtitles = data["subtitles"]
                auto_langs = sorted((subtitles.get("auto_generated") or {}).keys())
                uploader_langs = sorted((subtitles.get("uploader_provided") or {}).keys())
                summary = (f"Video ID: {video_id}\n"
                           f"Auto-generated languages: {', '.join(auto_langs) if auto_langs else 'none'}\n"
                           f"Uploader-provided languages: {', '.join(uploader_langs) if uploader_langs else 'none'}")
            else:
                summary = f"Video ID: {video_id}"
        except json.JSONDecodeError:
            summary = f"Video ID: {video_id}"

        base_response = (f"YouTube subtitles retrieved:\n"
                         f"{summary}\n"
                         f"Saved to: {file_path}")

        if return_content:
            return f"{base_response}\n\nCONTENT:\n{response.text}"
        else:
            return base_response


# Backward-compatible aliases for the pre-1.0.0 transcript naming
YouTubeTranscriptInput = YouTubeSubtitlesInput
YouTubeTranscriptTool = YouTubeSubtitlesTool


# ======================================================================================
# Tool 13: Fast Search API
# ======================================================================================

class FastSearchInput(BaseModel):
    """Input model for the Fast Search tool."""
    search: str = Field(description="The search query text to send to Google")
    params: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="""Optional parameters dictionary for the Fast Search API. Must be a proper dictionary/object.
            Examples:
            {"country_code": "gb", "language": "en"}
            {"page": 2}"""
    )
    results_folder: Optional[str] = Field(
        default="scraping_results",
        description="Base folder to save results"
    )
    return_content: Optional[bool] = Field(
        default=False,
        description="Whether to return the search results in the response. Must be set to True if the agent needs to read the contents."
    )

    @field_validator('params', mode='before')
    @classmethod
    def validate_params(cls, v: Any) -> Any:
        return str_to_dict_validator(v)


class FastSearchTool(BaseTool):
    """
    Performs a lightweight, low-latency Google search optimized for sub-second responses.
    All results are automatically saved to a JSON file.
    """
    name: str = "fast_search"
    description: str = (
        "Performs a lightweight, low-latency Google search optimized for sub-second responses. "
        "Returns organic results (rank, title, link, description, extensions) and top_stories. "
        "Cheaper and faster than the full Google Search API, but without detailed SERP types, "
        "multi-page aggregation, or AI Overviews - use google_search when those are needed. "
        "All results are automatically saved to a JSON file. "
        "Use return_content=True to get the results back for analysis."
        """
        SUPPORTED PARAMS:
        - "country_code": "us" / "gb" / "de" etc. - ISO 3166-1 alpha-2 code to localize results (default: "us")
        - "language": "en" / "es" / "fr" etc. - Language of the results (default: "en")
        - "page": 2 - Page number for pagination (default: 1)
        - "tag": "my-label" - Arbitrary label returned in response headers; does not affect scraping

        Cost: 10 credits per successful request.
        """
    )
    args_schema: Type[BaseModel] = FastSearchInput
    api_key: str

    def _run(self, search: str, params: Optional[Dict[str, Any]] = None,
             results_folder: str = "scraping_results", return_content: bool = False) -> str:
        params = params or {}
        api_url = "https://app.scrapingbee.com/api/v1/fast_search"
        request_params = {'search': search, **params}

        try:
            response = requests.get(api_url, params=request_params, headers={'User-Agent': 'LangChain', 'Authorization': f'Bearer {self.api_key}'}, timeout=120)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return f"Error during Fast Search API call: {getattr(e.response, 'text', str(e))}"

        folder_path = create_results_folder(results_folder)
        filename = f"fast_search_{sanitize_filename(search)}.json"
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

        save_scraping_metadata(folder_path, f"fast_search:{search}", params, "search_results", filename=filename)

        try:
            results = json.loads(response.text)
            result_count = len(results.get("organic", []))
        except json.JSONDecodeError:
            result_count = "unknown (invalid JSON)"

        base_response = (f"Fast search complete:\n"
                         f"Query: \"{search}\"\n"
                         f"Organic results: {result_count}\n"
                         f"Saved to: {file_path}")

        if return_content:
            return f"{base_response}\n\nCONTENT:\n{response.text}"
        else:
            return base_response


# ======================================================================================
# Tool 14: Amazon Pricing API
# ======================================================================================

class AmazonPricingInput(BaseModel):
    """Input model for the Amazon Pricing tool."""
    asin: str = Field(description="The 10-character ASIN (Amazon Standard Identification Number) of the product")
    params: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="""Optional parameters for the Amazon Pricing API. Must be a proper dictionary/object.
            Examples:
            {"domain": "co.uk", "currency": "GBP"}
            {"zip_code": "90210", "light_request": false}"""
    )
    results_folder: Optional[str] = Field(
        default="scraping_results",
        description="Base folder to save results"
    )
    return_content: Optional[bool] = Field(
        default=False,
        description="Whether to return the pricing data in the response"
    )

    @field_validator('params', mode='before')
    @classmethod
    def validate_params(cls, v: Any) -> Any:
        return str_to_dict_validator(v)


class AmazonPricingTool(BaseTool):
    """
    Retrieves current pricing and offers for a specific Amazon product using its ASIN.
    All results are automatically saved to a JSON file.
    """
    name: str = "amazon_pricing"
    description: str = (
        "Retrieves current pricing and offers for a specific Amazon product using its ASIN "
        "(the 10-character identifier in an Amazon /dp/ASIN URL). "
        "Each offer can include condition, price/currency, shipping price, seller info, "
        "delivery text, and delivery options. All results are automatically saved to a JSON file. "
        "Use return_content=True to get the results back for analysis."
        """
        SUPPORTED PARAMS:
        - "add_html": true/false - Include the full HTML of the page in the JSON response (default: false).
        - "country": "us" / "gb" / "de" etc. - Two-letter country code for geolocation. Do NOT set it to the same country as the selected domain (e.g., country=fr with domain=fr returns 400); use zip_code instead.
        - "currency": "USD" / "GBP" / "EUR" etc. - Three-letter currency code (ISO 4217) to display prices in (conversion may be unavailable for some domains/products).
        - "device": "desktop" - Only desktop is documented for this endpoint.
        - "domain": "com" / "co.uk" / "de" etc. - The Amazon top-level domain to use for the request (default: com).
        - "language": "en" / "es" / "fr" etc. - Language code for the request.
        - "light_request": true/false - Perform a light, faster request. Set to false to force a full JavaScript render which may yield more data (default: true).
        - "tag": "my-label" - Arbitrary label returned in response headers; does not affect scraping.
        - "zip_code": "90210" - Postal or ZIP code for geolocation to see local availability, shipping, and regional prices.

        Cost: 5 credits for a light request, 15 for a rendered request.
        """
    )
    args_schema: Type[BaseModel] = AmazonPricingInput
    api_key: str

    def _run(self, asin: str, params: Optional[Dict[str, Any]] = None,
             results_folder: str = "scraping_results", return_content: bool = False) -> str:
        params = params or {}
        api_url = "https://app.scrapingbee.com/api/v1/amazon/pricing"
        request_params = {'asin': asin, **params}

        try:
            response = requests.get(api_url, params=request_params, headers={'User-Agent': 'LangChain', 'Authorization': f'Bearer {self.api_key}'}, timeout=120)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return f"Error during Amazon Pricing API call: {getattr(e.response, 'text', str(e))}"

        folder_path = create_results_folder(results_folder)
        filename = f"amazon_pricing_{sanitize_filename(asin)}.json"
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

        save_scraping_metadata(folder_path, f"amazon_pricing:{asin}", params, "pricing_data", filename=filename)

        try:
            results = json.loads(response.text)
            offer_count = len(results.get("pricing", []))
        except json.JSONDecodeError:
            offer_count = "unknown (invalid JSON)"

        base_response = (f"Amazon pricing data retrieved:\n"
                         f"ASIN: \"{asin}\"\n"
                         f"Offers: {offer_count}\n"
                         f"Saved to: {file_path}")

        if return_content:
            return f"{base_response}\n\nCONTENT:\n{response.text}"
        else:
            return base_response


# ======================================================================================
# Tool 15: Gemini API
# ======================================================================================

class GeminiInput(BaseModel):
    """Input model for the Gemini tool."""
    prompt: str = Field(description="The prompt you want to send to Gemini")
    params: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional parameters for the Gemini API"
    )
    results_folder: Optional[str] = Field(
        default="scraping_results",
        description="Base folder to save results"
    )
    return_content: Optional[bool] = Field(
        default=False,
        description="Whether to return the API response in the output"
    )

    @field_validator('params', mode='before')
    @classmethod
    def validate_params(cls, v: Any) -> Any:
        return str_to_dict_validator(v)


class GeminiTool(BaseTool):
    """
    Interacts with the Gemini API for conversational AI tasks. The response is saved to a JSON file.
    """
    name: str = "gemini"
    description: str = (
        "Interacts with the Gemini API for conversational AI tasks. "
        "The response is saved to a JSON file and contains prompt, results_markdown, "
        "results_text, citations, and optional full_html. "
        "Citations are not guaranteed and are more likely for web-grounded answers; when present, "
        "each citation has title, url, description, and text."
        """
        SUPPORTED PARAMS:
        - "add_html": true/false - Include the full HTML (full_html) in the results (default: false).
        - "country_code": "us" / "gb" / "de" etc. - Two-letter country code that sets the request geolocation.
        - "tag": "my-label" - Arbitrary label returned in response headers; does not affect the model behavior.

        Cost: 15 credits per successful request. Unlike chat_gpt, there is no "search" parameter.
        """
    )
    args_schema: Type[BaseModel] = GeminiInput
    api_key: str

    def _run(self, prompt: str, params: Optional[Dict[str, Any]] = None,
             results_folder: str = "scraping_results", return_content: bool = False) -> str:
        params = params or {}
        api_url = "https://app.scrapingbee.com/api/v1/gemini"
        headers = {'User-Agent': 'LangChain', 'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'}
        request_params = {'prompt': prompt, **params}

        try:
            response = requests.get(api_url, headers=headers, params=request_params, timeout=180)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return f"Error during Gemini API call: {getattr(e.response, 'text', str(e))}"

        folder_path = create_results_folder(results_folder)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gemini_response_{timestamp}.json"
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

        # For metadata, we omit the potentially very large message payload for cleaner logs
        logged_params = {**params, "prompt": "omitted for brevity"}
        save_scraping_metadata(folder_path, "gemini_api", logged_params, "api_response", filename=filename)

        base_response = (f"Gemini API call successful:\n"
                         f"Saved to: {file_path}")

        if return_content:
            return f"{base_response}\n\nCONTENT:\n{response.text}"
        else:
            return base_response
