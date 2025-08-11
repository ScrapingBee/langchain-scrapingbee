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
        "For non-text files, use 'render_js=false'"
        "Before running ai_query and ai_extract_rules, scrapingbee converts the HTML content to markdown. So the ai model only have access to markdown not the html"
        "EXTREMELY IMPORTANT: Must use valid parameters and instructions, do not make up non-existent parameters and instructions"
        "Can also perform Google Search with custom_google=true"
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
        - "country_code": "us"/"gb"/"de" etc. - Premium proxy location (ISO 3166)
        - "custom_google": true - Must use it when scraping Google domains (20 credits)
        - "device": "desktop"/"mobile" - Device simulation
        - "extract_rules": CSS/XPath extraction rules, example: '{"title": "h1", "links": {"selector": "a", "type": "list", "output": "@href"}}'
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
        - "forward_headers_pure": true - Forward only your specified headers
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
                - Stealth proxy limitations: No infinite_scroll, evaluate, custom headers/cookies
            ]
        - "json_response": true - Wrap response in JSON format with metadata. This can also be used to find internal xhr requests
        - "own_proxy": "protocol://user:pass@host:port" - Use your own proxy
        - "premium_proxy": true - Use premium proxy pool (10-25 credits)
        - "render_js": true/false - Enable JS rendering (default: true)
        - "return_page_source": true - Return original HTML before JS rendering
        - "scraping_config": config_name (Must only use if provided by user) - Use a pre-saved request configuration on your request
        - "screenshot": true - Screenshot of viewport
        - "screenshot_full_page": true - Full page screenshot  
        - "screenshot_selector": "css-selector" - Screenshot specific element
        - "session_id": 123 - Use persistent session (1-10000000, lasts 5 minutes)
        - "stealth_proxy": true - Use stealth proxies for difficult sites or when the previous request fails (75 credits)
        - "timeout": 60000 - Request timeout in milliseconds (1000-140000)
        - "transparent_status_code": true - Return original HTTP status codes
        - "wait": 3000 - Wait milliseconds before capture (0-35000)
        - "wait_browser": "domcontentloaded"/"load"/"networkidle0"/"networkidle2" - Browser wait condition
        - "wait_for": "css-selector" - Wait for element to appear
        - "window_height": 1080 - Viewport height in pixels
        - "window_width": 1920 - Viewport width in pixels

        UNSUPPORTED PARAMS:
        - Anything that is not mentioned in the above list

        SCRAPING STRATEGY:
        - Research Well using Google Search API: When you lack sufficient information, use Google Search API to find it first before attempting to scrape pages. You can use it multiple times before and in-between scrapes
        - Prefer AI Extraction Rules over Data Extraction rules: Use extract_rules if selector is known, otherwise use extract_rules to get the body in html to find the selector, do not use ai_query or ai_extract_rules for finding selector and do not guess selector
        - Use 
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
        
        api_url = "https://app.scrapingbee.com/api/v1/"
        request_params = {'api_key': self.api_key, 'url': url, **processed_params}
        
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
{"nb_results": 20, "language": "es"}
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
        "Performs Google searches across different search types (web, news, images, maps). "
        "All results are automatically saved to disk to conserve AI tokens. "
        "For image searches: Downloads base64 images and saves image URLs to image_links.txt. "
        "For other searches: Saves JSON results to file. "
        "Use return_content=True to return the search result in response for AI analysis when needed."
        "Use params for specific searches: {'search_type': 'news'} or location: {'country_code': 'gb'}. "
        "Supports pagination, language preferences, and result filtering. "
        "params should be a valid dictionary. "
        """
        SUPPORTED PARAMS:
        - "add_html": true - Include full HTML of result pages in response
        - "country_code": "us"/"gb"/"de" etc. - Country for localized results (default: "us")
        - "device": "desktop"/"mobile" - Device type for search (default: "desktop")
        - "extra_params": "safe=active&filter=0" - Additional Google URL parameters
        - "language": "en"/"es"/"fr" etc. - Language for results (default: "en")
        - "light_request": true/false default is true, disable it if not getting required results
        - "nb_results": 50 - Number of results to return (default: 100, max varies)
        - "nfpr": true - Exclude auto-corrected spelling results
        - "page": 2 - Page number for pagination (default: 1)
        - "search_type": "classic"/"news"/"maps"/"images" - Type of Google search (default: "classic")
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
        request_params = {'api_key': self.api_key, 'search': search, **params}
        
        try:
            response = requests.get(api_url, params=request_params, headers={'User-Agent': 'LangChain'}, timeout=120)
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
            result_count = max([len(results.get("organic_results", [])), len(results.get("news_results", [])), len(results.get("maps_results", []))])
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
        "Checks current ScrapingBee API usage statistics including remaining credits, "
        "used credits, concurrency limits, and account status. Takes no parameters."
    )
    api_key: str

    def _run(self) -> str:
        api_url = "https://app.scrapingbee.com/api/v1/usage"
        params = {'api_key': self.api_key}
        
        try:
            response = requests.get(api_url, params=params, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            error_detail = getattr(e.response, 'text', str(e)) if hasattr(e, 'response') else str(e)
            return f"Error checking usage: {error_detail}"