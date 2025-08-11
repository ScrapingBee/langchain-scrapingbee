import os
import pytest
import tempfile
import glob
import shutil
from typing import Type

from langchain_tests.integration_tests import ToolsIntegrationTests
from langchain_scrapingbee.tools import ScrapeUrlTool, GoogleSearchTool, CheckUsageTool


class TestScrapeUrlToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> Type[ScrapeUrlTool]:
        return ScrapeUrlTool

    @property
    def tool_constructor_params(self) -> dict:
        api_key = os.environ.get('SCRAPINGBEE_API_KEY')
        if not api_key:
            pytest.skip("SCRAPINGBEE_API_KEY environment variable not set")
        return {"api_key": api_key}

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.
        """
        return {
            "url": "https://httpbin.org/json",
            "params": {},
            "headers": {},
            "results_folder": "temp_folder",
            "custom_filename": None,
            "return_content": True
        }

    def test_scrape_simple_webpage(self):
        """Test scraping a simple webpage that returns JSON"""
        tool = self.tool_constructor(**self.tool_constructor_params)
        
        with tempfile.TemporaryDirectory(prefix="scrapingbee_test_") as temp_dir:
            result = tool._run(
                url="https://httpbin.org/json",
                results_folder=temp_dir,
                return_content=True
            )
            
            assert "Text content saved and loaded" in result
            assert "slideshow" in result  # httpbin.org/json contains this field
            assert "CONTENT:" in result
            
            # Verify files were created
            created_folders = glob.glob(os.path.join(temp_dir, "*"))
            assert len(created_folders) > 0, "No folders created"

    def test_scrape_with_screenshot(self):
        """Test taking a screenshot of a webpage"""
        tool = self.tool_constructor(**self.tool_constructor_params)
        
        with tempfile.TemporaryDirectory(prefix="scrapingbee_test_") as temp_dir:
            result = tool._run(
                url="https://httpbin.org/html",
                params={"screenshot": True},
                results_folder=temp_dir,
                return_content=False
            )
            
            assert "Binary content saved successfully" in result
            assert "image/png" in result or "bytes" in result
            
            # Verify files were created
            created_folders = glob.glob(os.path.join(temp_dir, "*"))
            assert len(created_folders) > 0, "No folders created"

    def test_scrape_with_extract_rules(self):
        """Test structured data extraction using CSS selectors"""
        tool = self.tool_constructor(**self.tool_constructor_params)
        
        with tempfile.TemporaryDirectory(prefix="scrapingbee_test_") as temp_dir:
            result = tool._run(
                url="https://httpbin.org/html",
                params={
                    "extract_rules": {"title": "title", "h1": "h1"}
                },
                results_folder=temp_dir,
                return_content=True
            )
            
            assert "Text content saved and loaded" in result
            assert "Herman Melville" in result  # From httpbin.org/html
            
            # Verify files were created
            created_folders = glob.glob(os.path.join(temp_dir, "*"))
            assert len(created_folders) > 0, "No folders created"

    def test_scrape_with_custom_headers(self):
        """Test scraping with custom headers"""
        tool = self.tool_constructor(**self.tool_constructor_params)
        
        with tempfile.TemporaryDirectory(prefix="scrapingbee_test_") as temp_dir:
            result = tool._run(
                url="https://httpbin.org/headers",
                headers={"Custom-Header": "test-value"},
                results_folder=temp_dir,
                return_content=True
            )
            
            assert "Text content saved and loaded" in result
            assert "Custom-Header" in result
            
            # Verify files were created
            created_folders = glob.glob(os.path.join(temp_dir, "*"))
            assert len(created_folders) > 0, "No folders created"


class TestGoogleSearchToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> Type[GoogleSearchTool]:
        return GoogleSearchTool

    @property
    def tool_constructor_params(self) -> dict:
        api_key = os.environ.get('SCRAPINGBEE_API_KEY')
        if not api_key:
            pytest.skip("SCRAPINGBEE_API_KEY environment variable not set")
        return {"api_key": api_key}

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.
        """
        return {
            "search": "langchain python",
            "params": {"nb_results": 5},
            "results_folder": "temp_folder",
            "return_content": False
        }

    def test_google_web_search(self):
        """Test basic web search functionality"""
        tool = self.tool_constructor(**self.tool_constructor_params)
        
        with tempfile.TemporaryDirectory(prefix="temp_folder") as temp_dir:
            result = tool._run(
                search="python programming",
                params={"nb_results": 3, "search_type": "classic"},
                results_folder=temp_dir,
                return_content=True
            )
            
            assert "Search complete" in result
            assert "organic_results" in result
            assert "python" in result.lower()
            
            # Verify files were created
            created_folders = glob.glob(os.path.join(temp_dir, "*"))
            assert len(created_folders) > 0, "No folders created"

    def test_google_news_search(self):
        """Test news search functionality"""
        tool = self.tool_constructor(**self.tool_constructor_params)
        
        with tempfile.TemporaryDirectory(prefix="temp_folder") as temp_dir:
            result = tool._run(
                search="artificial intelligence",
                params={"nb_results": 3, "search_type": "news"},
                results_folder=temp_dir,
                return_content=True
            )
            
            assert "Search complete" in result
            assert "news" in result
            assert "news_results" in result or "organic_results" in result
            
            # Verify files were created
            created_folders = glob.glob(os.path.join(temp_dir, "*"))
            assert len(created_folders) > 0, "No folders created"

    def test_google_image_search(self):
        """Test image search functionality"""
        tool = self.tool_constructor(**self.tool_constructor_params)
        
        with tempfile.TemporaryDirectory(prefix="temp_folder") as temp_dir:
            result = tool._run(
                search="python logo",
                params={"nb_results": 3, "search_type": "images"},
                results_folder=temp_dir,
                return_content=False  # Images can be large
            )
            
            assert "Image search complete" in result
            assert ("base64 images" in result or "image links" in result)
            
            # Verify files were created
            created_folders = glob.glob(os.path.join(temp_dir, "*"))
            assert len(created_folders) > 0, "No folders created"

    def test_google_search_with_country_code(self):
        """Test search with specific country localization"""
        tool = self.tool_constructor(**self.tool_constructor_params)
        
        with tempfile.TemporaryDirectory(prefix="temp_folder") as temp_dir:
            result = tool._run(
                search="weather today",
                params={"nb_results": 3, "country_code": "gb", "language": "en"},
                results_folder=temp_dir,
                return_content=True
            )
            
            assert "Search complete" in result
            assert "Results:" in result
            
            # Verify files were created
            created_folders = glob.glob(os.path.join(temp_dir, "*"))
            assert len(created_folders) > 0, "No folders created"

    def test_google_maps_search(self):
        """Test maps search functionality"""
        tool = self.tool_constructor(**self.tool_constructor_params)
        
        with tempfile.TemporaryDirectory(prefix="temp_folder") as temp_dir:
            result = tool._run(
                search="restaurants near Times Square NYC",
                params={"nb_results": 3, "search_type": "maps"},
                results_folder=temp_dir,
                return_content=True
            )
            
            assert "Search complete" in result
            assert ("maps_results" in result or "organic_results" in result)
            
            # Verify files were created
            created_folders = glob.glob(os.path.join(temp_dir, "*"))
            assert len(created_folders) > 0, "No folders created"


class TestCheckUsageToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> Type[CheckUsageTool]:
        return CheckUsageTool

    @property
    def tool_constructor_params(self) -> dict:
        api_key = os.environ.get('SCRAPINGBEE_API_KEY')
        if not api_key:
            pytest.skip("SCRAPINGBEE_API_KEY environment variable not set")
        return {"api_key": api_key}

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.
        CheckUsageTool takes no parameters, so return empty dict.
        """
        return {}

    def test_check_usage_success(self):
        """Test successful usage check with real API"""
        tool = self.tool_constructor(**self.tool_constructor_params)
        
        result = tool._run()
        
        # Should contain usage information
        assert any(key in result for key in ["used", "remaining", "credits", "limit"])
        # Should be valid JSON or contain usage stats
        assert "{" in result or "credits" in result.lower()


# Additional integration tests for edge cases and real-world scenarios
class TestAdvancedIntegrationScenarios:
    """
    Advanced integration tests that test realistic usage patterns
    and edge cases with real API calls.
    """
    
    def setup_method(self):
        """Setup for each test method"""
        self.api_key = os.environ.get('SCRAPINGBEE_API_KEY')
        if not self.api_key:
            pytest.skip("SCRAPINGBEE_API_KEY environment variable not set")

    def test_scrape_javascript_heavy_site(self):
        """Test scraping a JavaScript-heavy website"""
        tool = ScrapeUrlTool(api_key=self.api_key)
        
        with tempfile.TemporaryDirectory(prefix="temp_folder") as temp_dir:
            result = tool._run(
                url="https://httpbin.org/delay/2",  # Simulates slow loading
                params={
                    "render_js": True,
                    "wait": 3000,
                    "wait_browser": "networkidle2"
                },
                results_folder=temp_dir,
                return_content=True
            )
            
            assert "Text content saved and loaded" in result
            assert len(result) > 100  # Should have substantial content

    def test_scrape_with_premium_proxy(self):
        """Test scraping with premium proxy for geo-location"""
        tool = ScrapeUrlTool(api_key=self.api_key)
        
        with tempfile.TemporaryDirectory(prefix="temp_folder") as temp_dir:
            result = tool._run(
                url="https://httpbin.org/ip",
                params={
                    "premium_proxy": True,
                    "country_code": "gb"
                },
                results_folder=temp_dir,
                return_content=True
            )
            
            assert "Text content saved and loaded" in result
            assert "origin" in result  # httpbin.org/ip returns IP info

    def test_error_handling_invalid_url(self):
        """Test error handling with invalid URL"""
        tool = ScrapeUrlTool(api_key=self.api_key)
        
        with tempfile.TemporaryDirectory(prefix="temp_folder") as temp_dir:
            result = tool._run(
                url="https://this-domain-does-not-exist-12345.com",
                results_folder=temp_dir,
                return_content=False
            )
            
            assert "Error" in result or "failed" in result.lower()

    def test_error_handling_invalid_api_key(self):
        """Test error handling with invalid API key"""
        tool = ScrapeUrlTool(api_key="invalid_api_key_123")
        
        with tempfile.TemporaryDirectory(prefix="temp_folder") as temp_dir:
            result = tool._run(
                url="https://httpbin.org/json",
                results_folder=temp_dir,
                return_content=False
            )
            
            assert "Error" in result

    def test_file_and_metadata_creation(self):
        """Test that files and metadata are actually created"""
        tool = ScrapeUrlTool(api_key=self.api_key)
        
        with tempfile.TemporaryDirectory(prefix="temp_folder") as temp_dir:
            result = tool._run(
                url="https://httpbin.org/json",
                results_folder=temp_dir,
                return_content=False
            )
            
            assert "Text content saved successfully" in result
            
            # Check that files were created
            created_folders = glob.glob(os.path.join(temp_dir, "*"))
            assert len(created_folders) > 0
            
            # Check for HTML file and metadata
            folder_path = created_folders[0]
            html_files = glob.glob(os.path.join(folder_path, "*.html"))
            metadata_files = glob.glob(os.path.join(folder_path, "*.jsonl"))
            
            assert len(html_files) > 0, "No HTML files created"
            assert len(metadata_files) > 0, "No metadata files created"

    def test_large_content_handling(self):
        """Test handling of large content responses"""
        tool = ScrapeUrlTool(api_key=self.api_key)
        
        with tempfile.TemporaryDirectory(prefix="temp_folder") as temp_dir:
            # Test with a page that returns substantial content
            result = tool._run(
                url="https://httpbin.org/stream/20",  # Returns 20 lines of JSON
                results_folder=temp_dir,
                return_content=False  # Don't return content to avoid large response
            )
            
            assert "Text content saved successfully" in result
            assert "characters" in result  # Should mention character count


# Pytest configuration for integration tests
@pytest.fixture(scope="session", autouse=True)
def check_api_key():
    """Check if API key is available before running integration tests"""
    if not os.environ.get('SCRAPINGBEE_API_KEY'):
        pytest.skip("Integration tests require SCRAPINGBEE_API_KEY environment variable", allow_module_level=True)

@pytest.fixture(autouse=True)
def cleanup_test_folders():
    """Clean up any test folders that might have been created"""
    yield
    # Cleanup after each test
    if os.path.exists("temp_folder"):
        shutil.rmtree("temp_folder")

