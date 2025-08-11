from typing import Type
import requests
from unittest.mock import Mock, patch, mock_open

from langchain_tests.unit_tests import ToolsUnitTests

# Import your tools
from langchain_scrapingbee.tools import ScrapeUrlTool, GoogleSearchTool, CheckUsageTool


class TestScrapeUrlToolUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> Type[ScrapeUrlTool]:
        return ScrapeUrlTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"api_key": "test_api_key"}

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.
        """
        return {
            "url": "https://example.com",
            "params": {"screenshot": True, "wait": 2000},
            "headers": {"User-Agent": "test-agent"},
            "results_folder": "test_results",
            "custom_filename": "test_page.png",
            "return_content": False
        }

    @patch('langchain_scrapingbee.tools.requests.get')
    @patch('langchain_scrapingbee.tools.create_results_folder')
    @patch('langchain_scrapingbee.tools.save_scraping_metadata')
    @patch('builtins.open', new_callable=mock_open)
    def test_scrape_url_binary_content(self, mock_file, mock_save_metadata, mock_create_folder, mock_requests_get):
        """Test binary content handling (screenshots)"""
        # Setup mocks
        mock_response = Mock()
        mock_response.content = b'fake_png_data'
        mock_response.headers = {'Content-Type': 'image/png'}
        mock_response.raise_for_status.return_value = None
        mock_requests_get.return_value = mock_response
        mock_create_folder.return_value = '/tmp/test_folder'

        # Create tool instance
        tool = self.tool_constructor(**self.tool_constructor_params)
        
        # Test binary content
        result = tool._run(
            url="https://example.com",
            params={"screenshot": True}
        )
        
        assert "Binary content saved successfully" in result
        mock_requests_get.assert_called_once()
        mock_file.assert_called_once()
        mock_save_metadata.assert_called_once()

    @patch('langchain_scrapingbee.tools.requests.get')
    @patch('langchain_scrapingbee.tools.create_results_folder')
    @patch('langchain_scrapingbee.tools.save_scraping_metadata')
    @patch('builtins.open', new_callable=mock_open, read_data='<html><body>Test content</body></html>')
    def test_scrape_url_text_content(self, mock_file, mock_save_metadata, mock_create_folder, mock_requests_get):
        """Test text content handling"""
        # Setup mocks
        mock_response = Mock()
        mock_response.text = '<html><body>Test content</body></html>'
        mock_response.headers = {'Content-Type': 'text/html'}
        mock_response.raise_for_status.return_value = None
        mock_requests_get.return_value = mock_response
        mock_create_folder.return_value = '/tmp/test_folder'

        # Create tool instance
        tool = self.tool_constructor(**self.tool_constructor_params)
        
        # Test text content
        result = tool._run(
            url="https://example.com",
            params={},
            return_content=True
        )
        
        assert "Text content saved and loaded" in result
        assert "Test content" in result
        mock_file.assert_called()
        mock_save_metadata.assert_called_once()


class TestGoogleSearchToolUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> Type[GoogleSearchTool]:
        return GoogleSearchTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"api_key": "test_api_key"}

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.
        """
        return {
            "search": "langchain python tutorial",
            "params": {"nb_results": 10, "country_code": "us"},
            "results_folder": "search_results",
            "return_content": False
        }

    @patch('langchain_scrapingbee.tools.requests.get')
    @patch('langchain_scrapingbee.tools.create_results_folder')
    @patch('langchain_scrapingbee.tools.save_scraping_metadata')
    @patch('builtins.open', new_callable=mock_open)
    def test_google_search_regular(self, mock_file, mock_save_metadata, mock_create_folder, mock_requests_get):
        """Test regular web search functionality"""
        # Setup mocks
        mock_response = Mock()
        mock_response.text = '{"organic_results": [{"title": "Test", "url": "https://test.com"}]}'
        mock_response.raise_for_status.return_value = None
        mock_requests_get.return_value = mock_response
        mock_create_folder.return_value = '/tmp/test_folder'

        # Create tool instance
        tool = self.tool_constructor(**self.tool_constructor_params)
        
        # Test regular search
        result = tool._run(
            search="test query",
            params={"search_type": "classic"}
        )
        
        assert "Search complete" in result
        assert "classic" in result  # Changed from "web" to "classic" to match the actual search_type
        mock_file.assert_called()
        mock_save_metadata.assert_called_once()

    @patch('langchain_scrapingbee.tools.requests.get')
    @patch('langchain_scrapingbee.tools.create_results_folder')
    @patch('langchain_scrapingbee.tools.save_scraping_metadata')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    def test_google_search_images(self, mock_makedirs, mock_file, mock_save_metadata, mock_create_folder, mock_requests_get):
        """Test image search functionality"""
        # Setup mocks
        mock_response = Mock()
        mock_response.json.return_value = {
            "images": [
                {
                    "image": "https://example.com/image1.jpg",
                    "title": "Test Image 1",
                    "position": 1
                },
                {
                    "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
                    "title": "Test Image 2",
                    "position": 2
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_requests_get.return_value = mock_response
        mock_create_folder.return_value = '/tmp/test_folder'

        # Create tool instance
        tool = self.tool_constructor(**self.tool_constructor_params)
        
        # Test image search
        result = tool._run(
            search="test images",
            params={"search_type": "images"}
        )
        
        assert "Image search complete" in result
        mock_makedirs.assert_called()
        mock_file.assert_called()
        mock_save_metadata.assert_called_once()


class TestCheckUsageToolUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> Type[CheckUsageTool]:
        return CheckUsageTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"api_key": "test_api_key"}

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.
        CheckUsageTool takes no parameters, so return empty dict.
        """
        return {}

    @patch('langchain_scrapingbee.tools.requests.get')
    def test_check_usage_success(self, mock_requests_get):
        """Test successful usage check"""
        # Setup mocks
        mock_response = Mock()
        mock_response.text = '{"used_credits": 100, "remaining_credits": 900}'
        mock_response.raise_for_status.return_value = None
        mock_requests_get.return_value = mock_response

        # Create tool instance
        tool = self.tool_constructor(**self.tool_constructor_params)
        
        # Test usage check
        result = tool._run()
        
        assert "used_credits" in result
        assert "remaining_credits" in result

    @patch('langchain_scrapingbee.tools.requests.get')
    def test_check_usage_error(self, mock_requests_get):
        """Test error handling in usage check"""
        # Setup mock to raise an exception with a response attribute
        mock_response = Mock()
        mock_response.text = "API Error Response"
        
        mock_exception = requests.exceptions.RequestException("API Error")
        mock_exception.response = mock_response
        mock_requests_get.side_effect = mock_exception

        # Create tool instance
        tool = self.tool_constructor(**self.tool_constructor_params)
        
        # Test error handling
        result = tool._run()
        
        assert "Error checking usage" in result


# Additional unit tests for utility functions
class TestUtilityFunctions:
    """Test utility functions used by the tools"""
    
    def test_sanitize_filename(self):
        """Test filename sanitization"""
        from langchain_scrapingbee.tools import sanitize_filename
        
        # Test normal URL
        result = sanitize_filename("https://example.com/page?param=value")
        assert result == "example.com_page_param_value"
        
        # Test with special characters
        result = sanitize_filename("https://test.com/file@#$%")
        assert result == "test.com_file____"
    
    def test_stringify_nested_objects(self):
        """Test parameter stringification"""
        from langchain_scrapingbee.tools import stringify_nested_objects
        
        params = {
            "simple": "value",
            "nested_dict": {"key": "value"},
            "nested_list": [1, 2, 3]
        }
        
        result = stringify_nested_objects(params)
        
        assert result["simple"] == "value"
        assert result["nested_dict"] == '{"key": "value"}'
        assert result["nested_list"] == "[1, 2, 3]"
    
    def test_str_to_dict_validator(self):
        """Test string to dictionary validation"""
        from langchain_scrapingbee.tools import str_to_dict_validator
        
        # Test JSON string
        result = str_to_dict_validator('{"key": "value"}')
        assert result == {"key": "value"}
        
        # Test empty string
        result = str_to_dict_validator('')
        assert result == {}
        
        # Test URL parameters
        result = str_to_dict_validator('screenshot=true&wait=2000')
        assert result == {"screenshot": True, "wait": 2000}
        
        # Test Python dict literal
        result = str_to_dict_validator("{'key': True}")
        assert result == {"key": True}