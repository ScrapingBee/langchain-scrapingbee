* 1.0.1 - All requests now authenticate via the recommended `Authorization: Bearer` header instead of the deprecated `api_key` query parameter
* 1.0.0 - Synced with the current ScrapingBee API surface:
  * Added `FastSearchTool` (Fast Search API), `AmazonPricingTool` (Amazon Pricing API), and `GeminiTool` (Gemini API)
  * Renamed `YouTubeTranscriptTool` to `YouTubeSubtitlesTool` and moved it to the `/youtube/subtitles` endpoint with the `subtitle_origin` parameter (`YouTubeTranscriptTool` and `transcript_origin` still work as deprecated aliases)
  * Removed `YouTubeTrainabilityTool` — the `/youtube/trainability` endpoint no longer exists
  * Refreshed tool descriptions: HTML API auto mode (`mode=auto`, `max_cost`), new Google search types (`lens`, `shopping`, `ai_mode`, `ads`) and filters, Amazon pagination/sorting/category/merchant filters, Walmart corrections, per-request costs, and `tag` support
* 0.2.0 - Added YouTube APIs — Search, Metadata, Trainability, Transcript
* 0.1.0 — Initial version
