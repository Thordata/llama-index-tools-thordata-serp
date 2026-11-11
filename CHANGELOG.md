# Changelog

## 0.2.0

- Retrieve failed-task diagnostics once within the remaining wait budget and expose safe upstream reasons and `service_code` without resubmitting the task.
- Treat Amazon `page_turning: null` like an omitted page count, matching LangChain, and document an explicit single-page example and timeout handling.
- Send `api_source=sdk` and `source_platform=llama` as Web Scraper POST form fields.
- Use English throughout package text and documentation; redirect the former localized tutorial to the English edition.
- Allow Amazon category requests to omit `page_turning` and use the server default, while validating explicitly supplied page counts.
- Accept `Average Rating` and `Featured Recommendations` as aliases for the existing Amazon sort values.
- Add the synchronous Web Scraper tool for the seven supported Thordata spiders.
- Preserve long business fields while redacting credentials from records.
- Parse concatenated JSON array chunks returned by Web Scraper downloads.
- Add a larger Web Scraper response budget, safe integer validation, and deadline timeout errors.
- Add regression coverage for long records, concatenated downloads, response limits, and timeout classification.

## 0.1.0

- Add schema-driven Thordata SERP tools for LlamaIndex.
