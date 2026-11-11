import json
from urllib.parse import parse_qs

import pytest
import responses

from llama_index.tools.thordata_serp import ThordataSerpToolSpec
from llama_index.tools.thordata_serp.web_scraper import (
    WEB_SCRAPER_BUILDER_URL,
    WEB_SCRAPER_DOWNLOAD_URL,
    WEB_SCRAPER_TASK_API_BASE_URL,
    validate_web_scraper_params,
)


SPIDER = "amazon_product_by-category-url"
URL = "https://www.amazon.com/s?i=baby-products-intl-ship"
OMITTED = object()


@pytest.mark.parametrize(
    "sort_input, expected_sort",
    [
        (None, "Best Sellers"),
        ("Best Sellers", "Best Sellers"),
        ("Newest Arrivals", "Newest Arrivals"),
        ("Avg. Customer Review", "Avg. Customer Review"),
        ("Price: High to Low", "Price: High to Low"),
        ("Price: Low to High", "Price: Low to High"),
        ("Featured", "Featured"),
        ("Average Rating", "Avg. Customer Review"),
        ("Featured Recommendations", "Featured"),
    ],
)
@pytest.mark.parametrize("pages", [OMITTED, None, 1, 2], ids=["omitted", "null", "one-page", "two-pages"])
@responses.activate
def test_amazon_tool_posts_compatible_parameters(sort_input, expected_sort, pages):
    task_id = "amazon-compatibility-task"
    download = f"{WEB_SCRAPER_DOWNLOAD_URL}?task_id={task_id}&type=json"
    responses.add(
        responses.POST, WEB_SCRAPER_BUILDER_URL,
        json={"code": 200, "data": {"task_id": task_id}},
    )
    responses.add(
        responses.POST, f"{WEB_SCRAPER_TASK_API_BASE_URL}/tasks-status",
        json={"code": 200, "data": [{"task_id": task_id, "status": "Ready"}]},
    )
    responses.add(
        responses.POST, f"{WEB_SCRAPER_TASK_API_BASE_URL}/tasks-download",
        json={"code": 200, "data": {"tasks_ids": task_id, "download": download}},
    )
    responses.add(responses.GET, download, json=[{"title": "Baby product"}])
    params = {"url": URL}
    if sort_input is not None:
        params["sort_by"] = sort_input
    if pages is not OMITTED:
        params["page_turning"] = pages
    original = params.copy()

    output = json.loads(ThordataSerpToolSpec("test-api-key").scrape(SPIDER, params))

    assert output["ok"] is True
    assert output["records"] == [{"title": "Baby product"}]
    assert params == original
    assert len(responses.calls) == 4
    request = responses.calls[0].request
    body = request.body.decode() if isinstance(request.body, bytes) else request.body
    form = parse_qs(body)
    expected = {"url": URL, "sort_by": expected_sort}
    if pages is not OMITTED and pages is not None:
        expected["page_turning"] = pages
    assert json.loads(form["spider_parameters"][0]) == [expected]
    assert form["api_source"] == ["sdk"]
    assert "api-source" not in form
    assert form["source_platform"] == ["llama"]
    assert "api-source" not in request.headers
    assert "api_source" not in request.headers
    assert "source_platform" not in request.headers


@pytest.mark.parametrize("pages, expected", [(1, 1), ("2", 2), (9007199254740991, 9007199254740991)])
def test_amazon_explicit_page_count_preserves_integer_validation(pages, expected):
    assert validate_web_scraper_params(SPIDER, {"url": URL, "page_turning": pages}) == {
        "url": URL, "page_turning": expected, "sort_by": "Best Sellers",
    }


@pytest.mark.parametrize(
    "extra",
    [
        {"page_turning": ""},
        {"page_turning": " "},
        {"page_turning": 0},
        {"page_turning": -1},
        {"page_turning": 1.5},
        {"page_turning": True},
        {"page_turning": "abc"},
        {"page_turning": 9007199254740992},
        {"sort_by": "Other"},
        {"sort_by": "average rating"},
        {"sort_by": []},
        {"unexpected": "field"},
    ],
)
@responses.activate
def test_amazon_invalid_parameters_are_rejected_before_submission(extra):
    result = json.loads(ThordataSerpToolSpec("test-api-key").scrape(SPIDER, {"url": URL, **extra}))

    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_parameters"
    assert result["error"]["stage"] == "validate"
    assert len(responses.calls) == 0
