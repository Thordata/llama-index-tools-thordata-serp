from llama_index.core.tools.tool_spec.base import BaseToolSpec

from llama_index.tools.thordata_serp import ThordataSerpToolSpec, __version__


def test_public_import_contract():
    assert issubclass(ThordataSerpToolSpec, BaseToolSpec)
    assert ThordataSerpToolSpec.spec_functions == [
        "list_engines",
        "get_engine_schema",
        "search",
    ]
    assert __version__ == "0.1.0"
