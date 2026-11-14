"""Python AST extraction for Application Readiness."""

from __future__ import annotations

from pathlib import Path

from helpers import make_project
from samyak.app.extractors.python_ast import extract_python_observations
from samyak.app.models import NOTICE_FILE_PARSE_FAILED, EvidenceType
from samyak.app.observations import ObservationKind
from samyak.app.pipeline import extract_application


def _refs(source: str, path: str = "app.py"):
    return [
        item
        for item in extract_python_observations(path, source)
        if item.kind is ObservationKind.MODEL_REFERENCE
    ]


def test_direct_model_keyword() -> None:
    source = "client.create(model='some-model')\n"
    refs = _refs(source)
    assert len(refs) == 1
    assert refs[0].location.detected_value == "some-model"
    assert refs[0].location.config_key == "model"
    assert refs[0].location.evidence_type is EvidenceType.PYTHON_KEYWORD
    assert refs[0].location.line_start == 1
    assert refs[0].location.line_end == 1
    assert refs[0].location.symbol == "create"
    assert refs[0].provider_id is None


def test_model_id_keyword() -> None:
    source = "client.run(model_id='abc-id')\n"
    refs = _refs(source)
    assert len(refs) == 1
    assert refs[0].location.detected_value == "abc-id"
    assert refs[0].location.config_key == "model_id"


def test_messages_create_model() -> None:
    source = "client.messages.create(model='some-model')\n"
    refs = _refs(source)
    assert refs[0].location.detected_value == "some-model"
    assert refs[0].location.symbol == "create"


def test_dictionary_model_slots() -> None:
    source = "config = {'model': 'dict-model', 'model_id': 'dict-id', 'model_name': 'dict-name'}\n"
    refs = _refs(source)
    values = {item.location.config_key: item.location.detected_value for item in refs}
    assert values == {
        "model": "dict-model",
        "model_id": "dict-id",
        "model_name": "dict-name",
    }
    assert all(item.location.evidence_type is EvidenceType.PYTHON_LITERAL for item in refs)
    assert all(item.location.line_start == 1 for item in refs)


def test_exact_ast_line_numbers_on_indented_kwargs() -> None:
    source = "client.create(\n    model='lined-model',\n)\n"
    refs = _refs(source)
    assert refs[0].location.line_start == 2
    assert refs[0].location.line_end == 2


def test_comments_do_not_produce_observations() -> None:
    source = "# client.create(model='from-comment')\nvalue = 1\n"
    assert _refs(source) == []


def test_docstrings_do_not_produce_observations() -> None:
    source = '''"""client.create(model='from-docstring')"""

def f():
    """model='also-doc'"""
    return 1
'''
    assert _refs(source) == []


def test_same_file_constant_resolves_to_model_reference() -> None:
    source = "MODEL = 'some-model'\nclient.create(model=MODEL)\n"
    refs = _refs(source)
    assert len(refs) == 1
    assert refs[0].location.detected_value == "some-model"
    assert refs[0].expression == "MODEL"
    assert refs[0].resolution_method is not None
    assert refs[0].resolution_method.value == "same_file_constant"
    assert refs[0].resolution_source is not None
    assert refs[0].resolution_source.line_start == 1
    assert refs[0].resolution_source.detected_value == "some-model"
    assert refs[0].resolution_source.symbol == "MODEL"
    assert refs[0].location.line_start == 2
    operations = [
        item
        for item in extract_python_observations("app.py", source)
        if item.kind is ObservationKind.AI_OPERATION
    ]
    assert operations == []


def test_malformed_python_notice_does_not_abort_other_files(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {
            "ok.py": "client.create(model='kept')\n",
            "bad.py": "def (\n",
        },
    )
    extraction = extract_application(project)
    values = [
        item.location.detected_value
        for item in extraction.observations
        if item.kind is ObservationKind.MODEL_REFERENCE
    ]
    assert values == ["kept"]
    assert extraction.report.summary.parse_failures == 1
    assert any(item.code == NOTICE_FILE_PARSE_FAILED for item in extraction.report.notices)
    assert any(item.path == "bad.py" for item in extraction.report.notices)


def test_openai_import_attaches_same_file_provider_hint() -> None:
    source = (
        "from openai import OpenAI\n"
        "\n"
        "client = OpenAI()\n"
        "client.chat.completions.create(model='gpt-test')\n"
    )
    observations = extract_python_observations("bot.py", source)
    imports = [item for item in observations if item.kind is ObservationKind.PROVIDER_SDK_IMPORT]
    refs = [item for item in observations if item.kind is ObservationKind.MODEL_REFERENCE]
    assert len(imports) == 1
    assert imports[0].provider_id == "openai"
    assert imports[0].location.evidence_type is EvidenceType.PYTHON_IMPORT
    assert imports[0].location.line_start == 1
    assert refs[0].provider_id == "openai"
    assert refs[0].location.detected_value == "gpt-test"


def test_anthropic_import_attaches_same_file_provider_hint() -> None:
    source = (
        "from anthropic import Anthropic\n"
        "\n"
        "client = Anthropic()\n"
        "client.messages.create(model='claude-test')\n"
    )
    observations = extract_python_observations("bot.py", source)
    refs = [item for item in observations if item.kind is ObservationKind.MODEL_REFERENCE]
    assert refs[0].provider_id == "anthropic"


def test_no_cross_file_provider_inference(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {
            "imports.py": "from openai import OpenAI\n",
            "calls.py": "client.create(model='other-file')\n",
        },
    )
    extraction = extract_application(project)
    refs = [
        item for item in extraction.observations if item.kind is ObservationKind.MODEL_REFERENCE
    ]
    assert len(refs) == 1
    assert refs[0].location.path == "calls.py"
    assert refs[0].provider_id is None


def test_provider_hint_is_source_evidence_not_catalog() -> None:
    source = "import openai\nclient.create(model='src-only')\n"
    refs = _refs(source)
    payload = refs[0].to_dict()
    assert payload["provider_id"] == "openai"
    assert "samyak_id" not in payload
    assert "lifecycle" not in payload
    assert "catalog" not in payload
    assert refs[0].location.evidence_type is EvidenceType.PYTHON_KEYWORD


def test_conflicting_same_file_imports_do_not_attach_provider() -> None:
    source = "import openai\nimport anthropic\nclient.create(model='ambiguous')\n"
    refs = _refs(source)
    assert refs[0].provider_id is None
