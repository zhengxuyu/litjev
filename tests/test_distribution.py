"""Keep public examples and package branding aligned with the runnable API."""

import json
import tomllib
from pathlib import Path

from fastapi.testclient import TestClient

from litjev.api import create_app
from litjev.schema import SystemOneRequest

ROOT = Path(__file__).resolve().parents[1]


def test_readme_request_is_a_valid_schema():
    request = SystemOneRequest.model_validate_json((ROOT / "examples/request.json").read_text())
    schema = request.to_schema()
    assert schema.names == tuple(f"q{i}" for i in range(1, 11))
    assert len(schema["q1"].choices) == 3
    client = TestClient(create_app(lambda: None))
    assert client.get("/example").json() == request.model_dump()


def test_public_package_metadata_and_bundled_frontend():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["name"] == "litjev"
    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == ["LICENSE", "NOTICE", "THIRD_PARTY_LICENSES/*.txt"]
    assert project["scripts"]["litjev"] == "litjev.cli:serve"
    client = TestClient(create_app(lambda: None))
    assert "LitJev" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert json.loads(client.get("/health").text)["model_loaded"] is False


def test_attribution_and_citation_are_present():
    assert "Version 2.0, January 2004" in (ROOT / "LICENSE").read_text()
    assert "Copyright 2026 ZhengxuYu" in (ROOT / "NOTICE").read_text()
    citation = (ROOT / "CITATION.cff").read_text()
    assert 'license: Apache-2.0' in citation
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert f'version: "{project["version"]}"' in citation
    readme = (ROOT / "README.md").read_text()
    assert readme.rfind("## Citation") > readme.rfind("## License")
