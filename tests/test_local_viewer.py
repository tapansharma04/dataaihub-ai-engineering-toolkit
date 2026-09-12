"""Tests for the local run-history HTTP viewer and related CLI."""

from __future__ import annotations

import http.client
import json
import socket
import threading
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from http.client import HTTPResponse
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from helpers import make_corpus
from samyak import analyze_corpus
from samyak.cli import main
from samyak.corpus.report import render_html_report, render_json_report, render_text_report
from samyak.server.app import ViewerBindError, create_server, viewer_url
from samyak.server.dashboard import render_dashboard, wrap_report_html
from samyak.store.filesystem import FileRunStore
from samyak.store.models import RunMetadata, RunPage


class _HTMLStructureParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.tags: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)


def _analyze(tmp_path: Path):
    corpus = make_corpus(
        tmp_path,
        {"doc.txt": "Viewer test document with enough content about warehouse logistics.\n" * 3},
    )
    return analyze_corpus(corpus), corpus


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_server(store: FileRunStore):
    server = create_server(store, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _stop_server(server, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _get(url: str) -> HTTPResponse:
    return urlopen(url, timeout=5)


def test_server_binds_to_loopback(tmp_path: Path) -> None:
    store = FileRunStore(root=tmp_path / "cache")
    server = create_server(store, host="127.0.0.1", port=0)
    try:
        host, port = server.server_address[:2]
        assert host == "127.0.0.1"
        assert server.socket.getsockname()[0] == "127.0.0.1"
        assert port != 0
    finally:
        server.server_close()


def test_port_unavailable_fails_cleanly(tmp_path: Path) -> None:
    store = FileRunStore(root=tmp_path / "cache")
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    try:
        port = int(blocker.getsockname()[1])
        with pytest.raises(ViewerBindError, match="could not bind"):
            create_server(store, host="127.0.0.1", port=port)
    finally:
        blocker.close()


def test_empty_history_index(tmp_path: Path) -> None:
    store = FileRunStore(root=tmp_path / "cache")
    server, thread = _start_server(store)
    try:
        with _get(viewer_url(server)) as response:
            html = response.read().decode("utf-8")
            assert response.status == 200
        assert "Run History" in html
        assert "Local Samyak workspace" in html
        assert "Data stored on this machine" in html
        assert "No saved runs yet" in html
        assert "https://" not in html
        assert "http://" not in html
        parser = _HTMLStructureParser()
        parser.feed(html)
        assert "table" not in parser.tags
    finally:
        _stop_server(server, thread)


def test_index_lists_runs_and_report_route(tmp_path: Path) -> None:
    report, _ = _analyze(tmp_path)
    store = FileRunStore(root=tmp_path / "cache")
    meta = store.save_run(report, duration_seconds=1.5)
    server, thread = _start_server(store)
    try:
        base = viewer_url(server).rstrip("/")
        with _get(base + "/runs") as response:
            index = response.read().decode("utf-8")
        assert meta.run_id in index
        assert str(meta.files_discovered) in index
        assert str(meta.files_analyzed) in index
        assert str(meta.findings_count) in index
        assert str(meta.load_errors) in index
        assert str(meta.discovery_errors) in index
        assert "completed" in index
        assert f"/runs/{meta.run_id}" in index
        assert "https://" not in index
        assert "cdn" not in index.lower()

        with _get(base + f"/runs/{meta.run_id}") as response:
            page = response.read().decode("utf-8")
        expected = render_html_report(report)
        assert expected.startswith("<!DOCTYPE html>")
        assert "Corpus Intelligence Report" in page
        assert "← Run History" in page
        assert "Local workspace" in page
        assert "Corpus path" not in page
        for finding in report.findings:
            assert finding.code in page
            assert finding.title in page
        assert report.summary.corpus_root in page
        assert str(tmp_path) not in page
    finally:
        _stop_server(server, thread)


def test_missing_run_returns_clean_not_found(tmp_path: Path) -> None:
    store = FileRunStore(root=tmp_path / "cache")
    server, thread = _start_server(store)
    try:
        url = viewer_url(server).rstrip("/") + "/runs/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        with pytest.raises(HTTPError) as exc:
            urlopen(url, timeout=5)
        assert exc.value.code == 404
        body = exc.value.read().decode("utf-8")
        assert "Not found" in body
        assert "Traceback" not in body
        assert "<script>" not in body
    finally:
        _stop_server(server, thread)


def test_corrupt_run_does_not_hide_healthy_run(tmp_path: Path) -> None:
    report, _ = _analyze(tmp_path)
    store = FileRunStore(root=tmp_path / "cache")
    healthy = store.save_run(report, duration_seconds=0.8)
    bad = tmp_path / "cache" / "runs" / "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    bad.mkdir(parents=True)
    (bad / "metadata.json").write_text("{nope", encoding="utf-8")
    server, thread = _start_server(store)
    try:
        with _get(viewer_url(server).rstrip("/") + "/runs") as response:
            index = response.read().decode("utf-8")
        assert healthy.run_id in index
        assert "could not be read" in index
        with pytest.raises(HTTPError) as exc:
            urlopen(
                viewer_url(server).rstrip("/") + "/runs/bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                timeout=5,
            )
        assert exc.value.code == 409
        assert "could not be read" in exc.value.read().decode("utf-8")
    finally:
        _stop_server(server, thread)


def test_dashboard_escapes_labels() -> None:
    run_id = "cccccccc-cccc-cccc-cccc-cccccccccccc"
    page = RunPage(
        runs=(
            RunMetadata(
                run_id=run_id,
                created_at="2026-01-01T00:00:00+00:00",
                completed_at="2026-01-01T00:00:01+00:00",
                duration_seconds=1.0,
                samyak_version="0.1.0",
                report_schema_version=1,
                status="completed",
                capability="corpus",
                corpus_label='<script>alert("x")</script>',
                findings_count=1,
                files_discovered=1,
                files_analyzed=1,
                load_errors=0,
                discovery_errors=0,
            ),
        ),
        total=1,
        offset=0,
        limit=50,
    )
    html = render_dashboard(page)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert 'alert("x")' not in html
    assert "table-wrap" in html
    assert "overflow-x:auto" in html
    assert "overflow-x:hidden" in html


def test_history_pagination_links(tmp_path: Path) -> None:
    report, _ = _analyze(tmp_path)
    store = FileRunStore(root=tmp_path / "cache")
    base = datetime(2026, 3, 1, tzinfo=UTC)
    for index in range(51):
        store.save_run(
            report,
            duration_seconds=index,
            created_at=base + timedelta(seconds=index),
            completed_at=base + timedelta(seconds=index),
        )
    server, thread = _start_server(store)
    try:
        with _get(viewer_url(server).rstrip("/") + "/runs") as response:
            first = response.read().decode("utf-8")
        assert "Older runs" in first
        assert "/runs?offset=50" in first
        assert "Showing 1–50 of 51" in first
        with _get(viewer_url(server).rstrip("/") + "/runs?offset=50") as response:
            second = response.read().decode("utf-8")
        assert "Newer runs" in second
        assert "Showing 51–51 of 51" in second
        conn = http.client.HTTPConnection(*server.server_address[:2], timeout=5)
        try:
            conn.request("GET", "/?offset=50")
            redirected = conn.getresponse()
            redirected.read()
            assert redirected.status == 302
            assert redirected.getheader("Location") == "/runs?offset=50"
        finally:
            conn.close()
    finally:
        _stop_server(server, thread)


def test_existing_html_renderer_has_no_viewer_chrome(tmp_path: Path) -> None:
    report, _ = _analyze(tmp_path)
    html = render_html_report(report)
    assert "← Run History" not in html
    assert "sv-nav" not in html
    assert html.startswith("<!DOCTYPE html>")


def test_cli_save_keeps_json_stdout_clean(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], samyak_cache_dir: Path
) -> None:
    report, corpus = _analyze(tmp_path)
    expected = render_json_report(report)
    code = main(["corpus", str(corpus), "--format", "json", "--save", "--no-progress"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == expected
    json.loads(captured.out)
    assert "Saved run" in captured.err
    assert "samyak view" in captured.err
    assert "Saved run" not in captured.out
    payload = json.loads(captured.out)
    assert "corpus_path" not in payload
    assert "corpus_path" not in payload.get("summary", {})
    page = FileRunStore(root=samyak_cache_dir).list_runs()
    assert page.total == 1
    stored = FileRunStore(root=samyak_cache_dir).get_run(page.runs[0].run_id)
    assert stored.report.to_dict() == report.to_dict()
    assert stored.metadata.corpus_label == corpus.name
    assert stored.metadata.corpus_path is not None
    assert stored.metadata.corpus_path.endswith(corpus.name)
    assert stored.metadata.corpus_path not in captured.out
    report_file = (samyak_cache_dir / "runs" / stored.metadata.run_id / "report.json").read_text(
        encoding="utf-8"
    )
    assert stored.metadata.corpus_path not in report_file


def test_cli_without_save_does_not_persist(
    tmp_path: Path, samyak_cache_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, corpus = _analyze(tmp_path)
    code = main(["corpus", str(corpus), "--format", "json", "--no-progress"])
    captured = capsys.readouterr()
    assert code == 0
    json.loads(captured.out)
    assert captured.err == ""
    assert FileRunStore(root=samyak_cache_dir).list_runs().runs == ()


def test_cli_save_keeps_text_stdout_unchanged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report, corpus = _analyze(tmp_path)
    expected = render_text_report(report)
    code = main(["corpus", str(corpus), "--format", "text", "--save", "--no-progress"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == expected
    assert "Saved run" in captured.err


def test_cli_view_no_open_and_port(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    samyak_cache_dir: Path,
) -> None:
    opened: list[str] = []
    monkeypatch.setattr("samyak.server.app.webbrowser.open", opened.append)

    def stop_immediately(self) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("samyak.server.app.ViewerServer.serve_forever", stop_immediately)
    port = _free_port()
    code = main(["view", "--port", str(port), "--no-open"])
    captured = capsys.readouterr()
    assert code == 0
    assert opened == []
    assert f"127.0.0.1:{port}" in captured.out
    assert "Local Samyak workspace" in captured.out


def test_cli_view_opens_browser_by_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    opened: list[str] = []
    monkeypatch.setattr("samyak.server.app.webbrowser.open", opened.append)

    def stop_immediately(self) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("samyak.server.app.ViewerServer.serve_forever", stop_immediately)
    port = _free_port()
    code = main(["view", "--port", str(port)])
    assert code == 0
    assert len(opened) == 1
    assert opened[0].startswith("http://127.0.0.1:")
    assert str(port) in opened[0]


def test_cli_view_port_in_use(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("samyak.server.app.webbrowser.open", lambda _url: None)
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    try:
        port = int(blocker.getsockname()[1])
        code = main(["view", "--port", str(port), "--no-open"])
        captured = capsys.readouterr()
        assert code == 2
        assert "error:" in captured.err
        assert "Traceback" not in captured.err
        assert str(port) in captured.err
    finally:
        blocker.close()


def test_cli_view_invalid_port(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["view", "--port", "0", "--no-open"])
    captured = capsys.readouterr()
    assert code == 2
    assert "error:" in captured.err
    assert "Traceback" not in captured.err


def test_cli_view_help() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["view", "--help"])
    assert exc.value.code == 0


def _sample_metadata(
    *,
    corpus_label: str = "sample-corpus",
    corpus_path: str | None = "/tmp/example/sample-corpus",
    status: str = "completed",
    run_id: str = "cccccccc-cccc-cccc-cccc-cccccccccccc",
) -> RunMetadata:
    return RunMetadata(
        run_id=run_id,
        created_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:00:01+00:00",
        duration_seconds=1.0,
        samyak_version="0.1.0",
        report_schema_version=1,
        status=status,
        capability="corpus",
        corpus_label=corpus_label,
        findings_count=1,
        files_discovered=1,
        files_analyzed=1,
        load_errors=0,
        discovery_errors=0,
        corpus_path=corpus_path,
    )


def test_index_shows_corpus_name_not_full_path() -> None:
    path = "/Users/demo/projects/foo/examples/sample-corpus"
    page = RunPage(
        runs=(_sample_metadata(corpus_label="sample-corpus", corpus_path=path),),
        total=1,
        offset=0,
        limit=50,
    )
    html = render_dashboard(page)
    assert "sample-corpus" in html
    assert path not in html
    assert "Corpus path" not in html
    assert "<th>Corpus</th>" in html
    assert "Compare selected runs" not in html
    detail = wrap_report_html("<body>report</body>", page.runs[0])
    assert "Corpus path" in detail
    assert path in detail
    assert ">Corpus<" in detail or "Corpus</dt>" in detail


def test_viewer_escapes_corpus_path_and_status() -> None:
    meta = _sample_metadata(
        corpus_label="<script>alert(1)</script>",
        corpus_path='javascript:alert(1)/<img src=x onerror=alert(1)>/"quoted"',
        status='completed"><script>x</script>',
        run_id="cccccccc-cccc-cccc-cccc-cccccccccccc",
    )
    index = render_dashboard(RunPage(runs=(meta,), total=1, offset=0, limit=50))
    detail = wrap_report_html("<!DOCTYPE html><body></body>", meta)
    for html in (index, detail):
        assert "<script>" not in html
        assert "<img src=x" not in html
        assert "&lt;script&gt;" in html
    assert "&lt;img" in detail
    assert "<img src=x onerror=alert(1)>" not in detail
    assert 'href="javascript:' not in detail
    assert "javascript:alert(1)" in detail


def test_cli_save_and_output_together(
    tmp_path: Path, samyak_cache_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, corpus = _analyze(tmp_path)
    output = tmp_path / "written.json"
    code = main(
        [
            "corpus",
            str(corpus),
            "--format",
            "json",
            "--save",
            "--no-progress",
            "-o",
            str(output),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == ""
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["product"] == "samyak"
    assert "corpus_path" not in payload
    stored = FileRunStore(root=samyak_cache_dir).list_runs()
    assert stored.total == 1
    assert stored.runs[0].corpus_path is not None


def test_cli_output_failure_after_successful_save(
    tmp_path: Path, samyak_cache_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, corpus = _analyze(tmp_path)
    code = main(
        [
            "corpus",
            str(corpus),
            "--format",
            "json",
            "--save",
            "--no-progress",
            "-o",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "failed to write output" in captured.err
    assert FileRunStore(root=samyak_cache_dir).list_runs().total == 1


def test_cli_save_failure_still_writes_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, corpus = _analyze(tmp_path)
    output = tmp_path / "out.json"

    def boom(self, *args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr("samyak.store.filesystem.FileRunStore.save_run", boom)
    code = main(
        [
            "corpus",
            str(corpus),
            "--format",
            "json",
            "--save",
            "--no-progress",
            "-o",
            str(output),
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "failed to save run" in captured.err
    json.loads(output.read_text(encoding="utf-8"))


def test_cli_html_save_keeps_stdout_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report, corpus = _analyze(tmp_path)
    expected = render_html_report(report)
    code = main(["corpus", str(corpus), "--format", "html", "--save", "--no-progress"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == expected
    assert "Saved run" in captured.err
    assert "← Run History" not in captured.out


def test_without_save_html_and_json_omit_machine_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, corpus = _analyze(tmp_path)
    abs_path = str(corpus.absolute())
    for fmt in ("json", "html"):
        code = main(["corpus", str(corpus), "--format", fmt, "--no-progress"])
        captured = capsys.readouterr()
        assert code == 0
        assert abs_path not in captured.out
        assert captured.err == ""


def test_path_traversal_and_direct_file_requests_are_rejected(tmp_path: Path) -> None:
    report, _ = _analyze(tmp_path)
    store = FileRunStore(root=tmp_path / "cache")
    meta = store.save_run(report, duration_seconds=0.2)
    server, thread = _start_server(store)
    try:
        host, port = server.server_address[:2]
        rejected = [
            "/runs/../etc/passwd",
            "/runs/%2e%2e/%2e%2e/etc/passwd",
            "/runs/%2e%2e%2fetc/passwd",
            f"/runs/{meta.run_id}/report.json",
            f"/runs/{meta.run_id}%2freport.json",
            f"/runs/{meta.run_id}%5creport.json",
            "/runs/not-a-uuid",
            "/compare/../etc/passwd",
            "/compare/%2e%2e/etc/passwd",
            "/models/../etc/passwd",
            "/models/%2e%2e/etc/passwd",
            "/favicon.ico",
        ]
        for suffix in rejected:
            conn = http.client.HTTPConnection(host, port, timeout=5)
            try:
                conn.request("GET", suffix)
                response = conn.getresponse()
                body = response.read().decode("utf-8")
            finally:
                conn.close()
            assert response.status == 404, suffix
            assert "Traceback" not in body
        with _get(viewer_url(server).rstrip("/") + f"/runs/{meta.run_id}") as response:
            assert response.status == 200
    finally:
        _stop_server(server, thread)


def test_head_matches_get_without_body(tmp_path: Path) -> None:
    store = FileRunStore(root=tmp_path / "cache")
    server, thread = _start_server(store)
    try:
        url = viewer_url(server)
        with _get(url) as get_response:
            get_len = get_response.headers.get("Content-Length")
        request = Request(url, method="HEAD")
        with urlopen(request, timeout=5) as head_response:
            assert head_response.status == 200
            assert head_response.read() == b""
            assert head_response.headers.get("Content-Length") == get_len
    finally:
        _stop_server(server, thread)


def test_create_server_rejects_non_loopback(tmp_path: Path) -> None:
    store = FileRunStore(root=tmp_path / "cache")
    with pytest.raises(ViewerBindError, match="127.0.0.1"):
        create_server(store, host="0.0.0.0", port=0)


def test_inconsistent_run_opens_as_error_not_mixed_report(tmp_path: Path) -> None:
    report, corpus = _analyze(tmp_path)
    store = FileRunStore(root=tmp_path / "cache")
    meta = store.save_run(report, duration_seconds=0.3, corpus_path=str(corpus.absolute()))
    data = json.loads(
        (tmp_path / "cache" / "runs" / meta.run_id / "metadata.json").read_text(encoding="utf-8")
    )
    data["files_analyzed"] = data["files_analyzed"] + 9
    (tmp_path / "cache" / "runs" / meta.run_id / "metadata.json").write_text(
        json.dumps(data), encoding="utf-8"
    )
    server, thread = _start_server(store)
    try:
        with _get(viewer_url(server).rstrip("/") + "/runs") as response:
            index = response.read().decode("utf-8")
        assert meta.run_id not in index
        assert "could not be read" in index
        with pytest.raises(HTTPError) as exc:
            urlopen(viewer_url(server).rstrip("/") + f"/runs/{meta.run_id}", timeout=5)
        assert exc.value.code == 409
        body = exc.value.read().decode("utf-8")
        assert "could not be read" in body
        assert "Traceback" not in body
        assert str(tmp_path / "cache") not in body
    finally:
        _stop_server(server, thread)


def test_malicious_filename_is_escaped_in_viewer(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "<img src=x onerror=alert(1)>.txt").write_text("", encoding="utf-8")
    (corpus / "ok.txt").write_text(
        "A normal document with enough content about billing policies.\n" * 3,
        encoding="utf-8",
    )
    report = analyze_corpus(corpus)
    store = FileRunStore(root=tmp_path / "cache")
    meta = store.save_run(report, duration_seconds=0.2)
    server, thread = _start_server(store)
    try:
        with _get(viewer_url(server).rstrip("/") + f"/runs/{meta.run_id}") as response:
            html = response.read().decode("utf-8")
        assert "<img src=x onerror=alert(1)>" not in html
        assert "&lt;img src=x onerror=alert(1)&gt;" in html
    finally:
        _stop_server(server, thread)


def test_viewer_escapes_persisted_finding_fields(tmp_path: Path) -> None:
    report, _ = _analyze(tmp_path)
    store = FileRunStore(root=tmp_path / "cache")
    meta = store.save_run(report, duration_seconds=0.2)
    run_dir = tmp_path / "cache" / "runs" / meta.run_id
    payload = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    malicious = {
        "code": "<script>alert(1)</script>",
        "category": "injection",
        "severity": "HIGH",
        "title": "<img src=x onerror=alert(1)>",
        "message": 'javascript:alert(1) & < > "',
        "why_it_matters": "<script>alert(1)</script>",
        "recommendation": "<img src=x onerror=alert(1)>",
        "evidence": {"note": "<script>alert(1)</script>"},
        "affected_documents": ["<img src=x onerror=alert(1)>.txt"],
    }
    payload["findings"] = [malicious]
    (run_dir / "report.json").write_text(json.dumps(payload), encoding="utf-8")
    data = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    data["findings_count"] = 1
    (run_dir / "metadata.json").write_text(json.dumps(data), encoding="utf-8")
    server, thread = _start_server(store)
    try:
        with _get(viewer_url(server).rstrip("/") + f"/runs/{meta.run_id}") as response:
            html = response.read().decode("utf-8")
        assert response.status == 200
        assert "<script>" not in html
        assert "<img src=x onerror=alert(1)>" not in html
        assert "&lt;script&gt;" in html
        assert "&lt;img src=x onerror=alert(1)&gt;" in html
        assert 'href="javascript:' not in html
        assert "javascript:alert(1)" in html
    finally:
        _stop_server(server, thread)


def _two_runs(tmp_path: Path):
    report, corpus = _analyze(tmp_path)
    store = FileRunStore(root=tmp_path / "cache")
    t0 = datetime(2026, 6, 1, tzinfo=UTC)
    first = store.save_run(
        report,
        duration_seconds=0.2,
        created_at=t0,
        completed_at=t0,
        corpus_path=str(corpus.absolute()),
        corpus_label=corpus.name,
    )
    second = store.save_run(
        report,
        duration_seconds=0.4,
        created_at=t0 + timedelta(seconds=12),
        completed_at=t0 + timedelta(seconds=12),
        corpus_path=str(corpus.absolute()),
        corpus_label=corpus.name,
    )
    return store, first, second, corpus


def test_history_index_offers_compare_for_two_runs(tmp_path: Path) -> None:
    store, first, second, corpus = _two_runs(tmp_path)
    server, thread = _start_server(store)
    try:
        with _get(viewer_url(server).rstrip("/") + "/runs") as response:
            html = response.read().decode("utf-8")
        assert 'action="/compare"' in html
        assert "Compare selected runs" in html
        assert f'value="{first.run_id}"' in html
        assert f'value="{second.run_id}"' in html
        assert str(corpus.absolute()) not in html
        assert "Select exactly two runs" in html
    finally:
        _stop_server(server, thread)


def test_compare_route_renders_summary(tmp_path: Path) -> None:
    store, first, second, corpus = _two_runs(tmp_path)
    server, thread = _start_server(store)
    try:
        base = viewer_url(server).rstrip("/")
        url = f"{base}/compare?left={first.run_id}&right={second.run_id}"
        with _get(url) as response:
            html = response.read().decode("utf-8")
            assert response.status == 200
        assert "Run comparison" in html
        assert first.run_id in html
        assert second.run_id in html
        assert "Baseline" in html
        assert "Current" in html
        assert "Older run" not in html
        assert "Newer run" not in html
        assert "Files discovered" in html
        assert "Finding changes" in html
        assert "Unchanged findings" in html
        assert corpus.name in html
        assert "Cross-corpus" not in html
        assert str(tmp_path / "cache") not in html
        assert "<script>" not in html
        assert "https://" not in html
        run_form = f"{base}/compare?run={second.run_id}&run={first.run_id}"
        with _get(run_form) as response:
            again = response.read().decode("utf-8")
        assert "Baseline" in again
        assert first.run_id in again
        start = again.index("Baseline")
        current_at = again.index("Current", start)
        assert first.run_id in again[start:current_at]
        swapped = f"{base}/compare?left={second.run_id}&right={first.run_id}"
        with _get(swapped) as response:
            explicit = response.read().decode("utf-8")
        start = explicit.index("Baseline")
        current_at = explicit.index("Current", start)
        assert second.run_id in explicit[start:current_at]
        assert "Supported files" in html
    finally:
        _stop_server(server, thread)


def test_compare_requires_exactly_two_distinct_runs(tmp_path: Path) -> None:
    store, first, _, _ = _two_runs(tmp_path)
    server, thread = _start_server(store)
    try:
        base = viewer_url(server).rstrip("/")
        cases = [
            f"{base}/compare",
            f"{base}/compare?left={first.run_id}",
            f"{base}/compare?left={first.run_id}&right={first.run_id}",
            f"{base}/compare?run={first.run_id}",
            (
                f"{base}/compare?run={first.run_id}&run={first.run_id}"
                "&run=cccccccc-cccc-cccc-cccc-cccccccccccc"
            ),
        ]
        for url in cases:
            with pytest.raises(HTTPError) as exc:
                urlopen(url, timeout=5)
            assert exc.value.code == 400
            body = exc.value.read().decode("utf-8")
            assert "Cannot compare" in body
            assert "Traceback" not in body
            assert str(tmp_path / "cache") not in body
    finally:
        _stop_server(server, thread)


def test_compare_missing_and_malformed_ids(tmp_path: Path) -> None:
    store, first, _, _ = _two_runs(tmp_path)
    server, thread = _start_server(store)
    try:
        base = viewer_url(server).rstrip("/")
        urls = [
            f"{base}/compare?left={first.run_id}&right=aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            f"{base}/compare?left=not-a-uuid&right={first.run_id}",
            f"{base}/compare?left=../etc/passwd&right={first.run_id}",
        ]
        for url in urls:
            with pytest.raises(HTTPError) as exc:
                urlopen(url, timeout=5)
            assert exc.value.code == 404
            body = exc.value.read().decode("utf-8")
            assert "Not found" in body
            assert "Traceback" not in body
    finally:
        _stop_server(server, thread)


def test_compare_rejects_corrupt_run(tmp_path: Path) -> None:
    store, first, second, _ = _two_runs(tmp_path)
    data = json.loads(
        (tmp_path / "cache" / "runs" / second.run_id / "metadata.json").read_text(encoding="utf-8")
    )
    data["findings_count"] = data["findings_count"] + 4
    (tmp_path / "cache" / "runs" / second.run_id / "metadata.json").write_text(
        json.dumps(data), encoding="utf-8"
    )
    server, thread = _start_server(store)
    try:
        url = viewer_url(server).rstrip("/") + f"/compare?left={first.run_id}&right={second.run_id}"
        with pytest.raises(HTTPError) as exc:
            urlopen(url, timeout=5)
        assert exc.value.code == 409
        body = exc.value.read().decode("utf-8")
        assert "could not be read" in body
        assert "Files discovered" not in body
        assert str(tmp_path / "cache") not in body
        assert "Traceback" not in body
    finally:
        _stop_server(server, thread)


def test_compare_escapes_finding_and_path_fields(tmp_path: Path) -> None:
    report, _ = _analyze(tmp_path)
    store = FileRunStore(root=tmp_path / "cache")
    t0 = datetime(2026, 7, 1, tzinfo=UTC)
    first = store.save_run(
        report,
        duration_seconds=0.1,
        created_at=t0,
        completed_at=t0,
        corpus_path="/<img src=x onerror=alert(1)>/<script>alert(1)</script>",
        corpus_label="<script>alert(1)</script>",
    )
    second = store.save_run(
        report,
        duration_seconds=0.2,
        created_at=t0 + timedelta(seconds=3),
        completed_at=t0 + timedelta(seconds=3),
        corpus_path="javascript:alert(1)",
        corpus_label="docs",
    )
    payload = json.loads(
        (tmp_path / "cache" / "runs" / second.run_id / "report.json").read_text(encoding="utf-8")
    )
    payload["findings"] = [
        {
            "code": "<script>alert(1)</script>",
            "category": "injection",
            "severity": "HIGH",
            "title": "<img src=x onerror=alert(1)>",
            "message": "javascript:alert(1)",
            "why_it_matters": "n",
            "recommendation": "<script>alert(1)</script>",
            "evidence": {"count": 1, "affected_document_count": 1},
            "affected_documents": ["<img src=x onerror=alert(1)>.txt"],
        }
    ]
    (tmp_path / "cache" / "runs" / second.run_id / "report.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    meta = json.loads(
        (tmp_path / "cache" / "runs" / second.run_id / "metadata.json").read_text(encoding="utf-8")
    )
    meta["findings_count"] = 1
    (tmp_path / "cache" / "runs" / second.run_id / "metadata.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )
    server, thread = _start_server(store)
    try:
        url = viewer_url(server).rstrip("/") + f"/compare?left={first.run_id}&right={second.run_id}"
        with _get(url) as response:
            html = response.read().decode("utf-8")
        assert "<script>" not in html
        assert "<img src=x onerror=alert(1)>" not in html
        assert "&lt;script&gt;" in html
        assert "&lt;img" in html
        assert 'href="javascript:' not in html
        assert "javascript:alert(1)" in html
        assert "Different saved corpus locations" in html
    finally:
        _stop_server(server, thread)
