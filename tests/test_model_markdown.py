"""Pipe-table splitting. No network."""

from samyak.model.markdown import pipe_tables, unwrap_blockquote_pipe_row


def test_ordinary_pipe_table_columns_stay_aligned() -> None:
    text = "\n".join(
        [
            "| Shutdown date | Model / system | Recommended replacement |",
            "| --- | --- | --- |",
            "| Dec 11, 2026 | `gpt-4.5-preview` | `gpt-5.6-sol` |",
        ]
    )
    tables = pipe_tables(text)
    assert len(tables) == 1
    assert tables[0] == (
        {
            "Shutdown date": "Dec 11, 2026",
            "Model / system": "`gpt-4.5-preview`",
            "Recommended replacement": "`gpt-5.6-sol`",
        },
    )


def test_escaped_pipe_stays_inside_the_cell() -> None:
    text = "\n".join(
        [
            "| Shutdown date | Model / system | Recommended replacement |",
            "| --- | --- | --- |",
            "| Dec 11, 2026 | `gpt-5` \\| `gpt-5-2025-08-07` | `gpt-5.6-sol` |",
        ]
    )
    row = pipe_tables(text)[0][0]
    assert row["Shutdown date"] == "Dec 11, 2026"
    assert row["Model / system"] == "`gpt-5` \\| `gpt-5-2025-08-07`"
    assert row["Recommended replacement"] == "`gpt-5.6-sol`"


def test_escaped_backslash_does_not_escape_the_following_pipe() -> None:
    text = "\n".join(
        [
            "| Left | Right |",
            "| --- | --- |",
            "| keep \\\\ | next |",
        ]
    )
    row = pipe_tables(text)[0][0]
    assert row["Left"] == "keep \\\\"
    assert row["Right"] == "next"


def test_raw_blockquote_is_not_an_ordinary_pipe_table() -> None:
    text = "\n".join(
        [
            "> | Model | Context |",
            "> | --- | --- |",
            "> | `fireworks/qwen3-embedding-8b` | 40k |",
        ]
    )
    assert pipe_tables(text) == ()


def test_blockquote_pipe_row_unwraps_one_and_nested_markers() -> None:
    assert unwrap_blockquote_pipe_row("> | Model | ID |") == "| Model | ID |"
    assert unwrap_blockquote_pipe_row("> > | Model | ID |") == "| Model | ID |"
    prose = "> The label | Model | is not a row."
    assert unwrap_blockquote_pipe_row(prose) == prose
    ordinary = "| Model | ID |"
    assert unwrap_blockquote_pipe_row(ordinary) == ordinary


def test_unwrapped_blockquote_table_keeps_headers_and_escaped_pipes() -> None:
    text = "\n".join(
        unwrap_blockquote_pipe_row(line)
        for line in (
            "> | Model | Snapshot |",
            "> | --- | --- |",
            "> | `gpt-5` \\| `gpt-5-2025-08-07` | keep > inside |",
        )
    )
    row = pipe_tables(text)[0][0]
    assert row == {
        "Model": "`gpt-5` \\| `gpt-5-2025-08-07`",
        "Snapshot": "keep > inside",
    }


def test_cell_whitespace_is_stripped_and_columns_stay_aligned() -> None:
    text = "\n".join(
        [
            "| Model | Replacement |",
            "| --- | --- |",
            "|  gpt-5  |  gpt-5.6-sol  |",
        ]
    )
    row = pipe_tables(text)[0][0]
    assert row == {"Model": "gpt-5", "Replacement": "gpt-5.6-sol"}
