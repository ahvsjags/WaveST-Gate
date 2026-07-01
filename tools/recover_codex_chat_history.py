from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


CODEX_HOME = Path.home() / ".codex"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PATH = PROJECT_ROOT.resolve()
OUTPUT_ROOT = PROJECT_ROOT / "reports" / "codex_chat_recovery"


@dataclass
class TranscriptItem:
    timestamp: str
    role: str
    text: str
    line_no: int
    phase: str | None = None


@dataclass
class ToolItem:
    timestamp: str
    kind: str
    name: str
    summary: str
    line_no: int


@dataclass
class SessionRecord:
    session_id: str
    cwd: str
    source_log: Path
    created_at: str
    title: str
    preview: str
    transcript: list[TranscriptItem]
    tools: list[ToolItem]
    goals: list[dict[str, Any]]


def normalize_path(value: str) -> str:
    value = value.replace("\\\\?\\", "")
    return value.replace("/", "\\").rstrip("\\").casefold()


def is_project_cwd(cwd: str) -> bool:
    cwd_norm = normalize_path(cwd)
    root_norm = normalize_path(str(PROJECT_PATH))
    return cwd_norm == root_norm or cwd_norm.startswith(root_norm + "\\")


def safe_json_loads(line: str) -> dict[str, Any] | None:
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def iter_session_files() -> list[Path]:
    files: list[Path] = []
    for base in (CODEX_HOME / "sessions", CODEX_HOME / "archived_sessions"):
        if base.exists():
            files.extend(base.rglob("*.jsonl"))
    return sorted(files)


def redacted(text: str) -> str:
    if not text:
        return text

    patterns = [
        # SSH one-liners pasted as host, port, password.
        (
            r"(ssh\s+-p\s+\d+\s+[^\s,，]+(?:[,，]\s*\d+)?[,，]\s*)([\x21-\x7E]{6,})",
            r"\1[REDACTED_SECRET]",
        ),
        (r"(?i)(password|passwd|pwd|密码)\s*[:=：]\s*([^\s，,;]{4,})", r"\1: [REDACTED_SECRET]"),
        (r"(?i)(OPENAI_API_KEY|API_KEY|SECRET|TOKEN)\s*[:=]\s*([A-Za-z0-9_\-]{8,})", r"\1=[REDACTED_SECRET]"),
    ]
    for pattern, repl in patterns:
        text = re.sub(pattern, repl, text)
    return text


def clean_message_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    if stripped.startswith("<environment_context>") and stripped.endswith("</environment_context>"):
        return ""
    if stripped.startswith("<permissions instructions>"):
        return ""
    if stripped.startswith("<app-context>"):
        return ""
    if stripped.startswith("# AGENTS.md instructions for "):
        match = re.search(r"## My request for Codex:\s*(.*)", stripped, flags=re.DOTALL)
        return redacted(match.group(1).strip()) if match else ""
    return redacted(stripped)


def extract_content_text(content: Any) -> str:
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for chunk in content:
        if not isinstance(chunk, dict):
            continue
        chunk_type = chunk.get("type")
        if chunk_type in {"input_text", "output_text", "text"}:
            parts.append(str(chunk.get("text", "")))
        elif chunk_type in {"input_image", "localImage", "output_image"}:
            path = chunk.get("path") or chunk.get("image_url") or "[image]"
            parts.append(f"[image] {path}")
    return clean_message_text("\n".join(part for part in parts if part))


def load_state_rows() -> dict[str, dict[str, Any]]:
    db_path = CODEX_HOME / "state_5.sqlite"
    if not db_path.exists():
        return {}

    rows: dict[str, dict[str, Any]] = {}
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        for row in connection.execute("SELECT * FROM threads"):
            data = dict(row)
            cwd = str(data.get("cwd") or "")
            if is_project_cwd(cwd):
                rows[str(data["id"])] = data
    return rows


def load_goal_rows() -> dict[str, list[dict[str, Any]]]:
    db_path = CODEX_HOME / "goals_1.sqlite"
    if not db_path.exists():
        return {}

    rows: dict[str, list[dict[str, Any]]] = {}
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        for row in connection.execute("SELECT * FROM thread_goals ORDER BY updated_at_ms ASC"):
            data = dict(row)
            rows.setdefault(str(data["thread_id"]), []).append(data)
    return rows


def human_time(timestamp: str | None, millis: int | None = None) -> str:
    if timestamp:
        return timestamp
    if millis:
        return datetime.fromtimestamp(millis / 1000).isoformat(timespec="seconds")
    return ""


def compact_summary(value: Any, limit: int = 800) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, default=str)
    value = redacted(value).strip()
    value = re.sub(r"\s+", " ", value)
    return value[:limit] + ("..." if len(value) > limit else "")


def compact_title(value: str, fallback: str) -> str:
    value = redacted(value).strip() or fallback
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    title = lines[0] if lines else fallback
    title = re.sub(r"\s+", " ", title)
    if title.lower().startswith("记住这个内容：ssh "):
        return "远程连接测试和投稿编辑包下载建议"
    return title[:120]


def summarize_tool_output(output: Any) -> str:
    output_text = str(output or "")
    if not output_text:
        return "[tool output omitted]"

    exit_match = re.search(r"Exit code:\s*(-?\d+)", output_text)
    wall_match = re.search(r"Wall time:\s*([^\r\n]+)", output_text)
    parts = []
    if exit_match:
        parts.append(f"exit={exit_match.group(1)}")
    if wall_match:
        parts.append(f"wall={wall_match.group(1).strip()}")
    if parts:
        return "[tool output omitted; " + ", ".join(parts) + "]"
    return "[tool output omitted]"


def parse_session_file(
    session_file: Path,
    state_rows: dict[str, dict[str, Any]],
    goal_rows: dict[str, list[dict[str, Any]]],
) -> SessionRecord | None:
    with session_file.open("r", encoding="utf-8", errors="replace") as handle:
        header = safe_json_loads(handle.readline())
        if not header:
            return None
        payload = header.get("payload", {})
        if not isinstance(payload, dict):
            return None

        cwd = str(payload.get("cwd") or "")
        session_id = str(payload.get("session_id") or payload.get("id") or "")
        if not cwd or not session_id or not is_project_cwd(cwd):
            return None

        state = state_rows.get(session_id, {})
        transcript: list[TranscriptItem] = []
        tools: list[ToolItem] = []
        goals: list[dict[str, Any]] = []

        handle.seek(0)
        for line_no, line in enumerate(handle, start=1):
            event = safe_json_loads(line)
            if not event:
                continue
            event_type = event.get("type")
            event_payload = event.get("payload", {})
            if not isinstance(event_payload, dict):
                continue
            timestamp = str(event.get("timestamp") or "")

            if event_type == "event_msg" and event_payload.get("type") == "thread_goal_updated":
                goal = event_payload.get("goal")
                if isinstance(goal, dict):
                    goals.append({"timestamp": timestamp, "line_no": line_no, **goal})
                continue

            if event_type != "response_item":
                continue

            item_type = event_payload.get("type")
            if item_type == "message":
                role = str(event_payload.get("role") or "")
                if role not in {"user", "assistant"}:
                    continue
                text = extract_content_text(event_payload.get("content"))
                if not text:
                    continue
                phase = event_payload.get("phase")
                transcript.append(
                    TranscriptItem(
                        timestamp=timestamp,
                        role=role,
                        text=text,
                        line_no=line_no,
                        phase=str(phase) if phase else None,
                    )
                )
            elif item_type in {"function_call", "function_call_output", "custom_tool_call", "custom_tool_call_output"}:
                name = str(event_payload.get("name") or event_payload.get("call_id") or "")
                if str(item_type).endswith("_output"):
                    summary = summarize_tool_output(event_payload.get("output"))
                else:
                    summary = "[tool call arguments omitted]"
                tools.append(ToolItem(timestamp=timestamp, kind=str(item_type), name=name, summary=summary, line_no=line_no))

        all_goals = goals + goal_rows.get(session_id, [])
        raw_title = str(state.get("title") or state.get("first_user_message") or (transcript[0].text if transcript else session_id))
        title = compact_title(raw_title, session_id)
        preview = redacted(str(state.get("preview") or ""))
        return SessionRecord(
            session_id=session_id,
            cwd=cwd,
            source_log=session_file,
            created_at=human_time(str(payload.get("timestamp") or ""), state.get("created_at_ms")),
            title=title.strip(),
            preview=preview.strip(),
            transcript=transcript,
            tools=tools,
            goals=all_goals,
        )


def write_markdown(session: SessionRecord) -> str:
    md_path = OUTPUT_ROOT / f"{session.session_id}.md"
    lines: list[str] = [
        f"# {session.title[:120] or session.session_id}",
        "",
        "## Metadata",
        "",
        f"- Session ID: `{session.session_id}`",
        f"- CWD: `{session.cwd}`",
        f"- Created: `{session.created_at}`",
        f"- Source log: `{session.source_log}`",
        f"- Messages recovered: `{len(session.transcript)}`",
        f"- Tool events summarized: `{len(session.tools)}`",
        f"- Goals recovered: `{len(session.goals)}`",
        "",
        "## Goals",
        "",
    ]

    if session.goals:
        for goal in session.goals:
            objective = redacted(str(goal.get("objective") or goal.get("Objective") or ""))
            status = goal.get("status") or goal.get("Status") or ""
            lines.append(f"- `{status}` {objective}")
    else:
        lines.append("No goal records found for this session.")

    lines.extend(["", "## Transcript", ""])
    for item in session.transcript:
        phase = f" / {item.phase}" if item.phase else ""
        lines.extend(
            [
                f"### {item.role}{phase} - {item.timestamp}",
                "",
                item.text,
                "",
            ]
        )

    lines.extend(["## Tool Activity Summary", ""])
    if session.tools:
        for tool in session.tools:
            name = f" `{tool.name}`" if tool.name else ""
            lines.append(f"- `{tool.timestamp}` `{tool.kind}`{name}: {tool.summary}")
    else:
        lines.append("No tool events found.")

    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8-sig")
    return str(md_path)


def write_project_index(sessions: list[SessionRecord], markdown_paths: dict[str, str]) -> None:
    index_lines = [
        "# WaveST-Gate Codex Chat Recovery",
        "",
        f"- Project path: `{PROJECT_PATH}`",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Sessions recovered: `{len(sessions)}`",
        "",
        "## Sessions",
        "",
    ]
    for session in sessions:
        rel = Path(markdown_paths[session.session_id]).relative_to(OUTPUT_ROOT).as_posix()
        index_lines.append(
            f"- [{session.title[:80] or session.session_id}]({rel}) "
            f"- `{session.session_id}`, messages `{len(session.transcript)}`, tools `{len(session.tools)}`"
        )

    (OUTPUT_ROOT / "_project_index.md").write_text("\n".join(index_lines).rstrip() + "\n", encoding="utf-8-sig")


def write_json_index(sessions: list[SessionRecord]) -> None:
    payload = []
    for session in sessions:
        payload.append(
            {
                "session_id": session.session_id,
                "cwd": session.cwd,
                "source_log": str(session.source_log),
                "created_at": session.created_at,
                "title": session.title,
                "preview": session.preview,
                "goals": session.goals,
                "transcript": [item.__dict__ for item in session.transcript],
                "tool_activity": [item.__dict__ for item in session.tools],
            }
        )
    (OUTPUT_ROOT / "index.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8-sig",
    )


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    state_rows = load_state_rows()
    goal_rows = load_goal_rows()
    sessions: list[SessionRecord] = []

    for session_file in iter_session_files():
        record = parse_session_file(session_file, state_rows, goal_rows)
        if record:
            sessions.append(record)

    sessions.sort(key=lambda session: session.created_at)
    markdown_paths = {session.session_id: write_markdown(session) for session in sessions}
    write_project_index(sessions, markdown_paths)
    write_json_index(sessions)

    print(f"Recovered {len(sessions)} session(s) into {OUTPUT_ROOT}")
    for session in sessions:
        print(f"- {session.session_id}: {len(session.transcript)} messages, {len(session.tools)} tool events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
