"""
Session logger — records every prompt/response pair to disk.

Each session is a .jsonl file (one JSON object per line) stored in
the sessions/ directory. These become your most valuable training
data — real prompts you actually sent and responses you accepted.

Usage:
    from pipeline.logger import SessionLogger

    log = SessionLogger()
    log.start_session()
    log.record(prompt="explain this bug", response="...", accepted=True)
    log.end_session()
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

SESSIONS_DIR = Path(__file__).parent.parent / "sessions"


class SessionLogger:
    """
    Logs prompt/response pairs to a JSONL file.

    Each call to start_session() creates a new file.
    Each call to record() appends one entry to that file.
    """

    def __init__(self, sessions_dir: str | Path = SESSIONS_DIR) -> None:
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._session_id: str | None = None
        self._file_path: Path | None = None

    def start_session(self, model: str = "", task: str = "") -> str:
        """
        Start a new session and open a log file.

        Args:
            model: Model tag used in this session.
            task: Task mode (explain/edit/debug/etc).

        Returns:
            The session ID.
        """
        self._session_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{self._session_id}.jsonl"
        self._file_path = self.sessions_dir / filename

        # write session header
        self._append({
            "type": "session_start",
            "session_id": self._session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "task": task,
        })

        return self._session_id

    def record(
        self,
        prompt: str,
        response: str,
        accepted: bool = True,
        diff: str = "",
        context: str = "",
    ) -> None:
        """
        Record a single prompt/response pair.

        Args:
            prompt: The user's input.
            response: The model's output.
            accepted: Whether the user accepted this response (default True).
            diff: The actual diff applied, if any (most valuable signal).
            context: Any repo context injected into the prompt.
        """
        if not self._file_path:
            raise RuntimeError("Call start_session() before record()")

        self._append({
            "type": "turn",
            "session_id": self._session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt": prompt,
            "response": response,
            "accepted": accepted,
            "diff": diff,
            "context": context,
        })

    def end_session(self) -> None:
        """Mark the session as ended."""
        if not self._file_path:
            return
        self._append({
            "type": "session_end",
            "session_id": self._session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._session_id = None
        self._file_path = None

    def _append(self, entry: dict) -> None:
        with self._file_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# CLI — python -m pipeline.logger (interactive test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log = SessionLogger()
    sid = log.start_session(model="qwen2.5-coder:1.5b", task="explain")
    print(f"Session started: {sid}")

    log.record(
        prompt="explain the stream function in ollama_client.py",
        response="The stream() function sends a POST request to /v1/chat/completions...",
        accepted=True,
    )

    log.end_session()
    print(f"Session saved to: {log.sessions_dir}")
