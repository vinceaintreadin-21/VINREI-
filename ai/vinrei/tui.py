"""
tui.py — Interactive TUI for vinrei using Textual.

A rich terminal interface with:
  - Chat history panel (scrollable)
  - Input box at the bottom
  - Syntax highlighted code blocks
  - Keyboard shortcuts

Usage:
    python -m vinrei.tui
    python -m vinrei.tui --model vinrei:v1
    python -m vinrei.tui --model vinrei:v1 --repo .
"""

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, RichLog, Label
from textual.containers import Horizontal
from textual.binding import Binding
from textual import work
from rich.markdown import Markdown
from rich.syntax import Syntax
import re
import sys

sys.path.insert(0, ".")
from vinrei.ollama_client import DEFAULT_MODEL
from vinrei.prompts import build as build_prompt
from vinrei.guardrails import check_prompt
from vinrei import agent as agent_mod


class VinreiTUI(App):
    """Main TUI application."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #chat-log {
        border: solid $accent;
        height: 1fr;
        margin: 0 1;
        padding: 1;
    }

    #input-row {
        height: 3;
        margin: 0 1;
    }

    #prompt-input {
        width: 1fr;
        border: solid $accent;
    }

    #mode-label {
        width: 12;
        height: 3;
        content-align: center middle;
        border: solid $accent;
        margin-left: 1;
        color: $accent;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear", "Clear"),
        Binding("ctrl+e", "cycle_mode", "Mode"),
        Binding("escape", "focus_input", "Focus input"),
    ]

    MODES = ["general", "explain", "edit", "debug", "search", "diff"]

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        task: str | None = None,
        repo: str | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self._mode = task or "general"
        self._mode_index = self.MODES.index(self._mode) if self._mode in self.MODES else 0
        self._repo = repo

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield RichLog(id="chat-log", highlight=True, markup=True, wrap=True)
        with Horizontal(id="input-row"):
            yield Input(
                placeholder="Ask vinrei anything... (Ctrl+E to change mode)",
                id="prompt-input",
            )
            yield Label(self._mode, id="mode-label")
        yield Footer()

    def on_mount(self) -> None:
        self.title = f"vinrei — {self.model}"
        log = self.query_one("#chat-log", RichLog)
        repo_info = f" | repo: `{self._repo}`" if self._repo else ""
        log.write(Markdown(
            f"# vinrei\n"
            f"Model: `{self.model}` | Mode: `{self._mode}`{repo_info}\n"
            f"Type your prompt and press Enter. "
            f"**Ctrl+E** to cycle modes, **Ctrl+L** to clear.\n---"
        ))
        self.query_one("#prompt-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt:
            return

        event.input.clear()
        log = self.query_one("#chat-log", RichLog)

        # check injection
        is_safe, reason = check_prompt(prompt)
        if not is_safe:
            log.write(f"[red][guardrails] Blocked — {reason}[/red]")
            return

        # show user prompt
        log.write(f"\n[bold cyan]You:[/bold cyan] {prompt}\n")
        log.write("[bold green]vinrei:[/bold green]")

        self._stream_response(prompt)

    @work(thread=True)
    def _stream_response(self, prompt: str) -> None:
        """Run agent loop in a background thread."""
        log = self.query_one("#chat-log", RichLog)

        # inject repo context if --repo was passed
        full_prompt = prompt
        if self._repo:
            from vinrei import repo as repo_mod
            ctx = repo_mod.context(self._repo)
            full_prompt = f"{ctx}\n\n{prompt}"

        import time
        start = time.perf_counter()

        try:
            # always run through agent — it decides which tools to use
            response = agent_mod.run(
                full_prompt,
                root=self._repo or ".",
                model=self.model,
                task=self._mode if self._mode != "general" else None,
                verbose=False,
            )
        except Exception as e:
            self.call_from_thread(log.write, f"\n[red][error] {e}[/red]")
            return

        elapsed = time.perf_counter() - start
        token_count = len(response.split())
        tps = token_count / elapsed if elapsed > 0 else 0

        if "```" in response:
            self.call_from_thread(self._render_code_blocks, response)
        else:
            self.call_from_thread(log.write, response)

        self.call_from_thread(log.write, f"[dim]({tps:.1f} t/s)[/dim]\n")

    def _render_code_blocks(self, response: str) -> None:
        """Render response with syntax-highlighted code blocks."""
        log = self.query_one("#chat-log", RichLog)
        log.write("─" * 40)
        parts = re.split(r"```(\w+)?\n(.*?)```", response, flags=re.DOTALL)
        for i, part in enumerate(parts):
            if i % 3 == 0:
                if part.strip():
                    log.write(part.strip())
            elif i % 3 == 2:
                lang = parts[i - 1] or "python"
                log.write(Syntax(part.strip(), lang, theme="monokai", line_numbers=True))

    def action_clear(self) -> None:
        self.query_one("#chat-log", RichLog).clear()

    def action_cycle_mode(self) -> None:
        self._mode_index = (self._mode_index + 1) % len(self.MODES)
        self._mode = self.MODES[self._mode_index]
        self.query_one("#mode-label", Label).update(self._mode)
        log = self.query_one("#chat-log", RichLog)
        log.write(f"[dim]Mode switched to: {self._mode}[/dim]")

    def action_focus_input(self) -> None:
        self.query_one("#prompt-input", Input).focus()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="vinrei TUI")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL)
    parser.add_argument("--task", "-t", default=None)
    parser.add_argument("--repo", "-r", default=None, help="Repo root for file context")
    args = parser.parse_args()

    app = VinreiTUI(model=args.model, task=args.task, repo=args.repo)
    app.run()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="vinrei TUI")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL)
    parser.add_argument("--task", "-t", default=None)
    parser.add_argument("--repo", "-r", default=None, help="Repo root for file context")
    args = parser.parse_args()

    app = VinreiTUI(model=args.model, task=args.task, repo=args.repo)
    app.run()
