"""
Tool loop / agent.

The model can request tools by emitting JSON blocks in its response:

    ```tool
    {"tool": "read", "path": "vinrei/chat.py"}
    ```

Supported tools:
    read  — read a file
    grep  — search the codebase
    edit  — overwrite a file with new content
    run   — run a shell command (sandboxed to the repo root)

The agent loop:
    1. Send prompt to model
    2. Parse tool calls from the response
    3. Execute each tool
    4. Feed results back to the model
    5. Repeat until the model replies with no tool calls
"""

import json
import re 
import subprocess 
from pathlib import Path 

from vinrei.ollama_client import DEFAULT_MODEL, complete 
from vinrei.prompts import build as build_prompt
from vinrei.repo import read as repo_read, grep as repo_grep
from vinrei.guardrails import validate_tool_call, check_prompt

#Max iterations
MAX_TURNS = 8

# Regex to find ```tool ... ``` blocks in model output
TOOL_BLOCK_RE = re.compile(r"```tool\s*(\{.*?\})\s*```", re.DOTALL)

#Tools
def _tool_read(args: dict, root: Path) -> str:
    path = args.get("path")
    if not path:
        return "[error] read: missing 'path'"
    full = root / path 
    return repo_read(full)

def _tool_grep(args: dict, root: Path) -> str:
    pattern = args.get("pattern")
    if not pattern: 
        return "[error] grep: missing 'pattern'"
    include = args.get("include", "*.py")
    results = repo_grep(pattern, root=root, include=include)
    if not results:
        return f"[grep] No matches for '{pattern}'"
    lines = [f"{r['file']}:{r['line']}: {r['text']}" for r in results[:20]]
    return "\n".join(lines)

def _tool_edit(args: dict, root: Path) -> str: 
    path = args.get("pattern")
    content = args.get("content")
    if not path:
        return "[error] grep: missing 'path'"
    if content is None: 
        return "[error] edit: missing 'content'"
    full = root / path

    try:
        full.resolve().relative_to(root.resolve())
    except ValueError:
        return f"[error] edit: path '{path}' is outside the repo root"
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content)
    return f"[edit] Wrote {full}"

def _tool_run(args: dict, root: Path) -> str:
    command = args.get("command")
    if not command:
        return "[error] run: missing 'command'"
    blocked = ["rm -rf", "sudo", "mkfs", "dd ", "shutdown", "reboot"]
    for b in blocked: 
        if b in command:
            return f"[error] run: blocked command '{command}'"
    try: 
        result = subprocess.run(
            command, 
            shell=True,
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(root),
        )
        output = result.stdout + result.stderr
        return output[:2000] if output else "[run] (no output)"
    except subprocess.TimeoutExpired:
        return "[error] run: command timed out"
    
#Tools dispatcher

TOOLS = {
    "read": _tool_read,
    "grep": _tool_grep,
    "edit": _tool_edit,
    "run": _tool_run,
}

def _parse_tool_calls(text: str) -> list[dict]:
    """Extract all tool call dicts from a model response."""
    calls = []
    for match in TOOL_BLOCK_RE.finditer(text):
        try:
            calls.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            continue
    return calls


def _execute_tools(calls: list[dict], root: Path) -> str:
    """Run each tool call and return a combined result string."""
    results = []
    for call in calls:
        tool_name = call.get("tool")
        if tool_name not in TOOLS:
            results.append(f"[error] Unknown tool '{tool_name}'")
            continue

        #validate before executing 
        is_safe, reason = validate_tool_call(tool_name, call, repo_root=root)
        if not is_safe:
            results.append(f"[blocked] {tool_name}: {reason}")
            continue 

        result = TOOLS[tool_name](call, root)
        results.append(f"[tool:{tool_name}]\n{result}")
    return "\n\n".join(results)

#Agent loop 
def run(
    prompt: str,
    root: str | Path = ".",
    model: str = DEFAULT_MODEL,
    task: str | None = None,
    verbose: bool = False,
) -> str:
    """
    Run the agent loop for a given prompt.

    Sends the prompt to the model, executes any tool calls it makes,
    feeds results back, and repeats until no more tool calls or MAX_TURNS.

    Args:
        prompt: The user's request.
        root: Repo root for tool execution.
        model: Ollama model tag.
        task: Optional task mode for the system prompt.
        verbose: Print tool calls and results as they happen.

    Returns:
        The model's final response as a string.
    """
    is_safe, reason = check_prompt(prompt)
    if not is_safe:
        return f"[guardrails] Request blocked — {reason}"
        
    root = Path(root).resolve()
    system = build_prompt(task)

    tool_instructions = (
        "You have access to tools. Call them by emitting a fenced code block tagged 'tool'\n"
        "containing a JSON object. Example:\n\n"
        "```tool\n"
        '{"tool": "read", "path": "vinrei/chat.py"}\n'
        "```\n\n"
        "Available tools:\n"
        '- read : {"tool": "read", "path": "<relative path>"}\n'
        '- grep : {"tool": "grep", "pattern": "<regex>", "include": "<glob>"}\n'
        '- edit : {"tool": "edit", "path": "<relative path>", "content": "<full file content>"}\n'
        '- run  : {"tool": "run", "command": "<shell command>"}\n\n'
        "Call tools when you need more information. When you have enough, reply normally."
    )

    full_system = system + "\n\n" + tool_instructions

    messages = [{"role": "user", "content": prompt}]
    last_response = ""

    for turn in range(MAX_TURNS):
        conversation = "\n\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in messages
        )

        response = complete(conversation, model=model, system=full_system)
        last_response = response

        if verbose:
            print(f"\n[turn {turn + 1}]\n{response}")

        tool_calls = _parse_tool_calls(response)
        if not tool_calls:
            break

        tool_results = _execute_tools(tool_calls, root)

        if verbose:
            print(f"\n[tool results]\n{tool_results}")

        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"Tool results:\n{tool_results}"})

    return last_response

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="vinrei agent loop")
    parser.add_argument("prompt", help="What to ask the agent")
    parser.add_argument("--repo", "-r", default=".", help="Repo root")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL)
    parser.add_argument("--task", "-t", default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    result = run(
        args.prompt,
        root=args.repo,
        model=args.model,
        task=args.task,
        verbose=args.verbose,
    )
    print(result)
