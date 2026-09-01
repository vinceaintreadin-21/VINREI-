"""
guardrails.py — Input validation and sandboxed command execution.

Two responsibilities:
  1. Prompt injection detection — flag prompts that try to hijack the agent
  2. Command sandboxing — validate shell commands before the agent runs them

Used by agent.py before executing any tool call.
"""

import re
import shlex 
from pathlib import Path 

#Prompt injection detection 

INJECTION_PATTERNS = [
    r"ignore\b.{0,30}(instructions?|prompts?|rules?|constraints?)",
    r"you are now",
    r"forget (everything|all|your instructions)",
    r"new (instructions?|rules?|system prompt)",
    r"disregard (your |all )?(previous |prior )?(instructions?|rules?)",
    r"act as (a |an )?(different|unrestricted|jailbroken)",
    r"do anything now",
    r"dan mode",
    r"pretend (you are|to be) (a |an )?",
    r"override (safety|instructions?|rules?|constraints?)",
    r"system:\s*(you are|ignore|forget)",
]

_INJECTION_RE = re.compile(
    "|".join(INJECTION_PATTERNS),
    re.IGNORECASE,
)

def check_prompt(prompt: str) -> tuple[bool, str]:
    match = _INJECTION_RE.search(prompt)
    if match:
        return False, f"Prompt injection detected: '{match.group(0)}'"
    return True, ""

#Command sandboxing 

# Commands that are always blocked regardless of context
BLOCKED_COMMANDS = [
    "rm -rf",
    "rm -r",
    "sudo",
    "su ",
    "chmod 777",
    "chmod -R",
    "chown",
    "mkfs",
    "dd ",
    "dd\t",
    "> /dev/",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "curl | sh",
    "curl | bash",
    "wget | sh",
    "wget | bash",
    ":(){ :|:& };:",   # fork bomb
    "/etc/passwd",
    "/etc/shadow",
    "base64 -d",
    "eval ",
    "exec ",
    "os.system",
    "subprocess.call",
]

# Commands that are allowed — anything not matching blocked patterns
# that also stays within the repo root
ALLOWED_PREFIXES = [
    "python",
    "python3",
    "pytest",
    "ruff",
    "mypy",
    "git ",
    "git\t",
    "ls",
    "cat ",
    "cat\t",
    "echo ",
    "echo\t",
    "grep",
    "find",
    "head",
    "tail",
    "wc ",
    "diff",
    "test ",
]

# Max command length to prevent abuse
MAX_CMD_LENGTH = 512

def check_command(command: str, repo_root: Path | None = None) -> tuple[bool, str]:
    if not command or not command.strip():
        return False, "Empty command"
    
    if len(command) > MAX_CMD_LENGTH: 
        return False, f"Command too long ({len(command)}) > {MAX_CMD_LENGTH} chars"
    
    cmd_lower = command.lower()

    for pattern in BLOCKED_COMMANDS: 
        if pattern.lower() in cmd_lower:
            return False, f"blocked command pattern: '{pattern}'"
    
    #check for path traversal outside repo root
    if repo_root:
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = shlex.split()

        for token in tokens:
            if token.startswith("/") or ".." in token:
                candidate = Path(token).resolve()
                try:
                    candidate.relative_to(repo_root.resolve())
                except ValueError:
                    return False, f"Path '{token}' is outside the repo root"

    # check chained commands — block pipes to shells
    if re.search(r"\|\s*(sh|bash|zsh|fish|dash)", command):
        return False, "Piping to shell interpreter is not allowed"

    #check command substitution
    if re.search(r"`[^`]+`|\$\([^)]+\)", command):
        return False, "Command substitution is not allowed"

    return True, ""


#Combined check
def validate_tool_call(tool: str, args: dict, repo_root: Path | None = None) -> tuple[bool, str]:
    if tool == "run":
        command = args.get("command", "")
        return check_command(command, repo_root=repo_root)
    
    if tool == "edit":
        path = args.get("path", "")
        if not path:
            return False, "edit: missing path"
        if repo_root:
            full = (repo_root / path).resolve()
            try:
                full.relative_to(repo_root.resolve())
            except ValueError:
                return False, f"edit: path '{path}' is outside the repo root"

    if tool == "read":
        path = args.get("path", "")
        if repo_root and path:
            full = (repo_root / path).resolve()
            try:
                full.relative_to(repo_root.resolve())
            except ValueError:
                return False, f"read: path '{path}' is outside the repo root"

    return True, ""
