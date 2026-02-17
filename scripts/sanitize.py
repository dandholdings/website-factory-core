"""
sanitize.py — Input sanitization utilities for shell commands and workflow inputs.

Used by: GitHub Actions workflows and Python scripts to safely handle user inputs
containing spaces, quotes, special characters, and backslashes.
"""

import os
import re
import shlex
import subprocess
from typing import List, Union, Optional


def sanitize_shell_arg(value: str) -> str:
    """Sanitize a single argument for safe shell usage.
    
    Returns a properly quoted string that can be safely used in shell commands.
    Handles spaces, quotes, backslashes, and other special characters.
    
    Args:
        value: The string value to sanitize
        
    Returns:
        Properly shell-quoted string
    """
    if value is None:
        return "''"
    
    # Convert to string and strip surrounding quotes that GitHub might add
    s = str(value).strip()
    
    # Remove surrounding quotes if they match
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    
    # Use shlex.quote for proper shell escaping
    return shlex.quote(s)


def sanitize_for_yaml(value: str) -> str:
    """Sanitize a string for safe use in YAML values.
    
    Escapes special YAML characters and ensures proper quoting.
    
    Args:
        value: The string value to sanitize
        
    Returns:
        Safe YAML string value
    """
    if value is None:
        return ""
    
    s = str(value).strip()
    
    # Remove surrounding quotes if they match
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    
    # Escape special YAML characters
    if any(c in s for c in ":{}[],&*#?|-<>=!%@\\"):
        # Quote the string
        s = s.replace('"', '\\"')
        return f'"{s}"'
    
    return s


def sanitize_niche_input(value: str) -> str:
    """Specialized sanitization for niche inputs that may contain spaces/quotes.
    
    GitHub Actions sometimes adds extra quotes to workflow inputs.
    This function normalizes niche strings for consistent processing.
    
    Args:
        value: The niche string to sanitize
        
    Returns:
        Cleaned niche string
    """
    if value is None:
        return ""
    
    s = str(value).strip()
    
    # Remove surrounding quotes
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    
    # Normalize multiple spaces
    s = re.sub(r'\s+', ' ', s)
    
    return s


def sanitize_windows_path(value: str) -> str:
    """Sanitize Windows paths by converting backslashes to forward slashes.
    
    GitHub Actions runs on Linux, but inputs might contain Windows-style paths.
    
    Args:
        value: Path string to sanitize
        
    Returns:
        Path with forward slashes
    """
    if value is None:
        return ""
    
    s = str(value).strip()
    # Convert backslashes to forward slashes
    s = s.replace('\\', '/')
    # Remove surrounding quotes
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    
    return s


def build_safe_command(args: List[str]) -> str:
    """Build a safe shell command from a list of arguments.
    
    Each argument is properly quoted and escaped.
    
    Args:
        args: List of command arguments
        
    Returns:
        Safe shell command string
    """
    return ' '.join(sanitize_shell_arg(arg) for arg in args)


def run_safe_command(args: List[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a shell command safely with proper argument quoting.
    
    Uses subprocess.run with list arguments (not shell=True) for maximum safety.
    
    Args:
        args: List of command and arguments
        **kwargs: Additional arguments to subprocess.run
        
    Returns:
        CompletedProcess object
    """
    return subprocess.run(args, **kwargs)


def sanitize_workflow_inputs(inputs: dict) -> dict:
    """Sanitize all workflow inputs for safe use in scripts.
    
    Processes each input value based on its expected usage.
    
    Args:
        inputs: Dictionary of workflow inputs
        
    Returns:
        Dictionary with sanitized values
    """
    sanitized = {}
    
    for key, value in inputs.items():
        if value is None:
            sanitized[key] = ""
        elif isinstance(value, str):
            # Special handling for different types of inputs
            if 'niche' in key.lower():
                sanitized[key] = sanitize_niche_input(value)
            elif 'path' in key.lower() or 'dir' in key.lower() or 'slug' in key.lower():
                sanitized[key] = sanitize_windows_path(value)
            elif 'url' in key.lower():
                sanitized[key] = value.strip().strip('"\'')
            else:
                sanitized[key] = value.strip().strip('"\'')
        else:
            sanitized[key] = value
    
    return sanitized


# Convenience functions for common patterns
def get_sanitized_env(var_name: str, default: str = "") -> str:
    """Get and sanitize an environment variable.
    
    Args:
        var_name: Environment variable name
        default: Default value if variable is not set
        
    Returns:
        Sanitized value
    """
    value = os.environ.get(var_name, default)
    return sanitize_niche_input(value)


def safe_join_args(*args: str) -> str:
    """Safely join multiple arguments into a shell command string.
    
    Args:
        *args: Arguments to join
        
    Returns:
        Safe shell command string
    """
    return ' '.join(sanitize_shell_arg(arg) for arg in args if arg is not None)