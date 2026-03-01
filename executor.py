import inspect

from tools import read_file, write_file, edit_file, run_bash, list_files, search_files, web_search, get_last_edit_diff, was_last_bash_streamed
from ui import confirm_action, show_tool_call, show_tool_result, show_running_hint, show_diff

# Tools that require user confirmation before execution
DANGEROUS_TOOLS = {"write_file", "edit_file", "run_bash"}

# Tools that are safe to run in parallel (read-only, no side effects)
SAFE_TOOLS = {"read_file", "list_files", "search_files", "web_search"}


def is_safe_tool(name):
    """Check if a tool can be safely run in parallel."""
    return name in SAFE_TOOLS


skip_permissions = False

TOOL_MAP = {
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "run_bash": run_bash,
    "list_files": list_files,
    "search_files": search_files,
    "web_search": web_search,
}


def execute_tool(name, arguments):
    """Execute a tool by name with the given arguments.

    Returns the result string, or None if the user declined.
    """
    show_tool_call(name, arguments)

    if name not in TOOL_MAP:
        return f"Error: Unknown tool '{name}'"

    # Ask for confirmation on dangerous operations
    if name in DANGEROUS_TOOLS and not skip_permissions:
        approved, feedback = confirm_action(name, arguments)
        if not approved:
            return feedback

    func = TOOL_MAP[name]

    # Filter out unexpected keyword arguments before calling
    sig = inspect.signature(func)
    valid_params = set(sig.parameters.keys())
    extra = sorted(set(arguments.keys()) - valid_params)
    if extra:
        valid_str = ", ".join(valid_params)
        extra_str = ", ".join(extra)
        result = (
            f"Error: {name}() only accepts ({valid_str}). "
            f"Got unexpected arguments: {extra_str}.\n"
            "When writing HTML/XML/SVG files, ALL content must go inside ONE "
            "<parameter=content> tag as a single string. "
            "Do NOT split HTML attributes into separate <parameter=...> tags.\n"
            "Example:\n"
            "<function=write_file>\n"
            "<parameter=path>index.html</parameter>\n"
            "<parameter=content>\n"
            '<!DOCTYPE html>\n<html lang="en">\n<head>...</head>\n'
            "<body>...</body>\n</html>\n"
            "</parameter>\n"
            "</function>"
        )
        show_tool_result(name, result)
        return result

    if name == "run_bash":
        show_running_hint()
    try:
        result = func(**arguments)
    except TypeError as e:
        result = f"Error: {e}"
    # Show diff for successful edit_file
    if name == "edit_file" and result and not result.startswith("Error"):
        diff_info = get_last_edit_diff()
        if diff_info:
            _, old_content, new_content = diff_info
            show_diff(old_content, new_content, arguments.get("path", ""))
    # Compact result for streamed bash (output already shown line-by-line)
    if name == "run_bash" and was_last_bash_streamed():
        line_count = len(result.splitlines()) if result else 0
        # Extract exit code if present
        exit_code = 0
        if result and "[Exit code:" in result:
            import re
            m = re.search(r'\[Exit code: (\d+)\]', result)
            if m:
                exit_code = int(m.group(1))
        compact = f"[{line_count} lines shown above]"
        if exit_code != 0:
            compact += f" [Exit code: {exit_code}]"
        show_tool_result(name, compact)
    else:
        show_tool_result(name, result)
    return result
