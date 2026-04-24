"""Preprocessing utilities for ground-truth C source files.

Extracts function bodies from ground-truth .c files that contain stub
declarations followed by the actual target function.
"""

from __future__ import annotations


def extract_function_body(full_source: str, func_name: str | None = None) -> str:
    """Extract the main function body from a ground-truth .c file.

    Ground-truth files often include stub declarations before the actual
    function.  We take the *last* top-level function definition as the
    relevant body, which works for the AutoPatch dataset layout.

    Args:
        full_source: Complete file content.
        func_name:   Optional function name hint for matching.

    Returns:
        The extracted function body, or the full source as fallback.
    """
    # Find all top-level function bodies by matching balanced braces
    # after a line that looks like a function signature
    functions = []
    lines = full_source.split("\n")
    i = 0
    while i < len(lines):
        # rough heuristic: line with '(' and next non-empty has '{'
        line = lines[i].rstrip()
        if "(" in line and not line.startswith("#") and not line.startswith("//"):
            # scan forward for opening brace
            j = i
            while j < len(lines) and "{" not in lines[j]:
                j += 1
            if j < len(lines):
                # count braces to find end
                depth = 0
                start = j
                k = j
                found = False
                while k < len(lines):
                    depth += lines[k].count("{") - lines[k].count("}")
                    if depth == 0 and k > start:
                        functions.append((i, "\n".join(lines[i : k + 1])))
                        found = True
                        i = k + 1
                        break
                    k += 1
                if not found:
                    i += 1
                    continue
            else:
                i += 1
        else:
            i += 1

    if not functions:
        return full_source  # fallback: return everything

    # If a func_name hint is given, try to match it
    if func_name:
        for _, body in functions:
            if func_name in body.split("(")[0]:
                return body

    # Otherwise return the last function (the main one in AutoPatch files)
    return functions[-1][1]
