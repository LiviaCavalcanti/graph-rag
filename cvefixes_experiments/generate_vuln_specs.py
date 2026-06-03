#!/usr/bin/env python3
"""Generate vulnerability specification YAML files for each CVE.

For each CVE directory, reads info.json, original_code.txt, and vuln_patch.txt,
then analyzes the diff to identify the target function, vulnerable sink, and
required check (as a regex pattern).
"""

import difflib
import json
import os
import re
import yaml
from pathlib import Path


CVE_LIST_DIR = Path("CVE-list")
OUTPUT_DIR = Path("vulnerability_specs")


def read_file_content(path: Path) -> str:
    """Read file content, stripping markdown code fences if present."""
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace")
    # Strip markdown code fences
    content = re.sub(r"^```\w*\n", "", content)
    content = re.sub(r"\n```\s*$", "", content)
    return content.strip()


def get_added_lines(original: str, patched: str) -> list[str]:
    """Get lines added in the patch (present in patched but not in original)."""
    orig_lines = original.splitlines(keepends=True)
    patch_lines = patched.splitlines(keepends=True)
    differ = difflib.unified_diff(orig_lines, patch_lines, lineterm="")
    added = []
    for line in differ:
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:].strip())
    return [l for l in added if l]  # filter empty


def get_removed_lines(original: str, patched: str) -> list[str]:
    """Get lines removed in the patch (present in original but not in patched)."""
    orig_lines = original.splitlines(keepends=True)
    patch_lines = patched.splitlines(keepends=True)
    differ = difflib.unified_diff(orig_lines, patch_lines, lineterm="")
    removed = []
    for line in differ:
        if line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:].strip())
    return [l for l in removed if l]


def extract_function_calls(code: str) -> list[str]:
    """Extract function call names from code."""
    pattern = r"\b([a-zA-Z_]\w*)\s*\("
    return re.findall(pattern, code)


def identify_vulnerable_sink(cwe: str, original: str, patched: str, added_lines: list[str], removed_lines: list[str], function_name: str = "") -> str:
    """Identify the vulnerable sink based on CWE type and code analysis."""
    cwe_lower = cwe.lower()
    keywords = {"if", "for", "while", "return", "sizeof", "switch", "case", "else", function_name}

    if "use after free" in cwe_lower or "double free" in cwe_lower:
        # Look for free/release/destroy calls in original that lack guards
        free_patterns = [
            r"\b(\w*free\w*)\s*\(",
            r"\b(\w*release\w*)\s*\(",
            r"\b(\w*destroy\w*)\s*\(",
            r"\b(kfree|vfree|kfree_rcu)\s*\(",
            r"\b(\w*put\w*)\s*\(",
            r"\b(\w*deref\w*)\s*\(",
            r"\b(sk_free|sock_put)\s*\(",
        ]
        for pattern in free_patterns:
            matches = re.findall(pattern, original)
            if matches:
                for m in matches:
                    if m not in keywords:
                        return m
        # Look for pointer dereferences that happen after a potential free
        deref_pattern = r"(\w+)->(\w+)"
        matches = re.findall(deref_pattern, original)
        if matches:
            return f"{matches[0][0]}->{matches[0][1]}"

    elif "null pointer" in cwe_lower or "null dereference" in cwe_lower:
        # Look for pointer dereferences without null checks
        deref_pattern = r"(\w+)->(\w+)"
        matches = re.findall(deref_pattern, original)
        if matches:
            for var, member in matches:
                if f"if (!{var})" in patched or f"if ({var} == NULL)" in patched or f"if ({var})" in patched:
                    return f"{var}->{member}"
            # Check added lines for null checks to find what they guard
            for line in added_lines:
                null_match = re.search(r"if\s*\(\s*!(\w+)\s*\)", line)
                if null_match:
                    guarded_var = null_match.group(1)
                    for var, member in matches:
                        if var == guarded_var:
                            return f"{var}->{member}"
            return f"{matches[0][0]}->{matches[0][1]}"

    elif "buffer" in cwe_lower or "out-of-bounds" in cwe_lower or "overflow" in cwe_lower:
        # Look for dangerous string/buffer operations
        dangerous_ops = [
            r"\b(memcpy|memmove|strcpy|strncpy|sprintf|snprintf|memset|strcmp|strcat)\s*\(",
        ]
        for pattern in dangerous_ops:
            matches = re.findall(pattern, original)
            if matches:
                return matches[0]
        # Look for array accesses
        array_pattern = r"\b(\w+)\s*\[\s*[^]]+\]"
        matches = re.findall(array_pattern, original)
        if matches:
            for m in matches:
                if m not in keywords:
                    return f"{m}[...]"
        # Look for pointer arithmetic
        ptr_arith = r"(\w+)\s*\+\s*(\w+)"
        matches = re.findall(ptr_arith, original)
        if matches:
            return f"{matches[0][0]} + {matches[0][1]}"

    elif "integer overflow" in cwe_lower or "integer underflow" in cwe_lower:
        arith_pattern = r"(\w+\s*[+\-*]\s*\w+)"
        matches = re.findall(arith_pattern, original)
        if matches:
            return matches[0]

    elif "uninitialized" in cwe_lower:
        # Look for variable declarations without initialization that are later returned/used
        decl_pattern = r"(?:int|unsigned|long|char|size_t|ssize_t|u\d+|s\d+)\s+(\w+)\s*;"
        matches = re.findall(decl_pattern, original)
        if matches:
            for var in matches:
                if f"return {var}" in original:
                    return f"return {var}"
            return matches[0]

    elif "type confusion" in cwe_lower or "incorrect type" in cwe_lower:
        # Look at removed lines for the incorrect call
        if removed_lines:
            for line in removed_lines:
                line_calls = extract_function_calls(line)
                for c in line_calls:
                    if c not in keywords:
                        return c
        # Look for casts
        cast_pattern = r"\((\w+\s*\*?)\)\s*(\w+)"
        matches = re.findall(cast_pattern, original)
        if matches:
            return f"({matches[0][0]}){matches[0][1]}"

    elif "race condition" in cwe_lower or "concurrent" in cwe_lower:
        lock_patterns = [
            r"\b(spin_lock|mutex_lock|rcu_read_lock)\s*\(",
            r"(\w+)->(\w+)\s*=",
        ]
        for pattern in lock_patterns:
            matches = re.findall(pattern, original)
            if matches:
                return matches[0] if isinstance(matches[0], str) else "->".join(matches[0])

    elif "missing release" in cwe_lower or "resource leak" in cwe_lower:
        # Look for allocation calls that need corresponding frees
        alloc_patterns = [
            r"\b(\w*alloc\w*)\s*\(",
            r"\b(\w*create\w*)\s*\(",
            r"\b(\w*open\w*)\s*\(",
            r"\b(\w*get\w*)\s*\(",
        ]
        for pattern in alloc_patterns:
            matches = re.findall(pattern, original)
            if matches:
                for m in matches:
                    if m not in keywords:
                        return m

    # Fallback: prefer calls from removed/added lines over generic ones
    if removed_lines:
        for line in removed_lines:
            line_calls = extract_function_calls(line)
            for c in line_calls:
                if c not in keywords:
                    return c

    # Look for the most "dangerous" call in the function (near the change site)
    all_calls = extract_function_calls(original)
    calls = [c for c in all_calls if c not in keywords]

    # Prefer calls that appear near where changes were made
    if added_lines and calls:
        for line in added_lines:
            line_calls = extract_function_calls(line)
            for c in line_calls:
                if c in calls and c not in keywords:
                    return c

    if calls:
        return calls[0]

    return "unknown"


def generate_required_check_regex(cwe: str, original: str, patched: str, added_lines: list[str], sink: str) -> str:
    """Generate a regex pattern for the required check based on the patch."""
    cwe_lower = cwe.lower()

    if not added_lines:
        # No lines were added, look at modifications
        return ".*"  # Fallback

    # Look for conditional checks added in the patch
    check_lines = []
    for line in added_lines:
        # Skip braces-only lines, blank comments, etc.
        if line in ["{", "}", "/*", "*/", ""]:
            continue
        # Prioritize if-statements, null checks, bounds checks
        if re.match(r"\s*if\s*\(", line):
            check_lines.append(line)
        elif re.match(r"\s*else\s+if\s*\(", line):
            check_lines.append(line)
        elif "return" in line and ("NULL" in line or "err" in line.lower() or "-E" in line or "0" in line):
            check_lines.append(line)
        elif "goto" in line:
            check_lines.append(line)
        elif "= NULL" in line or "= 0" in line:
            check_lines.append(line)

    if not check_lines:
        # Use all non-trivial added lines
        check_lines = [l for l in added_lines if len(l) > 3 and l not in ["{", "}", "/*", "*/"]]

    if not check_lines:
        return ".*"

    # Build regex from the most important check line
    primary_check = check_lines[0]

    # Escape regex special chars but keep wildcards for variable names
    regex = re.escape(primary_check.strip())
    # Make variable names flexible with \w+ patterns
    regex = re.sub(r"\\-\\>", "->", regex)
    regex = re.sub(r"\\\(", r"\\s*\\(", regex)

    return regex


def build_readable_check(cwe: str, original: str, patched: str, added_lines: list[str], sink: str) -> dict:
    """Build a human-readable description of the required check plus regex."""
    cwe_lower = cwe.lower()

    # Gather the meaningful added lines (conditions, assignments, guards)
    meaningful_adds = []
    for line in added_lines:
        stripped = line.strip()
        if stripped in ["{", "}", "/*", "*/", "", "else {", "} else {"]:
            continue
        if stripped.startswith("*") and len(stripped) < 5:
            continue
        meaningful_adds.append(stripped)

    if not meaningful_adds:
        return {
            "description": "Patch modifies existing logic (no new lines added)",
            "pattern": ".*",
            "type": "modification"
        }

    # Categorize the check type
    check_type = "guard_condition"
    if any("= NULL" in l or "= 0" in l for l in meaningful_adds):
        if "use after free" in cwe_lower:
            check_type = "null_assignment_after_free"
    if any(re.match(r"if\s*\(", l) for l in meaningful_adds):
        check_type = "guard_condition"
    if any("= 0" in l and not "==" in l for l in meaningful_adds):
        if "uninitialized" in cwe_lower:
            check_type = "initialization"

    # Build a regex that captures the essential check
    # Take the first if-condition or the most meaningful line
    primary = None
    for line in meaningful_adds:
        if re.match(r"if\s*\(", line):
            primary = line
            break
    if not primary:
        primary = meaningful_adds[0]

    # Create a regex pattern
    # Escape and then relax for whitespace
    pattern = re.escape(primary)
    # Relax whitespace
    pattern = re.sub(r"\\ ", r"\\s+", pattern)
    # Keep operators readable
    pattern = pattern.replace(r"\-\>", "->")
    pattern = pattern.replace(r"\=\=", "==")
    pattern = pattern.replace(r"\!\=", "!=")
    pattern = pattern.replace(r"\<\=", "<=")
    pattern = pattern.replace(r"\>\=", ">=")

    return {
        "description": f"The patch adds: {primary}",
        "pattern": pattern,
        "type": check_type,
        "added_lines": meaningful_adds[:10]  # Cap at 10 lines
    }


def analyze_cve(cve_dir: Path) -> dict:
    """Analyze a single CVE and return its vulnerability spec."""
    info_path = cve_dir / "info.json"
    original_path = cve_dir / "original_code.txt"
    patched_path = cve_dir / "vuln_patch.txt"

    if not info_path.exists():
        return None

    with open(info_path) as f:
        info = json.load(f)

    original = read_file_content(original_path)
    patched = read_file_content(patched_path)

    if not original or not patched:
        return None

    added_lines = get_added_lines(original, patched)
    removed_lines = get_removed_lines(original, patched)

    cwe = info.get("cwe_id", "Unknown")
    function_name = info.get("function_name", "unknown")
    function_proto = info.get("function_prototype", "")

    sink = identify_vulnerable_sink(cwe, original, patched, added_lines, removed_lines, function_name)
    check = build_readable_check(cwe, original, patched, added_lines, sink)

    spec = {
        "cve_id": info.get("cve_id", cve_dir.name),
        "cwe_id": cwe,
        "target_function": {
            "name": function_name,
            "prototype": function_proto,
        },
        "vulnerable_sink": {
            "operation": sink,
            "description": f"Vulnerable {cwe.lower()} via {sink}",
        },
        "required_check": check,
        "language": info.get("programming_language", "c"),
    }

    return spec


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    cve_dirs = sorted(CVE_LIST_DIR.iterdir())
    specs = {}

    for cve_dir in cve_dirs:
        if not cve_dir.is_dir():
            continue

        print(f"Analyzing {cve_dir.name}...")
        spec = analyze_cve(cve_dir)
        if spec:
            # Write individual YAML file
            output_path = OUTPUT_DIR / f"{cve_dir.name}.yaml"
            with open(output_path, "w") as f:
                yaml.dump(spec, f, default_flow_style=False, sort_keys=False, width=120)
            specs[cve_dir.name] = spec
            print(f"  -> {output_path}")
        else:
            print(f"  -> SKIPPED (missing data)")

    # Write combined JSON file
    combined_path = OUTPUT_DIR / "all_vulnerability_specs.json"
    with open(combined_path, "w") as f:
        json.dump(specs, f, indent=2)

    print(f"\nGenerated {len(specs)} vulnerability specs in {OUTPUT_DIR}/")
    print(f"Combined file: {combined_path}")


if __name__ == "__main__":
    main()
