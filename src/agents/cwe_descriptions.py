"""
Static CWE knowledge base for the tool-calling agent.

Maps CWE IDs to human-readable descriptions and typical fix patterns,
enabling the agent to reason about *what category of fix* is needed before
generating code.
"""

from __future__ import annotations

CWE_KNOWLEDGE: dict[str, dict[str, str]] = {
    "CWE-20": {
        "name": "Improper Input Validation",
        "description": (
            "The product does not validate (or incorrectly validates) input that "
            "can affect control flow or data flow."
        ),
        "typical_fixes": (
            "Add bounds checks on user-supplied values; validate string lengths "
            "before use; reject unexpected/malformed input early; use safe parsing "
            "functions (e.g., kstrtoul instead of simple_strtoul)."
        ),
    },
    "CWE-190": {
        "name": "Integer Overflow or Wraparound",
        "description": (
            "An integer overflow or wraparound occurs when an integer value is "
            "incremented beyond its maximum, wrapping to a small or negative value."
        ),
        "typical_fixes": (
            "Add overflow checks before arithmetic (e.g., check_add_overflow); "
            "cap values at INT_MAX/SIZE_MAX; use wider types for intermediate "
            "computations; validate multiplied sizes before allocation."
        ),
    },
    "CWE-362": {
        "name": "Race Condition (Concurrent Execution Using Shared Resource)",
        "description": (
            "The product contains a code sequence that can run concurrently with "
            "other code, and the code sequence requires temporary exclusive access "
            "to a shared resource, but a timing window exists where the resource "
            "can be modified by another concurrent code sequence."
        ),
        "typical_fixes": (
            "Add proper locking (mutex, spinlock, inode_lock) around shared resource "
            "access; use atomic operations; use RCU for read-heavy paths; refactor "
            "into a helper that holds the lock for the critical section; increment "
            "reference counts before use to prevent concurrent free."
        ),
    },
    "CWE-400": {
        "name": "Uncontrolled Resource Consumption",
        "description": (
            "The product does not properly control the allocation/maintenance of a "
            "limited resource, allowing an actor to influence the amount consumed."
        ),
        "typical_fixes": (
            "Add size/count limits; validate allocation sizes; add timeouts; "
            "rate-limit operations; check return values from allocation functions."
        ),
    },
    "CWE-416": {
        "name": "Use After Free",
        "description": (
            "The product references memory after it has been freed, which can lead "
            "to program crashes, arbitrary code execution, or data corruption."
        ),
        "typical_fixes": (
            "Use reference counting (get/put pattern); set pointers to NULL after "
            "free; use RCU (call_rcu) for deferred free; use try_get_page/try_get_compound_head "
            "instead of unconditional get; reorder operations so free happens last."
        ),
    },
    "CWE-476": {
        "name": "NULL Pointer Dereference",
        "description": (
            "The product dereferences a pointer that it expects to be valid but is NULL."
        ),
        "typical_fixes": (
            "Add NULL check before dereference; use a safe accessor function that "
            "handles NULL internally (e.g., page_file_mapping instead of page->mapping); "
            "validate function return values before use; add bounds checks on indices "
            "used to look up pointers."
        ),
    },
    "CWE-787": {
        "name": "Out-of-bounds Write",
        "description": (
            "The product writes data past the end (or before the beginning) of the "
            "intended buffer."
        ),
        "typical_fixes": (
            "Add bounds checking before writes; validate buffer sizes against data "
            "lengths; use safe copy functions with explicit size limits; track a "
            "position counter and check against buffer length before each write; "
            "return error/FALSE when bounds would be exceeded."
        ),
    },
}


def get_cwe_info(cwe_id: str) -> dict[str, str]:
    """Return CWE knowledge for a given ID, or a generic fallback."""
    # Normalize: accept "CWE-362" or "362"
    if not cwe_id.startswith("CWE-"):
        cwe_id = f"CWE-{cwe_id}"
    return CWE_KNOWLEDGE.get(
        cwe_id,
        {
            "name": "Unknown CWE",
            "description": f"No detailed description available for {cwe_id}.",
            "typical_fixes": "Analyze the specific vulnerability context to determine the appropriate fix.",
        },
    )
