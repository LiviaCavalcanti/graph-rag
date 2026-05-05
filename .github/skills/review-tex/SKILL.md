---
name: review-tex
description: 'Review LaTeX Beamer meeting .tex files for quality. USE FOR: checking LaTeX errors/warnings, verifying agenda–content alignment, detecting overfull slides that should be split, grammar and spelling review, enforcing project Beamer conventions. Keywords: review, LaTeX, beamer, slides, grammar, agenda, overfull, split, lint, proofread, meetings, checkpoint.'
argument-hint: 'Path to the .tex file to review, or "all" to review every checkpoint'
---

# LaTeX Meeting Presentation Reviewer

## When to Use

- After creating or editing a checkpoint `.tex` file
- Before compiling a final PDF for a meeting
- When asked to proofread, lint, or review a presentation
- When a presentation feels "too dense" and may need splitting

## Input

The user provides either:
- A **specific `.tex` file path** (e.g., `meetings/checkpoint4/evaluation.tex`)
- The word **"all"** — review every `.tex` file under `meetings/`

If no path is given, default to the currently open file if it is a `.tex` under `meetings/`.

## Review Procedure

Perform **all five** checks below, in order. Report findings grouped by category. For each issue found, quote the relevant LaTeX snippet and state the fix.

---

### 1. LaTeX Errors & Warnings

**Goal:** Catch anything that would cause compilation failures or visual defects.

**Checks:**
- **Unmatched braces** — every `{` has a matching `}`. Pay special attention to nested `\textbf{...\textit{...}}` and TikZ nodes.
- **Unmatched environments** — every `\begin{X}` has a `\end{X}`. Common mismatches: `frame`, `columns`, `column`, `itemize`, `enumerate`, `tabular`, `tikzpicture`.
- **Missing required arguments** — e.g., `\chit{}` (empty cell highlight), `\textcolor{}{}` with missing color name.
- **Undeclared colors** — any `\textcolor{name}` or `\cellcolor{name}` where `name` is not in the preamble palette (`c1`–`c6`, `hit`, `miss`, `warn`, `black`, `white`, standard LaTeX colors).
- **Undeclared commands** — custom commands (`\chit`, `\cmiss`, `\cwarn`, `\czero`) must be defined in the preamble.
- **Bad table structure** — column count in header vs. data rows, missing `\\` at row ends, `&` count mismatches.
- **Escaped characters** — `%`, `_`, `&`, `#`, `$` must be escaped in text mode (but NOT in math mode or command names).
- **Fragile frame without `[fragile]`** — frames containing `\verb` or `verbatim` environments need `[fragile]`.
- **Missing packages** — commands used without the corresponding `\usepackage`. Cross-check the preamble.
- **Orphaned comments mid-line** — a `%` that accidentally truncates real content (e.g., a sentence ending with `shou` followed by a comment).

**How to check:** Read the file line by line. Use a mental brace/environment stack. Flag any imbalance.

---

### 2. Agenda–Content Alignment

**Goal:** Every agenda item must correspond to actual content, and every content section should be listed in the agenda.

**Checks:**
- **Parse the agenda frame** — extract each `\textbf{...}` item from the agenda `enumerate`.
- **Parse `\section{}` and `\begin{frame}{...}` titles** — build a list of actual content sections/frames.
- **Missing content** — agenda item exists but no matching section or frames follow. Flag as: *"Agenda lists 'X' but no corresponding section/frame was found."*
- **Missing agenda entry** — a `\section{}` or frame exists with no matching agenda item. Flag as: *"Section 'Y' has no agenda entry."*
- **Order mismatch** — agenda order should match the document order of sections. Flag if out of sequence.
- **Incomplete agenda descriptions** — an agenda `\textbf{...}` line that ends abruptly (e.g., trailing `\&` with no text after, or an empty `—` description). Flag as: *"Agenda item 'X' has an incomplete description."*

**How to check:** Extract agenda items and section/frame titles into two ordered lists. Diff them.

---

### 3. Slide Density / Overfull Detection

**Goal:** Identify frames that are too dense for a 10pt Beamer 16:9 slide and suggest splitting.

**Heuristics — flag a frame if ANY of these hold:**

| Signal | Threshold | Rationale |
|--------|-----------|-----------|
| Raw line count (between `\begin{frame}` and `\end{frame}`) | > 65 lines | Dense frames overflow vertically |
| Number of `\item` entries in a single list | > 8 items | Audience can't absorb >8 bullets |
| Number of table rows (`\\` inside `tabular`) | > 10 data rows | Table won't fit at `\scriptsize` |
| Nested two-column layout with BOTH columns containing a table | always | Two side-by-side tables rarely fit |
| Multiple `\section` or conceptual topics inside one frame | always | One idea per slide |
| `\tiny` or `\fontsize` used to force content to fit | always | Content should fit at `\scriptsize` minimum |

**Suggested split strategy:**
- **Dense table + interpretation** → split into "Results" slide (table only) + "Analysis" slide (interpretation columns).
- **Design + Results on one frame** → split into two frames following the experiment template (Design, then Results).
- **Long bullet list** → group into 2 slides by theme, or convert to a compact table.
- **Two-column frame where each column is independently complex** → give each column its own slide.

**Output format per flagged frame:**
```
OVERFULL: Frame "<title>" (lines X–Y)
  - Reason: <which threshold was exceeded>
  - Suggestion: <how to split>
```

---

### 4. Grammar & Spelling

**Goal:** Catch grammar mistakes, typos, and awkward phrasing in all visible text (frame titles, bullet points, table headers, TikZ labels, comments intended for the speaker).

**Checks:**
- **Spelling errors** — check every English word outside of LaTeX commands, math mode, and code literals (e.g., `\texttt{...}` content is exempt).
- **Subject-verb agreement** — especially in bullet points that start with a noun phrase.
- **Sentence fragments** — bullet items should be parallel in structure (all start with a verb, or all start with a noun phrase). Flag inconsistency within a single list.
- **Dangling modifiers** — e.g., "Running the experiment, the results show..." (the results didn't run the experiment).
- **Punctuation in lists** — within a single `itemize`/`enumerate`, items should be consistently punctuated (all with periods, or all without). Flag mixed styles.
- **Double spaces / double words** — "the the", "  " (two spaces not around a `~` or `\`).
- **British vs. American English consistency** — pick one and flag mixed usage (e.g., "normalisation" vs. "normalization").
- **Capitalization of frame titles** — should use Title Case consistently across the deck.
- **Abbreviations** — first use should be expanded; check for undefined acronyms (CWE, MRR, nDCG, PCA, etc. — if defined in an earlier frame, subsequent use is fine).
- **Incomplete sentences / truncated text** — text that appears cut off, possibly due to a stray `%` comment character.

**How to check:** Extract all text content (strip LaTeX commands), then review sentence by sentence.

---

### 5. Style Consistency (Project Conventions)

**Goal:** Ensure the presentation follows the established Beamer conventions from the `checkpoint-presentation` skill.

**Checks:**
- **Preamble** — must include the standard color palette (`c1`–`c6`, `hit`, `miss`, `warn`), Madrid theme, 169 aspect ratio, no navigation symbols, frame number footline.
- **Title frame** — subtitle must say "Checkpoint N" with the correct N for the directory.
- **Closing frame** — must use the standard `\begin{frame}[plain]` with the TikZ repo box.
- **Table style** — must use `booktabs` (`\toprule`, `\midrule`, `\bottomrule`). Flag any `\hline`.
- **Cell coloring** — use `\chit{}`, `\cmiss{}`, `\cwarn{}` for colored cells. Flag raw `\cellcolor{}` (should use the shorthand).
- **Key Findings slides** — should use FontAwesome icons (`\faIcon{...}`) and follow the bold-headline + explanation + italicized-implication pattern.
- **Section separators** — each experiment section should be preceded by `\section{...}`.
- **Spacing** — `\vspace{0.3em}` between items (0.5em only if ≤3 items).
- **Font sizing** — tables should use `\scriptsize` or `\small`; never `\tiny` for data.
- **Array stretch** — `\renewcommand{\arraystretch}{1.35}` in tables for readability.

---

## Output Format

Present the review as a structured report with these sections:

```
## Review: <filename>

### 1. LaTeX Errors & Warnings
<findings or "No issues found.">

### 2. Agenda–Content Alignment
<findings or "Agenda matches content.">

### 3. Overfull Slides
<findings or "All slides within density limits.">

### 4. Grammar & Spelling
<findings or "No issues found.">

### 5. Style Consistency
<findings or "Follows all conventions.">

### Summary
- Total issues: N
- Critical (compilation-breaking): N
- Recommended splits: N
- Grammar/style: N
```

## Severity Levels

Use these labels for each finding:

| Label | Meaning |
|-------|---------|
| **ERROR** | Will cause compilation failure or missing content |
| **WARNING** | Compiles but produces visual defect or incorrect output |
| **SUGGESTION** | Improvement for clarity, density, or style |

## Fix Mode

If the user asks to "fix" or "apply fixes", make the edits directly in the `.tex` file:
- For errors/warnings: fix them immediately.
- For overfull slides: split frames following the experiment template structure.
- For grammar: correct in-place.
- For style issues: update to match conventions.

Always show a summary of changes made after applying fixes.
