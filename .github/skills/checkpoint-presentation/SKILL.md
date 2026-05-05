---
name: checkpoint-presentation
description: 'Generate LaTeX Beamer checkpoint presentations for experiment results. USE FOR: creating new checkpoint slide decks, adding experiment sections with goal/metrics/rationale/results/key-findings, updating existing presentations with new results. Keywords: presentation, checkpoint, slides, beamer, LaTeX, meeting, results, experiment.'
argument-hint: 'Describe which experiments and results to include in the presentation'
---

# Checkpoint Presentation Generator

## When to Use

- Creating a new checkpoint presentation after running experiments
- Adding experiment result sections to an existing checkpoint deck
- Summarizing retrieval / embedding / patching / knowledge experiments for a meeting

## Output Location

Presentations go in `meetings/checkpointX/` where X is the **next** sequential number.

**Procedure to determine X:**
1. List `meetings/` directory
2. Find all existing `checkpointN` folders
3. Set X = max(N) + 1
4. Create `meetings/checkpointX/`

The main `.tex` file should be named descriptively (e.g., `evaluation.tex`, `embeddings.tex`, `pipeline.tex`).

## Visual Style (from checkpoint1)

All presentations **must** use the following Beamer setup to keep a consistent look across checkpoints.

### Preamble (copy verbatim into every new deck)

See [preamble template](./assets/preamble.tex) — includes:
- `\documentclass[aspectratio=169,10pt]{beamer}` with Madrid theme
- Color palette: `c1`–`c6`, `hit`, `miss`, `warn`
- Beamer color overrides (blue frame titles, white title text, no nav symbols)
- Cell-coloring commands: `\chit`, `\cmiss`, `\cwarn`
- Required packages: `tikz`, `booktabs`, `fontawesome5`, `xcolor`, `amsmath`, `colortbl`

### Title Frame

```latex
\title{\textbf{Graph-Based Vulnerability Retrieval}\\[4pt]}
\subtitle{Master's Thesis — Technical Progress · Checkpoint X}
\author{Lívia}
\institute{TU Munich · Siemens}
\date{\today}
```

### Closing Frame

```latex
\begin{frame}[plain]
\centering
\vspace{3em}
\begin{tikzpicture}
  \node[draw=c1!30,rounded corners=6pt,fill=c1!5,
        minimum width=9cm,minimum height=1.2cm,
        font=\small\sffamily\color{black!60},align=center]
    {Repository: \texttt{graph-rag/} \quad
     \texttt{<relevant run command>}};
\end{tikzpicture}
\end{frame}
```

## Slide Structure Per Experiment

Each experiment section follows this template. Read [experiment slide template](./assets/experiment_template.tex) for the full LaTeX. The structure is:

### 1. Experiment Design Slide (Goal + Metrics + Rationale)

Use a two-column layout:

- **Left column:**
  - **Goal** — 2–3 sentences explaining what the experiment tests
  - **Metrics** — description list of metrics used (e.g., hit@K, MRR, CWE recall)
  - **Rationale** (optional) — why this experiment matters or what hypothesis it tests

- **Right column:**
  - A TikZ bar chart or visual summary of the key metric, OR
  - A compact table summarizing the experimental grid (embedder × variant × backend)

### 2. Results Table Slide

- Use `booktabs` tables (`\toprule`, `\midrule`, `\bottomrule`)
- Color-code cells with `\chit{}` (green, good), `\cwarn{}` (yellow, moderate), `\cmiss{}` (red, bad)
- Group columns by experimental condition using `\cmidrule`
- Below the table, add a **3-column interpretation** using `\begin{columns}[T]` that explains the key takeaways per condition
- Font size: `\scriptsize` or `\small` for dense tables; `\renewcommand{\arraystretch}{1.35}` for spacing

### 3. Key Findings Slide

- Use a numbered list with FontAwesome icons for visual anchors:
  - `\faIcon{bug}` for bugs/problems found
  - `\faIcon{chart-bar}` for quantitative insights
  - `\faIcon{lightbulb}` for insights/ideas
  - `\faIcon{compress-arrows-alt}` for collapse/convergence findings
  - `\faIcon{code}` for code-level findings
  - `\faIcon{check-circle}` for confirmed hypotheses
- Each item: **bold headline** + 1–2 sentence explanation + *italicized fix or implication*
- Space items with `\vspace{0.3em}` (use 0.5em only if ≤3 items)

## Agenda Slide

If the presentation covers multiple experiments, include an agenda after the title:

```latex
\begin{frame}{Agenda}
  \begin{enumerate}
    \item[\textcolor{c1}{\faIcon{icon1}}]
      \textbf{Section title} — brief description
    \vspace{0.45em}
    \item[\textcolor{c3}{\faIcon{icon2}}]
      \textbf{Section title} — brief description
    % ... one item per section
  \end{enumerate}
\end{frame}
```

Assign each section a different color from the palette (`c1`–`c6`, `hit`, `miss`) and a relevant FontAwesome icon.

## Procedure: Creating a New Checkpoint

1. **Determine checkpoint number** — list `meetings/` and increment
2. **Create directory** — `meetings/checkpointX/`
3. **Copy preamble** — from [preamble template](./assets/preamble.tex)
4. **Set title/subtitle** — update checkpoint number
5. **Add agenda** — one entry per experiment section
6. **For each experiment:**
   a. Read experiment results (from `experiments/output/` or provided data)
   b. Write the Experiment Design slide (goal, metrics, rationale)
   c. Write the Results Table slide (data + 3-column interpretation)
   d. Write the Key Findings slide
7. **Add Next Steps frame** (optional) — short/medium-term items
8. **Add closing frame** with repo info

## Data Sources

Experiment results typically come from:
- `experiments/output/` — JSON result files
- Dashboard HTML files in `experiments/dashboard_scripts/`
- User-provided tables or summaries
- The experiment Python files in `experiments/` contain the setup details (grid, metrics, splits)

## Critical Patterns

### Table conventions
- Always use `booktabs` (never `\hline`)
- Percentages: `76.7\%` format
- Float scores: 3 decimal places (`.802`)
- Row highlighting: `\rowcolor{c1!12}` for best row
- Column grouping: `\cmidrule{2-4} \cmidrule{6-8}` with `&&` column spacers

### TikZ conventions
- Bar charts: horizontal bars, color-coded by performance tier
- Pipeline diagrams: left-to-right flow with `box` styles matching the palette
- Node styles: `rounded corners=3pt`, `font=\small\sffamily`

### Column layouts
- Two-column: `\begin{columns}[T]` with `0.48\textwidth` + `0.48\textwidth` or `0.52` + `0.45`
- Three-column interpretation: `0.32\textwidth` × 3

### Avoiding Overflow (critical — Beamer does NOT auto-paginate)

Beamer frames have a **fixed height**. Content that exceeds it silently overflows off the bottom. Always design for the frame boundary.

**Table overflow (most common):**
- Tables with 8+ data columns will overflow on 16:9 slides. Max ~8–9 narrow columns comfortably.
- **Drop redundant columns.** If values are identical across rows (e.g., hit@5 = 95.9% for all strategies), remove the column and add a footnote: `{\scriptsize\color{black!55} hit@5 identical across all (95.9\%) --- omitted.}`
- Use `\scriptsize` (not `\small`) for dense result tables.
- Use `\setlength{\tabcolsep}{3pt}` (not 4.5pt+) for wide tables.
- Use `\renewcommand{\arraystretch}{1.25}` (not 1.35) when vertical space is tight.
- For `p{}` column widths in two-column frames, keep total width under ~4.5cm per column (e.g., `p{2.0cm}p{4.5cm}` not `p{2.2cm}p{4.8cm}`).

**Text overflow below tables:**
- Three-column interpretation bullets below a table: use `\scriptsize` for headers (not `\small`), `\itemsep` of 1pt (not 2pt), and trim bullet text to single short lines.
- Reduce `\vspace` between table and columns to 0.2–0.3em (not 0.5em+).

**Key Findings slides:**
- Use `\vspace{0.3em}` between items (not 0.5em) — 4 items at 0.5em will overflow.
- Keep each finding to 2 lines of body text max. Trim verbose phrases (e.g., "individual embeddings before concatenation" → "before concatenation").
- If 4+ findings, consider `\scriptsize` for body text.

**General prevention:**
- After writing a frame, mentally estimate: title bar (~1cm) + each table row (~0.4cm at scriptsize) + vspace + columns. Total must fit ~8.5cm of usable height on 16:9 at 10pt.
- When in doubt, compile and check, or preemptively use smaller font sizes and tighter spacing.
