"""
Unified agent-harness CLI.

    python -m src.agents.harness <command> [options]

Commands
--------
  variants   List / describe / diff registered prompt variants.
  debug      Run ONE scenario and print the full trace (prompt, output, scores),
             or replay a saved InvocationRecord JSON.
  record     Run a suite live and record every LLM call to a cassette (needs
             AZURE_* keys). Build reusable offline fixtures this way.
  replay     Run a suite fully offline from a cassette. Deterministic.
  eval       Score generated patches — from a batch ``results.jsonl`` or from a
             fixture suite — with the unified evaluator.
  compare    N-way prompt-variant comparison over a fixture suite (offline via
             replay, or live).

Backends: --backend {mock,replay,record,live}. ``replay``/``record`` need
--cassette. ``mock`` and ``replay`` never touch the network.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── backend construction ─────────────────────────────────────────────


def _make_backend(
    kind: str, *, cassette: str | None = None, echo_patch: str | None = None
):
    from src.agents.backends import (
        Cassette,
        LiveBackend,
        MockBackend,
        RecordBackend,
        ReplayBackend,
    )

    if kind == "live":
        return LiveBackend(), None
    if kind == "mock":
        return (
            MockBackend.from_patch("(mock reasoning)", echo_patch or "// mock patch"),
            None,
        )
    if kind in ("replay", "record"):
        if not cassette:
            sys.exit(f"--backend {kind} requires --cassette PATH")
        cass = Cassette(cassette)
        if kind == "replay":
            return ReplayBackend(cass), cass
        return RecordBackend(LiveBackend(), cass), cass
    sys.exit(f"unknown backend: {kind}")


def _load(fixtures: str):
    from src.agents.harness.scenario import load_scenarios

    scns = load_scenarios(fixtures)
    if not scns:
        sys.exit(f"No scenarios found in {fixtures}")
    return scns


# ── commands ─────────────────────────────────────────────────────────


def cmd_variants(args) -> None:
    from src.agents.harness import registry

    if args.diff:
        a, b = args.diff
        for entry in registry.diff_variants(a, b):
            tag = "CHANGED" if entry["changed"] else "same"
            print(f"[{entry['index']}] {entry['role_a']}→{entry['role_b']}  {tag}")
            if entry.get("diff"):
                print(entry["diff"])
        return

    rows = registry.describe()
    print(f"{'variant':<16} {'msgs':>4} {'fingerprint':<14} roles")
    print("─" * 60)
    for r in rows:
        print(
            f"{r['variant']:<16} {r['n_messages']:>4} {r['fingerprint']:<14} "
            f"{','.join(r['roles'])}"
        )


def cmd_debug(args) -> None:
    from src.agents.harness.evaluator import AgentEvaluator
    from src.agents.harness.runner import AgentHarness, format_trace, replay_record

    # Replay a saved InvocationRecord (no LLM call).
    if args.from_record:
        reference = ""
        if args.reference_file:
            reference = Path(args.reference_file).read_text(errors="replace")
        card = replay_record(args.from_record, reference=reference)
        print(json.dumps(card.to_dict(), indent=2, default=str))
        return

    scns = _load(args.fixtures)
    scn = next((s for s in scns if s.id == args.id), None) if args.id else scns[0]
    if scn is None:
        sys.exit(f"scenario id {args.id!r} not found in {args.fixtures}")

    backend, _ = _make_backend(
        args.backend, cassette=args.cassette, echo_patch=scn.ground_truth
    )
    harness = AgentHarness(
        evaluator=AgentEvaluator(judge=args.judge),
        model_name=args.model,
        prompt_variant=args.variant,
    )
    trace = harness.run_scenario(scn, backend=backend)
    print(format_trace(trace))


def _run_suite(args, backend_kind: str):
    from src.agents.harness.evaluator import AgentEvaluator
    from src.agents.harness.runner import AgentHarness, save_traces

    scns = _load(args.fixtures)
    backend, cass = _make_backend(backend_kind, cassette=args.cassette)
    harness = AgentHarness(
        evaluator=AgentEvaluator(judge=getattr(args, "judge", False)),
        model_name=args.model,
        prompt_variant=args.variant,
    )
    print(
        f"Running {len(scns)} scenarios  (backend={backend_kind}, variant={args.variant})"
    )
    result = harness.run_suite(scns, backend=backend, progress=True)
    if cass is not None and backend_kind == "record":
        cass.save()
        print(f"Cassette saved: {cass.path}  ({len(cass)} entries)")
    print("\n── Summary " + "─" * 50)
    print(json.dumps(result.summary, indent=2, default=str))
    if args.out:
        save_traces(result.traces, args.out)
        Path(args.out).with_suffix(".summary.json").write_text(
            json.dumps(result.summary, indent=2, default=str)
        )
        print(f"Traces: {args.out}")
    return result


def cmd_record(args) -> None:
    _run_suite(args, "record")


def cmd_replay(args) -> None:
    _run_suite(args, "replay")


def cmd_eval(args) -> None:
    from src.agents.harness.evaluator import AgentEvaluator

    ev = AgentEvaluator(judge=args.judge)

    # Mode A: score an existing batch results.jsonl (no LLM needed).
    if args.results:
        cards = []
        with open(args.results) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                gen = rec.get("generated_patch") or ""
                ref = rec.get("ground_truth_patch") or ""
                if not gen:
                    continue
                cards.append(ev.score(gen, ref, scenario_id=rec.get("query_cve", "")))
        summary = ev.aggregate(cards)
        print(json.dumps(summary, indent=2, default=str))
        if args.out:
            Path(args.out).write_text(
                json.dumps(
                    {"summary": summary, "cards": [c.to_dict() for c in cards]},
                    indent=2,
                    default=str,
                )
            )
            print(f"Scorecards: {args.out}")
        return

    # Mode B: run a fixture suite through a backend and score it.
    if not args.fixtures:
        sys.exit("eval needs either --results <results.jsonl> or --fixtures <suite>")
    _run_suite(args, args.backend)


def cmd_compare(args) -> None:
    from src.agents.harness.compare import compare_variants
    from src.agents.harness.evaluator import AgentEvaluator

    scns = _load(args.fixtures)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    backend, _ = _make_backend(args.backend, cassette=args.cassette)
    report = compare_variants(
        scns,
        variants,
        backend=backend,
        model_name=args.model,
        evaluator=AgentEvaluator(judge=args.judge),
        baseline=args.baseline,
        progress=True,
    )
    print("\n" + report.render_table())
    if args.out:
        Path(args.out).write_text(json.dumps(report.to_dict(), indent=2, default=str))
        print(f"\nComparison: {args.out}")


# ── argument parser ──────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.agents.harness", description=__doc__
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp, *, needs_variant=True):
        sp.add_argument("--fixtures", default="tests/fixtures/agent/scenarios.json")
        if needs_variant:
            sp.add_argument("--variant", default="default", help="prompt variant")
        sp.add_argument("--model", default=None, help="model/deployment name override")
        sp.add_argument(
            "--backend", default="mock", choices=["mock", "replay", "record", "live"]
        )
        sp.add_argument(
            "--cassette", default=None, help="cassette path for replay/record"
        )
        sp.add_argument("--out", default=None, help="write traces/results here")

    sp = sub.add_parser("variants", help="list / diff prompt variants")
    sp.add_argument("--diff", nargs=2, metavar=("A", "B"), help="diff two variants")
    sp.set_defaults(func=cmd_variants)

    sp = sub.add_parser("debug", help="run one scenario, print full trace")
    add_common(sp)
    sp.add_argument("--id", default=None, help="scenario id (default: first)")
    sp.add_argument("--judge", action="store_true", help="also run LLM judge")
    sp.add_argument(
        "--from-record", default=None, help="replay a saved InvocationRecord JSON"
    )
    sp.add_argument(
        "--reference-file", default=None, help="ground-truth file for --from-record"
    )
    sp.set_defaults(func=cmd_debug)

    sp = sub.add_parser("record", help="run suite live, record a cassette")
    add_common(sp)
    sp.set_defaults(func=cmd_record, backend="record")

    sp = sub.add_parser("replay", help="run suite offline from a cassette")
    add_common(sp)
    sp.add_argument("--judge", action="store_true")
    sp.set_defaults(func=cmd_replay, backend="replay")

    sp = sub.add_parser("eval", help="score results.jsonl or a fixture suite")
    add_common(sp)
    sp.add_argument("--results", default=None, help="batch results.jsonl to score")
    sp.add_argument("--judge", action="store_true")
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("compare", help="N-way prompt-variant comparison")
    add_common(sp, needs_variant=False)
    sp.add_argument("--variants", required=True, help="comma-separated variant names")
    sp.add_argument("--baseline", default=None, help="baseline label for deltas")
    sp.add_argument("--judge", action="store_true")
    sp.set_defaults(func=cmd_compare)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
