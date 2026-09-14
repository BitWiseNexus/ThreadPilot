"""CLI: `python -m threadpilot.cli triage "..."`.

Output follows docs/design.md section 2. The governing rule there: a reviewer
must be able to see the evidence next to the claim, so a draft reply is never
shown without the precedent it was grounded in, and similarity scores are always
printed. A retrieval claim without its score is not inspectable.

Colour carries exactly one meaning - the decision. Amber for escalate rather
than red, because escalation is the correct, expected outcome for a large share
of traffic and red would frame correct caution as failure.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from . import config, retrieval
from .pipeline import TriageResult, triage_many


def _console(plain: bool):
    from rich.console import Console
    # no_color for piped output so redirected results diff cleanly
    return Console(no_color=plain, soft_wrap=True)


def render(res: TriageResult, console, *, show_k: int = 3) -> None:
    from rich.panel import Panel
    from rich.text import Text

    d = res.decision
    accent = "yellow" if d.is_escalate else "green"
    label = "ESCALATE" if d.is_escalate else "AUTO-HANDLE"

    console.print(Panel(Text(res.text), title="INBOUND", title_align="left",
                        border_style="grey50"))

    c = res.classification
    amb = f"  (vs {c.ambiguous_with})" if c.ambiguous_with else ""
    console.print(f"[bold]INTENT[/]     {c.intent}{amb}"
                  f"   confidence {c.confidence:.2f}")
    gates = "+".join(d.gate_codes) if d.gates_fired else "none (LLM proposal accepted)"
    console.print(f"[bold]DECISION[/]   [{accent}]{label}[/]   gate {gates}")
    if d.overrode_model:
        console.print(f"[dim]           gates overrode the model, which proposed "
                      f"{d.model_proposal}[/]")

    if res.precedents:
        lines = []
        for p in res.precedents[:show_k]:
            lines.append(f"[{p.similarity:.2f}] \"{p.customer_msg[:110]}\"")
            lines.append(f"    -> \"{p.brand_reply[:110]}\"  [dim]({p.reply_kind})[/]")
        title = f"PRECEDENT (top {min(show_k, len(res.precedents))} of {len(res.precedents)})"
        console.print(Panel("\n".join(lines), title=title, title_align="left",
                            border_style="grey50"))
    else:
        console.print("[dim]PRECEDENT   none retrieved[/]")

    console.print(Panel(Text(res.draft.reply or "(no draft produced)"),
                        title="DRAFT", title_align="left", border_style=accent))
    console.print(f"[bold]REASON[/]     {d.reason}\n")


def cmd_triage(args) -> int:
    console = _console(args.plain)
    try:
        index = retrieval.RetrievalIndex.load()
    except FileNotFoundError as e:
        sys.exit(str(e))

    texts = list(args.text)
    if args.file:
        texts += [l.strip() for l in open(args.file, encoding="utf-8") if l.strip()]
    if not texts:
        sys.exit("nothing to triage: pass text, or --file")

    results = triage_many(texts, index=index, k=args.k, offline=args.offline,
                          use_gates=not args.no_gates,
                          use_retrieval=not args.no_retrieval)

    if args.json:
        for r in results:
            print(json.dumps(r.to_dict(), ensure_ascii=False))
        return 0

    for r in results:
        render(r, console, show_k=args.show_k)

    n_esc = sum(1 for r in results if r.decision.is_escalate)
    console.print(f"[dim]{len(results)} triaged: {len(results)-n_esc} auto-handle, "
                  f"{n_esc} escalate[/]")
    return 0


def cmd_index(args) -> int:
    idx = retrieval.RetrievalIndex.load()
    print(json.dumps(idx.info, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="threadpilot", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("triage", help="triage one or more messages")
    t.add_argument("text", nargs="*", help="message text")
    t.add_argument("--file", help="file with one message per line")
    t.add_argument("-k", type=int, default=config.RETRIEVAL_K)
    t.add_argument("--show-k", type=int, default=3,
                   help="precedents to display (all are still retrieved)")
    t.add_argument("--json", action="store_true", help="JSONL output")
    t.add_argument("--plain", action="store_true", help="no colour, for piping")
    t.add_argument("--offline", action="store_true",
                   help="replay the committed cache; no API key needed")
    t.add_argument("--no-gates", action="store_true",
                   help="ablation: pure LLM decision, no deterministic gates")
    t.add_argument("--no-retrieval", action="store_true",
                   help="ablation: draft without precedent")
    t.set_defaults(func=cmd_triage)

    i = sub.add_parser("index", help="show retrieval index info")
    i.set_defaults(func=cmd_index)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING,
                        format="[%(levelname)s] %(message)s", stream=sys.stderr)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
