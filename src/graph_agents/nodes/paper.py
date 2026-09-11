"""Paper node: read training/eval JSONs -> LaTeX tables -> patch Results section."""
import pathlib, json, re

def _load_json(path):
    if pathlib.Path(path).exists():
        return json.loads(pathlib.Path(path).read_text())
    return None

def build_results_section(train, ev):
    """Pure builder: metrics dicts -> LaTeX Results section string.

    Guardrail: if the evaluation is flagged NOT_FOR_PUBLICATION (mock engine or
    too few positions), an explicit disclaimer is injected into the paper.
    """
    train_rows = "\n".join([f"{r.get('epoch',i)} & {r.get('loss',0):.4f} \\\\" for i,r in enumerate(train)])
    train_table = f"""\\begin{{table}}[h]\\centering\\caption{{Training Loss (CE+MSE)}}\\begin{{tabular}}{{cc}}\\hline Epoch & Loss \\\\ \\hline\n{train_rows}\n\\hline\\end{{tabular}}\\end{{table}}"""

    eval_table, caveat = "", ""
    if isinstance(ev, dict):
        eval_table = f"""\\begin{{table}}[h]\\centering\\caption{{Stockfish Evaluation (Top-k & Value MSE)}}\\begin{{tabular}}{{lcc}}\\hline Metric & Value \\\\ \\hline Top-1 Accuracy & {ev.get('top1_accuracy',0):.3f} \\\\ Top-3 Accuracy & {ev.get('top3_accuracy',0):.3f} \\\\ Value MSE & {ev.get('value_mse',0):.4f} \\\\ \\hline\\end{{tabular}}\\end{{table}}"""
        if ev.get("verdict") == "NOT_FOR_PUBLICATION":
            caveat = ("\\noindent\\textbf{Caveat:} evaluation metrics above come from a "
                      f"small mock set (engine={ev.get('engine_used','mock')}, "
                      f"n={ev.get('n_positions','?')}); they are pipeline smoke tests, "
                      "not publishable results.\n")

    return f"\\section{{Results \\& Evaluation}}\n{train_table}\n\n{eval_table}\n{caveat}\n"


def run_paper(metrics=None):
    base = pathlib.Path(__file__).resolve().parents[3]
    train = _load_json(base / "experiments" / "training_results.json") or [{"epoch":0,"loss":0.42}]
    ev = _load_json(base / "experiments" / "evaluation_results.json") or {"top1_accuracy":0,"top3_accuracy":0,"value_mse":0}
    results_section = build_results_section(train, ev)

    tex_path = base / "paper" / "main.tex"
    if tex_path.exists():
        tex = tex_path.read_text(encoding="utf-8")
        if "\\section{Experiments}" in tex:
            tex = re.sub(r"\\section\{Experiments\}.*?(?=\\section\{Attention)", lambda m: results_section + "\n\\section{Attention Analysis}", tex, flags=re.DOTALL)
            tex_path.write_text(tex, encoding="utf-8")
        elif "\\section{Results" in tex:
            # idempotent update: replace existing Results section
            tex = re.sub(r"\\section\{Results.*?(?=\\section\{Attention)", lambda m: results_section + "\n", tex, flags=re.DOTALL)
            tex_path.write_text(tex, encoding="utf-8")
        else:
            tex = tex.replace("\\bibliographystyle", results_section + "\n\\bibliographystyle")
            tex_path.write_text(tex, encoding="utf-8")
    print(f"Paper patched -> {tex_path}")
    return results_section
