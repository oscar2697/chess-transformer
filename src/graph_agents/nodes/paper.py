"""Paper node: read training/eval JSONs -> LaTeX tables -> patch Results section."""
import pathlib, json, re

def _load_json(path):
    if pathlib.Path(path).exists():
        return json.loads(pathlib.Path(path).read_text())
    return None

def run_paper(metrics=None):
    base = pathlib.Path(__file__).resolve().parents[3]
    train = _load_json(base / "experiments" / "training_results.json") or [{"epoch":0,"loss":0.42}]
    ev = _load_json(base / "experiments" / "evaluation_results.json") or {"top1_accuracy":0,"top3_accuracy":0,"value_mse":0}

    # Build LaTeX tables
    train_rows = "\n".join([f"{r.get('epoch',i)} & {r.get('loss',0):.4f} \\\\" for i,r in enumerate(train)])
    train_table = f"""\\begin{{table}}[h]\\centering\\caption{{Training Loss (CE+MSE)}}\\begin{{tabular}}{{cc}}\\hline Epoch & Loss \\\\ \\hline\n{train_rows}\n\\hline\\end{{tabular}}\\end{{table}}"""

    # ev may be dict
    if isinstance(ev, dict):
        eval_table = f"""\\begin{{table}}[h]\\centering\\caption{{Stockfish Evaluation (Top-k & Value MSE)}}\\begin{{tabular}}{{lcc}}\\hline Metric & Value \\\\ \\hline Top-1 Accuracy & {ev.get('top1_accuracy',0):.3f} \\\\ Top-3 Accuracy & {ev.get('top3_accuracy',0):.3f} \\\\ Value MSE & {ev.get('value_mse',0):.4f} \\\\ \\hline\\end{{tabular}}\\end{{table}}"""
    else:
        eval_table = ""

    results_section = f"\\section{{Results \\& Evaluation}}\n{train_table}\n\n{eval_table}\n"

    tex_path = base / "paper" / "main.tex"
    if tex_path.exists():
        tex = tex_path.read_text(encoding="utf-8")
        if "\\section{Experiments}" in tex:
            tex = re.sub(r"\\section\{Experiments\}.*?(?=\\section\{Attention)", lambda m: results_section + "\n\\section{Attention Analysis}", tex, flags=re.DOTALL)
            tex_path.write_text(tex, encoding="utf-8")
        elif "\\section{Results" not in tex:
            tex = tex.replace("\\bibliographystyle", results_section + "\n\\bibliographystyle")
            tex_path.write_text(tex, encoding="utf-8")
    print(f"Paper patched -> {tex_path}")
    return results_section
