#!/usr/bin/env python3
"""Draw the figure at the top of README.md from the result files.

    python3 tools/readme_figure.py

Reads models/<model>/recipe.toml and models/<model>/results/task-*.json. Writes docs/images/results-light.svg and
docs/images/results-dark.svg. It counts results by model kind and verdict only, so the figure names no runtime,
device or package. It counts the task results that judge a copied cell again (a [[cell]] with `z1_cell` or `z2_cell`:
the two groups of the earlier repos); a task result added later, with no copied cell behind it, is not in the figure.
"""
import json
import pathlib
import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
THEMES = {
    "light": {"text": "#0b0b0b", "muted": "#52514e", "pass": "#2a78d6", "fail": "#eb6834"},
    "dark": {"text": "#f0efec", "muted": "#c3c2b7", "pass": "#3987e5", "fail": "#d95926"},
}
ROWS = [
    ("llm", "LLM packages that passed a text-quality check"),
    ("vision", "vision and audio files that failed a raw-output comparison"),
]
TITLE = "Quick checks and the real job can disagree"
WIDTH, LEFT, BAR, GAP = 860, 0, 28, 2


def count():
    """Return {kind: [passed, failed]} over the task results that judge a copied cell again, in every model."""
    counts = {}
    for recipe in sorted((ROOT / "models").glob("*/recipe.toml")):
        with recipe.open("rb") as f:
            r = tomllib.load(f)
        kind = r["model"]["kind"]
        again = {c["id"] for c in r.get("cell") or [] if c.get("z1_cell") or c.get("z2_cell")}
        for result in sorted(recipe.parent.glob("results/task-*.json")):
            if result.stem not in again:
                continue
            verdict = json.loads(result.read_text(encoding="utf-8"))["line"]["pass"]
            if verdict is None:
                raise SystemExit(f"{result}: no verdict")
            counts.setdefault(kind, [0, 0])[0 if verdict else 1] += 1
    return counts


def text(x, y, body, size, fill, weight="400"):
    return f'<text x="{x:g}" y="{y:g}" font-size="{size}" font-weight="{weight}" fill="{fill}">{body}</text>'


def block(x, y, w, fill):
    return f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{BAR}" rx="4" fill="{fill}"/>'


def draw(counts, theme):
    c = THEMES[theme]
    biggest = max(sum(counts[k]) for k, _ in ROWS)
    unit = (WIDTH - 2 * LEFT) / biggest
    parts = [
        text(LEFT, 30, TITLE, 20, c["text"], "600"),
        block(LEFT, 44, 14, c["pass"]).replace(f'height="{BAR}"', 'height="14"'),
        text(LEFT + 20, 56, "Passed the task check", 14, c["muted"]),
        block(LEFT + 180, 44, 14, c["fail"]).replace(f'height="{BAR}"', 'height="14"'),
        text(LEFT + 200, 56, "Failed the task check", 14, c["muted"]),
    ]
    y = 96
    desc = []
    for kind, label in ROWS:
        passed, failed = counts[kind]
        total = passed + failed
        parts.append(text(LEFT, y, f"{total} {label}", 15, c["text"]))
        bar_y = y + 10
        w_pass, w_fail = passed * unit, failed * unit
        parts.append(block(LEFT, bar_y, w_pass - GAP / 2, c["pass"]))
        parts.append(block(LEFT + w_pass + GAP / 2, bar_y, w_fail - GAP / 2, c["fail"]))
        parts.append(text(LEFT + 10, bar_y + 19, f"{passed} passed", 14, "#ffffff", "600"))
        parts.append(text(LEFT + w_pass + GAP / 2 + 10, bar_y + 19, f"{failed} failed", 14, "#0b0b0b", "600"))
        desc.append(f"Of {total} {label}, {passed} passed the task check and {failed} failed it.")
        y += 72
    height = y - 72 + 10 + BAR + 12
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height:g}" width="{WIDTH}" height="{height:g}" '
        f'role="img" aria-labelledby="t d" font-family="{FONT}">\n'
        f'<title id="t">{TITLE}</title>\n<desc id="d">{" ".join(desc)}</desc>\n' + "\n".join(parts) + "\n</svg>\n"
    )


def main():
    counts = count()
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        path = OUT / f"results-{theme}.svg"
        path.write_text(draw(counts, theme), encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")
    for kind, (passed, failed) in sorted(counts.items()):
        print(f"{kind}: {passed} passed, {failed} failed")


if __name__ == "__main__":
    main()
