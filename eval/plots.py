"""Generate the comparison charts used in README.md.

Reads the most recent saved run per strategy from results/ and writes PNGs to
docs/images/. Re-run after any new eval:

    python eval/plots.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


RESULTS_DIR = Path('results')
OUT_DIR = Path('docs/images')

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_SOFT = '#52514e'
GRID = '#e3e2de'

ORDER = [
    'Baseline (dense only)',
    'Hybrid (dense + BM25)',
    'Reranked',
    'Hybrid + Reranked',
    'Decomposed',
]
SHORT = {
    'Baseline (dense only)': 'Baseline',
    'Hybrid (dense + BM25)': 'Hybrid',
    'Reranked': 'Reranked',
    'Hybrid + Reranked': 'Hybrid + Rerank',
    'Decomposed': 'Decomposed',
}
CATEGORIES = ['single-chunk', 'multi-hop', 'exact-symbol', 'ambiguous']


def load_latest_runs(results_dir=RESULTS_DIR):
    """Most recent saved run per strategy label."""
    latest = {}
    for folder in sorted(results_dir.iterdir()):
        summary = folder / 'summary.json'
        if not summary.exists():
            continue
        saved = json.loads(summary.read_text(encoding='utf-8'))
        label = saved['metrics']['label']
        run_id = saved['run']['run_id']
        if label not in latest or run_id > latest[label]['run']['run_id']:
            latest[label] = saved
    return latest


def _style(ax, xlabel=None, title=None, subtitle=None):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.spines['bottom'].set_linewidth(1)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=1, linestyle='-')
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SOFT, fontsize=9, labelpad=8)
    if title:
        ax.set_title(title, color=INK, fontsize=13, fontweight='semibold',
                     loc='left', pad=22 if subtitle else 12)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_SOFT,
                fontsize=9.5, va='bottom')


def _legend_below(ax, ncols):
    """Legend outside the plot area, so it can never overlap a mark."""
    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14),
                    ncols=ncols, frameon=False, fontsize=9.5,
                    handlelength=1.1, handleheight=1.1, columnspacing=1.8)
    for text in leg.get_texts():
        text.set_color(INK_SOFT)
    return leg


def chart_quality(runs, path):
    """Recall@5 and MRR per strategy."""
    labels = [s for s in ORDER if s in runs]
    recall = [runs[s]['metrics']['recall_at_k'] for s in labels]
    mrr = [runs[s]['metrics']['mrr'] for s in labels]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = range(len(labels))
    height = 0.20
    gap = 0.03

    ax.barh([i + (height + gap) / 2 for i in y], recall, height,
            color=SERIES[0], label='Recall@5', zorder=2)
    ax.barh([i - (height + gap) / 2 for i in y], mrr, height,
            color=SERIES[1], label='MRR', zorder=2)

    for i, (r, m) in enumerate(zip(recall, mrr)):
        ax.text(r + 0.012, i + (height + gap) / 2, f'{r:.3f}',
                va='center', ha='left', fontsize=9, color=INK)
        ax.text(m + 0.012, i - (height + gap) / 2, f'{m:.3f}',
                va='center', ha='left', fontsize=9, color=INK_SOFT)

    ax.set_yticks(list(y))
    ax.set_yticklabels([SHORT[s] for s in labels], color=INK, fontsize=10)
    ax.set_xlim(0, 0.88)
    ax.invert_yaxis()
    _style(ax, xlabel='score (higher is better)',
           title='Retrieval quality by strategy',
           subtitle='56 gold questions over the Pydantic docs; 46 answerable')
    _legend_below(ax, ncols=2)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


def chart_per_category(runs, path):
    """Recall@5 broken out by question category."""
    labels = [s for s in ORDER if s in runs]
    fig, ax = plt.subplots(figsize=(9, 5.2))

    n = len(labels)
    height = 0.58 / n
    gap = 0.022
    for si, strategy in enumerate(labels):
        per_cat = runs[strategy]['metrics']['per_category']
        vals = [per_cat.get(c, {}).get('recall_at_k', 0) for c in CATEGORIES]
        offset = (n - 1) / 2 * (height + gap) - si * (height + gap)
        ax.barh([i + offset for i in range(len(CATEGORIES))], vals, height,
                color=SERIES[si], label=SHORT[strategy], zorder=2)

    counts = {c: runs[labels[0]]['metrics']['per_category'].get(c, {}).get('n', 0)
              for c in CATEGORIES}
    ax.set_yticks(range(len(CATEGORIES)))
    ax.set_yticklabels([f'{c}  (n={counts[c]})' for c in CATEGORIES],
                       color=INK, fontsize=10)
    ax.set_xlim(0, 1.04)
    ax.invert_yaxis()
    _style(ax, xlabel='recall@5',
           title='Where each strategy wins and loses',
           subtitle='Reranking lifts single-chunk and ambiguous questions but costs multi-hop')
    _legend_below(ax, ncols=5)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


def chart_cost(runs, path):
    """Quality against wall-clock cost. One colour; identity comes from labels."""
    labels = [s for s in ORDER if s in runs]
    xs = [runs[s]['metrics']['latency_s']['total'] for s in labels]
    ys = [runs[s]['metrics']['recall_at_k'] for s in labels]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.scatter(xs, ys, s=110, color=SERIES[0], zorder=3,
               edgecolor=SURFACE, linewidth=2)

    offsets = {
        'Baseline (dense only)': (12, 7, 'left'),
        'Hybrid (dense + BM25)': (12, 7, 'left'),
        'Decomposed':            (12, 7, 'left'),
        'Reranked':              (0, 15, 'center'),
        'Hybrid + Reranked':     (0, -38, 'center'),
    }
    for x, y, s in zip(xs, ys, labels):
        dx, dy, ha = offsets.get(s, (12, 7, 'left'))
        ax.annotate(f'{SHORT[s]}\n{y:.3f} · {x:.0f}s',
                    (x, y), textcoords='offset points', xytext=(dx, dy),
                    ha=ha, fontsize=9, color=INK, linespacing=1.4)

    ax.set_xscale('log')
    ax.set_xlim(1.5, 800)
    ax.set_ylim(0.55, 0.80)
    _style(ax, xlabel='total time for 56 questions, log scale (seconds)',
           title='Quality against cost',
           subtitle='Hybrid fusion captures most of the gain for a fraction of the time')
    ax.set_ylabel('recall@5', color=INK_SOFT, fontsize=9, labelpad=8)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


def markdown_table(runs):
    """The table view that accompanies the charts (and provides label relief)."""
    labels = [s for s in ORDER if s in runs]
    head = ('| Strategy | Recall@5 | MRR | single-chunk | multi-hop | '
            'exact-symbol | ambiguous | Time (56 q) |')
    sep = '|---|---|---|---|---|---|---|---|'
    rows = [head, sep]
    for s in labels:
        m = runs[s]['metrics']
        pc = m['per_category']
        cells = [f'{pc.get(c, {}).get("recall_at_k", 0):.3f}' for c in
                 ['single-chunk', 'multi-hop', 'exact-symbol', 'ambiguous']]
        rows.append(f'| {SHORT[s]} | {m["recall_at_k"]:.3f} | {m["mrr"]:.3f} | '
                    + ' | '.join(cells)
                    + f' | {m["latency_s"]["total"]:.1f}s |')
    return '\n'.join(rows)


def main():
    runs = load_latest_runs()
    if not runs:
        raise SystemExit('No saved runs in results/. Run: python pipeline/eval.py')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chart_quality(runs, OUT_DIR / 'quality.png')
    chart_per_category(runs, OUT_DIR / 'per_category.png')
    chart_cost(runs, OUT_DIR / 'cost_vs_quality.png')

    print(f'strategies plotted: {[SHORT[s] for s in ORDER if s in runs]}')
    print(f'charts written to {OUT_DIR}/')
    print()
    print(markdown_table(runs))


if __name__ == '__main__':
    main()
