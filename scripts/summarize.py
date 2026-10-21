"""Render complete six-protocol results without selecting on the test set."""
from pathlib import Path
from release_utils import load


def summarize(source, destination):
    r = load(source)
    lines = ['# I37 evaluation', '', f'Status: {r["status"]}; best epoch: {r["best_epoch"]} (ID validation selection).', '',
             '| Protocol | MAE | RMSE | Samples |', '| --- | ---: | ---: | ---: |']
    for name, row in r['sets'].items():
        m = row['metrics']
        lines.append(f'| {name} | {m["mae"]:.6f} | {m["rmse"]:.6f} | {row["samples"]} |')
    if r['status'] == 'finished':
        m = {k: v['metrics']['mae'] for k, v in r['sets'].items()}
        score = ((m['unseen_combinations'] + m['unseen_combinations_repeat']) / 2
                 + (m['unseen_geometry'] + m['unseen_geometry_repeat']) / 2 + m['unseen_triple']) / 3
        lines += ['', f'Equal-category OOD MAE: {score:.6f}.', '',
                  'One training seed. Mask repeats are not independent training replicates.']
    Path(destination).write_text('\n'.join(lines) + '\n')
