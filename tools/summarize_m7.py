#!/usr/bin/env python3
"""M7: メモリの使い回しを入れる前(before_N)と後(after_N)を、条件ごとにまとめる。

  results/m7/before_<N>/<条件>_r1/ と after_<N>/<条件>_r1/ を、N = 1 から順に読む。
  入れる前の版は、HEAD を別の場所に作って(git archive HEAD)、交互に動かした(片方を先に全部やると、
  PC の状態の違いが混ざるため)。各回の最初の 20 フレームと最後の 5 フレームは除く。

使い方: python3 tools/summarize_m7.py [results/m7] [回数=6]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_latency import STAGES, load, pct  # noqa: E402

root = sys.argv[1] if len(sys.argv) > 1 else 'results/m7'
n_rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 6


def per_run(d, cond):
    p = os.path.join(root, d, cond + '_r1')
    data = {s: load(os.path.join(p, s + '.csv')) for s in STAGES}
    stamps = sorted(data['sensor_sim'])[20:-5]
    full = [s for s in stamps if all(s in data[x] for x in STAGES)]
    proc = {st: [(int(data[st][s]['done_ns']) - int(data[st][s]['recv_ns'])) / 1e6 for s in full]
            for st in STAGES}
    e2e = [(int(data['cluster'][s]['done_ns']) - s) / 1e6 for s in full]
    return proc, e2e, len(stamps) - len(full)


for cond in ['ipc_idle', 'ipc_load_split']:
    print(f'=== {cond}  ({n_rounds} rounds each, interleaved)')
    print(f'{"ver":7s} {"n":>5s} {"lost":>4s} | proc p50: {"crop":>6s} {"voxel":>6s} {"ground":>6s} '
          f'{"cluster":>7s} | voxel p99 | e2e p50    p99  p99.9    max   [ms]')
    res = {}
    for ver in ['before', 'after']:
        P = {s: [] for s in STAGES}
        E = []
        lost = 0
        rounds = []
        for k in range(1, n_rounds + 1):
            proc, e2e, lo = per_run(f'{ver}_{k}', cond)
            for s in STAGES:
                P[s] += proc[s]
            E += e2e
            lost += lo
            rounds.append((pct(e2e, 50), pct(e2e, 99), pct(proc['voxel_downsample'], 50)))
        res[ver] = rounds
        print(f'{ver:7s} {len(E):5d} {lost:4d} |           {pct(P["crop_box"], 50):6.3f} '
              f'{pct(P["voxel_downsample"], 50):6.3f} {pct(P["ground_removal"], 50):6.3f} '
              f'{pct(P["cluster"], 50):7.3f} | {pct(P["voxel_downsample"], 99):9.3f} | '
              f'{pct(E, 50):7.2f} {pct(E, 99):6.2f} {pct(E, 99.9):6.2f} {max(E):6.2f}')
        print('        per round voxel p50: ' + ' '.join(f'{r[2]:.2f}' for r in rounds))
        print('        per round e2e p50/p99: ' + '  '.join(f'{r[0]:.2f}/{r[1]:.2f}' for r in rounds))
    wins = sum(1 for b, a in zip(res['before'], res['after']) if a[2] < b[2])
    print(f'        rounds where after voxel p50 < before: {wins} / {n_rounds}')
