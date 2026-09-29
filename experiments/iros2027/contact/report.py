"""Read an existing run and print per-condition episode counts, not frame rates."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path, help='包含 manifest.json 的运行目录')
    args = parser.parse_args()
    manifest = json.loads((args.run / 'manifest.json').read_text())
    attempt = manifest['attempts'][-1]
    if attempt['status'] != 'completed':
        parser.error(f"latest attempt is {attempt['status']}; inspect its logs before interpreting results")
    report = json.loads((args.run / attempt['summary']).read_text())
    print('条件                         保持成功     掉落     最大单指力(N)')
    for name in sorted({r['condition'] for r in report['episodes']}):
        rows = [r for r in report['episodes'] if r['condition'] == name]
        print(f"{name:28s} {sum(r['retained'] for r in rows)}/{len(rows):<7} "
              f"{sum(r['dropped'] for r in rows):<8} {max(r['peak_finger_force_n'] for r in rows):.3f}")
    print('开发实验结果；不代表独立测试集或学习算法收益。')


if __name__ == '__main__':
    main()
