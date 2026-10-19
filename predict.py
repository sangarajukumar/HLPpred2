#!/usr/bin/env python3
"""HLPpred 2.0: predict hemolytic activity (classification) and potency (pHC50 regression) from a FASTA file."""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import default_paths, read_fasta, run  # noqa: E402


def parse_args():
    defaults = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fasta', required=True, help='input FASTA (standard amino acids, 1-39 residues)')
    parser.add_argument('--task', choices=['classification', 'regression', 'both'], default='both')
    parser.add_argument('--output', default='predictions.csv', help='output CSV (default: %(default)s)')
    parser.add_argument('--device', choices=['cuda', 'cpu'], help='default: cuda if available, else cpu')
    parser.add_argument('--models_dir', default=defaults['models_dir'], help='folder with the model archives')
    parser.add_argument('--esm_dir', default=defaults['esm_dir'], help='clone of facebookresearch/esm')
    parser.add_argument('--embedder_weights_dir', help='folder with ESM-1/1b/1v weight files (optional)')
    return parser.parse_args()


def print_table(df):
    columns = [('sequence_id', 'Sequence ID', 24), ('predicted_class', 'Class', 14),
               ('hemolytic_probability', 'P(hemolytic)', 13), ('predicted_pHC50', 'pHC50', 7),
               ('predicted_HC50_uM', 'HC50 (uM)', 10)]
    columns = [c for c in columns if c[0] in df.columns]
    line = '-' * (sum(w for *_, w in columns) + 3 * (len(columns) - 1))
    print(line)
    print(' | '.join(f'{title:<{w}}' for _, title, w in columns))
    print(line)
    for row in df.to_dict('records'):
        print(' | '.join(f'{str(row[key])[:w]:<{w}}' for key, _, w in columns))
    print(line)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    import torch
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        sys.exit('CUDA is not available; use --device cpu')
    tasks = {'classification', 'regression'} if args.task == 'both' else {args.task}

    names, seqs = read_fasta(args.fasta)
    logging.info('%d sequence(s), task: %s, device: %s', len(seqs), args.task, device)
    paths = {'models_dir': args.models_dir, 'esm_dir': args.esm_dir,
             'embedder_weights_dir': args.embedder_weights_dir}
    result, failed = run(names, seqs, tasks, device, paths)

    for task, items in failed.items():
        if items:
            logging.warning('%d %s base model(s) could not be scored (set to 0):\n  %s',
                            len(items), task, '\n  '.join(items))
    result.to_csv(args.output, index=False)
    print_table(result)
    logging.info('Predictions saved to %s', args.output)


if __name__ == '__main__':
    main()
