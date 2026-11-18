"""Run R7 paired proposer training using 400 audited MainEffectNAM caches."""
from __future__ import annotations
import argparse
from agnam.revision.residual_sensitivity_runner import run_realization


def main():
    p = argparse.ArgumentParser(description='R7 probability vs deviance residual sensitivity')
    p.add_argument('--scenario', choices=['S1','S2','S3','S4'])
    p.add_argument('--realization', type=int)
    p.add_argument('--all', action='store_true', help='Run all 80 scenario-realizations')
    p.add_argument('--force', action='store_true', help='Explicitly refit and overwrite cached R7 rankings')
    args = p.parse_args()
    if args.all:
        if args.scenario or args.realization:
            p.error('--all cannot be combined with --scenario/--realization')
        specs = [(s, r) for s in ('S1','S2','S3','S4') for r in range(1,21)]
    else:
        if not args.scenario or args.realization is None or not 1 <= args.realization <= 20:
            p.error('Specify --all or both --scenario S1..S4 and --realization 1..20')
        specs = [(args.scenario, args.realization)]
    for i,(scenario,realization) in enumerate(specs,1):
        print(f'[{i}/{len(specs)}] R7 {scenario} R{realization:02d}',flush=True)
        run_realization(scenario,realization,force=args.force)
    print('R7 RUN COMPLETE',len(specs),'realizations',flush=True)


if __name__ == '__main__':
    main()
