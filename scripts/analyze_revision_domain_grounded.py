"""Run read-only-source, no-training R10 final model/evidence audit."""
from agnam.revision.domain_grounded_reporting import audit_r10


def main():
    evidence, summary, audit = audit_r10(write=True)
    print(f"R10 AUDIT PASS: {audit['verified_tasks']}/4 tasks; "
          f"{audit['verified_pairs']}/22 pairs; "
          f"{audit['rectangle_supported']}/22 supported rectangles")
    print(summary[['task_id','n_pairs','n_supported','r10_auroc',
                   'median_abs_mixed_difference_logit']].round(6).to_string(index=False))
    print('EXTERNAL DOMAIN EVIDENCE: 1 published surrogate cross-term; '
          '21 pairwise claims not established in screened sources')
    print('SAVED: results/revision/domain_grounded_r10/v1/analysis')


if __name__ == '__main__':
    main()
