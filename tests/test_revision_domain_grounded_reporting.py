import pandas as pd
import pytest

from agnam.revision.domain_grounded_reporting import (
    add_keys, pair_key, validate_evidence_classes,
)


def test_pair_key_unordered_and_nonidentical():
    assert pair_key('b', 'a') == 'a|b'
    assert pair_key('a', 'b') == 'a|b'
    with pytest.raises(ValueError):
        pair_key('a','a')


def test_add_keys_rejects_duplicates_in_reversed_order():
    frame = pd.DataFrame({'task_id':[3,3], 'feature_j':['a','b'], 'feature_k':['b','a']})
    with pytest.raises(ValueError, match='duplicate'):
        add_keys(frame, name='test')


def test_add_keys_handles_valid_rows():
    frame = pd.DataFrame({'task_id':[3,3], 'feature_j':['a','c'], 'feature_k':['b','d']})
    out = add_keys(frame, name='test')
    assert out.pair_key.tolist() == ['a|b','c|d']


def _screening():
    rows = [{'task_id':146819,'feature_j':'vconst_corr','feature_k':'vconst_2',
             'source_review_status':'published_surrogate_term_identified',
             'source_url':'https://doi.org/10.5194/gmd-6-1157-2013'}]
    for i in range(21):
        rows.append({'task_id':3,'feature_j':f'x{i}','feature_k':f'y{i}',
                     'source_review_status':'pairwise_evidence_not_established_in_screened_sources',
                     'source_url':'https://example.org/doc'})
    return pd.DataFrame(rows)


def test_valid_evidence_screening():
    validate_evidence_classes(_screening())


def test_unknown_evidence_label_rejected():
    df = _screening()
    df.loc[1,'source_review_status']='proven_ground_truth'
    with pytest.raises(ValueError, match='Unknown'):
        validate_evidence_classes(df)


def test_positive_wrong_pair_rejected():
    df = _screening()
    df.loc[0,'feature_j']='other'
    with pytest.raises(ValueError, match='literature'):
        validate_evidence_classes(df)


def test_source_missing_rejected():
    df = _screening()
    df.loc[2,'source_url']=''
    with pytest.raises(ValueError, match='source URL'):
        validate_evidence_classes(df)


def test_audit_r10_end_to_end_with_mock_artifacts(tmp_path):
    """Run the entire reporting path on structurally realistic miniature files."""
    import json
    import numpy as np
    from pathlib import Path
    from agnam.revision.domain_grounded_reporting import audit_r10, sha256

    benchmark = tmp_path / 'results' / 'openml' / 'benchmark'
    root = tmp_path / 'results' / 'revision' / 'domain_grounded_r10' / 'v1'
    benchmark.mkdir(parents=True)
    root.mkdir(parents=True)
    master = benchmark / 'master_results.csv'
    pd.DataFrame({'task_id':[146819,3021,3,14952], 'agnam_auroc':[1.,1.,1.,1.]}).to_csv(master,index=False)
    master_hash = sha256(master)
    counts = {146819:2,3021:4,3:7,14952:9}
    support = {146819:2,3021:3,3:6,14952:9}
    inv_rows, source_rows, agg_rows = [], [], []
    for task_id,n_pairs in counts.items():
        task_dir = root / f'task_{task_id}'
        task_dir.mkdir()
        pair_file = benchmark / f'task_{task_id}_pairs.csv'
        pair_file.write_text('feature_j,feature_k\na,b\n',encoding='utf-8')
        model_rows=[]
        for idx in range(n_pairs):
            a,b = ('vconst_corr','vconst_2') if task_id==146819 and idx==0 else (f'a{idx}',f'b{idx}')
            role = 'locked_case_study' if task_id != 14952 else 'posthoc_supplement'
            base=dict(task_id=task_id,dataset_name=f'dataset-{task_id}',feature_type='numeric',
                      selection_role=role,feature_j=a,feature_k=b,selection_frequency=0.8,
                      source_pair_file_sha256=sha256(pair_file),external_evidence_status='not_yet_adjudicated',
                      external_evidence_citation='',external_evidence_notes='')
            inv_rows.append(base)
            src=base.copy()
            published=(task_id==146819 and idx==0)
            src['external_evidence_status']='published_surrogate_cross_term' if published else 'not_yet_adjudicated'
            src['source_review_status']='published_surrogate_term_identified' if published else 'pairwise_evidence_not_established_in_screened_sources'
            src['source_url']='https://doi.org/10.5194/gmd-6-1157-2013'
            source_rows.append(src)
            supported=idx<support[task_id]
            pair=dict(task_id=task_id,dataset_name=f'dataset-{task_id}',selection_role=role,
                      feature_j=a,feature_k=b,mean_abs_centered_contribution_logit=0.5+idx,
                      sd_centered_contribution_logit=0.1+idx,
                      p95_abs_centered_contribution_logit=1.+idx,
                      rectangle_supported=supported,
                      mixed_difference_logit=float(idx+0.1) if supported else float('nan'),
                      rectangle_cell_counts='3|3|3|3')
            model_rows.append(pair)
            agg_rows.append({**pair,'abs_mixed_difference_logit':abs(pair['mixed_difference_logit'])})
        pd.DataFrame(model_rows).to_csv(task_dir/'pair_model_evidence.csv',index=False)
        pred=pd.DataFrame({'test_row':[0,1], 'y_true':[0,1], 'agnam_probability':[0.1,0.9],
                           'agnam_logit':[-2.1972245773,2.1972245773]})
        pred.to_csv(task_dir/'test_predictions.csv',index=False)
        pd.DataFrame({'test_row':[0,1]}).to_csv(task_dir/'test_pair_contributions.csv',index=False)
        (task_dir/'final_agnam_state_dict.pt').write_bytes(b'mock-state')
        meta={
            'task_id':task_id,'master_sha256':master_hash,'pair_audit_sha256':sha256(pair_file),
            'r10_auroc':1.,'original_auroc':1.,'n_test':2,'n_retained_pairs':n_pairs,
            'max_decomposition_error_logit':0.,'max_prediction_error':0.,
            'retained_pairs':[[r['feature_j'],r['feature_k']] for r in model_rows],
            'output_sha256':{p.name:sha256(p) for p in task_dir.iterdir() if p.name!='metadata.json'},
        }
        (task_dir/'metadata.json').write_text(json.dumps(meta),encoding='utf-8')
    pd.DataFrame(inv_rows).to_csv(root/'pair_inventory.csv',index=False)
    pd.DataFrame(source_rows).to_csv(root/'domain_evidence_source_screening_draft.csv',index=False)
    pd.DataFrame(agg_rows).to_csv(root/'all_pair_model_evidence.csv',index=False)
    evidence, summary, audit = audit_r10(root=root,master=master,write=True)
    assert len(evidence)==22 and int(summary.n_supported.sum())==20
    assert audit['verified_tasks']==4
    assert (root/'analysis'/'analysis_manifest.json').is_file()
    # Actual source screening was created from a joined review matrix. Make
    # sure the complete auditor handles those redundant model columns.
    screening_path = root / 'domain_evidence_source_screening_draft.csv'
    screening = pd.read_csv(screening_path)
    model_aggregate = pd.read_csv(root / 'all_pair_model_evidence.csv')
    overlap = ['mean_abs_centered_contribution_logit',
               'rectangle_supported', 'mixed_difference_logit']
    joined_screen = screening.merge(
        model_aggregate[['task_id', 'feature_j', 'feature_k', *overlap]],
        on=['task_id', 'feature_j', 'feature_k'], validate='one_to_one',
    )
    joined_screen.to_csv(screening_path, index=False)
    evidence_joined, summary_joined, audit_joined = audit_r10(
        root=root, master=master, write=True,
    )
    assert len(evidence_joined) == 22
    assert 'mixed_difference_logit' in evidence_joined.columns
    assert 'mixed_difference_logit_x' not in evidence_joined.columns
    assert audit_joined['verified_pairs'] == 22

    corrupted_screen = joined_screen.copy()
    corrupted_screen.loc[0, 'mixed_difference_logit'] = 12.0
    corrupted_screen.to_csv(screening_path, index=False)
    with pytest.raises(ValueError, match='model evidence mismatch'):
        audit_r10(root=root, master=master, write=False)
    joined_screen.to_csv(screening_path, index=False)
    # A source hash mismatch must never be silently ignored.
    (benchmark/'task_3_pairs.csv').write_text('tampered',encoding='utf-8')
    with pytest.raises(ValueError,match='pair audit'):
        audit_r10(root=root,master=master,write=False)


def test_screening_draft_with_joined_model_fields_is_reconciled():
    """Real R10 screening CSV already contains some aggregated model fields."""
    from agnam.revision.domain_grounded_reporting import reconcile_screen_model_fields

    aggregate = pd.DataFrame({
        'task_id': [146819, 3021],
        'pair_key': ['vconst_2|vconst_corr', 'FTI|TSH'],
        'mean_abs_centered_contribution_logit': [1.074656, 0.032023],
        'rectangle_supported': [True, True],
        'mixed_difference_logit': [-0.089048, 0.000104],
    })
    screening = aggregate.copy()
    screening['source_review_status'] = [
        'published_surrogate_term_identified',
        'pairwise_evidence_not_established_in_screened_sources',
    ]
    columns = (
        'mean_abs_centered_contribution_logit',
        'rectangle_supported',
        'mixed_difference_logit',
    )
    cleaned = reconcile_screen_model_fields(
        screening, aggregate, model_columns=columns,
    )
    assert all(column not in cleaned for column in columns)
    assert 'source_review_status' in cleaned
    joined = cleaned.merge(
        aggregate[['task_id', 'pair_key', *columns]],
        on=['task_id', 'pair_key'], validate='one_to_one',
    )
    assert 'mixed_difference_logit' in joined.columns
    assert 'mixed_difference_logit_x' not in joined.columns

    tampered = screening.copy()
    tampered.loc[0, 'mixed_difference_logit'] = 123.0
    with pytest.raises(ValueError, match='model evidence mismatch'):
        reconcile_screen_model_fields(
            tampered, aggregate, model_columns=columns,
        )
