"""Pure parser validation for the additive groves prospective batch runner."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/run_revision_additive_groves_batch.py"
SPEC = importlib.util.spec_from_file_location("revision_ag_batch", PATH)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_train_recommendation_and_best():
    text = "Alpha = 0.01\nN = 8\n60 bagging iterations\nBest model:\nAlpha = 0.5\nN = 8\nSuggested action: ag_expand -n 16 -b 90"
    assert MOD.parse_model(text) == (0.5, 8, 60)
    assert MOD.suggested_action(text) == ("ag_expand", ["-n", "16", "-b", "90"])


def test_expand_sequential_and_ag_save():
    text = "210 bagging iterations\nBest model:\n  Alpha = 0.5\n N = 23\nSuggested action: ag_save -a 0.5 -n 23"
    assert MOD.parse_model(text) == (0.5, 23, 210)
    assert MOD.suggested_action(text) == ("ag_save", ["-a", "0.5", "-n", "23"])


def test_latest_action_wins():
    text = "Suggested action: ag_expand -n 16 -b 90\nSuggested action: ag_expand -b 140"
    assert MOD.suggested_action(text) == ("ag_expand", ["-b", "140"])


def test_ag_expand_rejects_bad_flags():
    for args in (["-n", "32", "-b"], ["-x", "22"], ["-b", "0"], ["-b", "garbage"]):
        with pytest.raises(ValueError):
            MOD.validate_expansion_action("ag_expand", args)
    assert MOD.validate_expansion_action("ag_expand", ["-b", "210"]) == ["-b", "210"]


def test_ag_fs_uses_last_perf():
    text = "Average performance: 0.812, std: 0.004\nAverage performance: 0.848208, std: 0.00231232"
    assert MOD.parse_feature_selection_performance(text) == (0.848208, 0.00231232)


def test_missing_recommendation_fails_closed():
    with pytest.raises(ValueError, match="Suggested action"):
        MOD.suggested_action("ROC on validation set = 0.788318")


def test_absent_model_rejected():
    with pytest.raises(ValueError, match="Missing best"):
        MOD.parse_model("Iteration 209 out of 210")


def test_expand_header_current_bagging_overrides_previous_model():
    text = ("ag_expand -n 32 -b 140\n"
            "Alpha = 0.01\nN = 32\n140 bagging iterations\nlayered mode\n"
            "Model already trained:\n Alpha = 0.01\n N = 16\n"
            " 90 bagging iterations\nPrevious model's ROC = 0.807324\n"
            "Iteration 1 out of 140 (begin)\nBest model:\n"
            "Alpha = 0.5\nN = 23\nSuggested action: ag_save -a 0.5 -n 23")
    assert MOD.parse_model(text) == (0.5, 23, 140)


def test_reject_ambiguous_bagging_in_current_header():
    text = ("Alpha = 0.01\nN = 32\n140 bagging iterations\n"
            "90 bagging iterations\nBest model:\nAlpha = 0.5\nN = 23")
    with pytest.raises(ValueError, match="exactly one current bagging"):
        MOD.parse_model(text)


def test_completed_mismatch_detected_by_training_checkpoint(tmp_path):
    from json import dumps
    d = tmp_path
    (d/'ag_expand_02.log').write_text(
        "140 bagging iterations\nModel already trained:\n90 bagging iterations\n"
        "Best model:\nAlpha = 0.5\nN = 23\n"
        "Suggested action: ag_save -a 0.5 -n 23\n", encoding='utf-8')
    checkpoint = {
        'stages': {
            'ag_expand_02': {'exit_code': 0},
            'ag_fs': {'exit_code': 0, 'command': ['ag_fs.exe','-a','0.5','-b','90']},
        },
    }
    (d/'ag_batch_checkpoint.json').write_text(dumps(checkpoint), encoding='utf-8')
    assert MOD._completed_bagging_mismatch(d) == (90, 140)


def test_completed_matching_bagging_not_flagged(tmp_path):
    from json import dumps
    d=tmp_path
    (d/'ag_train.log').write_text(
        "60 bagging iterations\nBest model:\nAlpha = 0.5\nN = 8\n"
        "Suggested action: ag_save -a 0.5 -n 8\n", encoding='utf-8')
    checkpoint={'stages': {'ag_train': {'exit_code':0},
                           'ag_fs': {'exit_code':0,'command':['ag_fs.exe','-b','60']}}}
    (d/'ag_batch_checkpoint.json').write_text(dumps(checkpoint), encoding='utf-8')
    assert MOD._completed_bagging_mismatch(d) is None


def test_repair_archives_original_outputs_and_keeps_grid_search(tmp_path):
    from json import dumps
    d=tmp_path
    reference=tmp_path/'ref.json'
    reference.write_text('{"locked":true}', encoding='utf-8')
    (d/'ag_expand_02.log').write_text(
        '140 bagging iterations\nModel already trained:\n90 bagging iterations\n'
        'Best model:\nAlpha = 0.5\nN = 23\n'
        'Suggested action: ag_save -a 0.5 -n 23',encoding='utf-8')
    names=['ag_fs.log','ag_interactions.log','ag_fs_model.bin','data.fs.attr',
           'additive_groves_manifest.json','additive_groves_summary.csv',
           'additive_groves_pair_scores.csv']
    for name in names:
        (d/name).write_text('ORIGINAL-'+name,encoding='utf-8')
    stages = {'ag_train': {'exit_code':0},'ag_expand_02': {'exit_code':0},
              'ag_fs': {'exit_code':0,'command':['ag_fs.exe','-b','90']},
              'ag_interactions': {'exit_code':0, 'command':['ag_interactions.exe','-b','90']}}
    manifest = {'scenario':'S1','treeextra_training_parameters': {'bagging_iterations':90}}
    (d/'additive_groves_manifest.json').write_text(dumps(manifest),encoding='utf-8')
    checkpoint={'scenario':'S1','reference_manifest_sha256':MOD.sha256(reference),
                'finished_analysis':True,'stages':stages,
                'outputs':{name: MOD.sha256(d/name) for name in names if name.startswith('additive_groves_')}}
    (d/'ag_batch_checkpoint.json').write_text(dumps(checkpoint), encoding='utf-8')
    result=MOD.prepare_bagging_repair(d,reference)
    assert result == {'actual':90, 'expected':140}
    assert not (d/'ag_fs.log').exists()
    assert not (d/'additive_groves_summary.csv').exists()
    saved=__import__('json').loads((d/'ag_batch_checkpoint.json').read_text())
    assert set(saved['stages']) == {'ag_train','ag_expand_02'}
    assert saved['bagging_repair']['correct_current_search_bagging'] == 140
    archived=d/'repair_archive'/'bagging_90_to_140_pre_repair'
    assert (archived/'ag_fs.log').read_text() == 'ORIGINAL-ag_fs.log'
    assert (archived/'additive_groves_summary.csv').read_text() == 'ORIGINAL-additive_groves_summary.csv'
    assert (d/'ag_expand_02.log').exists()
    with pytest.raises(FileNotFoundError, match='incomplete repair'):
        MOD.prepare_bagging_repair(d,reference)


def test_ag_expand_accepts_r05_real_recommendation():
    """S1 R05 ag_train's native suggestion expands alpha and model capacity."""
    stdout = (
        "Alpha = 0.01\nN = 8\n60 bagging iterations\n"
        "Best model:\nAlpha = 0.5\nN = 8\n"
        "Recommendation: relaxing model parameters might produce a better model.\n"
        "Suggested action: ag_expand -a 0.001 -n 16 -b 90\n"
    )
    action, flags = MOD.suggested_action(stdout)
    assert MOD.validate_expansion_action(action, flags) == ["-a", "0.001", "-n", "16", "-b", "90"]


def test_ag_expand_accepts_single_alpha_and_scientific_notation():
    assert MOD.validate_expansion_action("ag_expand", ["-a", "0.001"]) == ["-a", "0.001"]
    assert MOD.validate_expansion_action("ag_expand", ["-a", "1e-3", "-b", "210"]) == [
        "-a", "1e-3", "-b", "210"
    ]
    assert MOD.validate_expansion_action("ag_expand", ["-b", "210", "-a", ".0001"]) == [
        "-b", "210", "-a", ".0001"
    ]


@pytest.mark.parametrize("args", [
    ["-a", "0"], ["-a", "-0.01"], ["-a", "-1"],
    ["-a", "nan"], ["-a", "inf"], ["-a", "1e999"], ["-a", "1e-999"],
    ["-a", "nope"], ["-a", "0.001 garbage"],
    ["-a", "0.001", "-a", "0.01"],
    ["-n", "16", "-n", "32"], ["-b", "90", "-b", "140"],
    ["-n", "4.5"], ["-b", "0"], ["-x", "2"],
    ["-a"], [],
])
def test_ag_expand_rejects_unsafe_alpha_and_bad_parameters(args):
    with pytest.raises(ValueError, match="ag_expand"):
        MOD.validate_expansion_action("ag_expand", args)


def test_ag_expand_rejects_wrong_program_with_valid_alpha():
    with pytest.raises(ValueError, match="Not an expansion action"):
        MOD.validate_expansion_action("ag_train", ["-a", "0.001", "-n", "16", "-b", "90"])
