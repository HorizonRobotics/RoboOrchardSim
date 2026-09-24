## Copyright (c) 2024 Horizon Robotics. All Rights Reserved.

import copy
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).parents[2]
SKILL_ROOT = REPO_ROOT / ".agents/skills/analyzing-pick-failure-cases"


def load_core():
    module_path = SKILL_ROOT / "scripts/analysis_core.py"
    spec = importlib.util.spec_from_file_location(
        "pick_failure_analysis_core", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_eval_result():
    return {
        "episode_num": 3,
        "success_rate": 1 / 3,
        "episode_results": [
            {
                "seed": 100000,
                "success": False,
                "progress": 0.0,
                "steps": 800,
                "stop_reason": "max_steps",
                "metrics": {
                    "criteria_reached": {
                        "reach_pick": False,
                        "lift_pick": False,
                    }
                },
            },
            {
                "seed": 100001,
                "success": True,
                "progress": 1.0,
                "steps": 420,
                "stop_reason": "success",
                "metrics": {
                    "criteria_reached": {
                        "reach_pick": True,
                        "lift_pick": True,
                    }
                },
            },
            {
                "seed": 100002,
                "success": False,
                "progress": 0.5,
                "steps": 800,
                "stop_reason": "max_steps",
                "metrics": {
                    "criteria_reached": {
                        "reach_pick": True,
                        "lift_pick": False,
                    }
                },
            },
        ],
    }


def make_extracted_signals():
    frame_indexes = list(range(100))
    timestamps = [index * 100 for index in frame_indexes]
    return {
        "seed": 100000,
        "source_mcap": "/tmp/fake.mcap",
        "target": {
            "category": "usb_drive",
            "uuid": "target-uuid",
            "actor_type": "pick",
        },
        "camera_frames": {
            view: {
                "indexes": frame_indexes,
                "timestamps_ns": timestamps,
            }
            for view in ("ext1", "ext2", "wrist")
        },
        "gripper": {
            "status": "AVAILABLE",
            "samples": [
                {"timestamp_ns": 0, "position": 0.0},
                {"timestamp_ns": 100, "position": 0.1},
                {"timestamp_ns": 200, "position": 0.7},
                {"timestamp_ns": 300, "position": 0.78},
            ],
            "close_intervals": [
                {
                    "start_index": 0,
                    "end_index": 3,
                    "peak_position": 0.78,
                }
            ],
        },
    }


def make_valid_analysis():
    return {
        "schema_version": "1.0",
        "seed": 100000,
        "instruction": "Grab usb_drive.",
        "target": {"category": "usb_drive", "uuid": "target-uuid"},
        "evidence_window": {
            "start_frame": 10,
            "end_frame": 40,
            "frames": [10, 20, 30, 40],
            "views": ["ext1", "ext2", "wrist"],
        },
        "attribution": {
            "target_selection": "CORRECT",
            "gripper_action": "CLOSE_ATTEMPTED",
            "target_interaction": "CONTACT_NO_GRASP",
            "object_outcome": "STATIONARY",
            "cause": "MISALIGNED_GRASP",
        },
        "evidence": {
            "summary": "The gripper closes beside the visible target.",
            "target_visible_frames": [10, 40],
            "interaction_visible_frames": [20, 30],
            "outcome_frames": [30, 40],
        },
        "evaluation": {
            "success": False,
            "progress": 0.0,
            "reach_pick": True,
            "lift_pick": False,
            "steps": 800,
            "stop_reason": "max_steps",
        },
        "provenance": {
            "source": "LLM_RULE_HYBRID",
            "analysis_version": "1.0",
        },
        "annotations": [],
    }


def make_synthetic_images():
    colors = {
        "ext1": (90, 120, 160),
        "ext2": (120, 90, 160),
        "wrist": (160, 120, 90),
    }
    return {
        (frame, view): Image.new("RGB", (480, 270), colors[view])
        for frame in (10, 20, 30, 40)
        for view in ("ext1", "ext2", "wrist")
    }


def test_skill_metadata_pick_failure_request_describes_trigger():
    skill_text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")

    assert "name: analyzing-pick-failure-cases" in skill_text
    assert "MCAP" in skill_text
    assert "eval_result.json" in skill_text


def test_skill_resources_expected_files_exist():
    expected = {
        "scripts/analysis_core.py",
        "scripts/prepare_cases.py",
        "scripts/render_case.py",
        "scripts/validate_results.py",
        "scripts/aggregate_results.py",
        "references/label-schema.md",
        "references/evidence-rules.md",
        "references/analysis-schema.json",
    }

    missing = {
        relative
        for relative in expected
        if not (SKILL_ROOT / relative).is_file()
    }
    assert not missing


def test_select_episode_results_explicit_seeds_preserves_requested_order():
    core = load_core()

    selected = core.select_episode_results(
        make_eval_result(),
        seeds=[100002, 100000],
        all_failures=False,
        include_success=False,
    )

    assert [item["seed"] for item in selected] == [100002, 100000]


def test_select_episode_results_all_failures_excludes_success_controls():
    core = load_core()

    selected = core.select_episode_results(
        make_eval_result(),
        seeds=None,
        all_failures=True,
        include_success=False,
    )

    assert [item["seed"] for item in selected] == [100000, 100002]


def test_select_episode_results_missing_mode_raises_value_error():
    core = load_core()

    with pytest.raises(ValueError, match="exactly one"):
        core.select_episode_results(
            make_eval_result(),
            seeds=None,
            all_failures=False,
            include_success=False,
        )


def test_extract_evaluation_metrics_missing_fields_remain_none():
    core = load_core()

    metrics = core.extract_evaluation_metrics({"seed": 100000})

    assert metrics == {
        "success": None,
        "progress": None,
        "reach_pick": None,
        "lift_pick": None,
        "steps": None,
        "stop_reason": None,
    }


def test_reconcile_evaluation_task_progress_one_overrides_failed_result():
    core = load_core()
    evaluation = {
        "success": False,
        "progress": 0.5,
        "reach_pick": True,
        "lift_pick": False,
        "steps": 300,
        "stop_reason": "max_steps",
    }

    result = core.reconcile_evaluation_metrics(
        evaluation,
        {"task_progress": 1.0, "task_success": 1.0},
    )

    assert result == {
        "success": True,
        "progress": 1.0,
        "reach_pick": True,
        "lift_pick": True,
        "steps": 300,
        "stop_reason": "success",
        "source": "MCAP_TASK_PROGRESS",
        "mcap_task_progress": 1.0,
        "mcap_task_success": True,
        "overrode_external_result": True,
    }


def test_reconcile_evaluation_partial_task_progress_preserves_result():
    core = load_core()
    evaluation = {
        "success": False,
        "progress": 0.5,
        "reach_pick": True,
        "lift_pick": False,
        "steps": 300,
        "stop_reason": "max_steps",
    }

    result = core.reconcile_evaluation_metrics(
        evaluation,
        {"task_progress": 0.5, "task_success": 0.0},
    )

    assert result["success"] is False
    assert result["progress"] == 0.5
    assert result["source"] == "EVAL_RESULT"
    assert result["overrode_external_result"] is False


def test_discover_mcap_single_seed_match_returns_path(tmp_path):
    core = load_core()
    expected = (
        tmp_path
        / "records/run/episode_0000_seed_100000/episode0/env0_data.mcap"
    )
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"mcap")

    result = core.discover_mcap(tmp_path, 100000)

    assert result == expected


def test_detect_close_intervals_sustained_position_change_returns_interval():
    core = load_core()
    samples = [
        (0, 0.0),
        (1, 0.02),
        (2, 0.36),
        (3, 0.71),
        (4, 0.78),
        (5, 0.78),
    ]

    result = core.detect_close_intervals(
        samples,
        minimum_delta=0.3,
        closed_threshold=0.6,
        minimum_samples=2,
    )

    assert result == [
        {"start_index": 1, "end_index": 4, "peak_position": 0.78}
    ]


def test_summarize_gripper_trace_missing_trace_returns_unknown_status():
    core = load_core()

    result = core.summarize_gripper_trace([])

    assert result["status"] == "UNKNOWN"


def test_nearest_timestamp_index_target_between_frames_returns_closest_frame():
    core = load_core()

    result = core.nearest_timestamp_index([100, 200, 310], 260)

    assert result == 2


def test_render_candidate_sheet_frame_labels_remain_chronological(tmp_path):
    core = load_core()
    frames = [
        (20, Image.new("RGB", (160, 90), "red")),
        (10, Image.new("RGB", (160, 90), "blue")),
    ]

    metadata = core.render_candidate_sheet(
        frames=frames,
        view="ext1",
        output_path=tmp_path / "sheet.jpg",
    )

    assert metadata["frame_indexes"] == [10, 20]


def test_validate_analysis_valid_record_returns_no_errors():
    core = load_core()

    result = core.validate_analysis(
        make_valid_analysis(), make_extracted_signals()
    )

    assert result["errors"] == []


def test_validate_analysis_pushed_away_without_sequence_returns_error():
    core = load_core()
    analysis = make_valid_analysis()
    analysis["attribution"]["object_outcome"] = "PUSHED_AWAY"
    analysis["evidence"]["outcome_frames"] = []

    result = core.validate_analysis(analysis, make_extracted_signals())

    assert any("PUSHED_AWAY" in error for error in result["errors"])


def test_validate_analysis_wrong_object_requires_target_evidence():
    core = load_core()
    analysis = make_valid_analysis()
    analysis["attribution"]["target_selection"] = "WRONG_OBJECT"
    analysis["evidence"]["target_visible_frames"] = []

    result = core.validate_analysis(analysis, make_extracted_signals())

    assert any("target_visible_frames" in error for error in result["errors"])


def test_validate_analysis_nonchronological_frames_returns_error():
    core = load_core()
    analysis = make_valid_analysis()
    analysis["evidence_window"]["frames"] = [10, 30, 20, 40]

    result = core.validate_analysis(analysis, make_extracted_signals())

    assert any("chronological" in error for error in result["errors"])


def test_validate_analysis_task_progress_one_requires_success_attribution():
    core = load_core()
    analysis = make_valid_analysis()
    extracted = make_extracted_signals()
    extracted["metadata"] = {
        "task_progress": 1.0,
        "task_success": 1.0,
    }

    result = core.validate_analysis(analysis, extracted)

    assert any(
        "MCAP task_progress=1.0 requires evaluation.success=true" in error
        for error in result["errors"]
    )


def test_cause_values_approved_taxonomy_matches_exact_labels():
    core = load_core()

    assert core.CAUSE_VALUES == {
        "NONE",
        "SELECTED_DISTRACTOR",
        "OFF_TARGET_APPROACH",
        "NO_CLOSE_ATTEMPT",
        "MISALIGNED_GRASP",
        "UNSTABLE_GRASP",
        "PREMATURE_GRIPPER_OPENING",
        "INSUFFICIENT_LIFT_MOTION",
        "UNKNOWN",
    }


@pytest.mark.parametrize(
    "legacy_cause",
    [
        "MISSED_TARGET",
        "OPEN_GRIPPER_COLLISION",
        "EARLY_CLOSE",
        "LATE_CLOSE",
        "INSUFFICIENT_ENCLOSURE",
        "EARLY_RELEASE",
        "INSUFFICIENT_LIFT",
    ],
)
def test_validate_analysis_legacy_cause_returns_error(legacy_cause):
    core = load_core()
    analysis = make_valid_analysis()
    analysis["attribution"]["cause"] = legacy_cause

    result = core.validate_analysis(analysis, make_extracted_signals())

    assert f"Invalid cause: {legacy_cause}" in result["errors"]


@pytest.mark.parametrize(
    ("cause", "attribution", "success", "has_close"),
    [
        (
            "NONE",
            {
                "target_selection": "CORRECT",
                "gripper_action": "CLOSE_ATTEMPTED",
                "target_interaction": "STABLE_GRASP",
                "object_outcome": "LIFTED_SUCCESSFULLY",
            },
            True,
            True,
        ),
        (
            "SELECTED_DISTRACTOR",
            {"target_selection": "WRONG_OBJECT"},
            False,
            True,
        ),
        (
            "OFF_TARGET_APPROACH",
            {
                "target_selection": "CORRECT",
                "target_interaction": "NO_CONTACT",
            },
            False,
            True,
        ),
        (
            "NO_CLOSE_ATTEMPT",
            {
                "target_selection": "CORRECT",
                "gripper_action": "NO_CLOSE_ATTEMPT",
                "target_interaction": "CONTACT_NO_GRASP",
            },
            False,
            False,
        ),
        (
            "MISALIGNED_GRASP",
            {
                "target_selection": "CORRECT",
                "gripper_action": "CLOSE_ATTEMPTED",
                "target_interaction": "CONTACT_NO_GRASP",
            },
            False,
            True,
        ),
        (
            "UNSTABLE_GRASP",
            {
                "target_selection": "CORRECT",
                "gripper_action": "CLOSE_ATTEMPTED",
                "target_interaction": "TRANSIENT_GRASP",
            },
            False,
            True,
        ),
        (
            "PREMATURE_GRIPPER_OPENING",
            {
                "target_selection": "CORRECT",
                "gripper_action": "CLOSE_ATTEMPTED",
                "target_interaction": "TRANSIENT_GRASP",
                "object_outcome": "LIFTED_THEN_DROPPED",
            },
            False,
            True,
        ),
        (
            "INSUFFICIENT_LIFT_MOTION",
            {
                "target_selection": "CORRECT",
                "gripper_action": "CLOSE_ATTEMPTED",
                "target_interaction": "STABLE_GRASP",
                "object_outcome": "LIFTED_INSUFFICIENTLY",
            },
            False,
            True,
        ),
        ("UNKNOWN", {}, False, True),
    ],
)
def test_validate_analysis_consistent_cause_dimensions_return_no_cause_error(
    cause, attribution, success, has_close
):
    core = load_core()
    analysis = make_valid_analysis()
    analysis["attribution"].update(attribution)
    analysis["attribution"]["cause"] = cause
    analysis["evaluation"]["success"] = success
    if cause == "SELECTED_DISTRACTOR":
        analysis["evidence"]["distractor_interaction_frames"] = [20, 30]
    extracted = make_extracted_signals()
    if not has_close:
        extracted["gripper"]["samples"] = [
            {"timestamp_ns": 0, "position": 0.0},
            {"timestamp_ns": 100, "position": 0.0},
        ]
        extracted["gripper"]["close_intervals"] = []

    result = core.validate_analysis(analysis, extracted)

    assert not any("Cause " in error for error in result["errors"])


@pytest.mark.parametrize(
    ("cause", "attribution", "expected"),
    [
        (
            "OFF_TARGET_APPROACH",
            {"target_interaction": "CONTACT_NO_GRASP"},
            "NO_CONTACT",
        ),
        (
            "NO_CLOSE_ATTEMPT",
            {"gripper_action": "CLOSE_ATTEMPTED"},
            "NO_CLOSE_ATTEMPT",
        ),
        (
            "MISALIGNED_GRASP",
            {"target_interaction": "TRANSIENT_GRASP"},
            "CONTACT_NO_GRASP",
        ),
        (
            "UNSTABLE_GRASP",
            {"target_interaction": "CONTACT_NO_GRASP"},
            "TRANSIENT_GRASP",
        ),
        (
            "PREMATURE_GRIPPER_OPENING",
            {"target_interaction": "STABLE_GRASP"},
            "TRANSIENT_GRASP",
        ),
        (
            "INSUFFICIENT_LIFT_MOTION",
            {"object_outcome": "LIFTED_THEN_DROPPED"},
            "LIFTED_INSUFFICIENTLY",
        ),
    ],
)
def test_validate_analysis_inconsistent_cause_dimensions_return_error(
    cause, attribution, expected
):
    core = load_core()
    analysis = make_valid_analysis()
    analysis["attribution"].update(attribution)
    analysis["attribution"]["cause"] = cause

    result = core.validate_analysis(analysis, make_extracted_signals())

    assert any(
        error.startswith(f"Cause {cause}") and expected in error
        for error in result["errors"]
    )


def test_render_case_preserves_v5_layout_with_metric_row(tmp_path):
    core = load_core()

    output = core.render_case_image(
        analysis=make_valid_analysis(),
        images=make_synthetic_images(),
        output_path=tmp_path / "sequence_analysis.jpg",
    )

    with Image.open(output) as image:
        assert image.size == (1440, 1375)
        assert image.getpixel((10, 294)) == pytest.approx((18, 21, 26), abs=3)
        assert image.getpixel((10, 300)) == pytest.approx((5, 7, 10), abs=3)
        assert image.getpixel((10, 335)) == pytest.approx(
            (90, 120, 160), abs=3
        )


def test_format_metric_line_missing_values_uses_na():
    core = load_core()

    result = core.format_metric_line({"reach_pick": None})

    assert "REACH_PICK=N/A" in result


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("WRONG_OBJECT", (220, 55, 55)),
        ("NO_CLOSE_ATTEMPT", (137, 76, 202)),
        ("NO_CONTACT", (110, 118, 128)),
        ("PUSHED_AWAY", (220, 55, 55)),
        ("STABLE_GRASP", (30, 165, 96)),
    ],
)
def test_attribution_color_preserves_v5_semantics(label, expected):
    core = load_core()

    assert core.attribution_color(label) == expected


def test_normalize_annotations_out_of_bounds_coordinates_are_clipped():
    core = load_core()
    annotation = {
        "frame": 10,
        "view": "ext1",
        "kind": "box",
        "geometry": [-0.2, 0.1, 1.4, 0.8],
        "text": "TARGET",
    }

    result = core.normalize_annotation(annotation)

    assert result["geometry"] == [0.0, 0.1, 1.0, 0.8]


def test_project_point_identity_camera_returns_principal_point():
    core = load_core()

    result = core.project_point_to_image(
        point=[0.0, 0.0, 1.0],
        camera_translation=[0.0, 0.0, 0.0],
        camera_quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
        intrinsic_matrix=[500.0, 0.0, 640.0, 0.0, 500.0, 360.0, 0.0, 0.0, 1.0],
        image_size=(1280, 720),
    )

    assert result == {"u": 640.0, "v": 360.0, "normalized": [0.5, 0.5]}


def test_aggregate_analyses_reports_dimension_counts_and_seed_lists():
    core = load_core()
    wrong_object = make_valid_analysis()
    wrong_object["attribution"].update(
        {
            "target_selection": "WRONG_OBJECT",
            "target_interaction": "NO_CONTACT",
            "object_outcome": "STATIONARY",
            "cause": "SELECTED_DISTRACTOR",
        }
    )
    wrong_object["evidence"]["target_visible_frames"] = [10, 40]
    pushed_away = copy.deepcopy(make_valid_analysis())
    pushed_away["seed"] = 100002
    pushed_away["evaluation"]["progress"] = 0.5
    pushed_away["attribution"]["object_outcome"] = "PUSHED_AWAY"

    summary = core.aggregate_analyses(
        eval_result=make_eval_result(),
        analyses=[wrong_object, pushed_away],
        validation_results=[],
    )

    assert summary["attribution"]["target_selection"]["WRONG_OBJECT"] == {
        "count": 1,
        "seeds": [100000],
        "analyzed_failure_rate": 0.5,
        "all_failure_rate": 0.5,
    }


def test_aggregate_analyses_success_controls_not_in_failure_denominator():
    core = load_core()
    success_control = make_valid_analysis()
    success_control["seed"] = 100001
    success_control["evaluation"]["success"] = True
    failure = copy.deepcopy(make_valid_analysis())

    summary = core.aggregate_analyses(
        eval_result=make_eval_result(),
        analyses=[success_control, failure],
        validation_results=[],
    )

    assert summary["coverage"]["analyzed_failures"] == 1


def test_aggregate_analyses_analysis_success_overrides_stale_eval_result():
    core = load_core()
    corrected_success = make_valid_analysis()
    corrected_success["evaluation"].update(
        {
            "success": True,
            "progress": 1.0,
            "reach_pick": True,
            "lift_pick": True,
            "stop_reason": "success",
        }
    )
    corrected_success["attribution"].update(
        {
            "target_interaction": "STABLE_GRASP",
            "object_outcome": "LIFTED_SUCCESSFULLY",
            "cause": "NONE",
        }
    )

    summary = core.aggregate_analyses(
        eval_result=make_eval_result(),
        analyses=[corrected_success],
        validation_results=[],
    )

    assert summary["evaluation"]["success"] == 2
    assert summary["evaluation"]["failure"] == 1
    assert summary["coverage"]["analyzed_failures"] == 0


def test_skill_mcap_task_progress_one_is_authoritative_success_rule():
    skill_text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    evidence_text = (SKILL_ROOT / "references/evidence-rules.md").read_text(
        encoding="utf-8"
    )

    assert "task_progress=1.0" in skill_text
    assert "task_progress=1.0" in evidence_text
    assert "authoritative success" in evidence_text


def test_aggregate_analyses_includes_requested_cross_analysis():
    core = load_core()
    analysis = make_valid_analysis()

    summary = core.aggregate_analyses(
        eval_result=make_eval_result(),
        analyses=[analysis],
        validation_results=[],
    )

    assert summary["cross_analysis"]["target_selection_by_reach_pick"] == {
        "CORRECT": {"TRUE": {"count": 1, "seeds": [100000]}}
    }


def test_render_aggregate_overview_uses_compact_dynamic_dashboard(tmp_path):
    core = load_core()
    first = make_valid_analysis()
    second = copy.deepcopy(first)
    second["seed"] = 100002
    second["attribution"].update(
        {
            "target_selection": "WRONG_OBJECT",
            "target_interaction": "NO_CONTACT",
            "object_outcome": "PUSHED_AWAY",
            "cause": "SELECTED_DISTRACTOR",
        }
    )
    summary = core.aggregate_analyses(
        eval_result=make_eval_result(),
        analyses=[first, second],
        validation_results=[],
    )

    output = core.render_aggregate_overview(
        summary, tmp_path / "aggregate_overview.jpg"
    )

    with Image.open(output) as image:
        assert image.width == 1440
        assert 1050 <= image.height <= 1500
        assert image.height != 1800


def test_analysis_schema_json_is_valid_json():
    schema_path = SKILL_ROOT / "references/analysis-schema.json"

    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["title"] == "Pick Failure Analysis"


def test_analysis_schema_cause_enum_matches_approved_taxonomy():
    schema_path = SKILL_ROOT / "references/analysis-schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert (
        set(schema["properties"]["attribution"]["properties"]["cause"]["enum"])
        == load_core().CAUSE_VALUES
    )


def test_label_schema_each_cause_has_use_and_exclusion_boundaries():
    label_text = (SKILL_ROOT / "references/label-schema.md").read_text(
        encoding="utf-8"
    )

    for cause in load_core().CAUSE_VALUES:
        assert f"### `{cause}`" in label_text
    assert label_text.count("Use when:") == len(load_core().CAUSE_VALUES)
    assert label_text.count("Do not use when:") == len(
        load_core().CAUSE_VALUES
    )


def test_label_schema_slip_and_reopening_causes_use_gripper_state_boundary():
    label_text = (SKILL_ROOT / "references/label-schema.md").read_text(
        encoding="utf-8"
    )

    assert "remains closed through object loss" in label_text
    assert "reopens before or at object loss" in label_text


def test_skill_workflow_requires_continuous_evidence_and_validation():
    skill_text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")

    assert "exactly four" in skill_text
    assert "validate_results.py" in skill_text
    assert "UNKNOWN" in skill_text
