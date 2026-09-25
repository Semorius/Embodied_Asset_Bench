# Embodied Asset Bench

Core implementation of a unified physical executability benchmark for three
articulated-asset dataset interfaces. The evaluator covers loading, settling,
collision, pushing, joint articulation, grasp lifting, and task actuation.

The repository intentionally contains only reusable evaluation code. Dataset
assets, asset selections, manifests, machine configuration, experiment outputs,
server launchers, and internal diagnostics are supplied separately by each
experiment owner and are excluded from version control.

## Running the evaluator

Use Isaac Sim's Python runtime and provide an external evaluation root containing
the evaluator configuration and dataset manifests. The command-line entry point
is `src/isaac_eval.py`:

```bash
export RAW_EVAL_CONFIG=/path/to/evaluation/config.json
export RAW_EVAL_RESULTS=results.jsonl
python src/isaac_eval.py validate \
  --dataset <dataset> \
  --tests load_precheck,settle,collision,push_contact,joint_sweep,grasp_lift,task_actuation
```

The evaluator records numeric results separately from applicability and runtime
status. Blocked, missing, and not-applicable outcomes are never silently turned
into physical zero scores.

`src/geometry.py` contains geometry and contact helpers; `src/scoring.py`
contains metric aggregation and score semantics; `src/grasp_contract.py` defines
the authored grasp contract.

## External inputs

The external evaluation root must provide a `config.json`, one manifest per
dataset, and the referenced assets. Do not commit those files when they contain
asset selections, paths, hashes, or experiment results.

