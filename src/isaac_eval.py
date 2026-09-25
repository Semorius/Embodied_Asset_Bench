from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
from collections import defaultdict, deque
from pathlib import Path
from manual_grasp import GraspBindingError


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONFIG_PATH = Path(os.environ.get("RAW_EVAL_CONFIG", ROOT / "configs" / "config.json"))
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
RUN_LABEL = os.environ.get("RAW_EVAL_RUN_LABEL", "v5")
RESULTS_NAME = os.environ.get("RAW_EVAL_RESULTS", "trials_v5.jsonl")


class EvaluationTimeout(RuntimeError):
    pass


class DeadlineApp:
    def __init__(self, app, timeout_seconds: float, label: str):
        self._app = app
        self.timeout_seconds = timeout_seconds
        self.label = label
        self.started = time.monotonic()

    def update(self, *args, **kwargs):
        if time.monotonic() - self.started >= self.timeout_seconds:
            raise EvaluationTimeout(
                f"{self.label} exceeded {self.timeout_seconds:.1f}s wall-clock limit"
            )
        result = self._app.update(*args, **kwargs)
        if time.monotonic() - self.started >= self.timeout_seconds:
            raise EvaluationTimeout(
                f"{self.label} exceeded {self.timeout_seconds:.1f}s wall-clock limit"
            )
        return result

    def __getattr__(self, name):
        return getattr(self._app, name)


def ensure_inside_root(path: Path) -> Path:
    root = ROOT.resolve()
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"拒绝写入实验目录之外: {resolved}")
    return resolved


def append_result(row: dict) -> None:
    if Path(RESULTS_NAME).name != RESULTS_NAME or not RESULTS_NAME.endswith(".jsonl"):
        raise ValueError(f"非法结果文件名: {RESULTS_NAME}")
    path = ensure_inside_root(ROOT / "runs" / RESULTS_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")


def json_value(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    if hasattr(value, "GetReal") and hasattr(value, "GetImaginary"):
        imaginary = value.GetImaginary()
        return [float(value.GetReal()), *(float(item) for item in imaginary)]
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, dict)):
        return list(value)
    if hasattr(value, "__len__") and hasattr(value, "__getitem__"):
        return [value[index] for index in range(len(value))]
    raise TypeError(f"不可序列化的结果类型: {type(value).__name__}")


def json_safe(value, _active=None):
    """将结果 payload 转成 JSON 基础类型，并截断诊断树中的循环引用。"""
    active = set() if _active is None else _active
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)

    identity = id(value)
    if identity in active:
        return "<circular_reference>"

    if isinstance(value, dict):
        active.add(identity)
        try:
            return {
                str(key): json_safe(item, active)
                for key, item in value.items()
            }
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple)):
        active.add(identity)
        try:
            return [json_safe(item, active) for item in value]
        finally:
            active.remove(identity)
    if isinstance(value, (set, frozenset)):
        active.add(identity)
        try:
            return [json_safe(item, active) for item in sorted(value, key=str)]
        finally:
            active.remove(identity)

    active.add(identity)
    try:
        converted = json_value(value)
        return json_safe(converted, active)
    finally:
        active.remove(identity)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluation_manifest_path() -> Path:
    annotated = ROOT / "manifests" / "evaluation_assets_v3.csv"
    return annotated if annotated.exists() else ROOT / "manifests" / "evaluation_assets_v2.csv"


def evaluation_rows() -> list[dict]:
    path = evaluation_manifest_path()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def configured_source_asset(dataset: str, asset: dict) -> Path | None:
    source = asset.get("source_asset")
    if not source:
        return None
    if dataset == "robophyscan":
        parts = source.replace("\\", "/").split("/")
        try:
            relative = parts[parts.index(asset["asset_id"]) + 1 :]
        except ValueError:
            relative = ["sim_ready", parts[-1]]
        return Path(CONFIG["datasets"][dataset]) / asset["asset_id"] / Path(*relative)
    if dataset == "artvip":
        return Path(CONFIG["datasets"][dataset]) / asset["asset_id"] / source_filename(source)
    return Path(source)


def source_filename(value: str) -> str:
    """Return a basename for paths authored on either Windows or POSIX."""
    return str(value).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def selected_assets(dataset: str, limit: int | None, start: int = 0, asset_id: str | None = None) -> list[dict]:
    rps = {row["asset_id"]: row for row in read_jsonl(ROOT / "manifests" / "robophyscan_v1.jsonl")}
    partnet = {
        row["asset_id"]: row for row in read_jsonl(ROOT / "manifests" / "partnet_mobility_v1.jsonl")
    }
    artvip = {row["asset_id"]: row for row in read_jsonl(ROOT / "manifests" / "artvip_v1.jsonl")}
    output = []
    sources = {"robophyscan": rps, "partnet_mobility": partnet, "artvip": artvip}
    for selection in evaluation_rows():
        if selection["dataset"] != dataset:
            continue
        item = sources[dataset].get(selection["asset_id"])
        if item:
            if dataset == "partnet_mobility":
                asset_dir = Path(CONFIG["datasets"]["partnet_mobility"]) / item["asset_id"]
                selection = {
                    **selection,
                    "source_asset": str(asset_dir / "mobility.urdf"),
                }
                item = {
                    **item,
                    "asset_dir": str(asset_dir),
                    "source_asset": str(asset_dir / "mobility.urdf"),
                    "semantics": str(asset_dir / "semantics.txt"),
                    "mobility": str(asset_dir / "mobility_v2.json"),
                }
            elif dataset == "artvip":
                item = {
                    **item,
                    "source_asset": str(configured_source_asset(dataset, item)),
                    "asset_dir": str(Path(CONFIG["datasets"][dataset]) / item["asset_id"]),
                }
            output.append({**item, "selection": selection})
    if asset_id is not None:
        output = [item for item in output if item["asset_id"] == asset_id]
    end = start + limit if limit else None
    return output[start:end]


def launch_config(cpu_only: bool, video: bool) -> dict:
    config = {
        "headless": True,
        "active_gpu": None,
        "physics_gpu": None,
        "multi_gpu": False,
        "disable_viewport_updates": not video,
        "renderer": "MinimalRendering" if cpu_only else "RayTracedLighting",
        "width": 640,
        "height": 480,
        "fast_shutdown": True,
    }
    if not cpu_only:
        gpu = int(os.environ.get("RAW_EVAL_GPU", "0"))
        config.update({"active_gpu": gpu, "physics_gpu": gpu})
    return config


def initialize_app():
    kit_log = ensure_inside_root(
        ROOT / "reports" / f"kit_{RUN_LABEL}_{os.getpid()}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    )
    sys.argv.append(f"--/log/file={kit_log}")
    kit_threads = os.environ.get("RAW_EVAL_KIT_THREADS")
    if kit_threads:
        sys.argv.extend(
            [
                f"--/plugins/carb.tasking.plugin/threadCount={kit_threads}",
                f"--/plugins/omni.tbb.globalcontrol/maxThreadCount={kit_threads}",
            ]
        )
    from isaacsim import SimulationApp

    cpu_only = os.environ.get("RAW_EVAL_CPU_ONLY") == "1"
    video = os.environ.get("RAW_EVAL_VIDEO") == "1"
    if cpu_only:
        sys.argv.append("--/app/renderer/enabled=false")
        print("isaac_cpu_only physics_gpu=none renderer=disabled", flush=True)
    else:
        gpu = int(os.environ.get("RAW_EVAL_GPU", "0"))
        # Keep worker GPU selection explicit at the Kit launcher boundary.
        sys.argv.extend([
            f"--/renderer/activeGpu={gpu}",
            f"--/physics/cudaDevice={gpu}",
            "--/renderer/multiGpu/enabled=false",
        ])
        print(f"isaac_gpu active={gpu} physics={gpu} multi_gpu=false", flush=True)
    return SimulationApp(launch_config(cpu_only, video))


def runtime_blocked(reason: str, *, phase: str, error: str | None = None, **details) -> dict:
    """Return a terminal runtime/evaluator record without turning it into an asset failure."""
    result = {
        # A blocked runtime attempt was not physically evaluated and must not
        # be converted into a score of zero by an aggregate metric.
        "applicable": False,
        "pass": False,
        "status": "evaluation_blocked",
        "reason": reason,
        "reason_class": "runtime" if reason.startswith(("cuda_", "physx_", "runtime_")) else "evaluator",
        "phase": phase,
        "retryable_runtime": False,
    }
    if error:
        result["error"] = error
    if details:
        result["diagnostics"] = details
    return result


def rigid_body_mass_or_blocked(dc, handle, path: str, phase: str):
    if not handle:
        return None, runtime_blocked(
            "runtime_rigid_body_handle_unavailable",
            phase=phase,
            rigid_body_path=path,
        )
    try:
        properties = dc.get_rigid_body_properties(handle)
        mass = getattr(properties, "mass", None)
        if mass is None or not math.isfinite(float(mass)) or float(mass) <= 0:
            return None, runtime_blocked(
                "runtime_rigid_body_properties_unavailable",
                phase=phase,
                rigid_body_path=path,
            )
        return float(mass), None
    except Exception as exc:
        return None, runtime_blocked(
            "runtime_rigid_body_properties_exception",
            phase=phase,
            error=f"{type(exc).__name__}: {exc}",
            rigid_body_path=path,
        )


def prepare_partnet_urdf(asset: dict, output_dir: Path) -> Path:
    source_dir = Path(asset["asset_dir"])
    prepared_dir = ensure_inside_root(output_dir / "source")
    urdf_text = Path(asset["source_asset"]).read_text(encoding="utf-8")
    filenames = re.findall(r'filename\s*=\s*"([^"]+)"', urdf_text)
    for filename in sorted(set(filenames)):
        relative = Path(filename)
        source_top = source_dir / relative.parts[0]
        target_top = prepared_dir / relative.parts[0]
        if source_top.is_dir() and not target_top.exists():
            shutil.copytree(source_top, target_top)
        safe_name = re.sub(r"[^A-Za-z0-9_.]", "_", relative.name)
        if safe_name != relative.name:
            safe_relative = relative.with_name(safe_name)
            shutil.copy2(prepared_dir / relative, prepared_dir / safe_relative)
            urdf_text = urdf_text.replace(filename, safe_relative.as_posix())
    images = source_dir / "images"
    target_images = prepared_dir / "images"
    if images.is_dir() and not target_images.exists():
        shutil.copytree(images, target_images)
    for texture in source_dir.iterdir():
        if texture.is_file() and texture.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            shutil.copy2(texture, prepared_dir / texture.name)
    prepared_urdf = prepared_dir / "mobility.urdf"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared_urdf.write_text(urdf_text, encoding="utf-8")
    return prepared_urdf


def convert_partnet(app, assets: list[dict]) -> None:
    import omni.kit.commands
    import omni.usd

    output_root = ensure_inside_root(ROOT / "derived_assets" / "partnet_mobility_v5")
    output_root.mkdir(parents=True, exist_ok=True)
    for index, asset in enumerate(assets, start=1):
        output_dir = ensure_inside_root(output_root / asset["asset_id"])
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / "asset.usd"
        flattened = output_dir / "evaluation_asset.usd"
        if flattened.exists():
            print(f"[{index}/{len(assets)}] exists {asset['asset_id']}")
            continue
        prepared_urdf = prepare_partnet_urdf(asset, output_dir)
        print(f"[{index}/{len(assets)}] importing {prepared_urdf}", flush=True)
        status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        if not status:
            raise RuntimeError("URDFCreateImportConfig failed")
        import_config.set_merge_fixed_joints(True)
        import_config.set_import_inertia_tensor(True)
        import_config.set_fix_base(False)
        import_config.set_convex_decomp(False)
        import_config.set_collision_from_visuals(False)
        import_config.set_distance_scale(1.0)
        import_config.set_make_default_prim(True)
        import_config.set_create_physics_scene(True)
        status, _prim_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=str(prepared_urdf),
            import_config=import_config,
            dest_path=str(destination),
            get_articulation_root=True,
        )
        if not status:
            print(f"[{index}/{len(assets)}] failed {asset['asset_id']}", flush=True)
            continue
        texture_output = ensure_inside_root(output_dir / "configuration" / "materials" / "textures")
        texture_output.mkdir(parents=True, exist_ok=True)
        for texture in (output_dir / "source" / "images").glob("*"):
            if texture.is_file() and texture.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                shutil.copy2(texture, texture_output / texture.name)
        from pxr import Usd

        imported_physics = output_dir / "configuration" / "asset_physics.usd"
        imported_stage = Usd.Stage.Open(str(imported_physics if imported_physics.exists() else destination))
        flattened = ensure_inside_root(flattened)
        imported_stage.Flatten().Export(str(flattened))
        omni.usd.get_context().close_stage()
        print(f"[{index}/{len(assets)}] converted {asset['asset_id']}", flush=True)
        app.update()


def open_stage(path: Path, app):
    from pxr import Usd

    # ponytail: static checks do not need an omni.usd context; direct USD loading
    # also avoids the synchronous context-open crash observed on Isaac Sim 5.1.
    return Usd.Stage.Open(str(path))


def open_stage_in_context(path: Path, app):
    import omni.usd

    context = omni.usd.get_context()
    context.disable_save_to_recent_files()
    try:
        opened = context.open_stage(str(path))
    finally:
        context.enable_save_to_recent_files()
    if not opened:
        raise RuntimeError(f"context_stage_open_failed: {path}")
    app.update()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError(f"context_stage_missing: {path}")
    return stage


def schemas(prim) -> set[str]:
    try:
        return set(prim.GetAppliedSchemas() or [])
    except Exception:
        return set()


def collision_enabled(prim) -> bool:
    if any("CollisionAPI" in schema for schema in schemas(prim)):
        return True
    attribute = prim.GetAttribute("physics:collisionEnabled")
    return bool(attribute and attribute.Get())


def stage_prims(stage, include_instance_proxies=False):
    if not include_instance_proxies:
        return stage.Traverse()
    from pxr import Usd

    return stage.Traverse(Usd.TraverseInstanceProxies())


def joint_metadata_names(type_name: str) -> tuple[str, ...]:
    common = ("physics:axis", "physics:lowerLimit", "physics:upperLimit")
    if type_name == "PhysicsRevoluteJoint":
        return common + (
            "drive:angular:physics:stiffness",
            "drive:angular:physics:damping",
            "drive:angular:physics:maxForce",
            "drive:angular:physics:targetPosition",
            "drive:angular:physics:targetVelocity",
            "drive:angular:physics:type",
        )
    if type_name == "PhysicsPrismaticJoint":
        return common + (
            "drive:linear:physics:stiffness",
            "drive:linear:physics:damping",
            "drive:linear:physics:maxForce",
            "drive:linear:physics:targetPosition",
            "drive:linear:physics:targetVelocity",
            "drive:linear:physics:type",
        )
    return ()


def classify_joint_drive(values: dict, authored: bool, command_attrs_valid: dict, drive_schema_applied: bool = True) -> str:
    if not drive_schema_applied or not authored:
        return "passive_joint"
    stiffness = values.get("stiffness")
    position = values.get("targetPosition")
    if (stiffness is not None and math.isfinite(stiffness) and stiffness > 0) or (position is not None and math.isfinite(position)):
        return "position_drive" if command_attrs_valid["targetPosition"] and command_attrs_valid["targetVelocity"] else "unsupported"
    if any(values.get(key) is not None for key in ("targetVelocity", "damping", "maxForce")):
        return "velocity_drive" if command_attrs_valid["targetVelocity"] else "unsupported"
    return "unsupported"


def joint_drive_metadata(joint_prim, joint_type: str) -> dict:
    """Read, but never alter, the authored DriveAPI contract for one finite DOF."""
    prefix = "linear" if joint_type == "translation" else "angular"
    drive_schema = f"PhysicsDriveAPI:{prefix}"
    drive_schema_applied = drive_schema in set(joint_prim.GetAppliedSchemas())
    names = {
        key: f"drive:{prefix}:physics:{key}"
        for key in ("stiffness", "damping", "maxForce", "targetPosition", "targetVelocity", "type")
    }
    values, attrs, authored = {}, {}, False
    for key, name in names.items():
        attr = joint_prim.GetAttribute(name)
        attrs[key] = attr
        valid = bool(attr and attr.IsValid())
        value = attr.Get() if valid else None
        values[key] = float(value) if key != "type" and value is not None else (str(value) if value is not None else None)
        authored |= valid and attr.HasAuthoredValueOpinion()
    command_valid = {key: bool(attr and attr.IsValid()) for key, attr in attrs.items()}
    wants_position = values["stiffness"] is not None and math.isfinite(values["stiffness"]) and values["stiffness"] > 0 or values["targetPosition"] is not None
    wants_velocity = not wants_position and any(values[key] is not None for key in ("targetVelocity", "damping", "maxForce"))
    created_targets = set()
    # A DriveAPI may omit a command target.  Author it only in the session
    # layer, then Clear it after the segment so the source contract is intact.
    if drive_schema_applied and authored and (wants_position or wants_velocity):
        from pxr import Sdf

        for key in (("targetPosition", "targetVelocity") if wants_position else ("targetVelocity",)):
            if not command_valid[key]:
                attrs[key] = joint_prim.CreateAttribute(names[key], Sdf.ValueTypeNames.Float)
                command_valid[key] = bool(attrs[key] and attrs[key].IsValid())
                created_targets.add(key)
    mode = classify_joint_drive(values, authored, command_valid, drive_schema_applied)
    return {"control_mode": mode, "metadata": values, "attrs": attrs, "authored": authored, "drive_schema": drive_schema, "drive_schema_applied": drive_schema_applied, "created_targets": created_targets}


def set_joint_drive_targets(drive: dict, position=None, velocity=None) -> None:
    """Write only DriveAPI command targets to the current session edit target."""
    if position is not None:
        drive["attrs"]["targetPosition"].Set(float(position))
    if velocity is not None:
        drive["attrs"]["targetVelocity"].Set(float(velocity))


def joint_segment_command(start, target, elapsed_seconds, segment_seconds, response_seconds):
    """Return a smooth absolute setpoint and its bounded target speed."""
    distance_to_target = float(target) - float(start)
    duration = max(float(segment_seconds) * 0.8, float(response_seconds), 1e-9)
    progress = max(0.0, min(1.0, float(elapsed_seconds) / duration))
    smooth = progress * progress * (3.0 - 2.0 * progress)
    speed = abs(distance_to_target) * 6.0 * progress * (1.0 - progress) / duration
    return float(start) + distance_to_target * smooth, speed


def restore_joint_drive_targets(drive: dict) -> None:
    original = drive["metadata"]
    for key, value in (("targetPosition", original["targetPosition"]), ("targetVelocity", original["targetVelocity"])):
        if key in drive.get("created_targets", set()) or value is None:
            drive["attrs"][key].Clear()
        else:
            drive["attrs"][key].Set(float(value))


def inertia_scale_ratio(mass, inertia, diagonal):
    if mass is None or inertia is None or diagonal is None:
        return None
    mass = float(mass)
    diagonal = float(diagonal)
    if not math.isfinite(mass) or not math.isfinite(diagonal) or mass <= 0 or diagonal <= 0:
        return None
    values = [float(value) for value in inertia]
    if not values or not all(math.isfinite(value) for value in values):
        return None
    return max(values) / (mass * diagonal * diagonal)


def mass_ratio(parent_mass, child_mass):
    if parent_mass is None or child_mass is None:
        return None
    values = [float(parent_mass), float(child_mass)]
    if not all(math.isfinite(value) and value > 0 for value in values):
        return None
    return max(values) / min(values)


def collapsed_collision_adjacency(edges, collider_bodies):
    collider_bodies = set(collider_bodies)
    graph = defaultdict(set)
    for first, second in edges:
        graph[first].add(second)
        graph[second].add(first)
    adjacent = set()
    for source in collider_bodies:
        pending = list(graph[source])
        visited = {source}
        while pending:
            node = pending.pop()
            if node in visited:
                continue
            visited.add(node)
            if node in collider_bodies:
                adjacent.add(frozenset((source, node)))
                continue
            pending.extend(graph[node] - visited)
    return adjacent


def asset_values(value) -> list:
    from pxr import Sdf

    if isinstance(value, Sdf.AssetPath):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, Sdf.AssetPath)]
    return []


def missing_asset_references(stage) -> list[dict]:
    from pxr import Sdf

    missing = []
    for prim in stage.Traverse():
        for attribute in prim.GetAttributes():
            values = asset_values(attribute.Get())
            if not values:
                continue
            stack = attribute.GetPropertyStack()
            layer = stack[0].layer if stack else stage.GetRootLayer()
            for value in values:
                raw = value.path
                if not raw or re.match(r"^[a-z]+://", raw, re.I):
                    continue
                resolved = value.resolvedPath or Sdf.ComputeAssetPathRelativeToLayer(layer, raw)
                if not resolved or not Path(resolved).is_file():
                    missing.append(
                        {
                            "prim": str(prim.GetPath()),
                            "attribute": attribute.GetName(),
                            "asset": raw,
                            "resolved": resolved or None,
                        }
                    )
    return missing


def list_op_items(editor) -> list:
    for method in ("GetAddedOrExplicitItems", "GetAppliedItems"):
        if hasattr(editor, method):
            return list(getattr(editor, method)())
    items = []
    for attribute in ("explicitItems", "prependedItems", "appendedItems", "addedItems"):
        items.extend(list(getattr(editor, attribute, []) or []))
    return items


def invalid_prim_references(stage) -> list[dict]:
    from pxr import Sdf

    invalid = []
    seen = set()
    for prim in stage.Traverse():
        for spec in prim.GetPrimStack():
            for reference in list_op_items(spec.referenceList):
                if not reference.primPath:
                    continue
                target_layer = spec.layer
                resolved = None
                if reference.assetPath:
                    resolved = Sdf.ComputeAssetPathRelativeToLayer(spec.layer, reference.assetPath)
                    target_layer = Sdf.Layer.FindOrOpen(resolved) if resolved else None
                key = (spec.layer.identifier, reference.assetPath, str(reference.primPath))
                if key in seen:
                    continue
                seen.add(key)
                if target_layer is None or target_layer.GetPrimAtPath(reference.primPath) is None:
                    invalid.append(
                        {
                            "authoring_layer": spec.layer.identifier,
                            "asset": reference.assetPath or None,
                            "resolved": resolved,
                            "target_prim": str(reference.primPath),
                            "composed_prim": str(prim.GetPath()),
                        }
                    )
    return invalid


def precheck(stage) -> dict:
    from pxr import UsdGeom, UsdPhysics

    rigid = []
    collision = []
    joints = []
    limited_nonfixed_joint_count = 0
    articulation_roots = []
    missing_joint_bodies = []
    for prim in stage_prims(stage, include_instance_proxies=True):
        applied = schemas(prim)
        path = str(prim.GetPath())
        if "PhysicsRigidBodyAPI" in applied:
            rigid.append(path)
        if "PhysicsArticulationRootAPI" in applied:
            articulation_roots.append(path)
        if collision_enabled(prim):
            collision.append(path)
        if prim.IsA(UsdPhysics.Joint):
            joints.append(path)
            joint = UsdPhysics.Joint(prim)
            if not prim.IsA(UsdPhysics.FixedJoint):
                lower_attr = prim.GetAttribute("physics:lowerLimit")
                upper_attr = prim.GetAttribute("physics:upperLimit")
                lower = lower_attr.Get() if lower_attr and lower_attr.IsValid() else None
                upper = upper_attr.Get() if upper_attr and upper_attr.IsValid() else None
                try:
                    if lower is not None and upper is not None and math.isfinite(float(lower)) and math.isfinite(float(upper)) and float(upper) > float(lower):
                        limited_nonfixed_joint_count += 1
                except (TypeError, ValueError):
                    pass
            body0 = joint.GetBody0Rel().GetTargets()
            body1 = joint.GetBody1Rel().GetTargets()
            if (not body0 and not body1) or (
                not prim.IsA(UsdPhysics.FixedJoint) and (not body0 or not body1)
            ):
                missing_joint_bodies.append(path)
    default_prim = stage.GetDefaultPrim()
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    composition_errors = (
        [str(error) for error in stage.GetCompositionErrors()]
        if hasattr(stage, "GetCompositionErrors")
        else []
    )
    missing_assets = missing_asset_references(stage)
    invalid_references = invalid_prim_references(stage)
    invalid_external_references = [
        item for item in invalid_references if item.get("asset")
    ]
    load_pass = bool(
        default_prim
        and math.isfinite(meters_per_unit)
        and meters_per_unit > 0
        and rigid
        and collision
        and not missing_joint_bodies
        and not composition_errors
        and not missing_assets
        and not invalid_external_references
    )
    return {
        "default_prim": str(default_prim.GetPath()) if default_prim else None,
        "rigid_bodies": rigid,
        "collisions": collision,
        "joints": joints,
        "limited_nonfixed_joint_count": limited_nonfixed_joint_count,
        "articulation_roots": articulation_roots,
        "missing_joint_bodies": missing_joint_bodies,
        "meters_per_unit": meters_per_unit,
        "composition_errors": composition_errors,
        "missing_asset_references": missing_assets,
        "invalid_prim_references": invalid_references,
        "invalid_external_prim_references": invalid_external_references,
        "load_pass": load_pass,
    }


def metric_rigid_bodies(stage, check: dict) -> list[str]:
    threshold = float(CONFIG["simulation"].get("metric_link_min_mass_kg", 0.0))
    kept = []
    for path in check.get("rigid_bodies", []):
        attribute = stage.GetPrimAtPath(path).GetAttribute("physics:mass")
        mass = attribute.Get() if attribute and attribute.HasAuthoredValueOpinion() else None
        if mass is None or not math.isfinite(float(mass)) or float(mass) >= threshold:
            kept.append(path)
    return kept or list(check.get("rigid_bodies", [])[:1])


def solved_metric_rigid_bodies(stage, check: dict) -> tuple[list[str], dict[str, float]]:
    from omni.isaac.dynamic_control import _dynamic_control

    threshold = float(CONFIG["simulation"].get("metric_link_min_mass_kg", 0.0))
    masses = {}
    dc = _dynamic_control.acquire_dynamic_control_interface()
    for path in check.get("rigid_bodies", []):
        try:
            handle = dc.get_rigid_body(path)
            mass = float(dc.get_rigid_body_properties(handle).mass) if handle else math.nan
            if math.isfinite(mass):
                masses[path] = mass
        except Exception:
            continue
    kept = [path for path in check.get("rigid_bodies", []) if masses.get(path, threshold) >= threshold]
    return (kept or metric_rigid_bodies(stage, check), masses)


def joint_moving_mass(stage, related_paths: set[str], runtime_masses: dict[str, float]):
    runtime_values = []
    for path in related_paths:
        try:
            mass = float(runtime_masses[path])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(mass) and mass > 0:
            runtime_values.append(mass)
    if runtime_values:
        return sum(runtime_values), "runtime_dynamic_control"

    authored_values = []
    for path in related_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            continue
        attribute = prim.GetAttribute("physics:mass")
        value = attribute.Get() if attribute and attribute.HasAuthoredValueOpinion() else None
        try:
            mass = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(mass) and mass > 0:
            authored_values.append(mass)
    if authored_values:
        return sum(authored_values), "usd_authored_mass"
    return None, None


def world_translation(stage, prim_path: str):
    from pxr import UsdGeom, Usd, Gf

    prim = stage.GetPrimAtPath(prim_path)
    matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    value = matrix.ExtractTranslation()
    return (float(value[0]), float(value[1]), float(value[2]))


def world_quaternion(stage, prim_path: str):
    from pxr import UsdGeom, Usd

    prim = stage.GetPrimAtPath(prim_path)
    quat = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim).ExtractRotationQuat()
    imaginary = quat.GetImaginary()
    values = (float(quat.GetReal()), *(float(value) for value in imaginary))
    norm = math.sqrt(sum(value * value for value in values))
    return tuple(value / norm for value in values)


def angular_distance(a, b) -> float:
    dot = min(1.0, abs(sum(x * y for x, y in zip(a, b))))
    return 2.0 * math.acos(dot)


def attach_stage(stage, app) -> None:
    import omni.usd
    from pxr import UsdUtils

    if omni.usd.get_context().get_stage() == stage:
        app.update()
        return
    cache = UsdUtils.StageCache.Get()
    stage_id = cache.GetId(stage)
    if not stage_id.IsValid():
        stage_id = cache.Insert(stage)
    omni.usd.get_context().attach_stage_with_callback(stage_id.ToLongInt())
    app.update()


def distance(a, b) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def add_session_physics(stage, ground_z: float):
    from pxr import Gf, UsdGeom, UsdPhysics

    stage.SetEditTarget(stage.GetSessionLayer())
    physics_scenes = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
    if not physics_scenes:
        scene = UsdPhysics.Scene.Define(stage, "/__raw_eval/PhysicsScene")
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr(abs(float(CONFIG["simulation"]["gravity"])))
        physics_scenes = [scene.GetPrim()]
    if os.environ.get("RAW_EVAL_CPU_ONLY") == "1":
        from isaacsim.core.simulation_manager import PhysxScene

        for physics_scene in physics_scenes:
            physx_scene = PhysxScene(physics_scene)
            physx_scene.set_enabled_gpu_dynamics(False)
            physx_scene.set_broadphase_type("SAP")
    ground = UsdGeom.Cube.Define(stage, "/__raw_eval/Ground")
    ground.CreateSizeAttr(1.0)
    ground.AddScaleOp().Set(Gf.Vec3f(1000, 1000, 0.05))
    ground.AddTranslateOp().Set(Gf.Vec3d(0, 0, ground_z - 0.025))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(ground.GetPrim()).CreateKinematicEnabledAttr(True)


def stage_bounds(stage, prim_path: str):
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    bounds = cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedBox()
    minimum = bounds.GetMin()
    maximum = bounds.GetMax()
    return tuple(map(float, minimum)), tuple(map(float, maximum))


def collision_bounds(stage, check: dict):
    bounds = []
    for path in check.get("collisions", []):
        try:
            minimum, maximum = stage_bounds(stage, path)
            values = (*minimum, *maximum)
            if all(math.isfinite(value) for value in values) and all(
                maximum[axis] > minimum[axis] for axis in range(3)
            ):
                bounds.append((minimum, maximum))
        except Exception:
            continue
    if not bounds:
        path = check["default_prim"] or check["rigid_bodies"][0]
        return stage_bounds(stage, path)
    minimum = tuple(min(item[0][axis] for item in bounds) for axis in range(3))
    maximum = tuple(max(item[1][axis] for item in bounds) for axis in range(3))
    return minimum, maximum


def object_link_collision_bounds(stage, check: dict, records=None) -> dict:
    """Return collision bounds grouped by the object rigid-body link.

    Collision geometry is assigned to the nearest declared rigid-body ancestor.
    Visual geometry and the synthetic ground prim are therefore excluded by
    construction.  The result is intentionally diagnostic-friendly so settling
    can identify which link defines the object's lowest point.
    """
    rigid_bodies = [str(path) for path in check.get("rigid_bodies", [])]
    grouped = {
        path: {"minimum": None, "maximum": None, "collision_paths": []}
        for path in rigid_bodies
    }
    entries = []
    if records is not None:
        entries = [
            (str(row.get("path")), str(row.get("rigid_body")), row.get("vertices"))
            for row in records
            if row.get("collision") and row.get("rigid_body")
        ]
    else:
        entries = [(str(path), None, None) for path in check.get("collisions", [])]
    for collision_path, explicit_owner, vertices in entries:
        owners = [explicit_owner] if explicit_owner in grouped else [
            path for path in rigid_bodies
            if collision_path == path or collision_path.startswith(path.rstrip("/") + "/")
        ]
        if not owners or owners[0] not in grouped:
            continue
        owner = owners[0] if explicit_owner else max(owners, key=len)
        if vertices is not None:
            try:
                minimum = tuple(float(min(vertex[axis] for vertex in vertices)) for axis in range(3))
                maximum = tuple(float(max(vertex[axis] for vertex in vertices)) for axis in range(3))
            except (TypeError, ValueError):
                continue
        else:
            try:
                minimum, maximum = stage_bounds(stage, collision_path)
            except Exception:
                continue
        values = (*minimum, *maximum)
        if not all(math.isfinite(value) for value in values):
            continue
        # Composed collision geometry can contain thin/planar prims; retain
        # them for link-union aggregation instead of dropping the whole link.
        if not any(maximum[axis] > minimum[axis] for axis in range(3)):
            continue
        item = grouped[owner]
        item["collision_paths"].append(collision_path)
        if item["minimum"] is None:
            item["minimum"], item["maximum"] = minimum, maximum
        else:
            item["minimum"] = tuple(min(item["minimum"][axis], minimum[axis]) for axis in range(3))
            item["maximum"] = tuple(max(item["maximum"][axis], maximum[axis]) for axis in range(3))
    valid = {
        path: item for path, item in grouped.items() if item["minimum"] is not None
    }
    if not valid:
        return {
            "links": {},
            "lowest_link_path": None,
            "lowest_link_z": None,
            "object_link_count": len(rigid_bodies),
            "valid_link_count": 0,
            "union_minimum": None,
            "union_maximum": None,
        }
    lowest_path = min(valid, key=lambda path: (valid[path]["minimum"][2], path))
    union_minimum = tuple(min(item["minimum"][axis] for item in valid.values()) for axis in range(3))
    union_maximum = tuple(max(item["maximum"][axis] for item in valid.values()) for axis in range(3))
    return {
        "links": valid,
        "lowest_link_path": lowest_path,
        "lowest_link_z": valid[lowest_path]["minimum"][2],
        "object_link_count": len(rigid_bodies),
        "valid_link_count": len(valid),
        "union_minimum": union_minimum,
        "union_maximum": union_maximum,
    }


def apply_settle_drop(stage, check: dict, ground_z: float, records=None) -> dict:
    """Move the session-layer object so its lowest collision link is airborne."""
    from pxr import Gf, UsdGeom

    bounds = object_link_collision_bounds(stage, check, records)
    if bounds["lowest_link_z"] is None:
        return {
            "applicable": False,
            "status": "evaluation_blocked",
            "reason": "settle_object_link_collision_bounds_unavailable",
            "object_link_bounds": bounds,
        }
    minimum = bounds["union_minimum"]
    maximum = bounds["union_maximum"]
    diagonal = distance(minimum, maximum)
    settings = CONFIG["simulation"]
    ratio = float(settings.get("settle_drop_height_diagonal_ratio", 0.25))
    lower = float(settings.get("settle_drop_height_min_m", 0.02))
    upper = float(settings.get("settle_drop_height_max_m", 0.10))
    drop_height = max(lower, min(upper, ratio * diagonal))
    target_z = float(ground_z) + drop_height
    translation_z = target_z - float(bounds["lowest_link_z"])
    root_path = check.get("default_prim") or (check.get("rigid_bodies") or [None])[0]
    root = stage.GetPrimAtPath(root_path) if root_path else None
    if root is None or not root.IsValid() or not root.IsA(UsdGeom.Xformable):
        return {
            "applicable": False,
            "status": "evaluation_blocked",
            "reason": "settle_drop_transform_unavailable",
            "object_link_bounds": bounds,
        }
    stage.SetEditTarget(stage.GetSessionLayer())
    xform = UsdGeom.Xformable(root)
    try:
        op = xform.AddTranslateOp(opSuffix="rawEvalSettleDrop", precision=UsdGeom.XformOp.PrecisionDouble)
        op.Set(Gf.Vec3d(0.0, 0.0, translation_z))
    except Exception as exc:
        return {
            "applicable": False,
            "status": "evaluation_blocked",
            "reason": "settle_drop_transform_unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "object_link_bounds": bounds,
        }
    shifted = object_link_collision_bounds(stage, check, mesh_records(stage))
    return {
        "applicable": True,
        "status": "complete",
        "drop_height_mode": "adaptive",
        "drop_height_target_m": drop_height,
        "drop_height_actual_m": shifted["lowest_link_z"] - float(ground_z) if shifted["lowest_link_z"] is not None else None,
        "target_lowest_link_z_m": target_z,
        "lowest_link_z_before_drop_m": bounds["lowest_link_z"],
        "lowest_link_z_after_release_m": shifted["lowest_link_z"],
        "lowest_link_path_before_drop": bounds["lowest_link_path"],
        "lowest_link_path_after_drop": shifted["lowest_link_path"],
        "object_link_count": bounds["object_link_count"],
        "valid_link_count": bounds["valid_link_count"],
        "drop_translation_m": [0.0, 0.0, translation_z],
        "drop_height_error_m": (shifted["lowest_link_z"] - target_z) if shifted["lowest_link_z"] is not None else None,
        "object_link_bounds": shifted,
    }


def observe_stability(stage, check: dict, app, maximum_seconds: float, ground_sensor=None, drop_info=None, collision_records=None, sampling_dt: float | None = None) -> dict:
    import numpy as np
    import omni.timeline
    from v5_geometry import support_patch

    settings = CONFIG["simulation"]
    dt = float(sampling_dt or settings["dt"])
    minimum_steps = max(1, int(float(settings["settle_min_seconds"]) / dt))
    maximum_steps = max(minimum_steps, int(maximum_seconds / dt))
    window_steps = max(2, int(float(settings["settle_window_seconds"]) / dt))
    metric_bodies, solved_masses = solved_metric_rigid_bodies(stage, check)
    bounds_path = check["default_prim"] or (metric_bodies[0] if metric_bodies else None)
    motion_root_path = metric_bodies[0] if metric_bodies else bounds_path
    if not bounds_path or not motion_root_path:
        return {"applicable": False, "pass": False, "reason": "no_root_or_rigid_body"}
    root_positions = deque(maxlen=window_steps)
    link_positions = {path: deque(maxlen=2) for path in metric_bodies}
    link_quaternions = {path: deque(maxlen=2) for path in metric_bodies}
    initial_minimum, initial_maximum = collision_bounds(stage, check)
    initial_link_bounds = object_link_collision_bounds(stage, check, collision_records)
    diagonal = distance(initial_minimum, initial_maximum)
    ground_z = float(settings.get("ground_z_m", 0.0))
    initial_ground_distance = initial_minimum[2] - ground_z
    final_linear_speed = final_angular_speed = root_drift = penetration = math.inf
    max_linear_speed = max_angular_speed = max_penetration = max_fall_speed = 0.0
    first_ground_contact = None
    energy = []
    contacts = deque(maxlen=max(1, int(1.0 / dt)))
    contact_witness_frames = []
    contact_witness_observed = False
    runtime_events = []
    finite = True
    settled_early = False
    step = 0
    timeline = omni.timeline.get_timeline_interface()
    start_time_seconds = float(timeline.get_current_time())
    if ground_sensor is None:
        runtime_events.append(
            {
                "phase": "initialization",
                "time_s": 0.0,
                "body": None,
                "event": "ground_contact_sensor_unavailable",
                "value": None,
                "threshold": None,
                "severe": False,
            }
        )
    for step in range(maximum_steps):
        app.update()
        elapsed_seconds = timeline_elapsed_seconds(timeline, start_time_seconds, (step + 1) * dt)
        root = world_translation(stage, motion_root_path)
        root_positions.append(root)
        for path in metric_bodies:
            link_positions[path].append(world_translation(stage, path))
            link_quaternions[path].append(world_quaternion(stage, path))
        values = [value for position in root_positions for value in position]
        values.extend(value for positions in link_positions.values() for position in positions for value in position)
        finite = all(math.isfinite(value) and abs(value) < 1e4 for value in values)
        if not finite:
            runtime_events.append({"phase": "fall" if first_ground_contact is None else "settle", "time_s": elapsed_seconds, "body": motion_root_path, "event": "nonfinite_state", "value": None, "threshold": None, "severe": True})
            break
        if all(len(items) >= 2 for items in link_positions.values()):
            linear = [distance(items[-2], items[-1]) / dt for items in link_positions.values()]
            angular = [angular_distance(link_quaternions[path][-2], link_quaternions[path][-1]) / dt for path in metric_bodies]
            final_linear_speed = max(linear, default=0.0)
            final_angular_speed = max(angular, default=0.0)
            max_linear_speed = max(max_linear_speed, final_linear_speed)
            max_angular_speed = max(max_angular_speed, final_angular_speed)
            max_fall_speed = max(max_fall_speed, max((max(0.0, -(items[-1][2] - items[-2][2]) / dt) for items in link_positions.values()), default=0.0))
            kinetic = sum(0.5 * solved_masses.get(path, 1.0) * (distance(link_positions[path][-2], link_positions[path][-1]) / dt) ** 2 for path in metric_bodies)
            energy.append(kinetic)
            phase = "fall" if first_ground_contact is None else "settle"
            if final_linear_speed > float(settings["runtime_linear_explosion_mps"]):
                runtime_events.append({"phase": phase, "time_s": elapsed_seconds, "body": motion_root_path, "event": "linear_velocity_explosion", "value": final_linear_speed, "threshold": settings["runtime_linear_explosion_mps"], "severe": True})
            if final_angular_speed > float(settings["runtime_angular_explosion_radps"]):
                runtime_events.append({"phase": phase, "time_s": elapsed_seconds, "body": motion_root_path, "event": "angular_velocity_explosion", "value": final_angular_speed, "threshold": settings["runtime_angular_explosion_radps"], "severe": True})
            teleport = max((distance(items[-2], items[-1]) for items in link_positions.values()), default=0.0)
            teleport_threshold = max(0.1, diagonal * float(settings.get("runtime_teleport_diagonal_ratio", 2.0)))
            if teleport > teleport_threshold:
                runtime_events.append({"phase": phase, "time_s": elapsed_seconds, "body": motion_root_path, "event": "abnormal_teleport", "value": teleport, "threshold": teleport_threshold, "severe": True})
        # Settling needs a body-paired raw contact witness.  Sensor summaries
        # cannot establish that the dropped object contacted the ground.
        frame = contact_frame(ground_sensor, metric_bodies, allow_summary_fallback=False) if ground_sensor is not None else {"contact": False, "contacts": [], "penetration": None}
        penetration = dynamic_contact_penetration(frame)
        max_penetration = max(max_penetration, penetration)
        if frame["contact"] and first_ground_contact is None:
            first_ground_contact = elapsed_seconds
        if frame["contact"]:
            contact_witness_observed = True
            contact_witness_frames.append(frame)
        contacts.append(frame)
        if len(root_positions) < window_steps:
            continue
        root_drift = max(distance(root_positions[0], position) for position in root_positions)
        stable = (
            root_drift < float(settings["root_drift_max_m"])
            and final_linear_speed < float(settings["linear_speed_max_mps"])
            and final_angular_speed < float(settings["angular_speed_max_radps"])
            and penetration < float(settings["penetration_max_m"])
        )
        if step + 1 >= minimum_steps + window_steps and stable:
            settled_early = True
            break
        if runtime_events:
            finite = False
            break
    passed = (
        finite
        and root_drift < float(settings["root_drift_max_m"])
        and final_linear_speed < float(settings["linear_speed_max_mps"])
        and final_angular_speed < float(settings["angular_speed_max_radps"])
        and max_penetration < float(settings["penetration_max_m"])
        and not any(event["severe"] for event in runtime_events)
    )
    contact_points = []
    support_frames = contact_witness_frames or list(contacts)
    for frame in support_frames:
        for contact in frame.get("contacts", []):
            for key in ("position", "point", "contact_point"):
                if key in contact and len(contact[key]) >= 2:
                    contact_points.append(contact[key][:2])
                    break
    # Never rebuild BBoxCache while the timeline is running.  The final support
    # estimate uses the pre-physics footprint; actual dynamic contact and
    # penetration are taken from PhysX contact reports above.
    patch = support_patch(contact_points, ((initial_minimum[0] + initial_maximum[0]) * 0.5, (initial_minimum[1] + initial_maximum[1]) * 0.5), diagonal)
    persistence = sum(frame.get("contact", False) for frame in contacts) / len(contacts) if contacts else 0.0
    patch["contact_persistence"] = persistence
    # This is only a provisional geometry signal while the timeline is active.
    # settle() refreshes it after stopping the timeline using post-contact link
    # bounds, so pre-drop height can never be mistaken for final support.
    current_lowest_link_z = initial_link_bounds.get("lowest_link_z")
    ground_clearance = math.inf if current_lowest_link_z is None else abs(float(current_lowest_link_z) - ground_z)
    geometry_support = current_lowest_link_z is not None and ground_clearance <= float(settings["penetration_max_m"]) and final_linear_speed < float(settings["linear_speed_max_mps"])
    stable_kinematics = (
        finite
        and root_drift < float(settings["root_drift_max_m"])
        and final_linear_speed < float(settings["linear_speed_max_mps"])
        and final_angular_speed < float(settings["angular_speed_max_radps"])
        and max_penetration < float(settings["penetration_max_m"])
        and not any(event["severe"] for event in runtime_events)
    )
    sleeping_contact_fallback = bool(contact_witness_observed and stable_kinematics)
    contact_report_support = bool(persistence > 0.0 or sleeping_contact_fallback)
    patch["contact_report_support"] = contact_report_support
    patch["contact_witness_observed"] = contact_witness_observed
    patch["sleeping_contact_fallback"] = sleeping_contact_fallback
    patch["geometry_support"] = geometry_support
    patch["support_source"] = (
        "contact_report"
        if persistence > 0.0
        else "sleeping_contact_fallback"
        if sleeping_contact_fallback
        else "geometry_fallback"
        if geometry_support
        else "unavailable"
    )
    patch["support_detector_unavailable"] = ground_sensor is None and not geometry_support
    if contact_report_support or geometry_support:
        patch["support_exists"] = True
    energy_initial = max(energy[: max(1, len(energy) // 4)], default=0.0)
    energy_final = sum(energy[-max(1, int(1.0 / dt)) :]) / max(1, min(len(energy), int(1.0 / dt))) if energy else 0.0
    energy_decay = max(0.0, min(1.0, 1.0 - energy_final / max(1e-12, energy_initial))) if energy_initial else 1.0
    return {
        "applicable": True,
        "pass": passed and not patch["support_detector_unavailable"],
        "root_drift_m": root_drift,
        "final_linear_speed_mps": final_linear_speed,
        "final_angular_speed_radps": final_angular_speed,
        "initial_ground_distance_m": initial_ground_distance,
        "lowest_link_z_before_drop_m": (drop_info or {}).get("lowest_link_z_before_drop_m", initial_link_bounds.get("lowest_link_z")),
        "lowest_link_path_before_drop": (drop_info or {}).get("lowest_link_path_before_drop", initial_link_bounds.get("lowest_link_path")),
        "drop_height_target_m": (drop_info or {}).get("drop_height_target_m"),
        "drop_height_actual_m": (drop_info or {}).get("drop_height_actual_m"),
        "first_ground_contact_observed": first_ground_contact is not None,
        "first_ground_contact_seconds": first_ground_contact,
        "maximum_fall_speed_mps": max_fall_speed,
        "maximum_linear_speed_mps": max_linear_speed,
        "maximum_angular_speed_radps": max_angular_speed,
        "maximum_ground_penetration_m": max_penetration,
        "final_ground_penetration_m": penetration,
        "penetration_scope": "physx_contact_report_only",
        "kinetic_energy_initial_proxy_j": energy_initial,
        "kinetic_energy_final_proxy_j": energy_final,
        "kinetic_energy_decay_score": energy_decay,
        "support": patch,
        "runtime_events": runtime_events,
        "severe_runtime_event": any(event["severe"] for event in runtime_events),
        "metric_rigid_bodies": metric_bodies,
        "solved_masses_kg": solved_masses,
        "ignored_tiny_mass_links": sorted(set(check["rigid_bodies"]) - set(metric_bodies)),
        "settle_elapsed_seconds": timeline_elapsed_seconds(timeline, start_time_seconds, (step + 1) * dt),
        "settled_before_timeout": settled_early,
        "thresholds": {
            "root_drift_max_m": settings["root_drift_max_m"],
            "linear_speed_max_mps": settings["linear_speed_max_mps"],
            "angular_speed_max_radps": settings["angular_speed_max_radps"],
            "penetration_max_m": settings["penetration_max_m"],
        },
    }


def settle(stage, check: dict, app) -> dict:
    import omni.timeline

    settings = CONFIG["simulation"]
    ground_z = float(settings.get("ground_z_m", 0.0))
    records = mesh_records(stage)
    drop_info = apply_settle_drop(stage, check, ground_z, records)
    if not drop_info.get("applicable", False):
        return drop_info
    add_session_physics(stage, ground_z)
    attach_stage(stage, app)
    from isaacsim.core.simulation_manager import SimulationManager

    SimulationManager.set_physics_dt(float(settings["dt"]))
    try:
        ground_sensor = create_contact_sensor("/__raw_eval/Ground", "ContactSensor")
    except Exception:
        ground_sensor = None
    timeline = omni.timeline.get_timeline_interface()
    # Some source stages have endTimeCode=4.  At 120 Hz that is a 33 ms
    # looping timeline, so its clock cannot represent a settling trial.
    timeline.set_start_time(0.0)
    timeline.set_end_time(float(settings["settle_max_seconds"]) + 1.0)
    timeline.set_looping(False)
    timeline.set_time_codes_per_second(1.0 / float(settings["dt"]))
    if ground_sensor is not None:
        ground_sensor.initialize()
    # The release, contact sensor, and first-contact clock must start together.
    timeline.play()
    result = observe_stability(
        stage,
        check,
        app,
        float(settings["settle_max_seconds"]),
        ground_sensor,
        drop_info,
        mesh_records(stage),
        # SimulationApp renders at 60 Hz, so one observation spans two 120 Hz physics steps.
        sampling_dt=2.0 * float(settings["dt"]),
    )
    timeline.stop()
    final_link_bounds = object_link_collision_bounds(stage, check, mesh_records(stage))
    final_lowest_z = final_link_bounds.get("lowest_link_z")
    final_clearance = math.inf if final_lowest_z is None else abs(float(final_lowest_z) - ground_z)
    support = result.setdefault("support", {})
    geometry_support = final_lowest_z is not None and final_clearance <= float(settings["penetration_max_m"]) and result.get("final_linear_speed_mps", math.inf) < float(settings["linear_speed_max_mps"])
    support["post_contact_lowest_link_z_m"] = final_lowest_z
    support["post_contact_lowest_link_path"] = final_link_bounds.get("lowest_link_path")
    support["geometry_support"] = geometry_support
    support["support_source"] = (
        "contact_report"
        if support.get("contact_persistence", 0.0) > 0.0
        else "sleeping_contact_fallback"
        if support.get("sleeping_contact_fallback")
        else "geometry_fallback"
        if geometry_support
        else "unavailable"
    )
    verdict = settle_support_verdict(
        support.get("contact_report_support"),
        geometry_support,
        final_lowest_z,
        ground_z,
        float(settings["penetration_max_m"]),
    )
    support.update(verdict)
    support["fallback_attempted"] = bool(
        support.get("contact_persistence", 0.0) <= 0.0
        and support.get("contact_witness_observed")
    )
    support["fallback_succeeded"] = bool(
        support.get("fallback_attempted") and support.get("support_exists")
    )
    support["fallback_result"] = (
        "support_confirmed" if support["fallback_succeeded"]
        else "support_rejected" if support["fallback_attempted"]
        else "not_attempted"
    )
    support["support_detector_unavailable"] = ground_sensor is None and not geometry_support and not support.get("contact_report_support")
    result["post_contact_lowest_link_z_m"] = final_lowest_z
    result["post_contact_lowest_link_path"] = final_link_bounds.get("lowest_link_path")
    result["initial_ground_distance_m"] = drop_info["drop_height_actual_m"]
    result["session_asset_translation_m"] = drop_info["drop_translation_m"]
    result["asset_transform_preserved"] = True
    if verdict["post_contact_collision_below_ground"]:
        result["reason"] = "post_contact_collision_below_ground"
    result["pass"] = bool(result.get("pass") and support["support_exists"] and not support["support_detector_unavailable"] and result.get("first_ground_contact_observed"))
    return result


def begin_physics(
    stage,
    check: dict,
    app,
    drop_height: float = 0.001,
    settle_seconds: float | None = None,
    return_articulation: bool = False,
    pause_after_settle: bool = False,
):
    import omni.timeline

    bounds_path = check["default_prim"] or check["rigid_bodies"][0]
    minimum, maximum = stage_bounds(stage, bounds_path)
    add_session_physics(stage, float(CONFIG["simulation"].get("ground_z_m", 0.0)))
    attach_stage(stage, app)
    timeline = omni.timeline.get_timeline_interface()
    # Source animation ranges must not loop or end a bounded physics trial.
    timeline.set_start_time(0.0)
    timeline.set_end_time(max(timeline.get_end_time(), 3600.0))
    timeline.set_looping(False)
    timeline.set_time_codes_per_second(1.0 / float(CONFIG["simulation"]["dt"]))
    timeline.play()
    for _ in range(3):
        app.update()
    try:
        ground_sensor = create_contact_sensor("/__raw_eval/Ground", "BeginPhysicsContactSensor")
        app.update()
        ground_sensor.initialize()
    except Exception:
        ground_sensor = None
    observe_stability(
        stage,
        check,
        app,
        float(settle_seconds or CONFIG["simulation"]["settle_max_seconds"]),
        ground_sensor,
    )
    runtime_articulation = modern_articulation(check) if return_articulation else None
    # Static geometry planning must observe a paused, settled snapshot.  This
    # also prevents high-frequency BBoxCache reads from racing Kit's USD loader.
    if pause_after_settle:
        timeline.pause()
    else:
        timeline.stop()
    for _ in range(2):
        app.update()
    if return_articulation:
        return timeline, minimum, maximum, runtime_articulation
    return timeline, minimum, maximum


def run_steps(app, seconds: float, callback=None) -> None:
    steps = max(1, int(seconds / float(CONFIG["simulation"]["dt"])))
    for index in range(steps):
        if callback:
            callback(index, steps)
        app.update()


def dynamic_articulation(dc, check: dict):
    candidates = [
        *check.get("articulation_roots", []),
        check.get("default_prim"),
        *check.get("rigid_bodies", [])[:1],
    ]
    for path in filter(None, candidates):
        handle = dc.get_articulation(path)
        if handle:
            return handle, path
    return None, None


def joint_descendants(stage, joint_paths: list[str], child_targets: list[str]) -> set[str]:
    from pxr import UsdPhysics

    edges = {}
    for joint_path in joint_paths:
        joint = UsdPhysics.Joint(stage.GetPrimAtPath(joint_path))
        parents = [str(path) for path in joint.GetBody0Rel().GetTargets()]
        children = [str(path) for path in joint.GetBody1Rel().GetTargets()]
        for parent in parents:
            edges.setdefault(parent, set()).update(children)
    related = set(child_targets)
    pending = list(child_targets)
    while pending:
        for child in edges.get(pending.pop(), ()):
            if child not in related:
                related.add(child)
                pending.append(child)
    return related


_LAST_ARTICULATION_ERRORS = []


class DynamicControlArticulationReader:
    backend = "dynamic_control_read_only"
    handles_initialized = True

    def __init__(self, dc, handle):
        import numpy as np
        from omni.isaac.dynamic_control import _dynamic_control

        self.dc = dc
        self.handle = handle
        self.dofs = [dc.get_articulation_dof(handle, index) for index in range(dc.get_articulation_dof_count(handle))]
        self.dof_names = [dc.get_dof_name(dof) for dof in self.dofs]
        self.num_dof = len(self.dofs)
        self.dof_properties = []
        for dof in self.dofs:
            props = dc.get_dof_properties(dof)
            self.dof_properties.append(
                {
                    "type": int(props.type),
                    "hasLimits": bool(getattr(props, "has_limits", True)),
                    "lower": float(props.lower),
                    "upper": float(props.upper),
                }
            )
        self._state_all = _dynamic_control.STATE_ALL
        self._np = np

    def initialize(self):
        return None

    def get_joint_positions(self):
        return self._np.asarray([self.dc.get_dof_state(dof, self._state_all).pos for dof in self.dofs])

    def get_joint_velocities(self):
        return self._np.asarray([self.dc.get_dof_state(dof, self._state_all).vel for dof in self.dofs])

    def get_measured_joint_efforts(self):
        return self._np.asarray([self.dc.get_dof_state(dof, self._state_all).effort for dof in self.dofs])


def modern_articulation(check: dict):
    global _LAST_ARTICULATION_ERRORS
    errors = []
    runtime_exposed_zero_dofs = False

    # ponytail: this evaluator already owns the timeline/PhysX lifecycle; use
    # its native articulation handle before requiring Isaac's high-level World.
    from omni.isaac.dynamic_control import _dynamic_control

    dc = _dynamic_control.acquire_dynamic_control_interface()
    handle, path = dynamic_articulation(dc, check)
    if handle and dc.get_articulation_dof_count(handle):
        _LAST_ARTICULATION_ERRORS = []
        return DynamicControlArticulationReader(dc, handle), path, None
    runtime_exposed_zero_dofs |= bool(handle)

    from isaacsim.core.prims import SingleArticulation

    for index, path in enumerate(
        dict.fromkeys(
            filter(
                None,
                [
                    *check.get("articulation_roots", []),
                    check.get("default_prim"),
                    *check.get("rigid_bodies", [])[:1],
                ],
            )
        )
    ):
        try:
            articulation = SingleArticulation(path, name=f"raw_eval_articulation_{index}")
            articulation.initialize()
            if articulation.handles_initialized and articulation.num_dof:
                articulation.backend = "single_articulation"
                return articulation, path, None
            runtime_exposed_zero_dofs |= bool(articulation.handles_initialized)
            errors.append(
                f"{path}: initialized handles={articulation.handles_initialized} num_dof={articulation.num_dof}"
            )
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
    handle, path = dynamic_articulation(dc, check)
    if handle and dc.get_articulation_dof_count(handle):
        errors.append("using dynamic_control read-only fallback because high-level articulation was unavailable")
        _LAST_ARTICULATION_ERRORS = errors
        print("modern_articulation fallback: " + " | ".join(errors), flush=True)
        return DynamicControlArticulationReader(dc, handle), path, None
    runtime_exposed_zero_dofs |= bool(handle)
    if errors:
        print("modern_articulation failed: " + " | ".join(errors), flush=True)
    _LAST_ARTICULATION_ERRORS = errors
    reason = (
        "articulation_handle_unavailable"
        if runtime_exposed_zero_dofs and check.get("limited_nonfixed_joint_count", 0) > 0
        else "no_limited_nonfixed_dof"
        if runtime_exposed_zero_dofs
        else "articulation_handle_unavailable"
    )
    return None, None, reason


def unavailable_articulation_result(reason: str) -> dict:
    result = {
        "applicable": reason != "no_limited_nonfixed_dof",
        "pass": False,
        "reason": reason,
        "articulation_errors": _LAST_ARTICULATION_ERRORS,
    }
    if reason == "articulation_handle_unavailable":
        result.update({"applicable": False, "status": "evaluation_blocked"})
    return result


def joint_pivot_world(stage, joint_prim):
    from pxr import Gf, UsdGeom, UsdPhysics

    joint = UsdPhysics.Joint(joint_prim)
    parents = joint.GetBody0Rel().GetTargets()
    local = joint_prim.GetAttribute("physics:localPos0").Get() or Gf.Vec3f(0, 0, 0)
    if not parents:
        return tuple(map(float, local))
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(parents[0]))
    return tuple(map(float, matrix.Transform(Gf.Vec3d(*local))))


def joint_force_geometry(stage, check: dict, dof_name: str):
    import numpy as np
    from pxr import UsdPhysics

    joint_prim, joint_type = matching_joint(stage, check, dof_name)
    if joint_prim is None or joint_type == "unknown":
        return None
    children = [str(path) for path in UsdPhysics.Joint(joint_prim).GetBody1Rel().GetTargets()]
    if not children:
        return None
    related = joint_descendants(stage, check["joints"], children)
    bounds = []
    for path in related:
        try:
            minimum, maximum = stage_bounds(stage, path)
            values = (*minimum, *maximum)
            if all(math.isfinite(value) and abs(value) < 1e6 for value in values):
                bounds.append((np.asarray(minimum), np.asarray(maximum)))
        except Exception:
            continue
    if not bounds:
        return None
    minimum = np.min([item[0] for item in bounds], axis=0)
    maximum = np.max([item[1] for item in bounds], axis=0)
    center = (minimum + maximum) * 0.5
    pivot = np.asarray(joint_pivot_world(stage, joint_prim))
    axis = joint_axis_world(stage, joint_prim)
    corners = [
        np.asarray((x, y, z))
        for x in (minimum[0], maximum[0])
        for y in (minimum[1], maximum[1])
        for z in (minimum[2], maximum[2])
    ]
    point = max(corners, key=lambda value: np.linalg.norm(np.cross(axis, value - pivot)))
    return {
        "prim": joint_prim,
        "type": joint_type,
        "child": children[0],
        "related": related,
        "axis": axis,
        "pivot": pivot,
        "point": point,
        "center": center,
    }


def joint_sweep(stage, check: dict, app) -> dict:
    import numpy as np
    from isaacsim.core.prims import RigidPrim
    from omni.isaac.dynamic_control import _dynamic_control

    if not check["joints"]:
        return {"applicable": False, "pass": False, "reason": "no_nonfixed_joint"}
    timeline, _minimum, _maximum, runtime_articulation = begin_physics(
        stage,
        check,
        app,
        settle_seconds=float(CONFIG["simulation"]["interaction_settle_seconds"]),
        return_articulation=True,
        pause_after_settle=True,
    )
    articulation, articulation_path, unavailable_reason = runtime_articulation
    if not articulation:
        timeline.stop()
        return unavailable_articulation_result(unavailable_reason)
    details = []
    settings = CONFIG["simulation"]
    dof_names = list(articulation.dof_names)
    settled_positions = np.asarray(articulation.get_joint_positions(), dtype=float).copy()
    for dof_offset, dof_name in enumerate(dof_names):
        if dof_offset and articulation.backend == "single_articulation":
            articulation.set_joint_positions(settled_positions)
            articulation.set_joint_velocities(np.zeros_like(settled_positions))
            for _ in range(3):
                app.update()
        index = list(articulation.dof_names).index(dof_name)
        props = articulation.dof_properties[index]
        lower = float(props["lower"])
        upper = float(props["upper"])
        authored_joint, authored_joint_type = matching_joint(stage, check, dof_name)
        joint_context = {
            "joint_path": str(authored_joint.GetPath()) if authored_joint else None,
            "joint_type": authored_joint_type,
            "dof_index": index,
            "lower_limit": lower,
            "upper_limit": upper,
            "failure_phase": "joint_geometry_or_control_initialization",
        }
        if not bool(props["hasLimits"]) or not all(map(math.isfinite, (lower, upper))):
            continue
        span = upper - lower
        if span <= 1e-8:
            continue
        geometry = joint_force_geometry(stage, check, dof_name)
        if geometry is None:
            details.append({"dof": dof_name, **joint_context, "child_body": None, "articulation_handle": articulation_path, "pass": False, "reason": "joint_child_geometry_unavailable", "status": "evaluation_blocked"})
            continue
        drive = joint_drive_metadata(geometry["prim"], geometry["type"])
        if drive["control_mode"] == "unsupported":
            details.append({"dof": dof_name, **joint_context, "child_body": geometry["child"], "articulation_handle": articulation_path, "pass": False, "status": "evaluation_blocked", "reason": "unsupported_authored_drive_protocol", "control_mode": "unsupported", "drive_schema": drive["drive_schema"], "drive_schema_applied": drive["drive_schema_applied"], "drive_metadata": drive["metadata"]})
            continue
        if articulation.backend == "single_articulation":
            child = RigidPrim(
                geometry["child"],
                name=f"raw_eval_joint_child_{index}",
                reset_xform_properties=False,
            )
            child.initialize()
        else:
            dc = _dynamic_control.acquire_dynamic_control_interface()
            child = dc.get_rigid_body(geometry["child"])
            if not child:
                details.append({"dof": dof_name, **joint_context, "child_body": geometry["child"], "articulation_handle": articulation_path, "pass": False, "reason": "joint_child_handle_unavailable", "status": "evaluation_blocked"})
                continue
        metric_bodies, runtime_masses = solved_metric_rigid_bodies(stage, check)
        moving_mass, moving_mass_source = joint_moving_mass(
            stage, geometry["related"], runtime_masses
        )
        if moving_mass is None:
            details.append({"dof": dof_name, **joint_context, "child_body": geometry["child"], "related_links": sorted(geometry["related"]), "articulation_handle": articulation_path, "pass": False, "reason": "joint_moving_mass_unavailable", "status": "evaluation_blocked", "control_mode": drive["control_mode"], "drive_metadata": drive["metadata"], "runtime_mass_paths": sorted(set(geometry["related"]) & runtime_masses.keys()), "authored_mass_paths": [path for path in sorted(geometry["related"]) if stage.GetPrimAtPath(path).IsValid() and stage.GetPrimAtPath(path).GetAttribute("physics:mass") and stage.GetPrimAtPath(path).GetAttribute("physics:mass").HasAuthoredValueOpinion()]})
            continue
        radius = max(0.01, float(np.linalg.norm(np.cross(geometry["axis"], geometry["point"] - geometry["pivot"]))))
        prismatic_force = min(float(settings["joint_force_max_n"]), float(settings["joint_force_mass_multiplier"]) * moving_mass * abs(float(settings["gravity"])))
        revolute_torque = min(float(settings["joint_torque_max_nm"]), float(settings["joint_torque_mass_radius_multiplier"]) * moving_mass * abs(float(settings["gravity"])) * radius)
        rigid_before = {path: world_translation(stage, path) for path in metric_bodies}
        root_path = metric_bodies[0]
        root_before = rigid_before[root_path]
        observed_positions = []
        observed_velocities = []
        observed_efforts = []
        observed_speed_scales = []
        reached = []
        segments = []
        unstable_reason = None
        target_speed = max(
            1e-6,
            span * float(settings["joint_completion_min"])
            / max(float(settings["joint_segment_seconds"]) * 0.8, float(settings["dt"])),
        )
        response_seconds = max(float(settings.get("joint_control_response_seconds", 0.1)), float(settings["dt"]))
        velocity_limit = float(settings["joint_velocity_explosion_mps"] if geometry["type"] == "translation" else settings["joint_velocity_explosion_radps"])
        soft_speed = max(2.0 * target_speed, span / response_seconds)
        original_targets = {"targetPosition": drive["metadata"]["targetPosition"], "targetVelocity": drive["metadata"]["targetVelocity"]}
        commanded_positions, commanded_velocities, terminal_reasons = [], [], []
        severe_event = None
        control_speed_overshoot = False
        estimated_demand = min(prismatic_force if geometry["type"] == "translation" else revolute_torque, float(drive["metadata"].get("maxForce") or math.inf))
        timeline.play()
        try:
            for target in (lower, upper, lower):
                segment_positions, terminal_reason = [], "segment_budget_exhausted"
                target_steps = max(1, int(float(settings["joint_segment_seconds"]) / float(settings["dt"])))
                segment_start = float(articulation.get_joint_positions()[index])
                for step in range(target_steps):
                    positions, velocities, efforts = articulation.get_joint_positions(), articulation.get_joint_velocities(), articulation.get_measured_joint_efforts()
                    state_pos, state_velocity, state_effort = float(positions[index]), float(velocities[index]), float(efforts[index])
                    observed_positions.append(state_pos); segment_positions.append(state_pos)
                    observed_velocities.append(abs(state_velocity)); observed_efforts.append(abs(state_effort))
                    if not all(map(math.isfinite, (state_pos, state_velocity, state_effort))):
                        unstable_reason, terminal_reason = "nonfinite_joint_state", "nonfinite_joint_state"
                        severe_event = {"dof": dof_name, "joint_path": str(geometry["prim"].GetPath()), "child_body": geometry["child"], "segment_index": len(terminal_reasons), "step": step, "time_s": step * float(settings["dt"]), "event": unstable_reason, "value": {"position": state_pos, "velocity": state_velocity, "effort": state_effort}, "threshold": None}
                        break
                    if abs(state_velocity) > velocity_limit:
                        unstable_reason, terminal_reason = "joint_velocity_explosion", "joint_velocity_explosion"
                        severe_event = {"dof": dof_name, "joint_path": str(geometry["prim"].GetPath()), "child_body": geometry["child"], "segment_index": len(terminal_reasons), "step": step, "time_s": step * float(settings["dt"]), "event": unstable_reason, "value": state_velocity, "threshold": velocity_limit}
                        break
                    if 1.0 - abs(state_pos - target) / span >= float(settings["joint_completion_min"]):
                        terminal_reason = "target_reached"
                        break
                    if abs(state_velocity) > soft_speed:
                        control_speed_overshoot, terminal_reason = True, "control_speed_overshoot"
                        if drive["control_mode"] != "passive_joint":
                            set_joint_drive_targets(drive, velocity=0.0)
                        break
                    sign = math.copysign(1.0, target - segment_start)
                    command, speed = joint_segment_command(
                        segment_start,
                        target,
                        step * float(settings["dt"]),
                        float(settings["joint_segment_seconds"]),
                        response_seconds,
                    )
                    commanded_velocities.append(sign * speed)
                    if drive["control_mode"] == "position_drive":
                        commanded_positions.append(command)
                        set_joint_drive_targets(drive, command, 0.0)
                    elif drive["control_mode"] == "velocity_drive":
                        set_joint_drive_targets(drive, velocity=sign * speed)
                    else:
                        velocity_error = sign * speed - state_velocity
                        response_force = moving_mass * velocity_error / response_seconds
                        effort = max(-prismatic_force, min(prismatic_force, response_force)) if geometry["type"] == "translation" else max(-revolute_torque / radius, min(revolute_torque / radius, response_force))
                        force = geometry["axis"] * effort if geometry["type"] == "translation" else np.cross(geometry["axis"], geometry["point"] - geometry["pivot"]) / radius * effort
                        point = geometry["center"] if geometry["type"] == "translation" else geometry["point"]
                        if not np.all(np.isfinite(force)) or not np.all(np.isfinite(point)):
                            unstable_reason, terminal_reason = "nonfinite_actuation_vector", "nonfinite_actuation_vector"
                            break
                        if articulation.backend == "single_articulation":
                            child.apply_forces_and_torques_at_pos(forces=np.asarray([force]), positions=np.asarray([point]), is_global=True)
                        else:
                            dc.apply_body_force(child, tuple(force), tuple(point), True)
                    app.update()
                if drive["control_mode"] != "passive_joint":
                    set_joint_drive_targets(drive, velocity=0.0)
                terminal_reasons.append(terminal_reason); segments.append(segment_positions)
                current = float(articulation.get_joint_positions()[index])
                reached.append(current if math.isfinite(current) else observed_positions[-1] if observed_positions else lower)
                if unstable_reason:
                    break
        finally:
            if drive["control_mode"] != "passive_joint":
                restore_joint_drive_targets(drive)
        while len(reached) < 3:
            reached.append(reached[-1] if reached else lower)
        upper_completion = max(0.0, 1.0 - abs(reached[1] - upper) / span)
        return_completion = max(0.0, 1.0 - abs(reached[2] - lower) / span)
        def monotonicity(values, sign):
            differences = [sign * (right - left) for left, right in zip(values, values[1:]) if abs(right - left) > 1e-9]
            return sum(value >= 0 for value in differences) / len(differences) if differences else 0.0
        forward_monotonicity = monotonicity(segments[1] if len(segments) > 1 else [], 1.0)
        return_monotonicity = monotonicity(segments[2] if len(segments) > 2 else [], -1.0)
        hysteresis = abs(reached[2] - lower) / span
        tolerance_floor = (
            float(settings["joint_limit_tolerance_min_m"])
            if geometry["type"] == "translation"
            else float(settings["joint_limit_tolerance_min_rad"])
        )
        tolerance = max(
            tolerance_floor,
            float(settings["joint_limit_tolerance_ratio"]) * span,
        )
        limit_violation = (
            unstable_reason is not None
            or
            min(observed_positions, default=lower) < lower - tolerance
            or max(observed_positions, default=upper) > upper + tolerance
        )
        child_targets = [geometry["child"]]
        joint_type = geometry["type"]
        related_links = geometry["related"]
        root_after = world_translation(stage, root_path)
        unrelated_motion = max(
            (
                distance(
                    tuple(rigid_before[path][axis] - root_before[axis] for axis in range(3)),
                    tuple(world_translation(stage, path)[axis] - root_after[axis] for axis in range(3)),
                )
                for path in metric_bodies
                if path != root_path and path not in related_links
            ),
            default=0.0,
        )
        velocity_threshold = float(settings["joint_velocity_explosion_radps"])
        if joint_type == "translation":
            velocity_threshold = float(settings["joint_velocity_explosion_mps"])
        detached = any(not stage.GetPrimAtPath(path).IsValid() for path in child_targets)
        passed = (
            upper_completion >= float(settings["joint_completion_min"])
            and return_completion >= float(settings["joint_completion_min"])
            and not limit_violation
            and not detached
            and unstable_reason is None
            and not control_speed_overshoot
            and max(observed_velocities, default=0.0) <= velocity_threshold
        )
        details.append(
            {
                "dof": dof_name,
                "joint_path": str(geometry["prim"].GetPath()),
                "dof_index": index,
                "lower_limit": lower,
                "upper_limit": upper,
                "child_body": geometry["child"],
                "articulation_handle": articulation_path,
                "joint_type": joint_type,
                "lower": lower,
                "upper": upper,
                "reached": reached,
                "upper_completion": upper_completion,
                "return_completion": return_completion,
                "forward_monotonicity": forward_monotonicity,
                "return_monotonicity": return_monotonicity,
                "normalized_hysteresis": hysteresis,
                "axis_pivot_valid": bool(np.all(np.isfinite(geometry["axis"])) and np.all(np.isfinite(geometry["pivot"])) and abs(float(np.linalg.norm(geometry["axis"])) - 1.0) <= 1e-3),
                "limit_violation": limit_violation,
                "detached_child": detached,
                "related_links": sorted(related_links),
                "moving_mass_kg": moving_mass,
                "moving_mass_source": moving_mass_source,
                "control_mode": drive["control_mode"],
                "drive_schema": drive["drive_schema"],
                "drive_schema_applied": drive["drive_schema_applied"],
                "drive_metadata": drive["metadata"],
                "original_drive_targets": original_targets,
                "commanded_target_position_range": [min(commanded_positions), max(commanded_positions)] if commanded_positions else None,
                "commanded_target_velocity_max": max(map(abs, commanded_velocities), default=0.0),
                "estimated_actuator_demand": estimated_demand,
                "authored_max_force": drive["metadata"].get("maxForce"),
                "control_speed_overshoot": control_speed_overshoot,
                "segment_terminal_reason": terminal_reasons,
                "first_severe_event": severe_event,
                "unrelated_link_motion_m": unrelated_motion,
                "max_velocity": max(observed_velocities, default=0.0),
                "max_effort": max(observed_efforts, default=0.0),
                "target_speed": target_speed,
                "minimum_speed_tracking_scale": min(observed_speed_scales, default=1.0),
                "commanded_force_n": prismatic_force if joint_type == "translation" else None,
                "commanded_torque_nm": revolute_torque if joint_type == "rotation" else None,
                "actuation": "authored_drive_targets" if drive["control_mode"] != "passive_joint" else "closed_loop_external_force_at_child_subtree",
                "articulation_reader": articulation.backend,
                "dof_initialization": "settled_snapshot_reset_between_dofs",
                "reason": unstable_reason or ("authored_drive_no_response" if drive["control_mode"] != "passive_joint" and max((abs(value - observed_positions[0]) for value in observed_positions), default=0.0) <= 1e-9 else None),
                "pass": passed,
            }
        )
    timeline.stop()
    if not details:
        return {"applicable": False, "pass": False, "reason": "no_limited_nonfixed_dof"}
    return {
        "applicable": True,
        "pass": all(item["pass"] for item in details),
        "articulation_path": articulation_path,
        "articulation_reader": articulation.backend,
        "joints": details,
        "thresholds": {
            "range_completion_min": settings["joint_completion_min"],
            "translation_limit_tolerance_min_m": settings["joint_limit_tolerance_min_m"],
            "rotation_limit_tolerance_min_rad": settings["joint_limit_tolerance_min_rad"],
            "unrelated_link_motion": "diagnostic_only; not a failure criterion",
        },
    }


def vector3(value) -> tuple[float, float, float]:
    return tuple(float(item) for item in value)


def duration_steps(seconds: float, dt: float) -> int:
    return max(1, int(math.ceil(float(seconds) / float(dt))))


def parallel_gripper_frame(authored_closing, authored_approach) -> dict:
    import numpy as np

    closing = np.asarray(authored_closing, dtype=float)
    closing /= max(1e-8, float(np.linalg.norm(closing)))
    approach = np.asarray(authored_approach, dtype=float)
    approach -= closing * float(np.dot(closing, approach))
    approach /= max(1e-8, float(np.linalg.norm(approach)))
    height = np.cross(closing, approach)
    height /= max(1e-8, float(np.linalg.norm(height)))
    return {
        "closing": closing,
        "approach": approach,
        "height": height,
        "rotation": np.column_stack((closing, approach, height)),
    }


def parallel_finger_centers(center, closing, approach, finger_depth, gap, inner_edge_inset):
    import numpy as np

    center = np.asarray(center, dtype=float)
    offset = np.asarray(closing, dtype=float) * (float(gap) * 0.5 + float(inner_edge_inset))
    return center - offset, center + offset


def _finger_projection_rectangles(descriptor: dict, side: str) -> list[dict]:
    """Return projected shell/pad rectangles for one finger, excluding base geometry."""
    import numpy as np

    inward = 1.0 if str(side) == "left" else -1.0
    rectangles = []
    for segment in descriptor["finger_segments"]:
        profile = [
            (inward * float(profile_x), float(profile_y))
            for profile_x, profile_y in segment["profile_vertices"]
        ]
        segment_height = float(segment["dimensions"][2])
        rectangles.append({
            "closing_min_m": min(point[0] for point in profile),
            "closing_max_m": max(point[0] for point in profile),
            "height_min_m": -segment_height * 0.5,
            "height_max_m": segment_height * 0.5,
            "role": "finger_shell",
        })

        pad_start = np.asarray(
            (
                inward * float(segment["inner_edge_start_inset"]),
                float(segment["longitudinal_start"]),
            ),
            dtype=float,
        )
        pad_end = np.asarray(
            (
                inward * float(segment["inner_edge_end_inset"]),
                float(segment["longitudinal_end"]),
            ),
            dtype=float,
        )
        tangent = pad_end - pad_start
        pad_length = float(np.linalg.norm(tangent))
        tangent /= max(pad_length, 1e-9)
        normal = np.asarray((tangent[1], -tangent[0]), dtype=float)
        if float(np.dot(normal, (inward, 0.0))) < 0.0:
            normal *= -1.0
        pad_thickness = float(segment["pad_thickness"])
        pad_center = (pad_start + pad_end) * 0.5 + normal * (pad_thickness * 0.5)
        pad_length += min(0.001, pad_thickness * 0.5)
        yaw = math.atan2(-float(tangent[0]), float(tangent[1]))
        half_x = (
            abs(math.cos(yaw)) * pad_thickness * 0.5
            + abs(math.sin(yaw)) * pad_length * 0.5
        )
        pad_x_min = float(pad_center[0]) - half_x
        pad_x_max = float(pad_center[0]) + half_x
        pad_half_height = segment_height * 0.45
        rectangles.append({
            "closing_min_m": pad_x_min,
            "closing_max_m": pad_x_max,
            "height_min_m": -pad_half_height,
            "height_max_m": pad_half_height,
            "role": "contact_pad",
        })
    return rectangles


def _finger_projection_bounds(descriptor: dict, side: str) -> dict:
    """Return one generated finger's closing/height bounds, excluding base geometry."""
    rectangles = _finger_projection_rectangles(descriptor, side)
    return {
        "closing_min_m": min(item["closing_min_m"] for item in rectangles),
        "closing_max_m": max(item["closing_max_m"] for item in rectangles),
        "height_min_m": min(item["height_min_m"] for item in rectangles),
        "height_max_m": max(item["height_max_m"] for item in rectangles),
    }


def _opening_projection_overlap(gap: float, object_bounds: dict, descriptor: dict, clearance: float) -> bool:
    """Check conservative 2-D collision projection overlap for both fingers at a symmetric gap."""
    inset = float(descriptor["max_inner_edge_inset"])
    object_x_min = float(object_bounds["closing_min_m"])
    object_x_max = float(object_bounds["closing_max_m"])
    object_z_min = float(object_bounds["height_min_m"])
    object_z_max = float(object_bounds["height_max_m"])
    for side in ("left", "right"):
        body_offset = (-1.0 if side == "left" else 1.0) * (
            float(gap) * 0.5 + inset
        )
        finger = _finger_projection_bounds(descriptor, side)
        finger_x_min = finger["closing_min_m"] + body_offset - float(clearance)
        finger_x_max = finger["closing_max_m"] + body_offset + float(clearance)
        finger_z_min = finger["height_min_m"] - float(clearance)
        finger_z_max = finger["height_max_m"] + float(clearance)
        height_overlap = min(finger_z_max, object_z_max) - max(
            finger_z_min, object_z_min
        )
        x_overlap = min(finger_x_max, object_x_max) - max(
            finger_x_min, object_x_min
        )
        if height_overlap > 0.0 and x_overlap > 0.0:
            return True
    return False


def _project_collision_points(collision_points, center, closing, approach) -> dict:
    import numpy as np

    points = np.asarray(collision_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("invalid_collision_projection_points")
    if not np.all(np.isfinite(points)):
        raise ValueError("nonfinite_collision_projection_points")
    center = np.asarray(center, dtype=float)
    closing = np.asarray(closing, dtype=float)
    closing /= max(1e-12, float(np.linalg.norm(closing)))
    approach = np.asarray(approach, dtype=float)
    approach -= closing * float(np.dot(closing, approach))
    approach /= max(1e-12, float(np.linalg.norm(approach)))
    height = np.cross(closing, approach)
    height /= max(1e-12, float(np.linalg.norm(height)))
    local = points - center
    closing_coordinates = local @ closing
    height_coordinates = local @ height
    return {
        "closing_min_m": float(np.min(closing_coordinates)),
        "closing_max_m": float(np.max(closing_coordinates)),
        "height_min_m": float(np.min(height_coordinates)),
        "height_max_m": float(np.max(height_coordinates)),
        "closing_axis_world": closing,
        "approach_axis_world": approach,
        "height_axis_world": height,
        "sample_count": int(len(points)),
    }


def grasp_opening_geometry(
    candidate_width,
    collision_minimum,
    collision_maximum,
    closing_axis,
    finger_thickness,
    pad_thickness,
    reference_max_opening=0.10,
    sample_count=5,
    *,
    approach_axis=None,
    grasp_center=None,
    collision_points=None,
    finger_descriptor=None,
    opening_clearance_m=0.001,
):
    """Compute a standard-gripper opening from collision geometry only."""
    import numpy as np

    minimum = np.asarray(collision_minimum, dtype=float)
    maximum = np.asarray(collision_maximum, dtype=float)
    axis = np.asarray(closing_axis, dtype=float)
    axis /= max(1e-12, float(np.linalg.norm(axis)))
    if minimum.shape != (3,) or maximum.shape != (3,) or np.any(maximum <= minimum):
        raise ValueError("invalid_collision_bounds_for_grasp_opening")
    corners = np.asarray(
        [
            [minimum[0] if bits & 1 else maximum[0],
             minimum[1] if bits & 2 else maximum[1],
             minimum[2] if bits & 4 else maximum[2]]
            for bits in range(8)
        ],
        dtype=float,
    )
    projections = corners @ axis
    swept_width = float(np.max(projections) - np.min(projections))
    candidate_width = None if candidate_width is None else float(candidate_width)
    finger_thickness = float(finger_thickness)
    pad_thickness = float(pad_thickness)
    reference_max_opening = float(reference_max_opening)
    if min(swept_width, finger_thickness, pad_thickness) < 0.0 or (
        candidate_width is not None and candidate_width < 0.0
    ):
        raise ValueError("negative_grasp_opening_geometry_input")

    if (
        collision_points is not None
        and approach_axis is not None
        and grasp_center is not None
        and finger_descriptor is not None
    ):
        projection = _project_collision_points(
            collision_points,
            grasp_center,
            axis,
            approach_axis,
        )
        clearance = max(0.0, float(opening_clearance_m))
        high = max(
            reference_max_opening,
            float(candidate_width or 0.0),
            2.0 * max(
                abs(projection["closing_min_m"]),
                abs(projection["closing_max_m"]),
            )
            + 2.0 * clearance
            + 2.0 * float(finger_descriptor["max_inner_edge_inset"])
            + 2.0 * finger_thickness,
        )
        if _opening_projection_overlap(high, projection, finger_descriptor, clearance):
            nonoverlap_gap = high
        else:
            low = 0.0
            for _ in range(48):
                middle = (low + high) * 0.5
                if _opening_projection_overlap(middle, projection, finger_descriptor, clearance):
                    low = middle
                else:
                    high = middle
            nonoverlap_gap = high
        # Projection separation alone omits the approach contact envelope.
        # Preserve the existing swept-width margin while honoring wider mesh bounds.
        swept_safe_gap = max(float(candidate_width or 0.0), swept_width) + 3.0 * finger_thickness
        requested = max(swept_safe_gap, nonoverlap_gap)
        applied = min(requested, reference_max_opening)
        projection_initial_overlap = _opening_projection_overlap(
            applied,
            projection,
            finger_descriptor,
            clearance,
        )
        return {
            "candidate_width_m": candidate_width,
            "collision_swept_width_m": projection["closing_max_m"] - projection["closing_min_m"],
            "pad_thickness_m": pad_thickness,
            "contact_clearance_m": 2.0 * clearance,
            "required_initial_gap_m": requested,
            "reference_max_opening_m": reference_max_opening,
            "initial_gap_requested_m": requested,
            "initial_gap_applied_m": applied,
            "opening_clamped": bool(applied < requested),
            "opening_supported": bool(requested <= reference_max_opening),
            "collision_swept_sample_count": projection["sample_count"],
            "collision_swept_axis_world": projection["closing_axis_world"].tolist(),
            "opening_projection_plane_normal_world": projection["approach_axis_world"].tolist(),
            "opening_projection_height_axis_world": projection["height_axis_world"].tolist(),
            "opening_object_projection_bounds_m": {
                "closing": [projection["closing_min_m"], projection["closing_max_m"]],
                "height": [projection["height_min_m"], projection["height_max_m"]],
            },
            "opening_nonoverlap_gap_m": nonoverlap_gap,
            "opening_swept_safe_gap_m": swept_safe_gap,
            "opening_initial_projection_overlap": projection_initial_overlap,
            "opening_base_excluded": True,
            "grasp_opening_source": "collision_mesh_projection_no_overlap",
        }

    requested = max(candidate_width if candidate_width is not None else 0.0, swept_width) + 3.0 * finger_thickness
    applied = min(requested, reference_max_opening)
    return {
        "candidate_width_m": candidate_width,
        "collision_swept_width_m": swept_width,
        "pad_thickness_m": pad_thickness,
        "contact_clearance_m": 3.0 * finger_thickness,
        "required_initial_gap_m": requested,
        "reference_max_opening_m": reference_max_opening,
        "initial_gap_requested_m": requested,
        "initial_gap_applied_m": applied,
        "opening_clamped": bool(applied < requested),
        "opening_supported": bool(requested <= reference_max_opening),
        "collision_swept_sample_count": max(1, int(sample_count)),
        "collision_swept_axis_world": axis.tolist(),
        "grasp_opening_source": "collision_swept_projection",
    }


def grasp_collision_union(collision_bounds: dict) -> dict:
    """Return the object-wide collision union used by authored grasp geometry."""
    links = collision_bounds.get("links", {}) if isinstance(collision_bounds, dict) else {}
    valid = {
        str(path): item for path, item in links.items()
        if item.get("minimum") is not None and item.get("maximum") is not None
    }
    if not valid:
        return {"minimum": None, "maximum": None, "links": [], "projection_links": {}}
    minimum = tuple(min(item["minimum"][axis] for item in valid.values()) for axis in range(3))
    maximum = tuple(max(item["maximum"][axis] for item in valid.values()) for axis in range(3))
    return {
        "minimum": minimum,
        "maximum": maximum,
        "links": sorted(valid),
        "projection_links": {
            path: {"minimum": tuple(item["minimum"]), "maximum": tuple(item["maximum"])}
            for path, item in sorted(valid.items())
        },
    }


def grasp_frame_to_palm_poses(center, closing, approach, finger_depth: float, palm_depth: float) -> dict:
    import numpy as np

    closing = np.asarray(closing, dtype=float)
    approach = np.asarray(approach, dtype=float)
    # Authored approach is the world-space motion direction from pregrasp to
    # grasp, and is also the gripper's positive depth/approach axis.
    gripper_approach = approach
    vertical = np.cross(closing, gripper_approach)
    grasp_pose = np.eye(4)
    grasp_pose[:3, :3] = np.column_stack((closing, gripper_approach, vertical))
    grasp_pose[:3, 3] = np.asarray(center, dtype=float)
    grasp_to_palm = np.eye(4)
    grasp_to_palm[1, 3] = -(float(finger_depth) + float(palm_depth)) * 0.5
    return {
        "evaluator_grasp_pose_world": grasp_pose,
        "grasp_frame_to_palm_transform": grasp_to_palm,
        "commanded_palm_pose_world": grasp_pose @ grasp_to_palm,
    }


def parallel_gripper_contact_contract(center, closing, approach, descriptor: dict, gap: float) -> dict:
    """Describe the generated fingers' inward pad planes in the authored grasp frame."""
    import numpy as np

    center = np.asarray(center, dtype=float)
    closing = np.asarray(closing, dtype=float)
    closing /= max(1e-8, float(np.linalg.norm(closing)))
    approach = np.asarray(approach, dtype=float)
    approach -= closing * float(np.dot(closing, approach))
    approach /= max(1e-8, float(np.linalg.norm(approach)))
    height = np.cross(closing, approach)
    depth = float(descriptor["finger_depth"])
    palm_depth = float(descriptor["palm_dimensions"][1])
    pad_thickness = max(
        (float(segment["pad_thickness"]) for segment in descriptor["finger_segments"]),
        default=0.0,
    )
    # The generated pads extend inward from the shell, so their object-facing
    # planes are separated by the requested jaw gap minus two pad thicknesses.
    half_gap = max(0.0, float(gap) * 0.5 - pad_thickness)
    return {
        "reference_point_world": center,
        "finger_center_world": center,
        "palm_origin_world": center - approach * ((depth + palm_depth) * 0.5),
        "left_pad_plane_center_world": center - closing * half_gap,
        "right_pad_plane_center_world": center + closing * half_gap,
        "closing_axis_world": closing,
        "approach_axis_world": approach,
        "height_axis_world": height,
        "pad_depth_center_local_m": 0.0,
        "pad_thickness_m": pad_thickness,
        "pad_depth_range_local_m": [-depth * 0.5, depth * 0.5],
        "pad_height_range_local_m": [-float(descriptor["finger_height"]) * 0.45, float(descriptor["finger_height"]) * 0.45],
        "finger_collision_depth_range_local_m": [-depth * 0.5, depth * 0.5],
        "finger_collision_height_range_local_m": [-float(descriptor["finger_height"]) * 0.5, float(descriptor["finger_height"]) * 0.5],
        "plastic_inner_plane_gap_m": max(0.0, float(gap)),
        "pad_surface_gap_m": max(0.0, float(gap) - 2.0 * pad_thickness),
        "source": "generated_gripper_geometry_descriptor",
    }


def grasp_pregrasp_center(center, approach, minimum, maximum, finger_depth: float, clearance: float):
    import numpy as np

    center = np.asarray(center, dtype=float)
    approach = np.asarray(approach, dtype=float)
    minimum = np.asarray(minimum, dtype=float)
    maximum = np.asarray(maximum, dtype=float)
    support = np.where(approach >= 0.0, minimum, maximum)
    outside_distance = float(np.dot(center - support, approach)) + float(clearance)
    distance = max(float(finger_depth), outside_distance, 0.0)
    return center - approach * distance, distance


def grasp_approach_collision(penetration, _object_motion, penetration_limit, _motion_limit) -> bool:
    return float(penetration) > float(penetration_limit)


def pose_arrival_error(commanded_position, commanded_orientation, actual_position, actual_orientation) -> dict:
    import numpy as np

    return {
        "position_error_m": float(
            np.linalg.norm(
                np.asarray(actual_position, dtype=float)
                - np.asarray(commanded_position, dtype=float)
            )
        ),
        "orientation_error_rad": float(
            np.linalg.norm(
                quaternion_rotation_error(actual_orientation, commanded_orientation)
            )
        ),
    }


def rigid_body_pose_dict(pose) -> dict:
    """Return a JSON-safe Dynamic Control pose snapshot for attempt diagnostics."""
    return {
        "position": [float(pose.p.x), float(pose.p.y), float(pose.p.z)],
        "orientation": [float(pose.r.x), float(pose.r.y), float(pose.r.z), float(pose.r.w)],
    }


def gripper_geometry_descriptor(
    geometry: str,
    finger_scale: float,
    asset_diagonal: float,
    closing_span_m: float = 0.0,
) -> dict:
    geometry = str(geometry).lower()
    if geometry not in {"flat", "wrap"}:
        raise ValueError(f"unsupported gripper geometry: {geometry}")
    diagonal = max(0.01, float(asset_diagonal))
    scale = max(0.1, float(finger_scale))
    # Keep the physical finger shell thickness fixed across the scale sweep.
    thickness = 0.005
    depth = scale * 0.08
    height = scale * min(0.06, max(0.02, diagonal * 0.15))
    pad_thickness = min(0.004, max(0.0015, thickness * 0.25))

    def segment(role, shape, start_y, end_y, start_inset, end_inset, start_width, end_width):
        profile = (
            (start_inset - start_width * 0.5, start_y),
            (start_inset + start_width * 0.5, start_y),
            (end_inset + end_width * 0.5, end_y),
            (end_inset - end_width * 0.5, end_y),
        )
        return {
            "profile_role": role,
            "shape": shape,
            "profile_vertices": profile,
            "longitudinal_start": start_y,
            "longitudinal_end": end_y,
            "longitudinal_offset": (start_y + end_y) * 0.5,
            "centerline_start_inset": start_inset,
            "centerline_end_inset": end_inset,
            "inner_edge_start_inset": start_inset + start_width * 0.5,
            "inner_edge_end_inset": end_inset + end_width * 0.5,
            "dimensions": (max(start_width, end_width), end_y - start_y, height),
            "pad_thickness": pad_thickness,
            "outer_material": "matte_black_plastic",
            "pad_material": "high_friction_black_rubber",
        }

    if geometry == "flat":
        segments = [
            segment(
                "straight",
                "rectangle",
                -depth * 0.5,
                depth * 0.5,
                0.0,
                0.0,
                thickness,
                thickness,
            )
        ]
    else:
        tip_y, lower_y, upper_y, root_y = (
            -depth * 0.5,
            -depth * 0.25,
            depth * 0.25,
            depth * 0.5,
        )
        bend = min(depth * 0.12, thickness * 1.25)
        root_width = thickness * 1.10
        middle_root_inset = -(root_width - thickness) * 0.5
        segments = [
            segment("tip", "rectangle", tip_y, lower_y, bend, 0.0, thickness, thickness),
            segment(
                "middle",
                "trapezoid",
                lower_y,
                upper_y,
                0.0,
                middle_root_inset,
                thickness,
                root_width,
            ),
            segment(
                "root",
                "rectangle",
                upper_y,
                root_y,
                middle_root_inset,
                bend + middle_root_inset,
                root_width,
                root_width,
            ),
        ]
    max_inner_edge_inset = max(
        max(item["inner_edge_start_inset"], item["inner_edge_end_inset"])
        for item in segments
    )
    # The palm and rail must cover the fingers at the largest supported jaw
    # opening, including the profile's outer inset and a small edge margin.
    safety_margin = 0.001
    closing_span_m = max(0.0, float(closing_span_m))
    base_closing_span = max(
        0.07,
        diagonal * 0.35,
        closing_span_m + 2.0 * max_inner_edge_inset + thickness + 2.0 * safety_margin,
    )
    rail_closing_span = max(0.08, diagonal * 0.40, base_closing_span)
    return {
        "geometry": geometry,
        "finger_scale": scale,
        "finger_thickness": thickness,
        "finger_depth": depth,
        "finger_height": height,
        "max_inner_edge_inset": max_inner_edge_inset,
        "palm_dimensions": (
            base_closing_span,
            min(0.06, max(0.025, diagonal * 0.12)),
            min(0.08, max(0.035, diagonal * 0.18)),
        ),
        "rail_dimensions": (
            rail_closing_span,
            min(0.025, max(0.012, diagonal * 0.05)),
            min(0.025, max(0.012, diagonal * 0.05)),
        ),
        "palm_material": "matte_black_plastic",
        "rail_material": "matte_black_plastic",
        "finger_segments": segments,
        "reference_model": "Franka Hand",
        "reference_max_opening_m": 0.20,
        "reference_finger_travel_m": 0.05,
        "reference_opening_enforced": True,
        "reference_parameters_source": "my_benchmark/franka_robolab.yml lock_joints",
        "closing_span_input_m": closing_span_m,
        "palm_closing_span_m": base_closing_span,
        "rail_closing_span_m": rail_closing_span,
    }


def common_link_grasp_candidates(link_bounds: dict, count: int) -> list[dict]:
    import numpy as np

    ranked = []
    for path, (minimum, maximum) in link_bounds.items():
        minimum = np.asarray(minimum, dtype=float)
        maximum = np.asarray(maximum, dtype=float)
        extent = maximum - minimum
        if not np.all(np.isfinite(extent)) or np.any(extent <= 0):
            continue
        ranked.append((float(np.prod(extent)), str(path), minimum, maximum, extent))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    if not ranked:
        return []
    output = []
    fractions = (0.5, 0.3, 0.7, 0.1, 0.9)
    for fraction in fractions:
        for _volume, path, minimum, maximum, extent in ranked:
            closing_axis = int(np.argmin(extent[:2]))
            approach_axis = 1 - closing_axis
            center = (minimum + maximum) * 0.5
            center[approach_axis] = minimum[approach_axis] + extent[approach_axis] * fraction
            closing = np.eye(3)[closing_axis]
            approach = -np.eye(3)[approach_axis]
            output.append(
                {
                    "center": center,
                    "closing": closing,
                    "approach": approach,
                    "width": float(extent[closing_axis]),
                    "target_rigid_body": path,
                    "same_rigid_body_required": False,
                    "source": "common_collision_link_bounds",
                    "annotation_diagnostics": {
                        "pose_valid": True,
                        "width_valid": True,
                        "fallback": "collision_bounds_by_rigid_link",
                    },
                }
            )
            if len(output) >= max(1, int(count)):
                return output
    return output


def common_grasp_geometry_score(attempts: list[dict]) -> dict:
    by_geometry = {
        geometry: max(
            (float(row.get("attempt_score", 0.0)) for row in attempts if row.get("applicable") and row.get("gripper_geometry") == geometry),
            default=0.0,
        )
        for geometry in ("flat", "wrap")
    }
    return {**by_geometry, "common": sum(by_geometry.values()) / len(by_geometry)}


def parallel_gripper_joint_contract(palm_center, finger_centers, closing, approach) -> dict:
    """Describe the shared prismatic-joint frame without touching USD state."""
    import numpy as np

    palm = np.asarray(palm_center, dtype=float)
    left, right = (np.asarray(value, dtype=float) for value in finger_centers)
    closing_axis = np.asarray(closing, dtype=float)
    closing_axis /= max(float(np.linalg.norm(closing_axis)), 1e-12)
    approach_axis = np.asarray(approach, dtype=float)
    approach_axis -= closing_axis * float(np.dot(closing_axis, approach_axis))
    approach_axis /= max(float(np.linalg.norm(approach_axis)), 1e-12)
    height_axis = np.cross(closing_axis, approach_axis)
    height_axis /= max(float(np.linalg.norm(height_axis)), 1e-12)
    basis = np.column_stack((closing_axis, approach_axis, height_axis))
    anchors = {
        "left": basis.T @ (left - palm),
        "right": basis.T @ (right - palm),
    }
    return {
        "basis": basis,
        "joint_axis_local": np.asarray((1.0, 0.0, 0.0)),
        "joint_axis_world": basis[:, 0],
        "local_anchor": anchors,
        "anchor_world": {"left": left, "right": right},
        "body1_local_anchor": np.zeros(3),
    }


def define_parallel_gripper(stage, candidate: dict, descriptor: dict, initial_gap: float, final_gap: float) -> dict:
    import numpy as np
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade

    stage.SetEditTarget(stage.GetSessionLayer())
    labels = {
        "plastic": "matte_black_plastic",
        "rubber": "high_friction_black_rubber",
    }

    def material(path, color, roughness, static_friction, dynamic_friction):
        value = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        value.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        physics = UsdPhysics.MaterialAPI.Apply(value.GetPrim())
        physics.CreateStaticFrictionAttr(float(static_friction))
        physics.CreateDynamicFrictionAttr(float(dynamic_friction))
        physics.CreateRestitutionAttr(0.0)
        return value

    plastic = material(
        "/__raw_eval/GripperPlasticMaterial",
        (0.025, 0.025, 0.025),
        0.86,
        0.45,
        0.40,
    )
    rubber_friction = float(CONFIG["simulation"]["grasp_gripper_static_friction"])
    rubber = material(
        "/__raw_eval/GripperRubberMaterial",
        (0.008, 0.008, 0.008),
        0.96,
        rubber_friction,
        rubber_friction,
    )

    orientation = tuple(float(value) for value in candidate["orientation"])
    quaternion = Gf.Quatf(orientation[3], orientation[0], orientation[1], orientation[2])
    center = np.asarray(candidate["center"], dtype=float)
    closing = np.asarray(candidate["closing"], dtype=float)
    approach = np.asarray(candidate["approach"], dtype=float)
    vertical = np.cross(closing, approach)
    depth = float(descriptor["finger_depth"])
    thickness = float(descriptor["finger_thickness"])
    palm_dimensions = np.asarray(descriptor["palm_dimensions"], dtype=float)
    rail_dimensions = np.asarray(descriptor["rail_dimensions"], dtype=float)
    finger_center = center
    palm_center = center - approach * ((depth + palm_dimensions[1]) * 0.5)
    finger_offset = closing * (
        float(initial_gap) * 0.5 + float(descriptor["max_inner_edge_inset"])
    )
    starts = (finger_center - finger_offset, finger_center + finger_offset)
    contact_contract = parallel_gripper_contact_contract(
        center,
        closing,
        approach,
        descriptor,
        initial_gap,
    )

    root = UsdGeom.Xform.Define(stage, "/__raw_eval/Gripper")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    def body(path, position, mass):
        value = UsdGeom.Xform.Define(stage, path)
        value.AddTranslateOp().Set(Gf.Vec3d(*map(float, position)))
        value.AddOrientOp().Set(quaternion)
        rigid = UsdPhysics.RigidBodyAPI.Apply(value.GetPrim())
        rigid.CreateKinematicEnabledAttr(False)
        UsdPhysics.MassAPI.Apply(value.GetPrim()).CreateMassAttr(float(mass))
        PhysxSchema.PhysxContactReportAPI.Apply(value.GetPrim()).CreateThresholdAttr(0.0)
        UsdPhysics.FilteredPairsAPI.Apply(value.GetPrim()).CreateFilteredPairsRel().AddTarget(
            Sdf.Path("/__raw_eval/Ground")
        )
        return value

    def cube(parent, name, dimensions, offset, yaw_degrees, visual_material, collision):
        value = UsdGeom.Cube.Define(stage, f"{parent}/{name}")
        value.CreateSizeAttr(1.0)
        value.AddTranslateOp().Set(Gf.Vec3d(*map(float, offset)))
        if yaw_degrees:
            value.AddRotateZOp().Set(float(yaw_degrees))
        value.AddScaleOp().Set(Gf.Vec3f(*map(float, dimensions)))
        UsdShade.MaterialBindingAPI.Apply(value.GetPrim()).Bind(visual_material)
        if collision:
            UsdPhysics.CollisionAPI.Apply(value.GetPrim())
            collision_api = PhysxSchema.PhysxCollisionAPI.Apply(value.GetPrim())
            collision_api.CreateContactOffsetAttr(0.001)
            collision_api.CreateRestOffsetAttr(0.0)
        return value

    def extruded_profile(parent, name, profile_vertices, height, visual_material, collision):
        profile = [(float(profile_x), float(profile_y)) for profile_x, profile_y in profile_vertices]
        signed_area = sum(
            profile[index][0] * profile[(index + 1) % len(profile)][1]
            - profile[(index + 1) % len(profile)][0] * profile[index][1]
            for index in range(len(profile))
        )
        if signed_area < 0.0:
            profile.reverse()
        half_height = float(height) * 0.5
        points = [Gf.Vec3f(x, y, -half_height) for x, y in profile]
        points.extend(Gf.Vec3f(x, y, half_height) for x, y in profile)
        count = len(profile)
        face_counts = [count, count] + [4] * count
        face_indices = list(reversed(range(count))) + list(range(count, count * 2))
        for index in range(count):
            following = (index + 1) % count
            face_indices.extend((index, following, following + count, index + count))
        value = UsdGeom.Mesh.Define(stage, f"{parent}/{name}")
        value.CreatePointsAttr(points)
        value.CreateFaceVertexCountsAttr(face_counts)
        value.CreateFaceVertexIndicesAttr(face_indices)
        value.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        UsdShade.MaterialBindingAPI.Apply(value.GetPrim()).Bind(visual_material)
        if collision:
            UsdPhysics.CollisionAPI.Apply(value.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(value.GetPrim()).CreateApproximationAttr("convexHull")
            collision_api = PhysxSchema.PhysxCollisionAPI.Apply(value.GetPrim())
            collision_api.CreateContactOffsetAttr(0.001)
            collision_api.CreateRestOffsetAttr(0.0)
        return value

    palm_path = "/__raw_eval/GripperPalm"
    left_path = "/__raw_eval/GripperLeft"
    right_path = "/__raw_eval/GripperRight"
    body(palm_path, palm_center, 1.5)
    body(left_path, starts[0], 0.25)
    body(right_path, starts[1], 0.25)
    cube(palm_path, "Palm", palm_dimensions, (0, 0, 0), 0, plastic, True)
    cube(
        palm_path,
        "Rail",
        rail_dimensions,
        (0, -palm_dimensions[1] * 0.5, 0),
        0,
        plastic,
        True,
    )

    for path, side in ((left_path, -1.0), (right_path, 1.0)):
        inward = -side
        for index, segment in enumerate(descriptor["finger_segments"]):
            segment_dimensions = np.asarray(segment["dimensions"], dtype=float)
            pad_thickness = float(segment["pad_thickness"])
            profile = [
                (inward * float(profile_x), float(profile_y))
                for profile_x, profile_y in segment["profile_vertices"]
            ]
            shell = extruded_profile(
                path,
                f"PlasticShell_{index}",
                profile,
                segment_dimensions[2],
                plastic,
                True,
            )
            shell.GetPrim().SetCustomDataByKey("raw_eval:material_role", labels["plastic"])
            pad_start = np.asarray(
                (
                    inward * float(segment["inner_edge_start_inset"]),
                    float(segment["longitudinal_start"]),
                ),
                dtype=float,
            )
            pad_end = np.asarray(
                (
                    inward * float(segment["inner_edge_end_inset"]),
                    float(segment["longitudinal_end"]),
                ),
                dtype=float,
            )
            tangent = pad_end - pad_start
            pad_length = float(np.linalg.norm(tangent))
            tangent /= max(pad_length, 1e-9)
            normal = np.asarray((tangent[1], -tangent[0]), dtype=float)
            if float(np.dot(normal, (inward, 0.0))) < 0.0:
                normal *= -1.0
            pad_center = (pad_start + pad_end) * 0.5 + normal * (pad_thickness * 0.5)
            pad_offset = (float(pad_center[0]), float(pad_center[1]), 0.0)
            yaw = math.degrees(math.atan2(-float(tangent[0]), float(tangent[1])))
            pad = cube(
                path,
                f"RubberPad_{index}",
                (
                    pad_thickness,
                    pad_length + min(0.001, pad_thickness * 0.5),
                    segment_dimensions[2] * 0.9,
                ),
                pad_offset,
                yaw,
                rubber,
                True,
            )
            UsdShade.MaterialBindingAPI.Apply(pad.GetPrim()).Bind(
                rubber,
                UsdShade.Tokens.strongerThanDescendants,
                "physics",
            )
            pad.GetPrim().SetCustomDataByKey("raw_eval:material_role", labels["rubber"])

    travel = max(0.0, (float(initial_gap) - float(final_gap)) * 0.5)
    joint_contract = parallel_gripper_joint_contract(
        palm_center,
        starts,
        closing,
        approach,
    )
    joint_paths = {}
    for name, path, start, lower, upper in (
        ("Left", left_path, starts[0], 0.0, travel),
        ("Right", right_path, starts[1], -travel, 0.0),
    ):
        joint = UsdPhysics.PrismaticJoint.Define(stage, f"/__raw_eval/Gripper{name}Joint")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(palm_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(path)])
        joint.CreateAxisAttr("X")
        joint.CreateLowerLimitAttr(float(lower))
        joint.CreateUpperLimitAttr(float(upper))
        joint.CreateCollisionEnabledAttr(False)
        local_anchor = joint_contract["local_anchor"][name.lower()]
        joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, local_anchor)))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateTargetPositionAttr(0.0)
        drive.CreateTargetVelocityAttr(0.0)
        drive.CreateStiffnessAttr(5_000.0)
        drive.CreateDampingAttr(100.0)
        drive.CreateMaxForceAttr(2.5)
        joint_paths[name.lower()] = str(joint.GetPath())

    return {
        "root_path": str(root.GetPath()),
        "palm_path": palm_path,
        "left_path": left_path,
        "right_path": right_path,
        "starts": starts,
        "palm_start": palm_center,
        "finger_dimensions": (thickness, depth, float(descriptor["finger_height"])),
        "mechanical_travel_m": travel,
        "contact_contract": {
            key: value.tolist() if hasattr(value, "tolist") else value
            for key, value in contact_contract.items()
        },
        "joint_paths": joint_paths,
        "joint_drive_max_force_n": 2.5,
        "joint_contract": {
            "joint_local_basis": joint_contract["basis"].tolist(),
            "joint_axis_local": joint_contract["joint_axis_local"].tolist(),
            "joint_axis_world": joint_contract["joint_axis_world"].tolist(),
            "left_joint_anchor_world": joint_contract["anchor_world"]["left"].tolist(),
            "right_joint_anchor_world": joint_contract["anchor_world"]["right"].tolist(),
            "left_joint_local_anchor": joint_contract["local_anchor"]["left"].tolist(),
            "right_joint_local_anchor": joint_contract["local_anchor"]["right"].tolist(),
        },
        "joint_axis_world": joint_contract["joint_axis_world"].tolist(),
        "initial_finger_center_delta_world": (starts[1] - starts[0]).tolist(),
        "palm_to_finger_vector_world": (finger_center - palm_center).tolist(),
        "geometry": descriptor["geometry"],
        "materials": {
            "palm": labels["plastic"],
            "rail": labels["plastic"],
            "finger_outer": labels["plastic"],
            "contact_pad": labels["rubber"],
        },
        "contact_region_static_friction": {"rubber": rubber_friction, "plastic": 0.45},
    }


def parallel_gripper_joint_targets(initial_gap: float, gap: float) -> tuple[float, float]:
    """Return the left/right prismatic targets for a symmetric jaw gap."""
    travel = max(0.0, (float(initial_gap) - float(gap)) * 0.5)
    return travel, -travel


def next_gripper_gap(current_gap: float, final_gap: float, closing_speed: float, dt: float, speed_scale: float) -> float:
    """Advance a jaw gap without reopening after a unilateral contact slowdown."""
    return max(
        float(final_gap),
        float(current_gap) - 2.0 * float(closing_speed) * float(dt) * float(speed_scale),
    )


def set_parallel_gripper_targets(stage, gripper: dict, targets: dict, max_force: float) -> None:
    """Set independent prismatic targets for the two dynamic fingers."""
    from pxr import UsdPhysics

    for name in ("left", "right"):
        joint = stage.GetPrimAtPath(gripper["joint_paths"][name])
        drive = UsdPhysics.DriveAPI.Get(joint, "linear")
        drive.GetTargetPositionAttr().Set(float(targets[name]))
        drive.GetMaxForceAttr().Set(max(0.0, float(max_force)))


def set_parallel_gripper_gap(stage, gripper: dict, initial_gap: float, gap: float, max_force: float) -> None:
    from pxr import UsdPhysics

    left_target, right_target = parallel_gripper_joint_targets(initial_gap, gap)
    for name, target in (("left", left_target), ("right", right_target)):
        joint = stage.GetPrimAtPath(gripper["joint_paths"][name])
        drive = UsdPhysics.DriveAPI.Get(joint, "linear")
        drive.GetTargetPositionAttr().Set(float(target))
        drive.GetMaxForceAttr().Set(max(0.0, float(max_force)))


def parallel_gripper_drive_targets(stage, gripper: dict) -> dict:
    """Read the current mirrored prismatic targets without changing force limits."""
    from pxr import UsdPhysics

    return {
        name: float(UsdPhysics.DriveAPI.Get(
            stage.GetPrimAtPath(gripper["joint_paths"][name]), "linear"
        ).GetTargetPositionAttr().Get() or 0.0)
        for name in ("left", "right")
    }


def set_parallel_gripper_drive_targets(stage, gripper: dict, targets: dict) -> None:
    """Hold existing prismatic targets during lift without reissuing a close command."""
    from pxr import UsdPhysics

    for name in ("left", "right"):
        UsdPhysics.DriveAPI.Get(
            stage.GetPrimAtPath(gripper["joint_paths"][name]), "linear"
        ).GetTargetPositionAttr().Set(float(targets[name]))


def grasp_vibration_statistics(samples: list[dict]) -> dict:
    """Summarize per-step lift diagnostics; these values never affect pass criteria."""
    import numpy as np

    if not samples:
        return {
            "object_lift_velocity_rms_mps": 0.0,
            "object_lift_acceleration_rms_mps2": 0.0,
            "object_lift_vertical_peak_to_peak_m": 0.0,
            "object_lateral_peak_to_peak_m": 0.0,
            "object_gripper_relative_vertical_peak_to_peak_m": 0.0,
            "object_gripper_relative_lateral_peak_to_peak_m": 0.0,
            "contact_force_peak_to_peak_n": 0.0,
        }
    object_positions = np.asarray([sample["actual_object_position_world"] for sample in samples], dtype=float)
    object_velocities = np.asarray([sample["object_velocity_world"] for sample in samples], dtype=float)
    relative = np.asarray([sample["object_relative_to_gripper_world"] for sample in samples], dtype=float)
    forces = np.asarray([sample["left_normal_force_n"] + sample["right_normal_force_n"] for sample in samples], dtype=float)
    dt = np.asarray([float(sample["dt_s"]) for sample in samples], dtype=float)
    acceleration = np.diff(object_velocities, axis=0) / np.maximum(dt[1:, None], 1e-12)
    lateral = np.linalg.norm(object_positions[:, :2] - object_positions[0, :2], axis=1)
    relative_lateral = np.linalg.norm(relative[:, :2] - relative[0, :2], axis=1)
    return {
        "object_lift_velocity_rms_mps": float(np.sqrt(np.mean(object_velocities[:, 2] ** 2))),
        "object_lift_acceleration_rms_mps2": float(np.sqrt(np.mean(acceleration[:, 2] ** 2))) if len(acceleration) else 0.0,
        "object_lift_vertical_peak_to_peak_m": float(np.ptp(object_positions[:, 2])),
        "object_lateral_peak_to_peak_m": float(np.ptp(lateral)),
        "object_gripper_relative_vertical_peak_to_peak_m": float(np.ptp(relative[:, 2])),
        "object_gripper_relative_lateral_peak_to_peak_m": float(np.ptp(relative_lateral)),
        "contact_force_peak_to_peak_n": float(np.ptp(forces)),
    }


def pusher_geometry_descriptor(size, direction) -> dict:
    dimensions = (size, size, size) if isinstance(size, (int, float)) else tuple(map(float, size))
    direction = tuple(map(float, direction))
    axis = max(range(3), key=lambda index: abs(direction[index]))
    if abs(direction[axis]) < 1e-9:
        raise ValueError("pusher direction must be non-zero")
    transverse = [dimensions[index] for index in range(3) if index != axis]
    rod_length = min(0.20, max(0.04, dimensions[axis] * 3.0))
    rod_radius = min(0.025, max(0.004, min(transverse) * 0.12))
    rod_offset = [0.0, 0.0, 0.0]
    rod_offset[axis] = math.copysign(
        dimensions[axis] * 0.5 + rod_length * 0.5,
        direction[axis],
    )
    return {
        "plate_dimensions": dimensions,
        "rod_axis": ("X", "Y", "Z")[axis],
        "rod_length": rod_length,
        "rod_radius": rod_radius,
        "rod_offset": tuple(rod_offset),
        "rod_collision": False,
        "material": "brushed_metal",
    }


def define_pusher(stage, center, size, direction, kinematic=False):
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade

    stage.SetEditTarget(stage.GetSessionLayer())
    descriptor = pusher_geometry_descriptor(size, direction)
    root = UsdGeom.Xform.Define(stage, "/__raw_eval/Pusher")
    root.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
    body_api = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    body_api.CreateKinematicEnabledAttr(bool(kinematic))
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(
        float(CONFIG["simulation"]["push_pusher_mass_kg"])
    )
    PhysxSchema.PhysxContactReportAPI.Apply(root.GetPrim()).CreateThresholdAttr(0.0)

    material = UsdShade.Material.Define(stage, "/__raw_eval/PusherMetalMaterial")
    shader = UsdShade.Shader.Define(stage, "/__raw_eval/PusherMetalMaterial/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.42, 0.45, 0.48))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.28)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.90)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    plate = UsdGeom.Cube.Define(stage, "/__raw_eval/Pusher/Plate")
    plate.CreateSizeAttr(1.0)
    plate.AddScaleOp().Set(Gf.Vec3f(*descriptor["plate_dimensions"]))
    UsdShade.MaterialBindingAPI.Apply(plate.GetPrim()).Bind(material)
    UsdPhysics.CollisionAPI.Apply(plate.GetPrim())
    collision_api = PhysxSchema.PhysxCollisionAPI.Apply(plate.GetPrim())
    collision_api.CreateContactOffsetAttr(0.001)
    collision_api.CreateRestOffsetAttr(0.0)

    rod = UsdGeom.Cylinder.Define(stage, "/__raw_eval/Pusher/Rod")
    rod.CreateAxisAttr(descriptor["rod_axis"])
    rod.CreateHeightAttr(float(descriptor["rod_length"]))
    rod.CreateRadiusAttr(float(descriptor["rod_radius"]))
    rod.AddTranslateOp().Set(Gf.Vec3d(*descriptor["rod_offset"]))
    UsdShade.MaterialBindingAPI.Apply(rod.GetPrim()).Bind(material)
    root.GetPrim().SetCustomDataByKey("raw_eval:material_role", descriptor["material"])
    root.GetPrim().SetCustomDataByKey("raw_eval:rod_collision", descriptor["rod_collision"])
    return str(root.GetPath())


def pusher_drive_command(settings: dict, pusher_mass_kg: float, object_mass_kg: float, speed_mps: float, elapsed_seconds: float) -> dict:
    """Return a bounded world-force magnitude for the dynamic Push pusher."""
    target_speed = float(settings["push_pusher_speed_mps"])
    ramp = max(1e-6, float(settings["push_force_ramp_seconds"]))
    response = max(1e-6, float(settings["push_speed_response_seconds"]))
    maximum_acceleration = float(settings["push_max_acceleration_mps2"])
    force_cap = min(
        float(settings["push_force_max_n"]),
        max(
            float(settings["push_force_min_n"]),
            float(settings["push_force_mass_multiplier"])
            * max(0.0, float(object_mass_kg))
            * abs(float(settings["gravity"])),
        ),
    )
    desired_speed = target_speed * min(1.0, max(0.0, float(elapsed_seconds)) / ramp)
    requested_acceleration = max(
        -maximum_acceleration,
        min(maximum_acceleration, (desired_speed - float(speed_mps)) / response),
    )
    requested_force = max(0.0, float(pusher_mass_kg)) * requested_acceleration
    applied_force = max(-force_cap, min(force_cap, requested_force))
    return {
        "desired_speed_mps": desired_speed,
        "measured_speed_mps": float(speed_mps),
        "requested_force_n": requested_force,
        "applied_force_n": applied_force,
        "force_cap_n": force_cap,
    }


def pusher_drive_force_vector(settings: dict, pusher_mass_kg: float, object_mass_kg: float, velocity, forward, elapsed_seconds: float) -> dict:
    """Track forward speed while damping every other pusher velocity component."""
    forward_values = tuple(float(value) for value in forward)
    velocity_values = tuple(float(value) for value in velocity)
    forward_speed = sum(value * axis for value, axis in zip(velocity_values, forward_values))
    command = pusher_drive_command(settings, pusher_mass_kg, object_mass_kg, forward_speed, elapsed_seconds)
    response = max(1e-6, float(settings["push_speed_response_seconds"]))
    requested = tuple(
        forward_values[index] * command["requested_force_n"]
        - float(pusher_mass_kg) * (velocity_values[index] - forward_values[index] * forward_speed) / response
        for index in range(3)
    )
    magnitude = math.sqrt(sum(value * value for value in requested))
    scale = min(1.0, command["force_cap_n"] / magnitude) if magnitude else 1.0
    applied = tuple(value * scale for value in requested)
    return {
        **command,
        "requested_force_vector_n": requested,
        "applied_force_vector_n": applied,
        "applied_force_magnitude_n": math.sqrt(sum(value * value for value in applied)),
    }


def define_kinematic_box(stage, path: str, center, dimensions, kinematic: bool = True, orientation=None):
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade

    stage.SetEditTarget(stage.GetSessionLayer())
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*center))
    if orientation is not None:
        x, y, z, w = map(float, orientation)
        cube.AddOrientOp().Set(Gf.Quatf(w, x, y, z))
    cube.AddScaleOp().Set(Gf.Vec3f(*dimensions))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    collision_api = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
    collision_api.CreateContactOffsetAttr(0.001)
    collision_api.CreateRestOffsetAttr(0.0)
    UsdPhysics.FilteredPairsAPI.Apply(cube.GetPrim()).CreateFilteredPairsRel().AddTarget(
        Sdf.Path("/__raw_eval/Ground")
    )
    body = UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    body.CreateKinematicEnabledAttr(kinematic)
    if not kinematic:
        UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(1.0)
    material = UsdShade.Material.Define(stage, "/__raw_eval/GripperPhysicsMaterial")
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr(2.0)
    material_api.CreateDynamicFrictionAttr(2.0)
    material_api.CreateRestitutionAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
        material,
        UsdShade.Tokens.strongerThanDescendants,
        "physics",
    )
    PhysxSchema.PhysxContactReportAPI.Apply(cube.GetPrim()).CreateThresholdAttr(0.0)
    return str(cube.GetPath())


def create_contact_sensor(parent_path: str, name: str):
    from isaacsim.sensors.physics import ContactSensor

    sensor = ContactSensor(
        prim_path=f"{parent_path}/{name}",
        name=name,
        frequency=1.0 / float(CONFIG["simulation"]["dt"]),
        min_threshold=0.0,
        max_threshold=1e9,
        radius=-1,
    )
    sensor.add_raw_contact_data_to_frame()
    return sensor


def contact_frame(sensor, target_paths: list[str] | None = None, allow_summary_fallback: bool = False) -> dict:
    frame = sensor.get_current_frame() or {}
    count = int(frame.get("number_of_contacts", 0))
    force = float(frame.get("force", frame.get("value", 0.0)) or 0.0)
    penetrations = []
    for contact in frame.get("contacts", []) or []:
        for key in ("distance", "separation"):
            value = contact.get(key) if isinstance(contact, dict) else getattr(contact, key, None)
            if value is not None and math.isfinite(float(value)) and float(value) < 0:
                penetrations.append(-float(value))
    contacts = []
    for contact in frame.get("contacts", []) or []:
        contacts.append(
            {
                key: (
                    value.tolist()
                    if hasattr(value, "tolist")
                    else list(value)
                    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes))
                    else value
                )
                for key, value in contact.items()
            }
        )
    summary_contact = bool(frame.get("in_contact", False) or count > 0 or force > 0)
    target_contact = None
    if target_paths is not None:
        target_contact = any(
            any(
                body == target or body.startswith(f"{target}/")
                for target in target_paths
                for body in (str(contact.get("body0", "")), str(contact.get("body1", "")))
            )
            for contact in contacts
        )
    contact = summary_contact if target_contact is None else target_contact
    contact_source = "sensor_summary" if target_contact is None else "target_raw"
    if allow_summary_fallback and target_contact is not None and not contacts:
        contact = summary_contact
        contact_source = "sensor_summary_fallback"
    return {
        "contact": contact,
        "contact_source": contact_source,
        "count": count,
        "force": force,
        "penetration": max(penetrations, default=None),
        "contacts": contacts,
    }


def dynamic_contact_penetration(frame: dict) -> float:
    """Return penetration reported by PhysX without querying live USD bounds."""
    try:
        value = float(frame.get("penetration") or 0.0)
    except (AttributeError, TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) and value > 0.0 else 0.0


def settle_support_verdict(contact_report_support, geometry_support, final_lowest_z, ground_z, penetration_max_m) -> dict:
    below_ground = final_lowest_z is not None and float(final_lowest_z) < float(ground_z) - float(penetration_max_m)
    return {
        "support_exists": bool((contact_report_support or geometry_support) and not below_ground),
        "post_contact_collision_below_ground": below_ground,
        "post_contact_ground_clearance_m": None if final_lowest_z is None else float(final_lowest_z) - float(ground_z),
    }


def timeline_elapsed_seconds(timeline, start_seconds: float, fallback_seconds: float) -> float:
    try:
        elapsed = float(timeline.get_current_time()) - float(start_seconds)
    except Exception:
        return float(fallback_seconds)
    return elapsed if math.isfinite(elapsed) and elapsed >= 0.0 else float(fallback_seconds)


def first_physx_link_contacts(stage, check, app):
    import omni.timeline
    from pxr import UsdPhysics

    add_session_physics(stage, float(CONFIG["simulation"].get("ground_z_m", 0.0)))
    attach_stage(stage, app)
    sensors = []
    for index, path in enumerate(check.get("rigid_bodies", [])):
        try:
            sensors.append((path, create_contact_sensor(path, f"InitialOverlapSensor_{index}")))
        except Exception:
            continue
    app.update()
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_time_codes_per_second(1.0 / float(CONFIG["simulation"]["dt"]))
    timeline.play()
    for _ in range(3):
        app.update()
    for _path, sensor in sensors:
        sensor.initialize()
    app.update()
    rigid = check.get("rigid_bodies", [])
    adjacent = set()
    for joint_path in check.get("joints", []):
        joint = UsdPhysics.Joint(stage.GetPrimAtPath(joint_path))
        for parent in joint.GetBody0Rel().GetTargets():
            for child in joint.GetBody1Rel().GetTargets():
                adjacent.add(frozenset((str(parent), str(child))))
    contacts = []
    for sensor_path, sensor in sensors:
        for contact in contact_frame(sensor)["contacts"]:
            body0, body1 = str(contact.get("body0", "")), str(contact.get("body1", ""))
            if not all(any(body == path or body.startswith(f"{path}/") for path in rigid) for body in (body0, body1)):
                continue
            owner0 = next(path for path in rigid if body0 == path or body0.startswith(f"{path}/"))
            owner1 = next(path for path in rigid if body1 == path or body1.startswith(f"{path}/"))
            if owner0 == owner1:
                continue
            penetration = None
            for key in ("distance", "separation"):
                value = contact.get(key)
                if value is not None and math.isfinite(float(value)) and float(value) < 0:
                    penetration = -float(value)
            contacts.append({"sensor_body": sensor_path, "body0": body0, "body1": body1, "owner0": owner0, "owner1": owner1, "adjacent_joint_links": frozenset((owner0, owner1)) in adjacent, "penetration_m": penetration})
    timeline.stop()
    maximum = max((row["penetration_m"] or 0.0 for row in contacts), default=0.0)
    nonadjacent_maximum = max((row["penetration_m"] or 0.0 for row in contacts if not row["adjacent_joint_links"]), default=0.0)
    return {"contact_count": len(contacts), "maximum_penetration_m": maximum, "nonadjacent_maximum_penetration_m": nonadjacent_maximum, "contacts": contacts, "severe": nonadjacent_maximum > float(CONFIG["simulation"]["penetration_max_m"])}


def aabb_overlap_depth(a_min, a_max, b_min, b_max) -> float:
    overlap = [min(a_max[i], b_max[i]) - max(a_min[i], b_min[i]) for i in range(3)]
    return max(0.0, min(overlap)) if all(value > 0 for value in overlap) else 0.0


def pusher_sweep_evidence(
    positions,
    dimensions,
    target_collision_paths,
    *,
    start_position=None,
    commanded_end_position=None,
    measured_end_position=None,
    contact_report_observed=None,
):
    """Collect bounded scene-query evidence without changing Push contact semantics."""
    endpoints = {
        "pusher_start_position": None if start_position is None else [float(value) for value in start_position],
        "pusher_commanded_end_position": None if commanded_end_position is None else [float(value) for value in commanded_end_position],
        "pusher_measured_end_position": None if measured_end_position is None else [float(value) for value in measured_end_position],
        "contact_report_observed": contact_report_observed,
    }
    try:
        import carb
        from omni.physx import get_physx_scene_query_interface

        hits = set()
        colliders = set()
        samples = []

        def on_hit(hit):
            body = str(getattr(hit, "rigid_body", ""))
            if body:
                hits.add(body)
            collider = str(getattr(hit, "collision", ""))
            if collider:
                colliders.add(collider)
            return True

        query = get_physx_scene_query_interface()
        half = tuple(float(value) * 0.5 for value in dimensions)
        for position in positions:
            result = query.overlap_box(
                carb.Float3(*half),
                carb.Float3(*(float(value) for value in position)),
                carb.Float4(0.0, 0.0, 0.0, 1.0),
                on_hit,
                False,
            )
            samples.append({"position": [float(value) for value in position], "query_hit": bool(result)})
        return {
            "status": "ok", "sample_count": len(samples), "samples": samples,
            "hit_rigid_bodies": sorted(hits), "hit_collider_paths": sorted(colliders),
            "target_collider_paths": sorted(target_collision_paths), **endpoints,
        }
    except Exception as exc:
        return {
            "status": "unavailable", "error": f"{type(exc).__name__}: {exc}",
            "target_collider_paths": sorted(target_collision_paths), **endpoints,
        }


def push_contact(stage, check: dict, app) -> dict:
    import numpy as np
    import omni.timeline
    import omni.usd
    from omni.isaac.dynamic_control import _dynamic_control

    settings = CONFIG["simulation"]
    directions = [
        np.asarray((1.0, 0.0, 0.0)),
        np.asarray((-1.0, 0.0, 0.0)),
        np.asarray((0.0, 1.0, 0.0)),
        np.asarray((0.0, -1.0, 0.0)),
    ][: int(settings["push_samples"])]
    source_path = Path(stage.GetRootLayer().realPath or stage.GetRootLayer().identifier)
    dc = _dynamic_control.acquire_dynamic_control_interface()
    samples = []
    for sample_index, direction in enumerate(directions):
        if sample_index:
            omni.timeline.get_timeline_interface().stop()
            stage = open_stage(source_path, app)
            check = precheck(stage)
        bounds_path = check["default_prim"] or check["rigid_bodies"][0]
        timeline, _minimum, _maximum = begin_physics(
            stage,
            check,
            app,
            settle_seconds=float(settings["interaction_settle_seconds"]),
        )
        timeline.stop()
        minimum, maximum = stage_bounds(stage, bounds_path)
        diagonal = float(np.linalg.norm(np.asarray(maximum) - np.asarray(minimum)))
        thickness = min(
            float(settings["push_pusher_size_max_m"]),
            max(
                float(settings["push_pusher_size_min_m"]),
                float(settings["push_pusher_size_diagonal_ratio"]) * diagonal,
            ),
        )
        extent = np.asarray(maximum) - np.asarray(minimum)
        axis = int(np.argmax(np.abs(direction)))
        dimensions = extent * 0.8
        dimensions[axis] = thickness
        dimensions = np.clip(dimensions, float(settings["push_pusher_size_min_m"]), 0.5)
        center = (np.asarray(minimum) + np.asarray(maximum)) * 0.5
        surface = center.copy()
        surface[axis] = maximum[axis] if direction[axis] > 0 else minimum[axis]
        start = surface + direction * (2.5 * thickness)
        guided_protocol = settings.get("push_drive_protocol", "legacy_velocity") == "guided_kinematic_velocity"
        pusher_path = define_pusher(stage, start, dimensions, direction, kinematic=guided_protocol)
        sensor = create_contact_sensor(pusher_path, f"ContactSensor_{sample_index}")
        app.update()
        timeline.play()
        for _ in range(3):
            app.update()
        sensor.initialize()
        pusher = dc.get_rigid_body(pusher_path)
        root = dc.get_rigid_body(check["rigid_bodies"][0])
        if not pusher or not root:
            timeline.stop()
            return {"applicable": True, "pass": False, "reason": "rigid_body_handle_unavailable"}
        dc.set_rigid_body_disable_gravity(pusher, True)
        dc.set_rigid_body_pose(
            pusher,
            _dynamic_control.Transform(tuple(start), (0.0, 0.0, 0.0, 1.0)),
        )
        dc.wake_up_rigid_body(pusher)
        pusher_view = None
        if guided_protocol:
            import omni.physics.tensors as tensors

            simulation_view = tensors.create_simulation_view("numpy")
            pusher_view = simulation_view.create_rigid_body_view(pusher_path)
            pusher_indices = np.asarray([0], dtype=np.uint32)
        metric_bodies, solved_masses = solved_metric_rigid_bodies(stage, check)
        target_collision_paths = [row["path"] for row in mesh_records(stage) if row["collision"] and row["rigid_body"] in metric_bodies]
        object_mass = sum(solved_masses.get(path, 0.0) for path in metric_bodies)
        pusher_mass = float(settings["push_pusher_mass_kg"])
        root_before = dc.get_rigid_body_pose(root).p
        root_rotation_before = world_quaternion(stage, check["rigid_bodies"][0])
        max_aabb_penetration = 0.0
        max_contact_penetration = None
        max_contact_force = 0.0
        contact_steps = 0
        active_steps = 0
        maximum_pusher_speed = 0.0
        maximum_measured_pusher_speed = 0.0
        maximum_applied_force = 0.0
        last_drive = None
        first_severe = None

        def drive_pusher(elapsed_seconds):
            nonlocal maximum_applied_force, last_drive
            velocity = dc.get_rigid_body_linear_velocity(pusher)
            forward = -direction
            if guided_protocol:
                # Kinematic guide: collision response cannot accelerate the pusher.
                distance_m = max(0.0, float(elapsed_seconds)) * float(settings["push_pusher_speed_mps"])
                pose = np.asarray(start, dtype=float) + forward * distance_m
                pusher_view.set_kinematic_targets(
                    np.asarray([[*pose, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
                    pusher_indices,
                )
                last_drive = {
                    "protocol": "guided_kinematic_velocity",
                    "desired_speed_mps": float(settings["push_pusher_speed_mps"]),
                    "commanded_distance_m": distance_m,
                }
                return
            if settings.get("push_drive_protocol", "legacy_velocity") == "legacy_velocity":
                dc.set_rigid_body_linear_velocity(pusher, tuple(forward * float(settings["push_pusher_speed_mps"])))
                last_drive = {"protocol": "legacy_velocity", "desired_speed_mps": float(settings["push_pusher_speed_mps"])}
                return
            drive = pusher_drive_force_vector(
                settings,
                pusher_mass,
                object_mass,
                (velocity.x, velocity.y, velocity.z),
                forward,
                elapsed_seconds,
            )
            pose = dc.get_rigid_body_pose(pusher).p
            apply_world_body_force(dc, pusher, drive["applied_force_vector_n"], (pose.x, pose.y, pose.z))
            dc.wake_up_rigid_body(pusher)
            maximum_applied_force = max(maximum_applied_force, drive["applied_force_magnitude_n"])
            last_drive = drive

        def observe(_step, _steps):
            nonlocal active_steps, contact_steps, max_aabb_penetration
            nonlocal max_contact_force, max_contact_penetration, maximum_pusher_speed
            nonlocal maximum_measured_pusher_speed, first_severe
            active_steps += 1
            pose = dc.get_rigid_body_pose(pusher).p
            pusher_center = np.asarray((pose.x, pose.y, pose.z))
            half = dimensions * 0.5
            max_aabb_penetration = max(
                max_aabb_penetration,
                aabb_overlap_depth(
                    pusher_center - half,
                    pusher_center + half,
                    np.asarray(minimum),
                    np.asarray(maximum),
                ),
            )
            velocity = dc.get_rigid_body_linear_velocity(pusher)
            maximum_pusher_speed = max(
                maximum_pusher_speed,
                float(np.linalg.norm((velocity.x, velocity.y, velocity.z))),
            )
            maximum_measured_pusher_speed = max(
                maximum_measured_pusher_speed,
                float(np.linalg.norm((velocity.x, velocity.y, velocity.z))),
            )
            if guided_protocol:
                maximum_pusher_speed = max(
                    maximum_pusher_speed,
                    float(settings["push_pusher_speed_mps"]),
                )
            speed = float(np.linalg.norm((velocity.x, velocity.y, velocity.z)))
            frame = contact_frame(sensor, check["rigid_bodies"])
            contact_penetration = dynamic_contact_penetration(frame)
            if not math.isfinite(speed) or contact_penetration > float(settings["push_penetration_max_m"]) * 4.0:
                first_severe = first_severe or {
                    "phase": "push",
                    "time_s": (_step + 1) * float(settings["dt"]),
                    "body": check["rigid_bodies"][0],
                    "event": "push_runtime_unstable",
                    "value": contact_penetration if math.isfinite(speed) else None,
                    "threshold": float(settings["push_penetration_max_m"]) * 4.0,
                    "evidence_source": "physx_contact_penetration" if math.isfinite(speed) else "nonfinite_pusher_speed",
                    "severe": True,
                    "pusher_position": pusher_center.tolist(),
                    "direction": (-direction).tolist(),
                }
            if frame["contact"]:
                contact_steps += 1
            max_contact_force = max(max_contact_force, frame["force"])
            if frame["penetration"] is not None:
                max_contact_penetration = max(
                    max_contact_penetration or 0.0,
                    frame["penetration"],
                )

        acquisition_steps = max(1, int(float(settings["push_acquire_timeout_seconds"]) / float(settings["dt"])))
        first_contact_step = None
        drive_start_time = float(timeline.get_current_time())
        first_contact_time = None
        for acquisition_step in range(acquisition_steps):
            elapsed = float(timeline.get_current_time()) - drive_start_time
            if elapsed >= float(settings["push_acquire_timeout_seconds"]):
                break
            drive_pusher(elapsed)
            app.update()
            observe(acquisition_step, acquisition_steps)
            if first_severe:
                break
            if contact_steps:
                first_contact_step = acquisition_step
                first_contact_time = float(timeline.get_current_time()) - drive_start_time
                break
        contact_steps = 0
        active_steps = 0
        if first_contact_step is not None:
            push_steps = max(1, int(float(settings["push_contact_seconds"]) / float(settings["dt"])))
            for push_step in range(push_steps):
                elapsed = float(timeline.get_current_time()) - drive_start_time
                if elapsed - first_contact_time >= float(settings["push_contact_seconds"]):
                    break
                drive_pusher(elapsed)
                app.update()
                observe(push_step, push_steps)
                if first_severe:
                    break
        root_after = dc.get_rigid_body_pose(root).p
        displacement = distance(
            (root_before.x, root_before.y, root_before.z),
            (root_after.x, root_after.y, root_after.z),
        )
        contact_fraction = contact_steps / active_steps if active_steps else 0.0
        contact = first_contact_step is not None
        driven_pusher_pose = dc.get_rigid_body_pose(pusher).p
        if not guided_protocol:
            dc.set_rigid_body_linear_velocity(pusher, (0.0, 0.0, 0.0))
        retracted = start + direction * (5 * thickness)
        if guided_protocol:
            # Retraction must clear the old target, or the next step moves back through the asset.
            retracted_pose = np.asarray([[*retracted, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
            pusher_view.set_transforms(retracted_pose, pusher_indices)
            pusher_view.set_kinematic_targets(retracted_pose, pusher_indices)
        else:
            dc.set_rigid_body_pose(
                pusher,
                _dynamic_control.Transform(tuple(retracted), (0.0, 0.0, 0.0, 1.0)),
            )
        release_speed = 0.0
        for body_path in metric_bodies:
            body = dc.get_rigid_body(body_path)
            if body:
                velocity = dc.get_rigid_body_linear_velocity(body)
                release_speed = max(release_speed, float(np.linalg.norm((velocity.x, velocity.y, velocity.z))))
        commanded_end = start + (-direction) * (
            float((last_drive or {}).get("commanded_distance_m", float(settings["push_pusher_speed_mps"]) * elapsed))
        )
        sweep_positions = [start + (-direction) * (float(settings["push_pusher_speed_mps"]) * float(settings["push_acquire_timeout_seconds"]) * fraction) for fraction in np.linspace(0.0, 1.0, 8)]
        sweep_positions.append(np.asarray((driven_pusher_pose.x, driven_pusher_pose.y, driven_pusher_pose.z)))
        sweep_evidence = pusher_sweep_evidence(
            sweep_positions,
            dimensions,
            target_collision_paths,
            start_position=start,
            commanded_end_position=commanded_end,
            measured_end_position=(driven_pusher_pose.x, driven_pusher_pose.y, driven_pusher_pose.z),
            contact_report_observed=contact,
        )
        sweep_hits_target = bool(
            set(sweep_evidence.get("hit_rigid_bodies", [])) & set(metric_bodies)
            or set(sweep_evidence.get("hit_collider_paths", [])) & set(target_collision_paths)
        )
        contact_reason = None if contact else (
            "contact_sensor_missed" if sweep_hits_target else "no_target_sweep_intersection"
        )
        try:
            recovery_sensor = create_contact_sensor(
                "/__raw_eval/Ground", "PushRecoveryContactSensor"
            )
            app.update()
            recovery_sensor.initialize()
        except Exception:
            recovery_sensor = None
        recovery = ({
            "pass": False,
            "severe_runtime_event": True,
            "runtime_events": [first_severe],
            "reason": "push_runtime_unstable",
        } if first_severe else observe_stability(
            stage,
            check,
            app,
            float(settings["push_recovery_seconds"]),
            recovery_sensor,
        ))
        post_recovery_speed = float(recovery.get("final_linear_speed_mps", math.inf))
        penetration_pass = (
            max_contact_penetration is None
            or max_contact_penetration < float(settings["push_penetration_max_m"])
        )
        root_rotation_after = world_quaternion(stage, check["rigid_bodies"][0])
        rotation = angular_distance(root_rotation_before, root_rotation_after)
        motion_bounded = math.isfinite(displacement) and math.isfinite(rotation) and not recovery.get("severe_runtime_event", False)
        passed = (
            contact
            and penetration_pass
            and motion_bounded
            and post_recovery_speed < float(settings["push_residual_speed_max_mps"])
            and recovery.get("pass", False)
        )
        samples.append(
            {
                "surface_point": surface.tolist(),
                "direction": (-direction).tolist(),
                "probe_speed_mps": float(settings["push_pusher_speed_mps"]),
                "pusher_controller": {
                    "protocol": settings["push_drive_protocol"],
                    "object_mass_kg": object_mass,
                    "pusher_mass_kg": pusher_mass,
                    "maximum_applied_force_n": maximum_applied_force,
                    "maximum_pusher_speed_mps": maximum_pusher_speed,
                    "maximum_measured_pusher_speed_mps": maximum_measured_pusher_speed,
                    "motion_source": "physx_kinematic_target" if guided_protocol else "dynamic_velocity_or_force",
                    "last_drive": last_drive,
                },
                "contact_acquisition_time_s": first_contact_time,
                "contact_acquisition_timeout_s": float(settings["push_acquire_timeout_seconds"]),
                "sustained_push_seconds": float(settings["push_contact_seconds"]),
                "contact": contact,
                "contact_reason": contact_reason,
                "sweep_hits_target": sweep_hits_target,
                "contact_fraction": contact_fraction,
                "maximum_contact_force_n": max_contact_force,
                "object_displacement_m": displacement,
                "object_rotation_rad": rotation,
                "motion_bounded": motion_bounded,
                "maximum_contact_penetration_m": max_contact_penetration,
                "maximum_aabb_overlap_m_diagnostic_only": max_aabb_penetration,
                "aabb_overlap_basis": "settled_initial_object_bounds",
                "release_speed_mps": release_speed,
                "post_recovery_speed_mps": post_recovery_speed,
                "residual_speed_mps": post_recovery_speed,
                "sweep_evidence": sweep_evidence,
                "recovery": recovery,
                "pass": passed,
                "first_severe_event": first_severe,
            }
        )
        timeline.stop()
    return {
        "applicable": True,
        "pass": sum(item["pass"] for item in samples) >= 3,
        "samples": samples,
        "penetration_scope": "physx_contact_report_only; initial_aabb_overlap_diagnostic_only",
        "parameters_source": "paper_success_rule_plus_reproduction_design_bbox_face_pusher",
    }


def native_grasp_candidates(stage, asset_prim_path: str) -> list[dict]:
    import numpy as np
    from pxr import Gf, UsdGeom
    from manual_grasp import CONVENTION, column_matrix, local_axes_from_matrix

    output = []
    grasps_path = f"{str(asset_prim_path).rstrip('/')}/grasps/"
    cache = UsdGeom.XformCache()
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if not prim_path.startswith(grasps_path):
            continue
        pose_attr = prim.GetAttribute("grasp:pose_matrix")
        pose = pose_attr.Get() if pose_attr and pose_attr.HasAuthoredValueOpinion() else None
        translate_attr = prim.GetAttribute("xformOp:translate")
        orient_attr = prim.GetAttribute("xformOp:orient")
        scale_attr = prim.GetAttribute("xformOp:scale")
        has_xform_pose = (
            translate_attr and translate_attr.HasAuthoredValueOpinion()
            and orient_attr and orient_attr.HasAuthoredValueOpinion()
        )
        frame_attr = prim.GetAttribute("grasp:frame_convention")
        manual = bool(frame_attr and frame_attr.HasAuthoredValueOpinion()
                      and str(frame_attr.Get()) == CONVENTION)
        if pose is None and not has_xform_pose and not manual:
            continue
        source = "grasp_attributes" if pose is not None else "authored_xform"
        if source == "grasp_attributes":
            matrix = column_matrix(pose)
            if manual:
                xform_world = cache.GetLocalToWorldTransform(prim)
                world_center = np.asarray(tuple(xform_world.Transform(Gf.Vec3d(0.0, 0.0, 0.0))))
                world_rotation = np.asarray(xform_world, dtype=float).T[:3, :3]
            else:
                world_center = np.asarray(asset_point_world(stage, asset_prim_path, matrix[:3, 3]))
                world_rotation = np.column_stack([
                    asset_direction_world(stage, asset_prim_path, matrix[:3, axis])
                    for axis in range(3)
                ])
            local_transform = matrix
        else:
            local_transform = np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation(), dtype=float)
            world_transform = cache.GetLocalToWorldTransform(prim)
            matrix = np.asarray(world_transform, dtype=float)
            world_center = np.asarray(tuple(world_transform.Transform(Gf.Vec3d(0.0, 0.0, 0.0))), dtype=float)
            world_rotation = np.column_stack([
                np.asarray(tuple(world_transform.TransformDir(Gf.Vec3d(*axis))), dtype=float)
                for axis in np.eye(3)
            ])
            if manual:
                matrix = column_matrix(world_transform * cache.GetLocalToWorldTransform(stage.GetPrimAtPath(asset_prim_path)).GetInverse())
                local_transform = matrix
        closing_attr = prim.GetAttribute("grasp:finger_closing")
        approach_attr = prim.GetAttribute("grasp:approach")
        width_attr = prim.GetAttribute("grasp:width")
        closing_attr_authored = bool(closing_attr and closing_attr.HasAuthoredValueOpinion())
        approach_attr_authored = bool(approach_attr and approach_attr.HasAuthoredValueOpinion())
        if manual:
            closing, approach = local_axes_from_matrix(matrix)
            relative_xform = column_matrix(cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(stage.GetPrimAtPath(asset_prim_path)).GetInverse())
            if not np.allclose(relative_xform, matrix, atol=1e-6, rtol=0):
                raise GraspBindingError(f"manual_grasp_pose_mismatch:{prim_path}")
            if closing_attr_authored and not np.allclose(np.asarray(closing_attr.Get(), dtype=float), closing, atol=1e-6):
                raise GraspBindingError(f"manual_grasp_axis_mismatch:{prim_path}:closing")
            if approach_attr_authored and not np.allclose(np.asarray(approach_attr.Get(), dtype=float), approach, atol=1e-6):
                raise GraspBindingError(f"manual_grasp_axis_mismatch:{prim_path}:approach")
        else:
            closing = closing_attr.Get() if closing_attr_authored else matrix[:3, 0]
            approach = approach_attr.Get() if approach_attr_authored else matrix[:3, 2]
        width = width_attr.Get() if width_attr and width_attr.HasAuthoredValueOpinion() else None
        component_attr = prim.GetAttribute("grasp:component")
        component = component_attr.Get() if component_attr and component_attr.HasAuthoredValueOpinion() else None
        rotation = np.column_stack([
            world_rotation[:, axis] / max(1e-12, float(np.linalg.norm(world_rotation[:, axis])))
            for axis in range(3)
        ])
        orthogonality_error = float(np.linalg.norm(rotation.T @ rotation - np.eye(3)))
        determinant = float(np.linalg.det(rotation))
        authored_pose_world = np.eye(4)
        authored_pose_world[:3, :3] = world_rotation
        authored_pose_world[:3, 3] = world_center
        # Read the asset-authored directions, then express them in world space
        # through the authored grasp pose.  The resulting approach axis is the
        # gripper center axis; closing is the bilateral finger axis.
        authored_closing_local = np.asarray(closing, dtype=float)
        authored_approach_local = np.asarray(approach, dtype=float)
        # The generated grasp pose uses its second frame axis for the
        # bilateral finger direction.  Preserve the authored attribute for
        # diagnostics, but bind the runtime closing axis to that frame axis.
        authored_closing_world = (
            asset_direction_world(stage, asset_prim_path, authored_closing_local)
            if manual else rotation[:, 1]
        )
        authored_approach_world = (
            asset_direction_world(stage, asset_prim_path, authored_approach_local)
            if manual or approach_attr_authored else rotation[:, 2]
        )
        authored_closing_world /= max(1e-12, float(np.linalg.norm(authored_closing_world)))
        authored_approach_world /= max(1e-12, float(np.linalg.norm(authored_approach_world)))
        gripper_frame = parallel_gripper_frame(authored_closing_world, authored_approach_world)
        scale = scale_attr.Get() if scale_attr and scale_attr.HasAuthoredValueOpinion() else (1.0, 1.0, 1.0)
        unit_scale = bool(np.allclose(np.asarray(scale, dtype=float), np.ones(3), atol=1e-6))
        output.append({
            "prim_path": prim_path,
            "center": world_center,
            "closing": gripper_frame["closing"],
            "approach": gripper_frame["approach"],
            "rotation": gripper_frame["rotation"],
            "authored_closing_world": authored_closing_world,
            "authored_approach_world": authored_approach_world,
            "authored_closing_local": authored_closing_local,
            "authored_approach_local": authored_approach_local,
            "authored_attribute_frame": "asset_local_manual_xz" if manual else ("grasp_local_pose" if (closing_attr_authored or approach_attr_authored) else "pose_columns"),
            "authored_pose_world": authored_pose_world,
            "width": float(width) if width is not None else None,
            "component": str(component) if component is not None else None,
            "orthogonality_error": orthogonality_error,
            "rotation_determinant": determinant,
            "pose_valid": bool(np.all(np.isfinite(matrix)) and unit_scale and orthogonality_error <= 1e-3 and determinant > 0),
            "width_valid": width is None or bool(math.isfinite(float(width)) and float(width) > 0),
            "grasp_pose_source": source,
            "authored_grasp_prim": prim_path,
            "authored_grasp_local_transform": local_transform,
            "authored_grasp_world_transform": authored_pose_world,
            "authored_grasp_axis_contract": CONVENTION if manual else "pose_frame_axis_1_to_world_finger_closing;authored_world_approach_to_world_center_axis",
            "authored_grasp_pose_valid": bool(np.all(np.isfinite(matrix)) and unit_scale and orthogonality_error <= 1e-3 and determinant > 0),
        })
    return output


def quaternion_xyzw(rotation) -> tuple[float, float, float, float]:
    from scipy.spatial.transform import Rotation

    return tuple(float(value) for value in Rotation.from_matrix(rotation).as_quat())


def asset_point_world(stage, prim_path: str, point):
    from pxr import Gf, UsdGeom

    value = UsdGeom.XformCache().GetLocalToWorldTransform(
        stage.GetPrimAtPath(prim_path)
    ).Transform(Gf.Vec3d(*map(float, point)))
    return tuple(map(float, value))


def asset_direction_world(stage, prim_path: str, direction):
    import numpy as np
    from pxr import Gf, UsdGeom

    value = UsdGeom.XformCache().GetLocalToWorldTransform(
        stage.GetPrimAtPath(prim_path)
    ).TransformDir(Gf.Vec3d(*map(float, direction)))
    result = np.asarray(tuple(value), dtype=float)
    norm = float(np.linalg.norm(result))
    return result / norm if norm > 1e-8 else result


def grasp_candidate(
    dataset: str,
    asset: dict,
    minimum,
    maximum,
    stage=None,
    asset_prim_path: str | None = None,
) -> dict | None:
    import numpy as np

    grasps = native_grasp_candidates(stage, asset_prim_path) if stage is not None and asset_prim_path else []
    rank = int(asset.get("_grasp_candidate_rank", 0))
    if grasps and rank < len(grasps):
        ordered = sorted(grasps, key=lambda item: item["prim_path"])
        grasp = ordered[rank]
        closing = grasp["closing"]
        approach = grasp["approach"]
        width = grasp["width"]
        source = "authored_usd_grasp"
        target_hint = grasp["component"]
        center = grasp["center"]
        rotation = grasp["rotation"]
        authored_pose_world = grasp["authored_pose_world"]
        annotation_diagnostics = {key: grasp[key] for key in ("prim_path", "component", "orthogonality_error", "rotation_determinant", "pose_valid", "width_valid", "authored_closing_local", "authored_approach_local", "authored_closing_world", "authored_approach_world", "authored_attribute_frame", "grasp_pose_source", "authored_grasp_prim", "authored_grasp_local_transform", "authored_grasp_world_transform", "authored_grasp_axis_contract", "authored_grasp_pose_valid")}
    else:
        return None
    closing_norm = float(np.linalg.norm(closing))
    if closing_norm < 1e-8 or (width is not None and width <= 0):
        return None
    closing = closing / closing_norm
    approach = approach - float(np.dot(approach, closing)) * closing
    approach_norm = float(np.linalg.norm(approach))
    if approach_norm < 1e-8:
        return None
    approach = approach / approach_norm
    orientation = quaternion_xyzw(
        np.column_stack((closing, approach, np.cross(closing, approach)))
    )
    return {
        "center": np.asarray(center, dtype=float),
        "closing": closing,
        "approach": approach,
        "width": width,
        "orientation": orientation,
        "source": source,
        "target_hint": target_hint,
        "authored_pose_world": authored_pose_world,
        "grasp_frame_convention": (
            "native_authored_x=closing,native_authored_z=approach;"
            "evaluator_x=closing,evaluator_y=approach"
            if source == "native_usd_grasp"
            else "x=closing,y=approach,z=closing_cross_approach"
        ),
        "annotation_diagnostics": annotation_diagnostics,
    }


def rigid_collision_link_bounds(stage, check: dict) -> dict:
    bounds = object_link_collision_bounds(stage, check, mesh_records(stage))
    return {
        path: (item["minimum"], item["maximum"])
        for path, item in bounds["links"].items()
        if all(math.isfinite(value) and abs(float(value)) < 1e6 for value in (*item["minimum"], *item["maximum"]))
        and all(item["maximum"][axis] >= item["minimum"][axis] for axis in range(3))
    }


def common_stage_grasp_candidate(stage, check: dict, rank: int, count: int) -> dict | None:
    import numpy as np

    candidates = common_link_grasp_candidates(
        rigid_collision_link_bounds(stage, check),
        count,
    )
    if rank < 0 or rank >= len(candidates):
        return None
    candidate = candidates[rank]
    closing = np.asarray(candidate["closing"], dtype=float)
    approach = np.asarray(candidate["approach"], dtype=float)
    candidate["orientation"] = quaternion_xyzw(
        np.column_stack((closing, approach, np.cross(closing, approach)))
    )
    return candidate


def native_common_grasp_candidate(stage, check: dict, asset: dict, dataset: str, rank: int, count: int) -> dict | None:
    """Use only an authored grasp and bind it to collision geometry without guessing."""
    import numpy as np

    root_path = check.get("default_prim") or check.get("rigid_bodies", [None])[0]
    native = native_grasp_candidates(stage, root_path) if root_path else []
    if not native:
        return None
    ordered = sorted(native, key=lambda item: item["prim_path"])
    if rank < 0 or rank >= min(len(ordered), max(1, int(count))):
        return None
    grasp = ordered[rank]
    if not grasp.get("pose_valid") or not grasp.get("width_valid"):
        return None
    candidate = grasp_candidate(
        dataset,
        {**asset, "_grasp_candidate_rank": rank},
        *stage_bounds(stage, root_path),
        stage,
        root_path,
    )
    if candidate is None:
        return None
    target_hint = str(grasp.get("component") or "").strip()
    rigid_paths = [str(path) for path in check.get("rigid_bodies", [])]
    target = target_hint if target_hint in rigid_paths else None
    target_source = "authored_component" if target is not None else None
    bounds = None

    # Annotation writers store semantic component names (for example ``Body``),
    # while converted assets expose generated rigid-body paths. Resolve the
    # semantic hint without rejecting an otherwise valid authored grasp when
    # several generated bodies share the same basename.
    if target is None and target_hint:
        matches = [
            path for path in rigid_paths
            if path.rsplit("/", 1)[-1].casefold() == target_hint.casefold()
        ]
        if len(matches) == 1:
            target = matches[0]
            target_source = "authored_component_basename"
        elif matches:
            bounds = object_link_collision_bounds(stage, check, mesh_records(stage))["links"]
            center = np.asarray(candidate["center"], dtype=float)
            ranked = []
            for path in matches:
                item = bounds.get(path)
                if not item or item.get("minimum") is None or item.get("maximum") is None:
                    continue
                midpoint = (
                    np.asarray(item["minimum"], dtype=float)
                    + np.asarray(item["maximum"], dtype=float)
                ) * 0.5
                ranked.append((float(np.linalg.norm(midpoint - center)), path))
            if ranked:
                target = min(ranked)[1]
                target_source = "authored_component_nearest_match"

    if target is None:
        if bounds is None:
            bounds = object_link_collision_bounds(stage, check, mesh_records(stage))["links"]
        target = min(
            (
                path for path, item in bounds.items()
                if path in rigid_paths
                if item.get("minimum") is not None and item.get("maximum") is not None
            ),
            key=lambda path: float(np.linalg.norm(
                (
                    np.asarray(bounds[path]["minimum"], dtype=float)
                    + np.asarray(bounds[path]["maximum"], dtype=float)
                ) * 0.5
                - np.asarray(candidate["center"], dtype=float)
            )),
            default=None,
        )
        if target is not None:
            target_source = "nearest_collision_rigid_body"
    if target is None:
        return None
    candidate["target_rigid_body"] = target
    candidate["same_rigid_body_required"] = False
    candidate["source"] = "authored_usd_grasp"
    candidate["native_grasp_prim"] = grasp["prim_path"]
    candidate["target_binding_source"] = target_source
    candidate["target_binding_status"] = "unique" if target else "unresolved"
    candidate["candidate_width_source"] = "authored" if candidate["width"] is not None else "target_collision_projection"
    candidate["annotation_diagnostics"] = {
        **candidate.get("annotation_diagnostics", {}),
        "native_component": grasp.get("component"),
        "native_target_binding": target,
    }
    return candidate


def set_body_pose(dc, handle, position, orientation) -> None:
    from omni.isaac.dynamic_control import _dynamic_control

    dc.set_rigid_body_pose(
        handle,
        _dynamic_control.Transform(tuple(float(value) for value in position), orientation),
    )


def apply_world_body_force(dc, handle, force, position) -> None:
    dc.apply_body_force(
        handle,
        tuple(float(value) for value in force),
        tuple(float(value) for value in position),
        True,
    )


def world_vector_to_body_local(vector_world, orientation_xyzw):
    """Convert a world-space vector to a body-local vector."""
    import numpy as np

    vector = np.asarray(vector_world, dtype=float)
    x, y, z, w = (float(value) for value in orientation_xyzw)
    quaternion = np.asarray((x, y, z, w), dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if vector.shape != (3,) or not np.all(np.isfinite(vector)) or norm <= 1e-12:
        return np.zeros(3, dtype=float)
    x, y, z, w = quaternion / norm
    qvec = np.asarray((x, y, z), dtype=float)
    # q^-1 * [v, 0] * q, expanded to avoid a runtime quaternion dependency.
    uv = np.cross(qvec, vector)
    uuv = np.cross(qvec, uv)
    return vector - 2.0 * w * uv + 2.0 * uuv


def apply_world_body_torque(dc, handle, torque, orientation_xyzw=None) -> None:
    """Apply a world-space torque through Isaac's local-coordinate API."""
    local_torque = (
        world_vector_to_body_local(torque, orientation_xyzw)
        if orientation_xyzw is not None
        else tuple(float(value) for value in torque)
    )
    dc.apply_body_torque(
        handle,
        tuple(float(value) for value in local_torque),
        True,
    )


def grasp_force_target(mass, gravity, left_friction, right_friction, safety_factor):
    support = max(1e-6, float(left_friction) + float(right_friction))
    return float(safety_factor) * float(mass) * abs(float(gravity)) / support


def virtual_palm_forces(
    positions,
    velocities,
    starts,
    closing_axis,
    desired_offset,
    desired_velocity,
    masses,
    inward_forces,
    feedforward,
    kp,
    kd,
    force_limit,
):
    import numpy as np

    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    starts = np.asarray(starts, dtype=float)
    closing_axis = np.asarray(closing_axis, dtype=float)
    center = positions.mean(axis=0)
    center_velocity = velocities.mean(axis=0)
    start_center = starts.mean(axis=0)
    center_acceleration = float(kp) * (
        start_center + np.asarray(desired_offset, dtype=float) - center
    ) + float(kd) * (np.asarray(desired_velocity, dtype=float) - center_velocity)
    forces = []
    for position, velocity, start, mass, inward in zip(
        positions, velocities, starts, masses, inward_forces
    ):
        relative_error = (start - start_center) - (position - center)
        relative_error -= closing_axis * float(np.dot(relative_error, closing_axis))
        relative_velocity = velocity - center_velocity
        relative_velocity -= closing_axis * float(np.dot(relative_velocity, closing_axis))
        servo = np.asarray(feedforward, dtype=float) + float(mass) * (
            center_acceleration
            + float(kp) * relative_error
            - float(kd) * relative_velocity
        )
        magnitude = float(np.linalg.norm(servo))
        if magnitude > float(force_limit) > 0:
            servo *= float(force_limit) / magnitude
        forces.append(servo + np.asarray(inward, dtype=float))
    return forces


def rigid_translation_force(
    current_position,
    current_velocity,
    desired_position,
    desired_velocity,
    mass,
    gravity,
    kp,
    kd,
    force_limit,
):
    """Compute a bounded force for translation without changing orientation."""
    import numpy as np

    current_position = np.asarray(current_position, dtype=float)
    current_velocity = np.asarray(current_velocity, dtype=float)
    desired_position = np.asarray(desired_position, dtype=float)
    desired_velocity = np.asarray(desired_velocity, dtype=float)
    force = float(mass) * (
        float(kp) * (desired_position - current_position)
        + float(kd) * (desired_velocity - current_velocity)
    )
    force += np.asarray((0.0, 0.0, float(mass) * abs(float(gravity))), dtype=float)
    magnitude = float(np.linalg.norm(force))
    limit = float(force_limit)
    clamped = bool(math.isfinite(limit) and limit > 0.0 and magnitude > limit)
    if clamped:
        force *= limit / max(magnitude, 1e-12)
    return {
        "force": force,
        "position_error": desired_position - current_position,
        "velocity_error": desired_velocity - current_velocity,
        "force_magnitude": float(np.linalg.norm(force)),
        "force_clamped": clamped,
    }


def finger_common_mode_stabilization_force(
    relative_positions,
    relative_velocities,
    initial_relative_positions,
    closing_axis,
    total_finger_mass,
    kp,
    kd,
    force_limit,
) -> dict:
    """Bound a shared jaw translation without commanding either jaw gap."""
    import numpy as np

    positions = np.asarray(relative_positions, dtype=float)
    velocities = np.asarray(relative_velocities, dtype=float)
    starts = np.asarray(initial_relative_positions, dtype=float)
    axis = np.asarray(closing_axis, dtype=float)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    position_error = float(np.dot(np.mean(starts - positions, axis=0), axis))
    velocity_error = float(np.dot(-np.mean(velocities, axis=0), axis))
    requested_force = axis * (0.5 * float(total_finger_mass)) * (
        float(kp) * position_error + float(kd) * velocity_error
    )
    magnitude = float(np.linalg.norm(requested_force))
    limit = max(0.0, float(force_limit))
    clamped = magnitude > limit > 0.0
    applied_force = requested_force * (limit / magnitude) if clamped else requested_force
    return {
        "force": applied_force,
        "position_error_m": position_error,
        "velocity_error_mps": velocity_error,
        "force_magnitude_n": float(np.linalg.norm(applied_force)),
        "force_clamped": clamped,
    }


def grasp_lift_controller_settings(required_lift_m: float) -> dict:
    """Read the isolated lift-controller experiment knobs without config edits."""
    import hashlib

    requested_controller = os.environ.get("RAW_EVAL_LIFT_CONTROLLER", "d6_drive").strip().lower()
    controller = requested_controller if requested_controller in {"force_pd", "d6_drive"} else "d6_drive"
    try:
        requested_height = float(os.environ.get("RAW_EVAL_COMMAND_LIFT_HEIGHT_M", required_lift_m))
    except (TypeError, ValueError):
        requested_height = float(required_lift_m)
    command_height = max(float(required_lift_m), requested_height)
    payload = {
        "lift_controller": controller,
        "requested_lift_controller": requested_controller,
        "lift_controller_config_error": (
            f"unsupported_controller:{requested_controller}"
            if requested_controller not in {"force_pd", "d6_drive"}
            else None
        ),
        "commanded_gripper_lift_m": command_height,
        "required_object_lift_m": float(required_lift_m),
    }
    payload["controller_config_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def create_world_anchored_d6_lift_drive(stage, palm_path: str, palm_position, palm_orientation, stiffness, damping, max_force) -> dict:
    """Create a session-layer world anchor with only its local Z drive free."""
    from pxr import Gf, Sdf, UsdPhysics

    stage.SetEditTarget(stage.GetSessionLayer())
    joint = UsdPhysics.Joint.Define(stage, "/__raw_eval/GripperLiftD6")
    joint.CreateBody1Rel().SetTargets([Sdf.Path(palm_path)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, palm_position)))
    x, y, z, w = map(float, palm_orientation)
    joint.CreateLocalRot0Attr(Gf.Quatf(w, x, y, z))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
    joint.CreateCollisionEnabledAttr(False)
    locked_dofs = ("transX", "transY", "rotX", "rotY", "rotZ")
    for dof in locked_dofs:
        limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), dof)
        limit.CreateLowAttr(1.0)
        limit.CreateHighAttr(-1.0)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "transZ")
    drive.CreateTypeAttr("force")
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateTargetVelocityAttr(0.0)
    drive.CreateStiffnessAttr(float(stiffness))
    drive.CreateDampingAttr(float(damping))
    drive.CreateMaxForceAttr(float(max_force))
    return {
        "status": "active",
        "path": str(joint.GetPath()),
        "target_position": drive.GetTargetPositionAttr(),
        "target_velocity": drive.GetTargetVelocityAttr(),
        "locked_dofs": list(locked_dofs),
        "stiffness": float(stiffness),
        "damping": float(damping),
        "max_force": float(max_force),
    }


def create_world_locked_palm_hold(stage, palm_path: str, palm_position, palm_orientation) -> dict:
    """Temporarily hold the palm at the first valid contact during close/seating."""
    from pxr import Gf, Sdf, UsdPhysics

    stage.SetEditTarget(stage.GetSessionLayer())
    joint = UsdPhysics.Joint.Define(stage, "/__raw_eval/GripperContactHold")
    joint.CreateBody1Rel().SetTargets([Sdf.Path(palm_path)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, palm_position)))
    x, y, z, w = map(float, palm_orientation)
    joint.CreateLocalRot0Attr(Gf.Quatf(w, x, y, z))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
    joint.CreateCollisionEnabledAttr(False)
    locked_dofs = ("transX", "transY", "transZ", "rotX", "rotY", "rotZ")
    for dof in locked_dofs:
        limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), dof)
        limit.CreateLowAttr(1.0)
        limit.CreateHighAttr(-1.0)
    return {"status": "active", "path": str(joint.GetPath()), "locked_dofs": list(locked_dofs)}


def set_world_anchored_d6_lift_target(drive: dict, position_m: float, velocity_mps: float) -> None:
    drive["target_position"].Set(float(position_m))
    drive["target_velocity"].Set(float(velocity_mps))


def d6_lift_load_compensation(required_lift_m: float, object_mass_kg: float, gravity_mps2: float, stiffness_npm: float, max_extra_m: float = 0.05) -> dict:
    """Return a bounded height lead for a finite-stiffness vertical drive.

    This compensates only the static position error of the existing force drive.
    It does not change the required object lift, drive force, mass, or friction.
    """
    required = max(0.0, float(required_lift_m))
    mass = max(0.0, float(object_mass_kg))
    gravity = abs(float(gravity_mps2))
    stiffness = float(stiffness_npm)
    cap = max(0.0, float(max_extra_m))
    if not all(math.isfinite(value) for value in (required, mass, gravity, stiffness, cap)) or stiffness <= 0.0:
        bias = 0.0
    else:
        bias = min(cap, mass * gravity / stiffness)
    return {
        "required_lift_m": required,
        "object_mass_kg": mass,
        "gravity_mps2": gravity,
        "stiffness_npm": stiffness,
        "load_bias_m": bias,
        "applied_target_lift_m": required + bias,
        "bias_clamped": bool(cap > 0.0 and mass * gravity / stiffness > cap) if stiffness > 0.0 else False,
    }


def quaternion_rotation_error(current_orientation, target_orientation):
    import numpy as np

    current = np.asarray(current_orientation, dtype=float)
    target = np.asarray(target_orientation, dtype=float)
    current /= max(float(np.linalg.norm(current)), 1e-12)
    target /= max(float(np.linalg.norm(target)), 1e-12)
    current_conjugate = np.asarray((-current[0], -current[1], -current[2], current[3]))
    vector = (
        target[3] * current_conjugate[:3]
        + current_conjugate[3] * target[:3]
        + np.cross(target[:3], current_conjugate[:3])
    )
    scalar = target[3] * current_conjugate[3] - float(
        np.dot(target[:3], current_conjugate[:3])
    )
    if scalar < 0:
        vector, scalar = -vector, -scalar
    length = float(np.linalg.norm(vector))
    return (
        vector / length * (2.0 * math.atan2(length, max(0.0, scalar)))
        if length > 1e-12
        else np.zeros(3)
    )


def virtual_palm_torque(
    current_orientation,
    target_orientation,
    angular_velocity,
    rotational_inertia,
    kp,
    kd,
    torque_limit,
):
    import numpy as np

    inertia = np.asarray(rotational_inertia, dtype=float)
    if inertia.ndim == 0:
        inertia = float(inertia)
    elif inertia.shape != (3,) or not np.all(np.isfinite(inertia)) or np.any(inertia <= 0):
        inertia = 1.0
    torque = inertia * (
        float(kp) * quaternion_rotation_error(current_orientation, target_orientation)
        - float(kd) * np.asarray(angular_velocity, dtype=float)
    )
    magnitude = float(np.linalg.norm(torque))
    if magnitude > float(torque_limit) > 0:
        torque *= float(torque_limit) / magnitude
    return torque


def rigid_body_rotational_inertia(stage, body_path, fallback_dimensions, mass):
    """Return a conservative diagonal inertia and its provenance.

    Authored USD inertia is preferred because it is the only value that can
    represent the composed collision body after PhysX mass recomputation.
    The fallback is the diagonal inertia of a uniform box, kept explicit so a
    missing authored value cannot be mistaken for runtime evidence.
    """
    import numpy as np

    fallback = np.asarray(
        [
            float(mass) * (float(fallback_dimensions[1]) ** 2 + float(fallback_dimensions[2]) ** 2) / 12.0,
            float(mass) * (float(fallback_dimensions[0]) ** 2 + float(fallback_dimensions[2]) ** 2) / 12.0,
            float(mass) * (float(fallback_dimensions[0]) ** 2 + float(fallback_dimensions[1]) ** 2) / 12.0,
        ],
        dtype=float,
    )
    try:
        prim = stage.GetPrimAtPath(str(body_path))
        attribute = prim.GetAttribute("physics:diagonalInertia") if prim and prim.IsValid() else None
        value = attribute.Get() if attribute and attribute.HasAuthoredValueOpinion() else None
        values = np.asarray(tuple(value), dtype=float) if value is not None else None
        if values is not None and values.shape == (3,) and np.all(np.isfinite(values)) and np.all(values > 0):
            return values, "authored_usd_diagonal_inertia"
    except (AttributeError, TypeError, ValueError):
        pass
    if fallback.shape == (3,) and np.all(np.isfinite(fallback)) and np.all(fallback > 0):
        return fallback, "box_geometry_fallback"
    return np.ones(3, dtype=float), "unit_inertia_fallback"


def grasp_clearance_result(clearances, ground_contacts, minimum, required_fraction):
    clearances = [float(value) for value in clearances if math.isfinite(float(value))]
    contacts = [bool(value) for value in ground_contacts]
    if not clearances or len(clearances) != len(contacts):
        return {"pass": False, "fraction": 0.0, "minimum": None, "ground_contact_fraction": 0.0}
    fraction = sum(
        clearance >= float(minimum) and not contact
        for clearance, contact in zip(clearances, contacts)
    ) / len(clearances)
    return {
        "pass": fraction >= float(required_fraction) and not any(contacts),
        "fraction": fraction,
        "minimum": min(clearances),
        "ground_contact_fraction": sum(contacts) / len(contacts),
    }


def contact_force_observation(contacts, target_paths, dt):
    """Return target contact and force evidence without conflating the two."""
    result = {
        "contact_observed": False,
        "force_observed": False,
        "force_source": "none",
        "raw_impulse_n_s": 0.0,
        "normal_force_n": 0.0,
        "contact_count": 0,
        "force_observation_status": "no_target_contact",
        "per_contact": [],
    }
    try:
        step = float(dt)
    except (TypeError, ValueError):
        step = 0.0
    if not math.isfinite(step) or step <= 0:
        result["force_observation_status"] = "invalid_dt"
        return result
    for contact in contacts or []:
        bodies = (str(contact.get("body0", "")), str(contact.get("body1", "")))
        if not any(
            body == target or body.startswith(f"{target}/")
            for target in target_paths
            for body in bodies
        ):
            continue
        result["contact_observed"] = True
        result["contact_count"] += 1
        item = {"force_observation_status": "missing_force_fields"}
        try:
            impulse = tuple(float(value) for value in contact["impulse"])
            normal = tuple(float(value) for value in contact["normal"])
            normal_length = math.sqrt(sum(value * value for value in normal))
            if len(impulse) != 3 or len(normal) != 3 or normal_length <= 1e-12:
                raise ValueError("invalid_contact_vector")
            normal_impulse = abs(sum(i * n for i, n in zip(impulse, normal)) / normal_length)
            if not math.isfinite(normal_impulse):
                raise ValueError("nonfinite_contact_impulse")
            item.update({
                "raw_impulse_n_s": normal_impulse,
                "normal_force_n": normal_impulse / step,
                "force_observation_status": "valid" if normal_impulse > 0.0 else "zero_impulse_contact",
            })
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            pass
        result["per_contact"].append(item)
        result["raw_impulse_n_s"] += float(item.get("raw_impulse_n_s", 0.0))
        result["normal_force_n"] += float(item.get("normal_force_n", 0.0))
    if not result["contact_observed"]:
        return result
    valid = [item for item in result["per_contact"] if item["force_observation_status"] == "valid"]
    if valid:
        result["force_observed"] = True
        result["force_source"] = "raw_impulse_projection"
        result["force_observation_status"] = "valid"
    elif any(item["force_observation_status"] == "zero_impulse_contact" for item in result["per_contact"]):
        result["force_observation_status"] = "zero_impulse_contact"
    else:
        result["force_observation_status"] = "missing_force_fields"
    return result


def raw_normal_contact_force(contacts, target_paths, dt):
    """Compatibility scalar for existing diagnostics and non-gating callers."""
    return float(contact_force_observation(contacts, target_paths, dt)["normal_force_n"])


def contact_torque_observation(contacts, target_paths, palm_center, dt):
    """Estimate target-contact torque about the palm from raw impulse data."""
    import numpy as np

    result = {
        "status": "no_target_contact",
        "source": "none",
        "contact_count": 0,
        "raw_impulse_n_s": 0.0,
        "normal_force_n": 0.0,
        "torque_world": [0.0, 0.0, 0.0],
        "torque_magnitude_nm": 0.0,
        "per_contact": [],
    }
    try:
        step = float(dt)
        center = np.asarray(palm_center, dtype=float)
    except (TypeError, ValueError):
        return {**result, "status": "invalid_dt"}
    if not math.isfinite(step) or step <= 0 or center.shape != (3,) or not np.all(np.isfinite(center)):
        return {**result, "status": "invalid_dt"}
    total_torque = np.zeros(3, dtype=float)
    for contact in contacts or []:
        bodies = (str(contact.get("body0", "")), str(contact.get("body1", "")))
        if not any(body == target or body.startswith(f"{target}/") for target in target_paths for body in bodies):
            continue
        result["contact_count"] += 1
        item = {"status": "missing_fields"}
        try:
            position = np.asarray(contact.get("position", contact.get("point", contact.get("contact_point"))), dtype=float)
            impulse = np.asarray(contact["impulse"], dtype=float)
            normal = np.asarray(contact["normal"], dtype=float)
            normal_length = float(np.linalg.norm(normal))
            if position.shape != (3,) or impulse.shape != (3,) or normal.shape != (3,) or normal_length <= 1e-12:
                raise ValueError("invalid_contact_vector")
            if not np.all(np.isfinite(position)) or not np.all(np.isfinite(impulse)) or not np.all(np.isfinite(normal)):
                raise ValueError("nonfinite_contact_vector")
            normal_unit = normal / normal_length
            signed_impulse = float(np.dot(impulse, normal_unit))
            if not math.isfinite(signed_impulse):
                raise ValueError("nonfinite_contact_impulse")
            force = normal_unit * (signed_impulse / step)
            arm = position - center
            torque = np.cross(arm, force)
            if not np.all(np.isfinite(torque)):
                raise ValueError("nonfinite_contact_torque")
            total_torque += torque
            result["raw_impulse_n_s"] += abs(signed_impulse)
            result["normal_force_n"] += abs(signed_impulse) / step
            item = {
                "status": "valid" if abs(signed_impulse) > 0.0 else "zero_impulse_contact",
                "position_world": position.tolist(),
                "arm_world_m": arm.tolist(),
                "normal_world": normal_unit.tolist(),
                "raw_impulse_n_s": signed_impulse,
                "normal_force_n": signed_impulse / step,
                "force_world_n": force.tolist(),
                "torque_world_nm": torque.tolist(),
            }
        except (KeyError, TypeError, ValueError):
            pass
        result["per_contact"].append(item)
    valid = [item for item in result["per_contact"] if item["status"] == "valid"]
    result["torque_world"] = total_torque.tolist()
    result["torque_magnitude_nm"] = float(np.linalg.norm(total_torque))
    if valid:
        result["status"] = "valid"
        result["source"] = "raw_impulse_contact_moment"
    elif result["contact_count"]:
        result["status"] = (
            "zero_impulse_contact"
            if any(item["status"] == "zero_impulse_contact" for item in result["per_contact"])
            else "contact_torque_unobservable"
        )
    return result


def bilateral_force_window(samples, force_target) -> dict:
    """Summarize target-only normal-force samples over one physical time window."""
    if not samples or not math.isfinite(float(force_target)) or float(force_target) <= 0:
        return {
            "left_average_normal_force_n": 0.0,
            "right_average_normal_force_n": 0.0,
            "target_contact": False,
            "force_ready": False,
        }
    left_values = [max(0.0, float(sample[0])) for sample in samples]
    right_values = [max(0.0, float(sample[1])) for sample in samples]
    left_average = sum(left_values) / len(left_values)
    right_average = sum(right_values) / len(right_values)
    target_contact = any(value > 0.0 for value in left_values) and any(
        value > 0.0 for value in right_values
    )
    return {
        "left_average_normal_force_n": left_average,
        "right_average_normal_force_n": right_average,
        "target_contact": target_contact,
        "force_ready": bool(
            target_contact
            and left_average >= float(force_target)
            and right_average >= float(force_target)
        ),
    }


def bilateral_force_window_fraction(samples, force_target, window_steps: int) -> tuple[float, list[dict]]:
    """Return the fraction of consecutive physical-force windows that remain bilateral."""
    window_steps = max(1, int(window_steps))
    windows = [
        bilateral_force_window(samples[index:index + window_steps], force_target)
        for index in range(0, len(samples), window_steps)
    ]
    fraction = sum(window["force_ready"] for window in windows) / len(windows) if windows else 0.0
    return fraction, windows


def bilateral_contact_fraction_from_pairs(samples, window_steps: int) -> float:
    """Fraction of physical samples with bilateral target contact."""
    del window_steps
    if not samples:
        return 0.0
    bilateral = sum(bool(left) and bool(right) for left, right in samples)
    return float(bilateral) / float(len(samples))


def bound_static_friction(stage, prim_path: str, fallback: float) -> tuple[float, str]:
    from pxr import Usd, UsdShade

    root = stage.GetPrimAtPath(prim_path)
    if not root:
        return float(fallback), "fallback"
    values = []
    for prim in Usd.PrimRange(root):
        attribute = prim.GetAttribute("physics:staticFriction")
        if attribute and attribute.HasAuthoredValueOpinion():
            values.append(float(attribute.Get()))
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(
            materialPurpose="physics"
        )
        if material:
            attribute = material.GetPrim().GetAttribute("physics:staticFriction")
            if attribute and attribute.HasAuthoredValueOpinion():
                values.append(float(attribute.Get()))
    values = [value for value in values if math.isfinite(value) and value > 0]
    return (min(values), "authored") if values else (float(fallback), "fallback")


def target_subtree_paths(target_rigid_body: str, rigid_bodies: list[str]) -> list[str]:
    """Return the authored target rigid body and its declared descendants."""
    target = str(target_rigid_body or "").rstrip("/")
    if not target:
        return []
    paths = [
        str(path)
        for path in rigid_bodies
        if str(path) == target or str(path).startswith(f"{target}/")
    ]
    return paths or [target]


def contact_target_path(frame: dict, target_paths: list[str]) -> str | None:
    for contact in frame.get("contacts", []):
        for body in (str(contact.get("body0", "")), str(contact.get("body1", ""))):
            for target in target_paths:
                if body == target or body.startswith(f"{target}/"):
                    return target
    return None


def gripper_pad_contact_diagnostics(contacts, contract: dict, side: str) -> list[dict]:
    """Project raw PhysX contacts into the generated, asset-independent pad frame."""
    import numpy as np

    reference = np.asarray(contract["reference_point_world"], dtype=float)
    closing = np.asarray(contract["closing_axis_world"], dtype=float)
    approach = np.asarray(contract["approach_axis_world"], dtype=float)
    height = np.asarray(contract["height_axis_world"], dtype=float)
    depth_min, depth_max = (float(value) for value in contract["pad_depth_range_local_m"])
    height_min, height_max = (float(value) for value in contract["pad_height_range_local_m"])
    output = []
    for contact in contacts:
        position = contact.get("position", contact.get("point"))
        normal = contact.get("normal")
        if position is None or normal is None:
            continue
        try:
            point = np.asarray(tuple(float(value) for value in position), dtype=float)
            unit_normal = np.asarray(tuple(float(value) for value in normal), dtype=float)
        except (TypeError, ValueError):
            continue
        normal_length = float(np.linalg.norm(unit_normal))
        if len(point) != 3 or len(unit_normal) != 3 or normal_length <= 1e-8:
            continue
        unit_normal /= normal_length
        local = point - reference
        closing_coordinate = float(np.dot(local, closing))
        depth_coordinate = float(np.dot(local, approach))
        height_coordinate = float(np.dot(local, height))
        expected_sign = -1.0 if side == "left" else 1.0
        output.append(
            {
                "position_world": point.tolist(),
                "normal_world": unit_normal.tolist(),
                "closing_coordinate_m": closing_coordinate,
                "depth_coordinate_m": depth_coordinate,
                "height_coordinate_m": height_coordinate,
                "normal_closing_projection": float(np.dot(unit_normal, closing)),
                "normal_closing_alignment": abs(float(np.dot(unit_normal, closing))),
                "normal_expected_side_alignment": expected_sign * float(np.dot(unit_normal, closing)),
                "within_pad_depth": depth_min <= depth_coordinate <= depth_max,
                "within_pad_height": height_min <= height_coordinate <= height_max,
            }
        )
    return output


def classify_finger_contact_region(position, contract: dict, side: str, tolerance_m: float = 0.002) -> dict:
    """Classify a generated finger contact without relying on the target asset."""
    import numpy as np

    point = np.asarray(position, dtype=float)
    reference = np.asarray(contract["reference_point_world"], dtype=float)
    closing = np.asarray(contract["closing_axis_world"], dtype=float)
    approach = np.asarray(contract["approach_axis_world"], dtype=float)
    height = np.asarray(contract["height_axis_world"], dtype=float)
    local = point - reference
    coordinates = {
        "closing_m": float(np.dot(local, closing)),
        "depth_m": float(np.dot(local, approach)),
        "height_m": float(np.dot(local, height)),
    }
    depth_min, depth_max = contract.get("finger_collision_depth_range_local_m", contract["pad_depth_range_local_m"])
    shell_min, shell_max = contract.get("finger_collision_height_range_local_m", contract["pad_height_range_local_m"])
    pad_depth_min, pad_depth_max = contract["pad_depth_range_local_m"]
    pad_height_min, pad_height_max = contract["pad_height_range_local_m"]
    in_shell = (
        float(depth_min) - tolerance_m <= coordinates["depth_m"] <= float(depth_max) + tolerance_m
        and float(shell_min) - tolerance_m <= coordinates["height_m"] <= float(shell_max) + tolerance_m
    )
    in_pad = (
        float(pad_depth_min) - tolerance_m <= coordinates["depth_m"] <= float(pad_depth_max) + tolerance_m
        and float(pad_height_min) - tolerance_m <= coordinates["height_m"] <= float(pad_height_max) + tolerance_m
    )
    if in_pad:
        region, material_role, status = "rubber_pad", "rubber", "classified"
    elif in_shell:
        region, material_role, status = "plastic_shell", "plastic", "classified"
    else:
        region, material_role, status = "outside_known_finger_collision", None, "outside_known_finger_collision"
    material_friction = contract.get("material_static_friction", {}).get(material_role) if material_role else None
    return {
        "side": str(side),
        "contact_region": region,
        "contact_material_role": material_role,
        "contact_static_friction": material_friction,
        "within_load_bearing_region": region in {"rubber_pad", "plastic_shell", "ambiguous_overlap"},
        "contact_region_classification_status": status,
        "local_coordinates_m": coordinates,
    }


def _acquire_grasp_once(stage, check: dict, app, dataset: str, asset: dict, recorder=None, target_paths=None, prepared_timeline=None) -> dict:
    import numpy as np
    from omni.isaac.dynamic_control import _dynamic_control

    bounds_path = check["default_prim"] or check["rigid_bodies"][0]
    candidate_rank = int(asset.get("_grasp_candidate_rank", 0))
    authored = sorted(native_grasp_candidates(stage, bounds_path), key=lambda item: item["prim_path"])
    if not authored:
        return {"applicable": False, "pass": False, "reason": "missing_authored_grasppose"}
    candidate_count = len(authored)
    if candidate_rank < 0 or candidate_rank >= candidate_count:
        return {"applicable": False, "pass": False, "reason": "authored_grasp_rank_unavailable"}
    if not authored[candidate_rank].get("pose_valid") or not authored[candidate_rank].get("width_valid"):
        return {"applicable": False, "pass": False, "reason": "invalid_authored_grasppose"}
    candidate = native_common_grasp_candidate(
        stage,
        check,
        asset,
        dataset,
        candidate_rank,
        candidate_count,
    )
    if candidate is None:
        return {"applicable": False, "pass": False, "reason": "authored_grasp_target_unresolved"}
    settings = CONFIG["simulation"]
    if prepared_timeline is None:
        timeline, _minimum, _maximum = begin_physics(
            stage, check, app,
            settle_seconds=float(settings["interaction_settle_seconds"]),
        )
    else:
        timeline = prepared_timeline
    timeline.stop()
    minimum, maximum = stage_bounds(stage, bounds_path)
    candidate = native_common_grasp_candidate(
        stage,
        check,
        asset,
        dataset,
        candidate_rank,
        candidate_count,
    )
    if candidate is None:
        return {"applicable": False, "pass": False, "reason": "authored_grasp_target_unresolved"}
    extent = np.asarray(maximum) - np.asarray(minimum)
    diagonal = float(np.linalg.norm(extent))
    finger_scale = float(asset.get("_grasp_finger_scale", 1.0))
    gripper_geometry = str(asset.get("_grasp_geometry", "flat"))
    descriptor = gripper_geometry_descriptor(gripper_geometry, finger_scale, diagonal)
    finger_thickness = float(descriptor["finger_thickness"])
    finger_depth = float(descriptor["finger_depth"])
    finger_height = float(descriptor["finger_height"])
    # Composed VHACD prim AABBs can be unresolved sentinels; mesh_records is
    # the existing collision-only vertex source used by settling diagnostics.
    records = mesh_records(stage)
    collision_bounds = object_link_collision_bounds(stage, check, records)
    target_collision = collision_bounds["links"].get(candidate["target_rigid_body"])
    collision_union = grasp_collision_union(collision_bounds)
    if target_collision is None or collision_union["minimum"] is None:
        timeline.stop()
        return {
            "applicable": True,
            "pass": False,
            "status": "evaluation_blocked",
            "reason": "evaluation_blocked",
            "reason_class": "evaluator",
            "retryable_runtime": False,
            "opening_reason": "collision_geometry_unavailable",
            "candidate_width_m": candidate.get("width"),
            "candidate_width_source": candidate.get("candidate_width_source"),
            "target_binding_source": candidate.get("target_binding_source"),
            "target_binding_status": candidate.get("target_binding_status"),
        }
    collision_points = [
        row["vertices"]
        for row in records
        if row.get("collision")
        and row.get("rigid_body") in collision_bounds.get("links", {})
    ]
    opening_geometry = grasp_opening_geometry(
        candidate["width"],
        collision_union["minimum"],
        collision_union["maximum"],
        candidate["closing"],
        finger_thickness,
        max(
            float(segment["pad_thickness"])
            for segment in descriptor["finger_segments"]
        ),
        descriptor["reference_max_opening_m"],
        approach_axis=candidate["approach"],
        grasp_center=candidate["center"],
        collision_points=(
            np.concatenate(collision_points, axis=0)
            if collision_points else None
        ),
        finger_descriptor=descriptor,
    )
    # Size the palm and rail for this candidate's actual supported opening.
    descriptor = gripper_geometry_descriptor(
        gripper_geometry,
        finger_scale,
        diagonal,
        opening_geometry["initial_gap_applied_m"],
    )
    finger_depth = float(descriptor["finger_depth"])
    finger_height = float(descriptor["finger_height"])
    # Keep authored approach semantics: pregrasp -> grasp moves along +approach.
    gripper_approach = np.asarray(candidate["approach"], dtype=float)
    gripper_orientation = quaternion_xyzw(
        np.column_stack(
            (
                candidate["closing"],
                gripper_approach,
                np.cross(candidate["closing"], gripper_approach),
            )
        )
    )
    frame_poses = grasp_frame_to_palm_poses(
        candidate["center"],
        candidate["closing"],
        candidate["approach"],
        finger_depth,
        float(descriptor["palm_dimensions"][1]),
    )
    dimensions = (finger_thickness, finger_depth, finger_height)
    initial_gap = opening_geometry["initial_gap_applied_m"]
    final_gap = 0.001
    approach_clearance = max(0.01, finger_thickness)
    pregrasp_center, approach_distance = grasp_pregrasp_center(
        candidate["center"],
        candidate["approach"],
        minimum,
        maximum,
        finger_depth,
        approach_clearance,
    )

    def finger_centers(center, gap):
        return parallel_finger_centers(
            center,
            candidate["closing"],
            gripper_approach,
            finger_depth,
            gap,
            descriptor["max_inner_edge_inset"],
        )

    spawn_candidate = {
        **candidate,
        "center": pregrasp_center,
        "approach": gripper_approach,
        "orientation": gripper_orientation,
    }
    gripper = define_parallel_gripper(stage, spawn_candidate, descriptor, initial_gap, final_gap)
    starts = tuple(np.asarray(value, dtype=float) for value in gripper["starts"])
    palm_path = gripper["palm_path"]
    left_path = gripper["left_path"]
    right_path = gripper["right_path"]
    palm_sensor = create_contact_sensor(palm_path, "ApproachContactSensor")
    left_sensor = create_contact_sensor(left_path, "ContactSensor")
    right_sensor = create_contact_sensor(right_path, "ContactSensor")
    try:
        ground_sensor = create_contact_sensor("/__raw_eval/Ground", "GraspContactSensor")
    except Exception:
        ground_sensor = None
    app.update()
    timeline.play()
    for _ in range(3):
        app.update()
    dc = _dynamic_control.acquire_dynamic_control_interface()
    palm = dc.get_rigid_body(palm_path)
    left = dc.get_rigid_body(left_path)
    right = dc.get_rigid_body(right_path)
    target_path = candidate["target_rigid_body"]
    # Nearest geometry is a tracking hint, not an authored contact restriction.
    close_target_paths = (
        target_subtree_paths(target_path, check["rigid_bodies"])
        if candidate.get("annotation_diagnostics", {}).get("native_component")
        else list(check["rigid_bodies"])
    )
    if target_paths is not None:
        close_target_paths = sorted(set(target_paths) & set(check["rigid_bodies"]))
    target_body = dc.get_rigid_body(target_path)
    if not palm or not left or not right or not target_body:
        timeline.stop()
        return runtime_blocked(
            "runtime_grasp_handle_unavailable",
            phase="grasp_lift",
            palm_path=palm_path,
            left_finger_path=left_path,
            right_finger_path=right_path,
            target_rigid_body=target_path,
        )
    body_handles = (
        (palm, palm_path, "palm"),
        (left, left_path, "left_finger"),
        (right, right_path, "right_finger"),
        (target_body, target_path, "target_body"),
    )

    def handle_blocked(failure_phase: str):
        """Stop before the next Dynamic Control call after handle expiry."""
        for body, body_path, handle_type in body_handles:
            _, blocked = rigid_body_mass_or_blocked(
                dc, body, body_path, failure_phase
            )
            if blocked:
                blocked.setdefault("diagnostics", {}).update({
                    "handle_type": handle_type,
                    "failure_phase": failure_phase,
                })
                timeline.stop()
                return blocked
        return None

    # Validate all dynamic-control bodies before entering the approach/close
    # loops. The same guard is reused after each simulation step.
    blocked = handle_blocked("grasp_handle_validation")
    if blocked:
        return blocked
    dc.set_rigid_body_disable_gravity(palm, True)
    dc.set_rigid_body_disable_gravity(left, True)
    dc.set_rigid_body_disable_gravity(right, True)
    set_parallel_gripper_gap(stage, gripper, initial_gap, initial_gap, 2.5)
    palm_sensor.initialize()
    left_sensor.initialize()
    right_sensor.initialize()
    if ground_sensor is not None:
        ground_sensor.initialize()
    half = np.asarray(dimensions) * 0.5
    object_min = np.asarray(minimum)
    object_max = np.asarray(maximum)
    approach_collision_free = all(
        aabb_overlap_depth(center - half, center + half, object_min, object_max) == 0
        for center in starts
    )
    orientation = gripper_orientation

    def record_frame():
        if recorder is not None:
            recorder.capture()

    approach_contacts = []
    approach_palm_contacts = []
    approach_contact_parts = set()
    approach_contact_bodies = set()
    approach_collision = False
    approach_trigger = None
    approach_completion_fraction = 0.0
    approach_first_contact_step = None
    approach_motion_before_contact = 0.0
    approach_max_penetration = 0.0
    approach_penetration_tolerance = float(settings["penetration_max_m"])
    approach_motion_tolerance = max(approach_penetration_tolerance, diagonal * 0.005)
    approach_steps = duration_steps(settings["grasp_approach_seconds"], settings["dt"])
    spawn_palm = np.asarray(gripper["palm_start"], dtype=float)
    target_start_pose = dc.get_rigid_body_pose(target_body).p
    target_start = np.asarray((target_start_pose.x, target_start_pose.y, target_start_pose.z))
    final_translation = np.asarray(candidate["center"], dtype=float) - pregrasp_center
    approach_contact_stable_steps = 0
    approach_contact_required_steps = max(3, duration_steps(0.1, settings["dt"]))
    approach_extension_budget = 0.0
    approach_extension_steps = 0
    approach_translation = np.zeros(3, dtype=float)
    approach_speed_samples = []
    approach_phase_transition_step = None
    approach_transition_speed_before_mps = None
    approach_transition_speed_after_mps = None
    approach_command_speed_mps = 0.0
    approach_cruise_speed_mps = 0.0
    approach_terminal_speed_mps = 0.0
    approach_deceleration_start_step = None
    approach_contact_stop = False
    approach_contact_stop_position = None
    approach_contact_stop_penetration_m = 0.0
    approach_contact_stop_object_motion_m = 0.0
    contact_hold = None
    distance = float(np.linalg.norm(final_translation))
    cruise_fraction = 0.70
    terminal_speed_ratio = 0.15
    profile_weights = []
    for index in range(max(1, approach_steps)):
        phase = (index + 1) / max(1, approach_steps)
        if phase <= cruise_fraction:
            weight = 1.0
        else:
            decel_phase = min(1.0, (phase - cruise_fraction) / (1.0 - cruise_fraction))
            smooth = decel_phase * decel_phase * (3.0 - 2.0 * decel_phase)
            weight = 1.0 - (1.0 - terminal_speed_ratio) * smooth
        profile_weights.append(weight)
    profile_total = max(sum(profile_weights), 1e-12)
    for step in range(approach_steps):
        alpha = sum(profile_weights[:step + 1]) / profile_total
        approach_command_speed_mps = distance * profile_weights[step] / profile_total / max(float(settings["dt"]), 1e-12)
        approach_cruise_speed_mps = distance / profile_total / max(float(settings["dt"]), 1e-12)
        approach_terminal_speed_mps = approach_cruise_speed_mps * terminal_speed_ratio
        if approach_deceleration_start_step is None and (step + 1) / max(1, approach_steps) > cruise_fraction:
            approach_deceleration_start_step = step
        approach_speed_samples.append(approach_command_speed_mps)
        translation = final_translation * alpha
        for handle, position in (
            (palm, spawn_palm + translation),
            (left, starts[0] + translation),
            (right, starts[1] + translation),
        ):
            set_body_pose(dc, handle, position, orientation)
            dc.set_rigid_body_linear_velocity(handle, (0.0, 0.0, 0.0))
            dc.set_rigid_body_angular_velocity(handle, (0.0, 0.0, 0.0))
        app.update()
        record_frame()
        blocked = handle_blocked("grasp_approach_after_update")
        if blocked:
            return blocked
        palm_frame = contact_frame(palm_sensor, check["rigid_bodies"])
        left_frame = contact_frame(left_sensor, check["rigid_bodies"])
        right_frame = contact_frame(right_sensor, check["rigid_bodies"])
        approach_contacts.append((left_frame["contact"], right_frame["contact"]))
        approach_palm_contacts.append(palm_frame["contact"])
        bilateral_target_contact = bool(left_frame["contact"] and right_frame["contact"])
        approach_contact_stable_steps = (
            approach_contact_stable_steps + 1 if bilateral_target_contact else 0
        )
        for name, frame in (
            ("palm", palm_frame),
            ("left_finger", left_frame),
            ("right_finger", right_frame),
        ):
            if frame["contact"]:
                approach_contact_parts.add(name)
            for contact in frame["contacts"]:
                approach_contact_bodies.update(
                    (str(contact.get("body0", "")), str(contact.get("body1", "")))
                )
        approach_max_penetration = max(
            approach_max_penetration,
            *(dynamic_contact_penetration(frame) for frame in (palm_frame, left_frame, right_frame)),
        )
        current_target_pose = dc.get_rigid_body_pose(target_body).p
        current_target = np.asarray(
            (current_target_pose.x, current_target_pose.y, current_target_pose.z)
        )
        current_object_motion = float(np.linalg.norm(current_target - target_start))
        target_contacts = tuple(
            contact_target_path(frame, check["rigid_bodies"])
            for frame in (palm_frame, left_frame, right_frame)
        )
        any_contact = any(path is not None for path in target_contacts)
        if any_contact and approach_first_contact_step is None:
            approach_first_contact_step = step
            approach_contact_stop = True
            approach_trigger = "contact_stop"
            approach_translation = translation.copy()
            approach_contact_stop_position = np.asarray((
                dc.get_rigid_body_pose(palm).p.x,
                dc.get_rigid_body_pose(palm).p.y,
                dc.get_rigid_body_pose(palm).p.z,
            ))
            approach_contact_stop_penetration_m = approach_max_penetration
            approach_contact_stop_object_motion_m = current_object_motion
            approach_collision = grasp_approach_collision(
                approach_max_penetration,
                current_object_motion,
                approach_penetration_tolerance,
                approach_motion_tolerance,
            )
            if approach_collision:
                approach_trigger = "penetration"
            dc.set_rigid_body_linear_velocity(palm, (0.0, 0.0, 0.0))
            dc.set_rigid_body_angular_velocity(palm, (0.0, 0.0, 0.0))
            contact_hold = create_world_locked_palm_hold(
                stage,
                palm_path,
                approach_contact_stop_position,
                (dc.get_rigid_body_pose(palm).r.x, dc.get_rigid_body_pose(palm).r.y,
                 dc.get_rigid_body_pose(palm).r.z, dc.get_rigid_body_pose(palm).r.w),
            )
            break
        if approach_first_contact_step is None:
            approach_motion_before_contact = max(
                approach_motion_before_contact,
                current_object_motion,
            )
        if grasp_approach_collision(
            approach_max_penetration,
            current_object_motion,
            approach_penetration_tolerance,
            approach_motion_tolerance,
        ):
            approach_collision = True
            approach_completion_fraction = alpha
            approach_trigger = "penetration"
            break
        approach_completion_fraction = alpha

    approach_phase_transition_step = approach_first_contact_step
    approach_transition_speed_before_mps = (
        approach_speed_samples[-1] if approach_speed_samples else 0.0
    )
    approach_transition_speed_after_mps = 0.0 if approach_contact_stop else None
    if not approach_collision and not approach_contact_stop:
        approach_trigger = "target_center_reached"

    target_arrival_pose = dc.get_rigid_body_pose(target_body).p
    target_arrival = np.asarray(
        (target_arrival_pose.x, target_arrival_pose.y, target_arrival_pose.z)
    )
    approach_object_motion = float(np.linalg.norm(target_arrival - target_start))
    approach_collision |= grasp_approach_collision(
        approach_max_penetration,
        approach_object_motion,
        approach_penetration_tolerance,
        approach_motion_tolerance,
    )
    palm_arrival_pose = dc.get_rigid_body_pose(palm)
    palm_arrival_error = pose_arrival_error(
        frame_poses["commanded_palm_pose_world"][:3, 3],
        gripper_orientation,
        (palm_arrival_pose.p.x, palm_arrival_pose.p.y, palm_arrival_pose.p.z),
        (palm_arrival_pose.r.x, palm_arrival_pose.r.y, palm_arrival_pose.r.z, palm_arrival_pose.r.w),
    )
    left_arrival_pose = dc.get_rigid_body_pose(left).p
    right_arrival_pose = dc.get_rigid_body_pose(right).p
    arrival_finger_centers = (
        np.asarray((left_arrival_pose.x, left_arrival_pose.y, left_arrival_pose.z)),
        np.asarray((right_arrival_pose.x, right_arrival_pose.y, right_arrival_pose.z)),
    )
    expected_arrival_finger_centers = finger_centers(candidate["center"], initial_gap)
    arrival_finger_center_error = max(
        float(np.linalg.norm(actual - expected))
        for actual, expected in zip(arrival_finger_centers, expected_arrival_finger_centers)
    )
    if approach_collision:
        timeline.stop()
        return {
            "applicable": True,
            "pass": False,
            "reason": "approach_collision",
            "candidate_source": candidate["source"],
            "gripper_geometry": gripper_geometry,
            "approach_collision_free": False,
            "approach_collision_free_aabb_diagnostic_only": approach_collision_free,
            "approach_contact_fraction": sum(
                palm_contact or left or right
                for palm_contact, (left, right) in zip(approach_palm_contacts, approach_contacts)
            ) / max(1, len(approach_contacts)),
            "approach_completion_fraction": approach_completion_fraction,
            "approach_trigger": approach_trigger,
            "approach_contact_parts": sorted(approach_contact_parts),
            "approach_contact_bodies": sorted(approach_contact_bodies),
            "approach_first_contact_step": approach_first_contact_step,
            "approach_contact_stable_steps": approach_contact_stable_steps,
            "approach_contact_required_steps": approach_contact_required_steps,
            "approach_extension_budget_m": approach_extension_budget,
            "approach_extension_applied_m": 0.0,
            "approach_command_speed_mps": approach_command_speed_mps,
            "approach_speed_samples_mps": approach_speed_samples,
            "approach_speed_min_mps": min(approach_speed_samples) if approach_speed_samples else 0.0,
            "approach_speed_max_mps": max(approach_speed_samples) if approach_speed_samples else 0.0,
            "approach_speed_variation_mps": (
                max(approach_speed_samples) - min(approach_speed_samples)
                if approach_speed_samples else 0.0
            ),
            "approach_cruise_speed_mps": approach_cruise_speed_mps,
            "approach_terminal_speed_mps": approach_terminal_speed_mps,
            "approach_deceleration_start_step": approach_deceleration_start_step,
            "approach_phase_transition_step": approach_phase_transition_step,
            "approach_transition_speed_before_mps": approach_transition_speed_before_mps,
            "approach_transition_speed_after_mps": approach_transition_speed_after_mps,
            "approach_contact_stop_position_world": (
                approach_contact_stop_position.tolist()
                if approach_contact_stop_position is not None else None
            ),
            "approach_contact_stop_penetration_m": approach_contact_stop_penetration_m,
            "approach_contact_stop_object_motion_m": approach_contact_stop_object_motion_m,
            "approach_command_speed_mps": approach_command_speed_mps,
            "approach_speed_samples_mps": approach_speed_samples,
            "approach_speed_min_mps": min(approach_speed_samples) if approach_speed_samples else 0.0,
            "approach_speed_max_mps": max(approach_speed_samples) if approach_speed_samples else 0.0,
            "approach_speed_variation_mps": (
                max(approach_speed_samples) - min(approach_speed_samples)
                if approach_speed_samples else 0.0
            ),
            "approach_phase_transition_step": approach_phase_transition_step,
            "approach_transition_speed_before_mps": approach_transition_speed_before_mps,
            "approach_transition_speed_after_mps": approach_transition_speed_after_mps,
            "approach_first_contact_step": approach_first_contact_step,
            "approach_motion_before_contact_m": approach_motion_before_contact,
            "approach_object_motion_m": approach_object_motion,
            "approach_object_motion_tolerance_m": approach_motion_tolerance,
            "approach_object_motion_failure_criterion": False,
            "approach_penetration_max_m": approach_max_penetration,
            "approach_penetration_tolerance_m": approach_penetration_tolerance,
            "approach_distance_m": approach_distance,
            "pregrasp_center_world": pregrasp_center.tolist(),
            "grasp_center_world": candidate["center"].tolist(),
            "grasp_approach_world": candidate["approach"].tolist(),
            "authored_closing_world": candidate.get("annotation_diagnostics", {}).get("authored_closing_world", candidate["closing"]).tolist() if hasattr(candidate.get("annotation_diagnostics", {}).get("authored_closing_world", candidate["closing"]), "tolist") else candidate.get("annotation_diagnostics", {}).get("authored_closing_world", candidate["closing"]),
            "authored_approach_world": candidate.get("annotation_diagnostics", {}).get("authored_approach_world", candidate["approach"]).tolist() if hasattr(candidate.get("annotation_diagnostics", {}).get("authored_approach_world", candidate["approach"]), "tolist") else candidate.get("annotation_diagnostics", {}).get("authored_approach_world", candidate["approach"]),
            "authored_closing_local": candidate.get("annotation_diagnostics", {}).get("authored_closing_local"),
            "authored_approach_local": candidate.get("annotation_diagnostics", {}).get("authored_approach_local"),
            "authored_attribute_frame": candidate.get("annotation_diagnostics", {}).get("authored_attribute_frame"),
            "gripper_closing_world": candidate["closing"].tolist(),
            "gripper_approach_world": gripper_approach.tolist(),
            "approach_motion_world": (final_translation / max(float(np.linalg.norm(final_translation)), 1e-8)).tolist(),
            "grasp_width_m": candidate["width"],
            "candidate_width_m": candidate.get("width"),
            "candidate_width_source": candidate.get("candidate_width_source"),
            "collision_swept_width_m": opening_geometry["collision_swept_width_m"],
            "target_binding_source": candidate.get("target_binding_source"),
            "target_binding_status": candidate.get("target_binding_status"),
            "grasp_pose_source": candidate["annotation_diagnostics"].get("grasp_pose_source"),
            "authored_grasp_prim": candidate["annotation_diagnostics"].get("authored_grasp_prim"),
        "initial_gripper_gap_m": initial_gap,
        "palm_closing_span_m": descriptor["palm_closing_span_m"],
        "rail_closing_span_m": descriptor["rail_closing_span_m"],
            "grasp_arrival_position_error_m": palm_arrival_error["position_error_m"],
            "grasp_arrival_orientation_error_rad": palm_arrival_error["orientation_error_rad"],
            "candidate_validation": candidate["annotation_diagnostics"],
        }
    approach_extension_applied = 0.0
    final_translation = approach_translation
    gripper["palm_start"] = spawn_palm + final_translation
    gripper["contact_contract"] = {
        key: value.tolist() if hasattr(value, "tolist") else value
        for key, value in parallel_gripper_contact_contract(
            candidate["center"],
            candidate["closing"],
            gripper_approach,
            descriptor,
            initial_gap,
        ).items()
    }
    gripper["contact_contract"]["material_static_friction"] = gripper["contact_region_static_friction"]

    def move_pair(start_center, end_center, start_gap, end_gap, seconds):
        contacts = []

        def move(step, steps):
            alpha = (step + 1) / steps
            center = start_center * (1 - alpha) + end_center * alpha
            gap = start_gap * (1 - alpha) + end_gap * alpha
            left_center, right_center = finger_centers(center, gap)
            set_body_pose(dc, left, left_center, orientation)
            set_body_pose(dc, right, right_center, orientation)
            contacts.append(
                (
                    contact_frame(left_sensor, check["rigid_bodies"])["contact"],
                    contact_frame(right_sensor, check["rigid_bodies"])["contact"],
                )
            )

        run_steps(app, seconds, move)
        return contacts

    close_contacts = []
    left_closed, right_closed = finger_centers(candidate["center"], initial_gap)
    close_initial_pose = {
        "palm": rigid_body_pose_dict(dc.get_rigid_body_pose(palm)),
        "left": rigid_body_pose_dict(dc.get_rigid_body_pose(left)),
        "right": rigid_body_pose_dict(dc.get_rigid_body_pose(right)),
        "target": rigid_body_pose_dict(dc.get_rigid_body_pose(target_body)),
    }
    contact_sensor_ready_at_close_start = bool(left_sensor is not None and right_sensor is not None)
    close_first_target_contact_step = None
    left_contact_step = right_contact_step = None
    left_contact_path = right_contact_path = None
    bound_target_paths = None
    left_any_contact = right_any_contact = False
    observed_contact_bodies = set()
    close_steps = max(
        1,
        int(float(settings["grasp_close_timeout_seconds"]) / float(settings["dt"])),
    )
    closing_speed = (initial_gap - final_gap) * 0.5 / float(settings["grasp_close_seconds"])
    object_mass = 0.0
    for path in check["rigid_bodies"]:
        handle = dc.get_rigid_body(path)
        if not handle:
            continue
        mass, blocked = rigid_body_mass_or_blocked(
            dc, handle, path, "grasp_object_mass"
        )
        if blocked:
            return blocked
        object_mass += mass
    if not math.isfinite(object_mass) or object_mass <= 0:
        timeline.stop()
        return {"applicable": False, "pass": False, "status": "evaluation_blocked",
                "reason": "grasp_object_mass_unavailable", "reason_class": "runtime"}
    fallback_friction = float(settings["grasp_default_static_friction"])
    gripper_friction = float(settings["grasp_gripper_static_friction"])
    left_gripper_friction = right_gripper_friction = gripper_friction
    left_object_friction = right_object_friction = fallback_friction
    left_friction_source = right_friction_source = "fallback"
    force_stable_steps = duration_steps(
        settings["grasp_force_stable_seconds"],
        settings["dt"],
    )
    force_samples = []
    close_force_windows = []
    close_gap_trace = []
    close_trace_stride = max(1, round(0.1 / float(settings["dt"])))
    force_threshold_reached = False
    bilateral_contact_ready = False
    bilateral_contact_stable_steps = 0
    bilateral_contact_window = []
    bilateral_contact_max_consecutive_steps = 0
    bilateral_contact_current_consecutive_steps = 0
    last_left_contact = last_right_contact = False
    force_target_n = None
    left_force_command_n = right_force_command_n = 0.0
    max_left_force_command_n = max_right_force_command_n = 0.0
    max_left_force_n = max_right_force_n = 0.0
    bilateral_contact_step = None
    first_target_raw_contacts = {}
    first_target_contact_summaries = {}
    first_target_pad_contacts = {}
    first_target_contact_regions = {}
    force_observation_statuses = {"left": set(), "right": set()}
    force_ramp_steps = max(
        1,
        int(float(settings["grasp_force_ramp_timeout_seconds"]) / float(settings["dt"])),
    )
    desired_gap = initial_gap
    independent_targets = {"left": 0.0, "right": 0.0}
    left_target_contact_step = right_target_contact_step = None
    close_bilateral_contact_step = None
    close_bilateral_target_stop = False
    left_target_frozen_after_contact = right_target_frozen_after_contact = False
    unilateral_progress_steps = unilateral_stall_steps = 0
    previous_measured_gap = initial_gap
    close_drive_force_limited = False
    for step in range(close_steps):
        blocked = handle_blocked("grasp_close_before_update")
        if blocked:
            return blocked
        step_travel = float(closing_speed) * float(settings["dt"])
        if not close_bilateral_target_stop:
            if not left_target_frozen_after_contact:
                independent_targets["left"] = min(
                    max(0.0, (initial_gap - final_gap) * 0.5),
                    independent_targets["left"] + step_travel,
                )
            if not right_target_frozen_after_contact:
                independent_targets["right"] = max(
                    -max(0.0, (initial_gap - final_gap) * 0.5),
                    independent_targets["right"] - step_travel,
                )
        desired_gap = initial_gap - independent_targets["left"] + independent_targets["right"]
        set_parallel_gripper_targets(stage, gripper, independent_targets, left_force_command_n or 2.5)
        app.update()
        record_frame()
        blocked = handle_blocked("grasp_close_after_update")
        if blocked:
            return blocked
        left_pose = dc.get_rigid_body_pose(left).p
        right_pose = dc.get_rigid_body_pose(right).p
        left_closed = np.asarray((left_pose.x, left_pose.y, left_pose.z))
        right_closed = np.asarray((right_pose.x, right_pose.y, right_pose.z))
        left_frame = contact_frame(left_sensor, close_target_paths)
        right_frame = contact_frame(right_sensor, close_target_paths)
        left_contact = left_frame["contact"]
        right_contact = right_frame["contact"]
        last_left_contact, last_right_contact = left_contact, right_contact
        left_any_contact |= left_frame["contact"]
        right_any_contact |= right_frame["contact"]
        for frame in (left_frame, right_frame):
            for contact in frame["contacts"]:
                observed_contact_bodies.update(
                    (str(contact.get("body0", "")), str(contact.get("body1", "")))
                )
        current_left_path = contact_target_path(left_frame, close_target_paths)
        current_right_path = contact_target_path(right_frame, close_target_paths)
        if (current_left_path or current_right_path) and contact_hold is None:
            palm_pose_at_contact = dc.get_rigid_body_pose(palm)
            contact_hold_position = np.asarray((
                palm_pose_at_contact.p.x,
                palm_pose_at_contact.p.y,
                palm_pose_at_contact.p.z,
            ))
            dc.set_rigid_body_linear_velocity(palm, (0.0, 0.0, 0.0))
            dc.set_rigid_body_angular_velocity(palm, (0.0, 0.0, 0.0))
            contact_hold = create_world_locked_palm_hold(
                stage,
                palm_path,
                contact_hold_position,
                (palm_pose_at_contact.r.x, palm_pose_at_contact.r.y,
                 palm_pose_at_contact.r.z, palm_pose_at_contact.r.w),
            )
            approach_contact_stop_position = contact_hold_position
            approach_contact_stop_penetration_m = max(
                approach_contact_stop_penetration_m,
                float(left_frame.get("penetration") or 0.0),
                float(right_frame.get("penetration") or 0.0),
            )
        for side, frame, target_path_value in (
            ("left", left_frame, current_left_path),
            ("right", right_frame, current_right_path),
        ):
            if target_path_value and side not in first_target_raw_contacts:
                first_target_raw_contacts[side] = next(
                    (
                        contact
                        for contact in frame["contacts"]
                        if target_path_value in (
                            str(contact.get("body0", "")),
                            str(contact.get("body1", "")),
                        )
                    ),
                    None,
                )
                first_target_contact_summaries[side] = {
                    "count": frame["count"],
                    "force": frame["force"],
                    "contact_source": frame["contact_source"],
                }
                first_target_pad_contacts[side] = gripper_pad_contact_diagnostics(
                    frame["contacts"], gripper["contact_contract"], side
                )
                position = first_target_raw_contacts[side].get("position") if first_target_raw_contacts[side] else None
                if position is not None:
                    first_target_contact_regions[side] = classify_finger_contact_region(
                        position, gripper["contact_contract"], side
                    )
                    region_friction = first_target_contact_regions[side].get("contact_static_friction")
                    if region_friction is not None:
                        if side == "left":
                            left_gripper_friction = float(region_friction)
                        else:
                            right_gripper_friction = float(region_friction)
        if left_contact and left_contact_step is None:
            left_contact_step = step
        if right_contact and right_contact_step is None:
            right_contact_step = step
        if current_left_path:
            left_contact_path = current_left_path
            if left_friction_source == "fallback":
                left_object_friction, left_friction_source = bound_static_friction(
                    stage, current_left_path, fallback_friction
                )
        if current_right_path:
            right_contact_path = current_right_path
            if right_friction_source == "fallback":
                right_object_friction, right_friction_source = bound_static_friction(
                    stage, current_right_path, fallback_friction
                )
        # Fingers commonly touch the same object on adjacent physical steps.
        # Bind each observed target path independently, then enable bilateral
        # force/stability checks once both sides have a target binding.
        if bound_target_paths is None and left_contact_path and right_contact_path:
            bound_target_paths = (left_contact_path, right_contact_path)
        if (current_left_path or current_right_path) and close_first_target_contact_step is None:
            close_first_target_contact_step = step
        if bound_target_paths is not None:
            if bilateral_contact_step is None:
                bilateral_contact_step = step
            left_effective_friction = min(left_object_friction, left_gripper_friction)
            right_effective_friction = min(right_object_friction, right_gripper_friction)
            force_target_n = grasp_force_target(
                object_mass,
                settings["gravity"],
                left_effective_friction,
                right_effective_friction,
                settings["grasp_force_safety_factor"],
            )
            if left_force_command_n == 0.0:
                left_force_command_n = right_force_command_n = force_target_n
            left_force = raw_normal_contact_force(
                left_frame["contacts"], bound_target_paths or close_target_paths, settings["dt"]
            )
            right_force = raw_normal_contact_force(
                right_frame["contacts"], bound_target_paths or close_target_paths, settings["dt"]
            )
            for side, frame in (("left", left_frame), ("right", right_frame)):
                force_observation_statuses[side].add(
                    contact_force_observation(
                        frame["contacts"], bound_target_paths or close_target_paths, settings["dt"]
                    )["force_observation_status"]
                )
            max_left_force_n = max(max_left_force_n, left_force)
            max_right_force_n = max(max_right_force_n, right_force)
            force_samples.append((left_force, right_force))
            if len(force_samples) > force_stable_steps:
                force_samples.pop(0)
            force_window = bilateral_force_window(force_samples, force_target_n)
            if len(force_samples) == force_stable_steps and (
                step % force_stable_steps == force_stable_steps - 1
            ):
                close_force_windows.append({"step": step, **force_window})
            command_step = float(settings["grasp_force_command_step_ratio"]) * force_target_n
            command_max = float(settings["grasp_force_command_max_ratio"]) * force_target_n
            if not force_window["force_ready"]:
                shared_command = min(
                    command_max,
                    max(left_force_command_n, right_force_command_n) + command_step,
                )
                left_force_command_n = right_force_command_n = shared_command
            force_threshold_reached = bool(
                len(force_samples) == force_stable_steps and force_window["force_ready"]
            )
            bilateral_sample = bool(current_left_path and current_right_path)
            bilateral_contact_window.append(bilateral_sample)
            if len(bilateral_contact_window) > force_stable_steps:
                bilateral_contact_window.pop(0)
            if bilateral_sample:
                bilateral_contact_stable_steps += 1
                bilateral_contact_current_consecutive_steps += 1
                bilateral_contact_max_consecutive_steps = max(
                    bilateral_contact_max_consecutive_steps,
                    bilateral_contact_current_consecutive_steps,
                )
            else:
                bilateral_contact_stable_steps = 0
                bilateral_contact_current_consecutive_steps = 0
            required_bilateral_samples = max(
                1,
                math.ceil(
                    float(settings["grasp_bilateral_contact_fraction_min"])
                    * len(bilateral_contact_window)
                ),
            )
            bilateral_contact_ready = bool(
                len(bilateral_contact_window) == force_stable_steps
                and sum(bilateral_contact_window) >= required_bilateral_samples
            )
            max_left_force_command_n = max(max_left_force_command_n, left_force_command_n)
            max_right_force_command_n = max(max_right_force_command_n, right_force_command_n)
        measured_gap = float(np.linalg.norm(right_closed - left_closed) - finger_thickness)
        if int(last_left_contact) + int(last_right_contact) == 1:
            unilateral_progress_steps += 1
            if measured_gap >= previous_measured_gap - 1e-7:
                unilateral_stall_steps += 1
        previous_measured_gap = measured_gap
        if step % close_trace_stride == 0 and len(close_gap_trace) < 256:
            finger_center_distance = float(np.linalg.norm(right_closed - left_closed))
            close_gap_trace.append({
                "step": step,
                "elapsed_seconds": (step + 1) * float(settings["dt"]),
                "finger_center_distance_m": finger_center_distance,
                "reported_gripper_gap_m": max(0.0, finger_center_distance - finger_thickness),
                "left_target_contact": bool(current_left_path),
                "right_target_contact": bool(current_right_path),
                "left_target_m": independent_targets["left"],
                "right_target_m": independent_targets["right"],
                "left_target_frozen": left_target_frozen_after_contact,
                "right_target_frozen": right_target_frozen_after_contact,
            })
        close_contacts.append((left_contact, right_contact))
        if current_left_path and left_target_contact_step is None:
            left_target_contact_step = step
            left_target_frozen_after_contact = True
        if current_right_path and right_target_contact_step is None:
            right_target_contact_step = step
            right_target_frozen_after_contact = True
        if current_left_path and current_right_path and close_bilateral_contact_step is None:
            close_bilateral_contact_step = step
            close_bilateral_target_stop = True
        if bilateral_contact_ready:
            break
        # A force window can take longer than the authored close trajectory.
        # Keep closing until the existing close timeout instead of leaving the
        # jaws half-open when the ramp window expires.
    seating_contacts = []
    seating_force_samples = []
    seating_steps = (
        duration_steps(settings["grasp_seating_seconds"], settings["dt"])
        if bilateral_contact_ready and bound_target_paths is not None
        else 0
    )
    for _ in range(seating_steps):
        blocked = handle_blocked("grasp_seating_before_update")
        if blocked:
            return blocked
        set_parallel_gripper_gap(
            stage, gripper, initial_gap, final_gap, left_force_command_n or 2.5
        )
        app.update()
        record_frame()
        blocked = handle_blocked("grasp_seating_after_update")
        if blocked:
            return blocked
        left_frame = contact_frame(left_sensor, bound_target_paths or close_target_paths)
        right_frame = contact_frame(right_sensor, bound_target_paths or close_target_paths)
        seating_contacts.append((
            bool(contact_target_path(left_frame, bound_target_paths or close_target_paths)),
            bool(contact_target_path(right_frame, bound_target_paths or close_target_paths)),
        ))
        seating_force_samples.append((
            raw_normal_contact_force(left_frame["contacts"], list(bound_target_paths), settings["dt"]),
            raw_normal_contact_force(right_frame["contacts"], list(bound_target_paths), settings["dt"]),
        ))
    seating_bilateral_contact_max_consecutive_steps = 0
    seating_bilateral_contact_current_consecutive_steps = 0
    for left_contact_observed, right_contact_observed in seating_contacts:
        if left_contact_observed and right_contact_observed:
            seating_bilateral_contact_current_consecutive_steps += 1
            seating_bilateral_contact_max_consecutive_steps = max(
                seating_bilateral_contact_max_consecutive_steps,
                seating_bilateral_contact_current_consecutive_steps,
            )
        else:
            seating_bilateral_contact_current_consecutive_steps = 0
    seating_force_fraction, seating_force_windows = bilateral_force_window_fraction(
        seating_force_samples,
        force_target_n,
        force_stable_steps,
    )
    seating_contact_fraction = bilateral_contact_fraction_from_pairs(seating_contacts, force_stable_steps)
    seating_pass = seating_contact_fraction >= float(settings["grasp_bilateral_contact_fraction_min"])
    bilateral_close_contact = (
        bilateral_contact_ready
        and bound_target_paths is not None
        and seating_pass
    )
    same_rigid_body_binding = bool(
        bound_target_paths and bound_target_paths[0] == bound_target_paths[1]
    )
    if same_rigid_body_binding:
        target_path = bound_target_paths[0]
        target_body = dc.get_rigid_body(target_path)
    # Keep both fingers as dynamic bodies after closing. Switching them to
    # kinematic bodies makes the hold contact report sparse and removes the
    # physical force path that the metric is meant to measure.
    if bilateral_close_contact:
        app.update()
        record_frame()
    return {
        "state": {
            "approach_collision_free": approach_collision_free,
            "approach_command_speed_mps": approach_command_speed_mps,
            "approach_completion_fraction": approach_completion_fraction,
            "approach_contact_bodies": approach_contact_bodies,
            "approach_contact_parts": approach_contact_parts,
            "approach_contact_required_steps": approach_contact_required_steps,
            "approach_contact_stable_steps": approach_contact_stable_steps,
            "approach_contact_stop": approach_contact_stop,
            "approach_contact_stop_object_motion_m": approach_contact_stop_object_motion_m,
            "approach_contact_stop_penetration_m": approach_contact_stop_penetration_m,
            "approach_contact_stop_position": approach_contact_stop_position,
            "approach_contacts": approach_contacts,
            "approach_cruise_speed_mps": approach_cruise_speed_mps,
            "approach_deceleration_start_step": approach_deceleration_start_step,
            "approach_distance": approach_distance,
            "approach_extension_applied": approach_extension_applied,
            "approach_extension_budget": approach_extension_budget,
            "approach_first_contact_step": approach_first_contact_step,
            "approach_max_penetration": approach_max_penetration,
            "approach_motion_tolerance": approach_motion_tolerance,
            "approach_object_motion": approach_object_motion,
            "approach_palm_contacts": approach_palm_contacts,
            "approach_penetration_tolerance": approach_penetration_tolerance,
            "approach_phase_transition_step": approach_phase_transition_step,
            "approach_speed_samples": approach_speed_samples,
            "approach_terminal_speed_mps": approach_terminal_speed_mps,
            "approach_transition_speed_after_mps": approach_transition_speed_after_mps,
            "approach_transition_speed_before_mps": approach_transition_speed_before_mps,
            "approach_trigger": approach_trigger,
            "arrival_finger_center_error": arrival_finger_center_error,
            "arrival_finger_centers": arrival_finger_centers,
            "bilateral_close_contact": bilateral_close_contact,
            "bilateral_contact_max_consecutive_steps": bilateral_contact_max_consecutive_steps,
            "bilateral_contact_ready": bilateral_contact_ready,
            "bilateral_contact_stable_steps": bilateral_contact_stable_steps,
            "bound_target_paths": bound_target_paths,
            "candidate": candidate,
            "close_bilateral_contact_step": close_bilateral_contact_step,
            "close_bilateral_target_stop": close_bilateral_target_stop,
            "close_drive_force_limited": close_drive_force_limited,
            "close_first_target_contact_step": close_first_target_contact_step,
            "close_force_windows": close_force_windows,
            "close_gap_trace": close_gap_trace,
            "close_initial_pose": close_initial_pose,
            "close_target_paths": close_target_paths,
            "collision_union": collision_union,
            "contact_hold": contact_hold,
            "contact_sensor_ready_at_close_start": contact_sensor_ready_at_close_start,
            "dc": dc,
            "descriptor": descriptor,
            "desired_gap": desired_gap,
            "diagonal": diagonal,
            "final_gap": final_gap,
            "final_translation": final_translation,
            "finger_thickness": finger_thickness,
            "first_target_contact_regions": first_target_contact_regions,
            "first_target_contact_summaries": first_target_contact_summaries,
            "first_target_pad_contacts": first_target_pad_contacts,
            "first_target_raw_contacts": first_target_raw_contacts,
            "force_observation_statuses": force_observation_statuses,
            "force_stable_steps": force_stable_steps,
            "force_target_n": force_target_n,
            "force_threshold_reached": force_threshold_reached,
            "frame_poses": frame_poses,
            "gripper": gripper,
            "gripper_approach": gripper_approach,
            "gripper_friction": gripper_friction,
            "gripper_geometry": gripper_geometry,
            "ground_sensor": ground_sensor,
            "handle_blocked": handle_blocked,
            "independent_targets": independent_targets,
            "initial_gap": initial_gap,
            "left": left,
            "left_any_contact": left_any_contact,
            "left_closed": left_closed,
            "left_contact_path": left_contact_path,
            "left_contact_step": left_contact_step,
            "left_force_command_n": left_force_command_n,
            "left_frame": left_frame,
            "left_friction_source": left_friction_source,
            "left_gripper_friction": left_gripper_friction,
            "left_object_friction": left_object_friction,
            "left_path": left_path,
            "left_sensor": left_sensor,
            "left_target_contact_step": left_target_contact_step,
            "left_target_frozen_after_contact": left_target_frozen_after_contact,
            "max_left_force_command_n": max_left_force_command_n,
            "max_left_force_n": max_left_force_n,
            "max_right_force_command_n": max_right_force_command_n,
            "max_right_force_n": max_right_force_n,
            "maximum": maximum,
            "minimum": minimum,
            "object_mass": object_mass,
            "observed_contact_bodies": observed_contact_bodies,
            "opening_geometry": opening_geometry,
            "orientation": orientation,
            "palm": palm,
            "palm_arrival_error": palm_arrival_error,
            "palm_path": palm_path,
            "pregrasp_center": pregrasp_center,
            "previous_measured_gap": previous_measured_gap,
            "record_frame": record_frame,
            "right": right,
            "right_any_contact": right_any_contact,
            "right_closed": right_closed,
            "right_contact_path": right_contact_path,
            "right_contact_step": right_contact_step,
            "right_force_command_n": right_force_command_n,
            "right_frame": right_frame,
            "right_friction_source": right_friction_source,
            "right_gripper_friction": right_gripper_friction,
            "right_object_friction": right_object_friction,
            "right_path": right_path,
            "right_sensor": right_sensor,
            "right_target_contact_step": right_target_contact_step,
            "right_target_frozen_after_contact": right_target_frozen_after_contact,
            "same_rigid_body_binding": same_rigid_body_binding,
            "seating_bilateral_contact_max_consecutive_steps": seating_bilateral_contact_max_consecutive_steps,
            "seating_contact_fraction": seating_contact_fraction,
            "seating_contacts": seating_contacts,
            "seating_force_samples": seating_force_samples,
            "seating_force_windows": seating_force_windows,
            "seating_pass": seating_pass,
            "settings": settings,
            "starts": starts,
            "target_body": target_body,
            "target_path": target_path,
            "timeline": timeline,
            "unilateral_progress_steps": unilateral_progress_steps,
            "unilateral_stall_steps": unilateral_stall_steps,
        },
    }


def acquire_grasp(stage, check, app, dataset, asset, recorder=None, target_paths=None, prepared_timeline=None):
    """Run the same acquisition controller before lift or joint actuation."""
    result = _acquire_grasp_once(stage, check, app, dataset, asset, recorder, target_paths, prepared_timeline)
    state = result.get("state")
    diagnostic = {
        "controller_version": "shared_grasp_acquisition_r26",
        "candidate_rank": int(asset.get("_grasp_candidate_rank", 0)),
        "candidate_prim_path": None,
        "target_paths": list(target_paths) if target_paths is not None else None,
        "pass": False,
        "failure_phase": result.get("failure_phase") or result.get("diagnostics", {}).get("failure_phase") or result.get("reason"),
    }
    if state is not None:
        candidate = state["candidate"]
        diagnostic.update({
            "candidate_prim_path": candidate["annotation_diagnostics"].get("authored_grasp_prim"),
            "target_rigid_body": candidate["target_rigid_body"],
            "target_paths": state["close_target_paths"],
            "approach_world": candidate["approach"].tolist(),
            "closing_world": candidate["closing"].tolist(),
            "width_m": float(candidate["width"]),
            "width_source": candidate.get("candidate_width_source"),
            "opening": state["opening_geometry"],
            "approach_collision_free": state["approach_collision_free"],
            "bilateral_contact_ready": state["bilateral_contact_ready"],
            "seating_contact_fraction": state["seating_contact_fraction"],
            "seating_pass": state["seating_pass"],
            "force_threshold_reached_diagnostic": state["force_threshold_reached"],
            "close_force_windows": state["close_force_windows"],
            "seating_force_windows": state["seating_force_windows"],
            "independent_targets": state["independent_targets"],
            "close_gap_trace": state["close_gap_trace"],
            "left_target_frozen": state["left_target_frozen_after_contact"],
            "right_target_frozen": state["right_target_frozen_after_contact"],
            "pass": bool(state["opening_geometry"]["opening_supported"] and state["bilateral_close_contact"]),
            "failure_phase": None if state["bilateral_close_contact"] else "grasp_close_or_seating",
        })
        if not state["opening_geometry"]["opening_supported"]:
            diagnostic["failure_phase"] = "grasp_opening"
    result["grasp_acquisition"] = diagnostic
    return result


def _grasp_lift_once(stage, check: dict, app, dataset: str, asset: dict, recorder=None) -> dict:
    import numpy as np
    from omni.isaac.dynamic_control import _dynamic_control

    acquisition = acquire_grasp(stage, check, app, dataset, asset, recorder)
    if "state" not in acquisition:
        return acquisition
    state = acquisition["state"]
    approach_collision_free = state["approach_collision_free"]
    approach_command_speed_mps = state["approach_command_speed_mps"]
    approach_contact_bodies = state["approach_contact_bodies"]
    approach_contact_parts = state["approach_contact_parts"]
    approach_contact_required_steps = state["approach_contact_required_steps"]
    approach_contact_stable_steps = state["approach_contact_stable_steps"]
    approach_contact_stop = state["approach_contact_stop"]
    approach_contact_stop_object_motion_m = state["approach_contact_stop_object_motion_m"]
    approach_contact_stop_penetration_m = state["approach_contact_stop_penetration_m"]
    approach_contact_stop_position = state["approach_contact_stop_position"]
    approach_contacts = state["approach_contacts"]
    approach_cruise_speed_mps = state["approach_cruise_speed_mps"]
    approach_deceleration_start_step = state["approach_deceleration_start_step"]
    approach_distance = state["approach_distance"]
    approach_extension_applied = state["approach_extension_applied"]
    approach_extension_budget = state["approach_extension_budget"]
    approach_first_contact_step = state["approach_first_contact_step"]
    approach_max_penetration = state["approach_max_penetration"]
    approach_motion_tolerance = state["approach_motion_tolerance"]
    approach_object_motion = state["approach_object_motion"]
    approach_palm_contacts = state["approach_palm_contacts"]
    approach_penetration_tolerance = state["approach_penetration_tolerance"]
    approach_phase_transition_step = state["approach_phase_transition_step"]
    approach_speed_samples = state["approach_speed_samples"]
    approach_terminal_speed_mps = state["approach_terminal_speed_mps"]
    approach_transition_speed_after_mps = state["approach_transition_speed_after_mps"]
    approach_transition_speed_before_mps = state["approach_transition_speed_before_mps"]
    approach_trigger = state["approach_trigger"]
    arrival_finger_center_error = state["arrival_finger_center_error"]
    arrival_finger_centers = state["arrival_finger_centers"]
    bilateral_close_contact = state["bilateral_close_contact"]
    bilateral_contact_max_consecutive_steps = state["bilateral_contact_max_consecutive_steps"]
    bilateral_contact_ready = state["bilateral_contact_ready"]
    bilateral_contact_stable_steps = state["bilateral_contact_stable_steps"]
    bound_target_paths = state["bound_target_paths"]
    candidate = state["candidate"]
    close_bilateral_contact_step = state["close_bilateral_contact_step"]
    close_bilateral_target_stop = state["close_bilateral_target_stop"]
    close_drive_force_limited = state["close_drive_force_limited"]
    close_first_target_contact_step = state["close_first_target_contact_step"]
    close_force_windows = state["close_force_windows"]
    close_gap_trace = state["close_gap_trace"]
    close_initial_pose = state["close_initial_pose"]
    close_target_paths = state["close_target_paths"]
    collision_union = state["collision_union"]
    contact_hold = state["contact_hold"]
    contact_sensor_ready_at_close_start = state["contact_sensor_ready_at_close_start"]
    dc = state["dc"]
    descriptor = state["descriptor"]
    desired_gap = state["desired_gap"]
    final_translation = state["final_translation"]
    finger_thickness = state["finger_thickness"]
    first_target_contact_regions = state["first_target_contact_regions"]
    first_target_contact_summaries = state["first_target_contact_summaries"]
    first_target_pad_contacts = state["first_target_pad_contacts"]
    first_target_raw_contacts = state["first_target_raw_contacts"]
    force_observation_statuses = state["force_observation_statuses"]
    force_stable_steps = state["force_stable_steps"]
    force_target_n = state["force_target_n"]
    force_threshold_reached = state["force_threshold_reached"]
    frame_poses = state["frame_poses"]
    gripper = state["gripper"]
    gripper_approach = state["gripper_approach"]
    gripper_friction = state["gripper_friction"]
    gripper_geometry = state["gripper_geometry"]
    ground_sensor = state["ground_sensor"]
    handle_blocked = state["handle_blocked"]
    independent_targets = state["independent_targets"]
    initial_gap = state["initial_gap"]
    left = state["left"]
    left_any_contact = state["left_any_contact"]
    left_closed = state["left_closed"]
    left_contact_path = state["left_contact_path"]
    left_contact_step = state["left_contact_step"]
    left_frame = state["left_frame"]
    left_friction_source = state["left_friction_source"]
    left_gripper_friction = state["left_gripper_friction"]
    left_object_friction = state["left_object_friction"]
    left_path = state["left_path"]
    left_sensor = state["left_sensor"]
    left_target_contact_step = state["left_target_contact_step"]
    left_target_frozen_after_contact = state["left_target_frozen_after_contact"]
    max_left_force_command_n = state["max_left_force_command_n"]
    max_left_force_n = state["max_left_force_n"]
    max_right_force_command_n = state["max_right_force_command_n"]
    max_right_force_n = state["max_right_force_n"]
    maximum = state["maximum"]
    minimum = state["minimum"]
    object_mass = state["object_mass"]
    observed_contact_bodies = state["observed_contact_bodies"]
    opening_geometry = state["opening_geometry"]
    orientation = state["orientation"]
    palm = state["palm"]
    palm_arrival_error = state["palm_arrival_error"]
    palm_path = state["palm_path"]
    pregrasp_center = state["pregrasp_center"]
    previous_measured_gap = state["previous_measured_gap"]
    record_frame = state["record_frame"]
    right = state["right"]
    right_any_contact = state["right_any_contact"]
    right_closed = state["right_closed"]
    right_contact_path = state["right_contact_path"]
    right_contact_step = state["right_contact_step"]
    right_frame = state["right_frame"]
    right_friction_source = state["right_friction_source"]
    right_gripper_friction = state["right_gripper_friction"]
    right_object_friction = state["right_object_friction"]
    right_path = state["right_path"]
    right_sensor = state["right_sensor"]
    right_target_contact_step = state["right_target_contact_step"]
    right_target_frozen_after_contact = state["right_target_frozen_after_contact"]
    same_rigid_body_binding = state["same_rigid_body_binding"]
    seating_bilateral_contact_max_consecutive_steps = state["seating_bilateral_contact_max_consecutive_steps"]
    seating_contact_fraction = state["seating_contact_fraction"]
    seating_contacts = state["seating_contacts"]
    seating_force_windows = state["seating_force_windows"]
    seating_pass = state["seating_pass"]
    settings = state["settings"]
    starts = state["starts"]
    target_body = state["target_body"]
    target_path = state["target_path"]
    timeline = state["timeline"]
    unilateral_progress_steps = state["unilateral_progress_steps"]
    unilateral_stall_steps = state["unilateral_stall_steps"]
    articulation, articulation_path = dynamic_articulation(dc, check)
    attachment_body = (
        dc.get_articulation_root_body(articulation)
        if articulation
        else target_body
    )
    attachment_body = attachment_body or target_body
    dofs = [
        dc.get_articulation_dof(articulation, index)
        for index in range(dc.get_articulation_dof_count(articulation))
    ] if articulation else []
    dof_names = [dc.get_dof_name(dof) for dof in dofs]
    dof_initial = np.asarray([
        float(dc.get_dof_state(dof, _dynamic_control.STATE_ALL).pos)
        for dof in dofs
    ])
    dof_max_delta = np.zeros(len(dofs))

    def observe_joint_motion():
        if dofs:
            current = np.asarray([
                float(dc.get_dof_state(dof, _dynamic_control.STATE_ALL).pos)
                for dof in dofs
            ])
            np.maximum(dof_max_delta, np.abs(current - dof_initial), out=dof_max_delta)

    root_before_pose = dc.get_rigid_body_pose(attachment_body)
    root_before = root_before_pose.p
    target_before_pose = dc.get_rigid_body_pose(target_body).p
    target_before = np.asarray((target_before_pose.x, target_before_pose.y, target_before_pose.z))
    gripper_before = (left_closed + right_closed) * 0.5
    palm_before_pose = dc.get_rigid_body_pose(palm)
    palm_before = np.asarray((palm_before_pose.p.x, palm_before_pose.p.y, palm_before_pose.p.z))
    left_before_pose = dc.get_rigid_body_pose(left).p
    right_before_pose = dc.get_rigid_body_pose(right).p
    left_before = np.asarray((left_before_pose.x, left_before_pose.y, left_before_pose.z))
    right_before = np.asarray((right_before_pose.x, right_before_pose.y, right_before_pose.z))
    # Lift diagnostics start after the seating window, not at the last close target.
    gripper_before = (left_before + right_before) * 0.5
    lift_start_joint_targets = parallel_gripper_drive_targets(stage, gripper)
    lift_start_finger_to_palm = {
        "left": (left_before - palm_before).copy(),
        "right": (right_before - palm_before).copy(),
    }
    # Lock each dynamic gripper body's arrival orientation independently.  A
    # prismatic drive constrains translation/closing, but does not guarantee
    # that the finger rigid body keeps the palm's world orientation under an
    # eccentric contact moment.
    palm_arrival_pose = dc.get_rigid_body_pose(palm)
    left_arrival_pose = dc.get_rigid_body_pose(left)
    right_arrival_pose = dc.get_rigid_body_pose(right)
    palm_target_orientation = tuple(float(value) for value in (
        palm_arrival_pose.r.x, palm_arrival_pose.r.y,
        palm_arrival_pose.r.z, palm_arrival_pose.r.w,
    ))
    left_target_orientation = tuple(float(value) for value in (
        left_arrival_pose.r.x, left_arrival_pose.r.y,
        left_arrival_pose.r.z, left_arrival_pose.r.w,
    ))
    right_target_orientation = tuple(float(value) for value in (
        right_arrival_pose.r.x, right_arrival_pose.r.y,
        right_arrival_pose.r.z, right_arrival_pose.r.w,
    ))
    required_lift = float(settings["grasp_lift_height_m"])
    controller_settings = grasp_lift_controller_settings(required_lift)
    lift_controller = controller_settings["lift_controller"]
    commanded_lift = float(controller_settings["commanded_gripper_lift_m"])
    lift = np.asarray((0.0, 0.0, commanded_lift))
    lift_contacts = []
    lift_force_samples = []
    lift_penetrations = []
    lift_seconds = max(float(settings["grasp_lift_seconds"]), float(settings["dt"]))
    lift_direction = lift / max(float(np.linalg.norm(lift)), 1e-12)
    requested_contact_torque_mode = os.environ.get("RAW_EVAL_CONTACT_TORQUE_MODE", "shadow").strip().lower()
    contact_torque_mode = requested_contact_torque_mode if requested_contact_torque_mode in {"shadow", "enabled"} else "shadow"
    try:
        orientation_pd_scale = min(200.0, max(1.0, float(os.environ.get("RAW_EVAL_ORIENTATION_PD_SCALE", "1.0"))))
    except (TypeError, ValueError):
        orientation_pd_scale = 1.0
    try:
        orientation_kd_scale = min(50.0, max(1.0, float(os.environ.get("RAW_EVAL_ORIENTATION_KD_SCALE", "1.0"))))
    except (TypeError, ValueError):
        orientation_kd_scale = 1.0
    orientation_axis_mode = os.environ.get(
        "RAW_EVAL_ORIENTATION_AXIS_MODE", "full"
    ).strip().lower()
    if orientation_axis_mode not in {"full", "dominant"}:
        orientation_axis_mode = "full"
    left_finger_mass, blocked = rigid_body_mass_or_blocked(dc, left, left_path, "grasp_lift")
    if blocked:
        timeline.stop()
        return {**blocked, "grasp_acquisition": acquisition["grasp_acquisition"]}
    right_finger_mass, blocked = rigid_body_mass_or_blocked(dc, right, right_path, "grasp_lift")
    if blocked:
        timeline.stop()
        return {**blocked, "grasp_acquisition": acquisition["grasp_acquisition"]}
    palm_mass, blocked = rigid_body_mass_or_blocked(dc, palm, palm_path, "grasp_lift")
    if blocked:
        timeline.stop()
        return {**blocked, "grasp_acquisition": acquisition["grasp_acquisition"]}
    effective_lift_mass = object_mass + left_finger_mass + right_finger_mass + palm_mass
    lift_force_limit_n = (
        float(settings["grasp_lift_force_mass_multiplier"])
        * effective_lift_mass
        * abs(float(settings["gravity"]))
        * 1.0
    )
    lift_drive = None
    d6_load_compensation = {
        "required_lift_m": required_lift,
        "object_mass_kg": object_mass,
        "gravity_mps2": abs(float(settings["gravity"])),
        "stiffness_npm": 0.0,
        "load_bias_m": 0.0,
        "applied_target_lift_m": commanded_lift,
        "bias_clamped": False,
    }
    d6_target_lift = commanded_lift
    d6_anchor_position_world = None
    d6_anchor_orientation_world = None
    d6_creation_to_first_step_delta_m = None
    d6_initial_constraint_impulse_diagnostic = "not_applicable"
    controller_status = "force_pd_active"
    controller_failure_reason = None
    if contact_hold is not None:
        stage.RemovePrim(contact_hold["path"])
    acquisition["grasp_acquisition"]["palm_hold_released_for_motion"] = True
    if lift_controller == "d6_drive":
        try:
            d6_stiffness = effective_lift_mass * float(settings["grasp_lift_servo_kp"])
            d6_load_compensation = d6_lift_load_compensation(
                commanded_lift,
                object_mass,
                settings["gravity"],
                d6_stiffness,
            )
            d6_target_lift = float(d6_load_compensation["applied_target_lift_m"])
            lift_drive = create_world_anchored_d6_lift_drive(
                stage,
                palm_path,
                palm_before,
                palm_target_orientation,
                d6_stiffness,
                effective_lift_mass * float(settings["grasp_lift_servo_kd"]),
                lift_force_limit_n,
            )
            set_world_anchored_d6_lift_target(lift_drive, 0.0, 0.0)
            d6_anchor_position_world = palm_before.tolist()
            d6_anchor_orientation_world = list(palm_target_orientation)
            d6_before_init = np.asarray((
                dc.get_rigid_body_pose(palm).p.x,
                dc.get_rigid_body_pose(palm).p.y,
                dc.get_rigid_body_pose(palm).p.z,
            ))
            app.update()
            d6_after_init = np.asarray((
                dc.get_rigid_body_pose(palm).p.x,
                dc.get_rigid_body_pose(palm).p.y,
                dc.get_rigid_body_pose(palm).p.z,
            ))
            d6_creation_to_first_step_delta_m = (d6_after_init - d6_before_init).tolist()
            d6_initial_constraint_impulse_diagnostic = "position_delta_only"
            controller_status = "d6_drive_active"
        except Exception as exc:
            controller_status = "d6_drive_unavailable"
            controller_failure_reason = f"{type(exc).__name__}: {exc}"
    palm_rotational_inertia, palm_inertia_source = rigid_body_rotational_inertia(
        stage,
        gripper["palm_path"],
        descriptor["palm_dimensions"],
        palm_mass,
    )
    palm_torque_limit_nm = lift_force_limit_n * max(descriptor["palm_dimensions"]) * 0.5
    lift_weight_feedforward_n = (
        float(settings["grasp_lift_weight_safety_factor"])
        * object_mass
        * abs(float(settings["gravity"]))
        * 0.5
    )
    max_left_lift_force_n = max_right_lift_force_n = 0.0
    max_gripper_lateral_drift_m = 0.0
    max_gripper_orientation_drift_rad = 0.0
    palm_orientation_drift_rad = 0.0
    left_finger_orientation_drift_rad = 0.0
    right_finger_orientation_drift_rad = 0.0
    translation_control_trace = []
    controller_trace = []
    lift_relative_finger_to_palm_trace = []
    lift_prismatic_target_trace = []
    motion_trace = []
    max_finger_to_palm_common_mode_drift_m = 0.0
    max_finger_to_palm_closing_drift_m = 0.0
    d6_common_mode_stabilization = lift_controller == "d6_drive"
    # This is the existing prismatic drive bound, not an additional actuator.
    d6_common_mode_force_limit_n = float(gripper["joint_drive_max_force_n"])
    orientation_control_trace = []
    orientation_control_steps = 0
    last_lift_contact_frames = ({"contacts": []}, {"contacts": []})
    last_contact_torque_observation = {
        "status": "no_target_contact",
        "source": "none",
        "contact_count": 0,
        "torque_world": [0.0, 0.0, 0.0],
        "torque_magnitude_nm": 0.0,
        "per_contact": [],
    }

    def apply_grasp_forces(desired_offset, desired_speed):
        nonlocal orientation_control_steps
        nonlocal max_left_lift_force_n, max_right_lift_force_n
        nonlocal max_gripper_lateral_drift_m, max_gripper_orientation_drift_rad
        nonlocal palm_orientation_drift_rad, left_finger_orientation_drift_rad
        nonlocal right_finger_orientation_drift_rad
        nonlocal last_lift_contact_frames, last_contact_torque_observation
        desired_offset = np.asarray(desired_offset, dtype=float)
        orientation_control_steps += 1
        # High orientation gains destabilize the loaded contact when applied
        # on the first lift step.  Ramp only the generic PD multiplier over a
        # fixed physical second; the authored gains, torque limit and contact
        # model remain unchanged.
        orientation_scale_alpha = min(
            1.0,
            orientation_control_steps * float(settings["dt"]) / 1.0,
        )
        effective_orientation_pd_scale = 1.0 + (
            orientation_pd_scale - 1.0
        ) * orientation_scale_alpha
        desired_velocity = lift_direction * float(desired_speed)
        set_parallel_gripper_drive_targets(stage, gripper, lift_start_joint_targets)
        controls = {}
        # The three gripper bodies receive the same world-space translation
        # controller.  The existing prismatic drives still exclusively set
        # jaw closure; this controller supplies no closing-axis command.
        if lift_controller == "force_pd":
            for name, handle, start in (
                ("palm", palm, palm_before),
                ("left", left, left_before),
                ("right", right, right_before),
            ):
                pose = dc.get_rigid_body_pose(handle).p
                velocity = dc.get_rigid_body_linear_velocity(handle)
                position = np.asarray((pose.x, pose.y, pose.z))
                linear_velocity = np.asarray((velocity.x, velocity.y, velocity.z))
                control = rigid_translation_force(
                    position,
                    linear_velocity,
                    np.asarray(start) + desired_offset,
                    desired_velocity,
                    float(dc.get_rigid_body_properties(handle).mass),
                    settings["gravity"],
                    settings["grasp_lift_servo_kp"],
                    settings["grasp_lift_servo_kd"],
                    lift_force_limit_n,
                )
                apply_world_body_force(dc, handle, control["force"], (pose.x, pose.y, pose.z))
                controls[name] = control
        elif lift_drive is not None:
            progress = (
                max(0.0, min(1.0, float(desired_offset[2]) / max(commanded_lift, 1e-12)))
                if commanded_lift > 0.0 else 1.0
            )
            d6_commanded_offset_z = float(desired_offset[2]) + float(d6_load_compensation["load_bias_m"]) * progress
            set_world_anchored_d6_lift_target(
                lift_drive,
                d6_commanded_offset_z,
                float(desired_velocity[2]),
            )
            if d6_common_mode_stabilization:
                palm_position = np.asarray((
                    dc.get_rigid_body_pose(palm).p.x,
                    dc.get_rigid_body_pose(palm).p.y,
                    dc.get_rigid_body_pose(palm).p.z,
                ))
                palm_velocity_value = dc.get_rigid_body_linear_velocity(palm)
                palm_velocity = np.asarray((
                    palm_velocity_value.x, palm_velocity_value.y, palm_velocity_value.z,
                ))
                finger_states = []
                for handle in (left, right):
                    pose = dc.get_rigid_body_pose(handle).p
                    velocity = dc.get_rigid_body_linear_velocity(handle)
                    finger_states.append((
                        np.asarray((pose.x, pose.y, pose.z)),
                        np.asarray((velocity.x, velocity.y, velocity.z)),
                        float(dc.get_rigid_body_properties(handle).mass),
                    ))
                common_mode = finger_common_mode_stabilization_force(
                    [position - palm_position for position, _velocity, _mass in finger_states],
                    [velocity - palm_velocity for _position, velocity, _mass in finger_states],
                    [lift_start_finger_to_palm["left"], lift_start_finger_to_palm["right"]],
                    candidate["closing"],
                    sum(mass for _position, _velocity, mass in finger_states),
                    settings["grasp_lift_servo_kp"],
                    settings["grasp_lift_servo_kd"],
                    d6_common_mode_force_limit_n,
                )
                for handle, (position, _velocity, _mass) in zip((left, right), finger_states):
                    apply_world_body_force(dc, handle, common_mode["force"], position)
                controls["finger_common_mode"] = common_mode
        palm_pose = dc.get_rigid_body_pose(palm)
        palm_angular_velocity = dc.get_rigid_body_angular_velocity(palm)
        palm_angular_velocity_world = np.asarray(
            (palm_angular_velocity.x, palm_angular_velocity.y, palm_angular_velocity.z),
            dtype=float,
        )
        palm_rotation_error = quaternion_rotation_error(
            (palm_pose.r.x, palm_pose.r.y, palm_pose.r.z, palm_pose.r.w),
            orientation,
        )
        palm_torque = (
            virtual_palm_torque(
                (palm_pose.r.x, palm_pose.r.y, palm_pose.r.z, palm_pose.r.w),
                orientation,
                palm_angular_velocity_world,
                palm_rotational_inertia,
                settings["grasp_palm_rotation_kp"],
                float(settings["grasp_palm_rotation_kd"]) * orientation_kd_scale,
                palm_torque_limit_nm,
            )
            if lift_controller == "force_pd"
            else np.zeros(3, dtype=float)
        )
        if orientation_axis_mode == "dominant":
            dominant_axis = int(np.argmax(np.abs(palm_rotation_error)))
            dominant_torque = np.zeros(3, dtype=float)
            dominant_torque[dominant_axis] = float(palm_torque[dominant_axis])
            palm_torque = dominant_torque
        palm_center_world = np.asarray((palm_pose.p.x, palm_pose.p.y, palm_pose.p.z), dtype=float)
        contact_torque = contact_torque_observation(
            list(last_lift_contact_frames[0].get("contacts", []))
            + list(last_lift_contact_frames[1].get("contacts", [])),
            list(bound_target_paths or check["rigid_bodies"]),
            palm_center_world,
            settings["dt"],
        )
        last_contact_torque_observation = contact_torque
        pd_torque = np.asarray(palm_torque, dtype=float)
        pd_torque *= effective_orientation_pd_scale
        pd_torque_magnitude = float(np.linalg.norm(pd_torque))
        if pd_torque_magnitude > palm_torque_limit_nm > 0.0:
            pd_torque *= palm_torque_limit_nm / pd_torque_magnitude
        compensation = (
            0.1 * np.asarray(contact_torque["torque_world"], dtype=float)
            if contact_torque_mode == "enabled" and contact_torque["status"] == "valid"
            else np.zeros(3, dtype=float)
        )
        requested_torque = pd_torque + compensation
        applied_torque = requested_torque.copy()
        torque_magnitude = float(np.linalg.norm(applied_torque))
        torque_clamped = torque_magnitude > palm_torque_limit_nm > 0.0
        if torque_clamped:
            applied_torque *= palm_torque_limit_nm / torque_magnitude
        if lift_controller == "force_pd":
            apply_world_body_torque(dc, palm, applied_torque)
        body_orientation_errors = {}
        for name, handle, target_orientation in (
            ("palm", palm, palm_target_orientation),
            ("left", left, left_target_orientation),
            ("right", right, right_target_orientation),
        ):
            body_pose = dc.get_rigid_body_pose(handle)
            error_vector = quaternion_rotation_error(
                (body_pose.r.x, body_pose.r.y, body_pose.r.z, body_pose.r.w),
                target_orientation,
            )
            body_orientation_errors[name] = {
                "error_vector_rad": error_vector.tolist(),
                "error_rad": float(np.linalg.norm(error_vector)),
                "orientation_xyzw": [body_pose.r.x, body_pose.r.y, body_pose.r.z, body_pose.r.w],
            }
        palm_orientation_drift_rad = max(
            palm_orientation_drift_rad,
            body_orientation_errors["palm"]["error_rad"],
        )
        left_finger_orientation_drift_rad = max(
            left_finger_orientation_drift_rad,
            body_orientation_errors["left"]["error_rad"],
        )
        right_finger_orientation_drift_rad = max(
            right_finger_orientation_drift_rad,
            body_orientation_errors["right"]["error_rad"],
        )
        max_gripper_orientation_drift_rad = max(
            palm_orientation_drift_rad,
            left_finger_orientation_drift_rad,
            right_finger_orientation_drift_rad,
        )
        left_pose = dc.get_rigid_body_pose(left).p
        right_pose = dc.get_rigid_body_pose(right).p
        positions = [
            np.asarray((left_pose.x, left_pose.y, left_pose.z)),
            np.asarray((right_pose.x, right_pose.y, right_pose.z)),
        ]
        expected_center = gripper_before + desired_offset
        center_error = (positions[0] + positions[1]) * 0.5 - expected_center
        lateral_error = center_error - lift_direction * float(np.dot(center_error, lift_direction))
        max_gripper_lateral_drift_m = max(
            max_gripper_lateral_drift_m, float(np.linalg.norm(lateral_error))
        )
        max_left_lift_force_n = max(max_left_lift_force_n, controls.get("left", {}).get("force_magnitude", 0.0))
        max_right_lift_force_n = max(max_right_lift_force_n, controls.get("right", {}).get("force_magnitude", 0.0))
        if len(translation_control_trace) < 256:
            translation_control_trace.append({
                "desired_offset_m": desired_offset.tolist(),
                "desired_velocity_mps": desired_velocity.tolist(),
                "palm_force_n": controls.get("palm", {}).get("force_magnitude", 0.0),
                "left_force_n": controls.get("left", {}).get("force_magnitude", 0.0),
                "right_force_n": controls.get("right", {}).get("force_magnitude", 0.0),
                "palm_force_clamped": controls.get("palm", {}).get("force_clamped", False),
                "left_force_clamped": controls.get("left", {}).get("force_clamped", False),
                "right_force_clamped": controls.get("right", {}).get("force_clamped", False),
            })
        if len(controller_trace) < 1024:
            palm_position = np.asarray((palm_pose.p.x, palm_pose.p.y, palm_pose.p.z))
            palm_linear_velocity = dc.get_rigid_body_linear_velocity(palm)
            controller_trace.append({
                "controller": lift_controller,
                "desired_offset_m": desired_offset.tolist(),
                "desired_velocity_mps": desired_velocity.tolist(),
                "actual_palm_offset_m": (palm_position - palm_before).tolist(),
                "d6_target_position_m": float(desired_offset[2]) if lift_drive else None,
                "d6_applied_target_position_m": (
                    float(desired_offset[2])
                    + float(d6_load_compensation["load_bias_m"])
                    * max(0.0, min(1.0, float(desired_offset[2]) / max(commanded_lift, 1e-12)))
                    if lift_drive else None
                ),
                "d6_target_velocity_mps": float(desired_velocity[2]) if lift_drive else None,
                "d6_active": bool(lift_drive),
                "d6_position_error_m": (
                    (
                        float(desired_offset[2])
                        + float(d6_load_compensation["load_bias_m"])
                        * max(0.0, min(1.0, float(desired_offset[2]) / max(commanded_lift, 1e-12)))
                        - float(palm_position[2] - palm_before[2])
                    )
                    if lift_drive else None
                ),
                "d6_actual_velocity_mps": float(palm_linear_velocity.z),
                "d6_velocity_error_mps": (
                    float(desired_velocity[2] - palm_linear_velocity.z) if lift_drive else None
                ),
                "d6_drive_force_limit_n": float(lift_drive["max_force"]) if lift_drive else None,
                "d6_load_bias_m": float(d6_load_compensation["load_bias_m"]) if lift_drive else 0.0,
            })
        if len(orientation_control_trace) < 256:
            orientation_control_trace.append({
                "orientation_target_world": list(orientation),
                "body_orientation_targets": {
                    "palm": list(palm_target_orientation),
                    "left": list(left_target_orientation),
                    "right": list(right_target_orientation),
                },
                "palm_error_vector_rad": palm_rotation_error.tolist(),
                "palm_error_rad": float(np.linalg.norm(palm_rotation_error)),
                "palm_angular_velocity_world": palm_angular_velocity_world.tolist(),
                "palm_torque_world": np.asarray(palm_torque, dtype=float).tolist(),
                "palm_torque_magnitude_nm": float(np.linalg.norm(palm_torque)),
                "pd_torque_world": pd_torque.tolist(),
                "orientation_pd_scale": orientation_pd_scale,
                "effective_orientation_pd_scale": effective_orientation_pd_scale,
                "orientation_kd_scale": orientation_kd_scale,
                "orientation_axis_mode": orientation_axis_mode,
                "contact_torque_world": contact_torque["torque_world"],
                "contact_torque_magnitude_nm": contact_torque["torque_magnitude_nm"],
                "requested_torque_world": requested_torque.tolist(),
                "applied_torque_world": applied_torque.tolist(),
                "torque_clamped": torque_clamped,
                "contact_torque_observation_status": contact_torque["status"],
                "contact_torque_source": contact_torque["source"],
                "contact_torque_contact_count": contact_torque["contact_count"],
                "palm_torque_limit_nm": palm_torque_limit_nm,
                "palm_rotational_inertia_kg_m2": np.asarray(palm_rotational_inertia, dtype=float).tolist(),
                "palm_inertia_source": palm_inertia_source,
                "body_orientation_errors": body_orientation_errors,
            })

    def append_motion_sample(phase_name, step, desired_offset, left_frame, right_frame):
        nonlocal max_finger_to_palm_common_mode_drift_m, max_finger_to_palm_closing_drift_m
        palm_pose = dc.get_rigid_body_pose(palm).p
        left_pose = dc.get_rigid_body_pose(left).p
        right_pose = dc.get_rigid_body_pose(right).p
        object_pose = dc.get_rigid_body_pose(target_body).p
        palm_position = np.asarray((palm_pose.x, palm_pose.y, palm_pose.z))
        left_position = np.asarray((left_pose.x, left_pose.y, left_pose.z))
        right_position = np.asarray((right_pose.x, right_pose.y, right_pose.z))
        object_position = np.asarray((object_pose.x, object_pose.y, object_pose.z))
        gripper_position = (left_position + right_position) * 0.5
        relative_now = {
            "left": left_position - palm_position,
            "right": right_position - palm_position,
        }
        relative_delta = {
            side: relative_now[side] - lift_start_finger_to_palm[side]
            for side in ("left", "right")
        }
        common_mode = (relative_delta["left"] + relative_delta["right"]) * 0.5
        max_finger_to_palm_common_mode_drift_m = max(
            max_finger_to_palm_common_mode_drift_m, float(np.linalg.norm(common_mode))
        )
        max_finger_to_palm_closing_drift_m = max(
            max_finger_to_palm_closing_drift_m,
            *(abs(float(np.dot(relative_delta[side], candidate["closing"]))) for side in ("left", "right")),
        )
        targets = parallel_gripper_drive_targets(stage, gripper)
        if len(lift_relative_finger_to_palm_trace) < 1024:
            lift_relative_finger_to_palm_trace.append({
                "phase": phase_name,
                "step": int(step),
                "left_delta_m": relative_delta["left"].tolist(),
                "right_delta_m": relative_delta["right"].tolist(),
                "common_mode_delta_m": common_mode.tolist(),
            })
            lift_prismatic_target_trace.append({"phase": phase_name, "step": int(step), **targets})
        if len(motion_trace) >= 1024:
            return
        object_velocity = dc.get_rigid_body_linear_velocity(target_body)
        palm_velocity = dc.get_rigid_body_linear_velocity(palm)
        left_velocity = dc.get_rigid_body_linear_velocity(left)
        right_velocity = dc.get_rigid_body_linear_velocity(right)
        motion_trace.append({
            "phase": phase_name,
            "step": int(step),
            "dt_s": float(settings["dt"]),
            "sim_time_s": float((step + 1) * float(settings["dt"])),
            "target_object_position_world": (target_before + np.asarray(desired_offset)).tolist(),
            "actual_object_position_world": object_position.tolist(),
            "target_palm_position_world": (palm_before + np.asarray(desired_offset)).tolist(),
            "actual_palm_position_world": palm_position.tolist(),
            "left_finger_position_world": left_position.tolist(),
            "right_finger_position_world": right_position.tolist(),
            "object_velocity_world": [object_velocity.x, object_velocity.y, object_velocity.z],
            "palm_velocity_world": [palm_velocity.x, palm_velocity.y, palm_velocity.z],
            "left_finger_velocity_world": [left_velocity.x, left_velocity.y, left_velocity.z],
            "right_finger_velocity_world": [right_velocity.x, right_velocity.y, right_velocity.z],
            "object_relative_to_gripper_world": (object_position - gripper_position).tolist(),
            "contact_left": bool(left_frame["contact"]),
            "contact_right": bool(right_frame["contact"]),
            "left_normal_force_n": raw_normal_contact_force(left_frame["contacts"], list(bound_target_paths), settings["dt"]),
            "right_normal_force_n": raw_normal_contact_force(right_frame["contacts"], list(bound_target_paths), settings["dt"]),
            "penetration_m": max(float(left_frame.get("penetration") or 0.0), float(right_frame.get("penetration") or 0.0)),
        })
    lift_steps = (
        max(1, int(float(settings["grasp_lift_seconds"]) / float(settings["dt"])))
        if bilateral_close_contact
        else 0
    )
    for step in range(lift_steps):
        blocked = handle_blocked("grasp_lift_before_update")
        if blocked:
            return {**blocked, "grasp_acquisition": acquisition["grasp_acquisition"]}
        phase = (step + 1) / lift_steps
        alpha = phase * phase * (3.0 - 2.0 * phase)
        alpha_speed = 6.0 * phase * (1.0 - phase) / lift_seconds
        apply_grasp_forces(lift * alpha, float(np.linalg.norm(lift)) * alpha_speed)
        app.update()
        record_frame()
        blocked = handle_blocked("grasp_lift_after_update")
        if blocked:
            return {**blocked, "grasp_acquisition": acquisition["grasp_acquisition"]}
        observe_joint_motion()
        left_frame = contact_frame(left_sensor, check["rigid_bodies"])
        right_frame = contact_frame(right_sensor, check["rigid_bodies"])
        last_lift_contact_frames = (left_frame, right_frame)
        lift_contacts.append((left_frame["contact"], right_frame["contact"]))
        lift_force_samples.append((
            raw_normal_contact_force(left_frame["contacts"], list(bound_target_paths), settings["dt"]),
            raw_normal_contact_force(right_frame["contacts"], list(bound_target_paths), settings["dt"]),
        ))
        append_motion_sample("lift", step, lift * alpha, left_frame, right_frame)
        lift_penetrations.append(
            max(
                float(left_frame.get("penetration") or 0.0),
                float(right_frame.get("penetration") or 0.0),
            )
        )
    hold_contacts = []
    hold_force_samples = []
    hold_samples = []
    hold_clearances = []
    hold_ground_contacts = []
    clearance_stride = max(1, round(1.0 / (20.0 * float(settings["dt"]))))
    hold_steps = (
        max(1, int(float(settings["grasp_hold_seconds"]) / float(settings["dt"])))
        if bilateral_close_contact
        else 0
    )
    ground_z = float(settings.get("ground_z_m", 0.0))
    for hold_step in range(hold_steps):
        blocked = handle_blocked("grasp_hold_before_update")
        if blocked:
            return {**blocked, "grasp_acquisition": acquisition["grasp_acquisition"]}
        apply_grasp_forces(lift, 0.0)
        app.update()
        record_frame()
        blocked = handle_blocked("grasp_hold_after_update")
        if blocked:
            return {**blocked, "grasp_acquisition": acquisition["grasp_acquisition"]}
        observe_joint_motion()
        left_frame = contact_frame(left_sensor, check["rigid_bodies"])
        right_frame = contact_frame(right_sensor, check["rigid_bodies"])
        last_lift_contact_frames = (left_frame, right_frame)
        hold_contacts.append((left_frame["contact"], right_frame["contact"]))
        hold_force_samples.append((
            raw_normal_contact_force(left_frame["contacts"], list(bound_target_paths), settings["dt"]),
            raw_normal_contact_force(right_frame["contacts"], list(bound_target_paths), settings["dt"]),
        ))
        append_motion_sample("hold", lift_steps + hold_step, lift, left_frame, right_frame)
        object_pose = dc.get_rigid_body_pose(target_body).p
        left_pose = dc.get_rigid_body_pose(left).p
        right_pose = dc.get_rigid_body_pose(right).p
        object_position = np.asarray((object_pose.x, object_pose.y, object_pose.z))
        gripper_position = (
            np.asarray((left_pose.x, left_pose.y, left_pose.z))
            + np.asarray((right_pose.x, right_pose.y, right_pose.z))
        ) * 0.5
        hold_samples.append(
            {
                "object_position": object_position,
                "gripper_position": gripper_position,
                "left_contact": bool(left_frame["contact"]),
                "right_contact": bool(right_frame["contact"]),
                "penetration_m": max(
                    float(left_frame.get("penetration") or 0.0),
                    float(right_frame.get("penetration") or 0.0),
                ),
            }
        )
        if hold_step % clearance_stride == 0 or hold_step == hold_steps - 1:
            # The object bounds were measured after settling while paused.
            # During lift/hold, translate that footprint by the authoritative
            # PhysX body pose instead of rebuilding a live BBoxCache.
            hold_clearances.append(
                float(minimum[2]) + float(object_position[2] - target_before[2]) - ground_z
            )
            ground_frame = contact_frame(ground_sensor, check["rigid_bodies"]) if ground_sensor is not None else {"contact": False}
            hold_ground_contacts.append(bool(ground_frame["contact"]))
    root_after = dc.get_rigid_body_pose(attachment_body).p
    target_after_pose = dc.get_rigid_body_pose(target_body).p
    target_after = np.asarray((target_after_pose.x, target_after_pose.y, target_after_pose.z))
    left_after_pose = dc.get_rigid_body_pose(left).p
    right_after_pose = dc.get_rigid_body_pose(right).p
    left_after = np.asarray(
        (left_after_pose.x, left_after_pose.y, left_after_pose.z)
    )
    right_after = np.asarray(
        (right_after_pose.x, right_after_pose.y, right_after_pose.z)
    )
    palm_after_pose = dc.get_rigid_body_pose(palm).p
    palm_after = np.asarray((palm_after_pose.x, palm_after_pose.y, palm_after_pose.z))
    gripper_after = (left_after + right_after) * 0.5
    relative_finger_to_palm_error_m = {
        "left": (left_after - palm_after - (left_before - palm_before)).tolist(),
        "right": (right_after - palm_after - (right_before - palm_before)).tolist(),
    }
    finger_axis_drifts = {}
    for side, start, final in (("left", starts[0], left_after), ("right", starts[1], right_after)):
        delta = np.asarray(final) - np.asarray(start)
        finger_axis_drifts[side] = {
            "closing_m": float(np.dot(delta, candidate["closing"])),
            "depth_m": float(np.dot(delta, gripper_approach)),
            "height_m": float(np.dot(delta, np.cross(candidate["closing"], gripper_approach))),
        }
    final_center_error = gripper_after - (gripper_before + lift)
    final_lateral_error = final_center_error - lift_direction * float(
        np.dot(final_center_error, lift_direction)
    )
    max_gripper_lateral_drift_m = max(
        max_gripper_lateral_drift_m, float(np.linalg.norm(final_lateral_error))
    )
    object_before = np.asarray((root_before.x, root_before.y, root_before.z))
    object_after = np.asarray((root_after.x, root_after.y, root_after.z))
    lifted = float(object_after[2] - object_before[2])
    relative_before = object_before - gripper_before
    relative_after = object_after - gripper_after
    slip = float(np.linalg.norm(relative_after - relative_before))
    gripper_lift = float(gripper_after[2] - gripper_before[2])
    target_link_lift = float(target_after[2] - target_before[2])
    required_lift = float(settings["grasp_lift_height_m"])
    bilateral_force_fraction_diagnostic, stable_force_windows = bilateral_force_window_fraction(
        lift_force_samples + hold_force_samples,
        force_target_n,
        force_stable_steps,
    )
    lift_force_fraction_diagnostic, lift_force_windows = bilateral_force_window_fraction(
        lift_force_samples,
        force_target_n,
        force_stable_steps,
    )
    hold_force_fraction_diagnostic, hold_force_windows = bilateral_force_window_fraction(
        hold_force_samples,
        force_target_n,
        force_stable_steps,
    )
    bilateral_contact_fraction = bilateral_contact_fraction_from_pairs(
        lift_contacts + hold_contacts,
        force_stable_steps,
    )
    lift_contact_fraction = bilateral_contact_fraction_from_pairs(lift_contacts, force_stable_steps)
    hold_contact_fraction = bilateral_contact_fraction_from_pairs(hold_contacts, force_stable_steps)
    left_hold_contact_fraction = (
        sum(
            all(left for left, _right in hold_contacts[index:index + force_stable_steps])
            for index in range(0, len(hold_contacts), force_stable_steps)
        ) / len(hold_force_windows)
        if hold_force_windows
        else 0.0
    )
    right_hold_contact_fraction = (
        sum(
            all(right for _left, right in hold_contacts[index:index + force_stable_steps])
            for index in range(0, len(hold_contacts), force_stable_steps)
        ) / len(hold_force_windows)
        if hold_force_windows
        else 0.0
    )
    clearance = grasp_clearance_result(
        hold_clearances,
        hold_ground_contacts,
        settings["grasp_ground_clearance_min_m"],
        settings["grasp_bilateral_contact_fraction_min"],
    )
    hold_relative_slip_m = max(
        (
            float(
                np.linalg.norm(
                    (sample["object_position"] - sample["gripper_position"])
                    - relative_before
                )
            )
            for sample in hold_samples
        ),
        default=0.0,
    )
    hold_penetration_max_m = max(
        (sample["penetration_m"] for sample in hold_samples),
        default=0.0,
    )
    lift_penetration_max_m = max(lift_penetrations, default=0.0)
    vibration_statistics = grasp_vibration_statistics(motion_trace)
    gripper_lateral_trajectory_pass = max_gripper_lateral_drift_m <= float(
        settings["grasp_gripper_lateral_drift_max_m"]
    )
    gripper_orientation_pass = max_gripper_orientation_drift_rad <= float(
        settings["grasp_gripper_orientation_drift_max_rad"]
    )
    gripper_trajectory_pass = gripper_lateral_trajectory_pass and gripper_orientation_pass
    hold_motion_pass = (
        clearance["pass"]
        and hold_relative_slip_m < float(settings["grasp_slip_max_m"])
        and gripper_trajectory_pass
    )
    hold_penetration_pass = hold_penetration_max_m <= float(settings["penetration_max_m"])
    contact_persistence_pass = (
        lift_contact_fraction >= float(settings["grasp_bilateral_contact_fraction_min"])
        and hold_contact_fraction >= float(settings["grasp_bilateral_contact_fraction_min"])
        and hold_motion_pass
        and hold_penetration_pass
    )
    load_bearing_pass = bool(
        bilateral_close_contact
        and target_link_lift >= required_lift
        and clearance["pass"]
        and slip < float(settings["grasp_slip_max_m"])
        and lift_contact_fraction >= float(settings["grasp_bilateral_contact_fraction_min"])
        and hold_contact_fraction >= float(settings["grasp_bilateral_contact_fraction_min"])
    )
    orientation_trajectory_pass = bool(gripper_trajectory_pass)
    contact_torque_observable = last_contact_torque_observation["status"] == "valid"
    attachment_maintained = (
        bilateral_close_contact
        and clearance["pass"]
        and slip < float(settings["grasp_slip_max_m"])
        and gripper_trajectory_pass
        and contact_persistence_pass
    )
    passed = (
        opening_geometry["opening_supported"]
        and bilateral_close_contact
        and attachment_maintained
    )
    failure_phase = (
        "passed"
        if passed
        else "grasp_opening_exceeds_reference"
        if not opening_geometry["opening_supported"]
        else "close_contact_window"
        if not bilateral_contact_ready
        else "seating_contact_window"
        if not seating_pass
        else "proof_lift_no_clearance"
        if target_link_lift < required_lift or not clearance["pass"]
        else "proof_lift_slip"
        if slip >= float(settings["grasp_slip_max_m"])
        else "lift_hold_contact_window"
        if not contact_persistence_pass
        else "hold_motion"
    )
    start_finger_distance = float(np.linalg.norm(starts[1] - starts[0]))
    final_finger_distance = float(np.linalg.norm(right_after - left_after))
    mechanical_travel_completed = bool(
        (start_finger_distance - final_finger_distance) * 0.5
        >= float(gripper["mechanical_travel_m"]) * 0.95
    )
    timeline.stop()
    return {
        "grasp_acquisition": acquisition["grasp_acquisition"],
        "applicable": True,
        "pass": passed,
        "load_bearing_pass": load_bearing_pass,
        "orientation_trajectory_pass": orientation_trajectory_pass,
        "contact_torque_observable": contact_torque_observable,
        "candidate_source": candidate["source"],
        "lift_target_path": target_path,
        "attachment_articulation_path": articulation_path,
        "approach_collision_free": True,
        "approach_collision_free_aabb_diagnostic_only": approach_collision_free,
        "approach_contact_fraction": (
            sum(
                palm_contact or left or right
                for palm_contact, (left, right) in zip(approach_palm_contacts, approach_contacts)
            )
            / len(approach_contacts)
            if approach_contacts
            else 0.0
        ),
        "approach_trigger": approach_trigger,
        "approach_contact_stable_steps": approach_contact_stable_steps,
        "approach_contact_required_steps": approach_contact_required_steps,
        "approach_extension_budget_m": approach_extension_budget,
        "approach_extension_applied_m": approach_extension_applied,
        "approach_command_speed_mps": approach_command_speed_mps,
        "approach_speed_samples_mps": approach_speed_samples,
        "approach_speed_min_mps": min(approach_speed_samples) if approach_speed_samples else 0.0,
        "approach_speed_max_mps": max(approach_speed_samples) if approach_speed_samples else 0.0,
        "approach_speed_variation_mps": (
            max(approach_speed_samples) - min(approach_speed_samples)
            if approach_speed_samples else 0.0
        ),
        "approach_phase_transition_step": approach_phase_transition_step,
        "approach_transition_speed_before_mps": approach_transition_speed_before_mps,
        "approach_transition_speed_after_mps": approach_transition_speed_after_mps,
        "approach_first_contact_step": approach_first_contact_step,
        "approach_cruise_speed_mps": approach_cruise_speed_mps,
        "approach_terminal_speed_mps": approach_terminal_speed_mps,
        "approach_deceleration_start_step": approach_deceleration_start_step,
        "approach_contact_stop": approach_contact_stop,
        "approach_contact_parts": sorted(approach_contact_parts),
        "approach_contact_bodies": sorted(approach_contact_bodies),
        "approach_contact_stop_position_world": (
            approach_contact_stop_position.tolist()
            if approach_contact_stop_position is not None else None
        ),
        "approach_contact_stop_penetration_m": approach_contact_stop_penetration_m,
        "approach_contact_stop_object_motion_m": approach_contact_stop_object_motion_m,
        "bilateral_close_contact": bilateral_close_contact,
        "gripper_geometry": gripper_geometry,
        "opening_geometry_source": opening_geometry.get(
            "grasp_opening_source", "object_collision_union_projection"
        ),
        "opening_projection_links": collision_union["links"],
        "opening_projection_by_link": collision_union["projection_links"],
        **opening_geometry,
        "opening_reason": None if opening_geometry["opening_supported"] else "grasp_opening_exceeds_reference",
        "gripper_material_roles": gripper["materials"],
        "gripper_structure": "dynamic_palm_mirrored_prismatic_fingers",
        "gripper_palm_path": palm_path,
        "gripper_reference": {
            key: descriptor[key]
            for key in (
                "reference_model",
                "reference_max_opening_m",
                "reference_finger_travel_m",
                "reference_opening_enforced",
                "reference_parameters_source",
            )
        },
        "left_target_contact_diagnostic": left_contact_step is not None,
        "right_target_contact_diagnostic": right_contact_step is not None,
        "left_contact_rigid_body": left_contact_path,
        "right_contact_rigid_body": right_contact_path,
        "bound_target_rigid_bodies": list(bound_target_paths or ()),
        "first_target_raw_contacts": first_target_raw_contacts,
        "first_target_contact_summaries": first_target_contact_summaries,
        "first_target_pad_contacts": first_target_pad_contacts,
        "first_target_contact_regions": first_target_contact_regions,
        "force_observation_status": {
            side: sorted(statuses) for side, statuses in force_observation_statuses.items()
        },
        "gripper_contact_contract": gripper["contact_contract"],
        "finger_axis_drifts_m": finger_axis_drifts,
        "relative_finger_to_palm_error_m": relative_finger_to_palm_error_m,
        "translation_control_trace": translation_control_trace,
        "controller_trace": controller_trace,
        "d6_anchor_position_world": d6_anchor_position_world,
        "d6_anchor_orientation_world": d6_anchor_orientation_world,
        "d6_initial_target_position_m": 0.0 if lift_drive else None,
        "d6_creation_to_first_step_delta_m": d6_creation_to_first_step_delta_m,
        "d6_initial_constraint_impulse_diagnostic": d6_initial_constraint_impulse_diagnostic,
        "d6_required_support_force_n": (
            float(d6_load_compensation["object_mass_kg"])
            * float(d6_load_compensation["gravity_mps2"])
            if lift_drive else None
        ),
        "d6_load_bias_m": float(d6_load_compensation["load_bias_m"]) if lift_drive else 0.0,
        "d6_applied_target_lift_m": float(d6_load_compensation["applied_target_lift_m"]) if lift_drive else None,
        "d6_load_bias_clamped": bool(d6_load_compensation["bias_clamped"]) if lift_drive else False,
        "left_lift_start_joint_target": lift_start_joint_targets["left"],
        "right_lift_start_joint_target": lift_start_joint_targets["right"],
        "lift_start_finger_to_palm_world": {
            side: value.tolist() for side, value in lift_start_finger_to_palm.items()
        },
        "lift_end_finger_to_palm_world": {
            "left": (left_after - palm_after).tolist(),
            "right": (right_after - palm_after).tolist(),
        },
        "lift_relative_finger_to_palm_trace": lift_relative_finger_to_palm_trace,
        "lift_prismatic_target_trace": lift_prismatic_target_trace,
        "max_finger_to_palm_common_mode_drift_m": max_finger_to_palm_common_mode_drift_m,
        "max_finger_to_palm_closing_drift_m": max_finger_to_palm_closing_drift_m,
        "d6_finger_common_mode_stabilization": d6_common_mode_stabilization,
        "d6_finger_common_mode_force_limit_n": d6_common_mode_force_limit_n,
        "motion_trace": motion_trace,
        **vibration_statistics,
        **controller_settings,
        "controller_status": controller_status,
        "controller_failure_reason": controller_failure_reason,
        "controller_repeat_index": int(os.environ.get("RAW_EVAL_CONTROLLER_REPEAT_INDEX", "0") or 0),
        "lift_drive": ({
            key: value for key, value in lift_drive.items()
            if key not in {"target_position", "target_velocity"}
        } if lift_drive else None),
        "orientation_control_trace": orientation_control_trace,
        "contact_torque_observation_status": last_contact_torque_observation["status"],
        "contact_torque_mode": contact_torque_mode,
        "contact_torque_control_applied": bool(contact_torque_mode == "enabled" and last_contact_torque_observation["status"] == "valid"),
        "contact_torque_control_disabled_reason": (
            "shadow_mode_until_contact_semantics_validated"
            if contact_torque_mode == "shadow"
            else "contact_torque_unobservable"
            if last_contact_torque_observation["status"] != "valid"
            else None
        ),
        "controller_baseline": "r21_bottle_orientation_finger_torque",
        "baseline_controller_preserved": contact_torque_mode == "shadow",
        "contact_torque_source": last_contact_torque_observation["source"],
        "contact_torque_world": last_contact_torque_observation["torque_world"],
        "contact_torque_magnitude_nm": last_contact_torque_observation["torque_magnitude_nm"],
        "contact_torque_contact_count": last_contact_torque_observation["contact_count"],
        "contact_torque_per_contact": last_contact_torque_observation["per_contact"],
        "orientation_target_world": list(orientation),
        "orientation_force_source": (
            "palm_world_torque_pd" if lift_controller == "force_pd"
            else "world_d6_lock_and_force_drive"
        ),
        "palm_inertia_source": palm_inertia_source,
        "palm_rotational_inertia_kg_m2": np.asarray(palm_rotational_inertia, dtype=float).tolist(),
        "same_rigid_body_binding": same_rigid_body_binding,
        "contact_binding_policy": "left_and_right_may_contact_different_asset_rigid_bodies",
        "initial_any_contact_both_fingers_diagnostic": left_any_contact and right_any_contact,
        "attempt_stage_identifier": stage.GetRootLayer().identifier,
        "contact_sensor_ready_at_close_start": contact_sensor_ready_at_close_start,
        "close_target_paths": close_target_paths,
        "close_target_scope": (
            "authored_component" if candidate.get("annotation_diagnostics", {}).get("native_component")
            else "asset_rigid_bodies"
        ),
        "close_initial_pose": close_initial_pose,
        "close_initial_gap_m": initial_gap,
        "close_first_target_contact_step": close_first_target_contact_step,
        "close_left_target_position_m": independent_targets["left"],
        "close_right_target_position_m": independent_targets["right"],
        "close_left_target_contact_step": left_target_contact_step,
        "close_bilateral_contact_step": close_bilateral_contact_step,
        "close_bilateral_target_stop": close_bilateral_target_stop,
        "close_right_target_contact_step": right_target_contact_step,
        "close_left_target_frozen_after_contact": left_target_frozen_after_contact,
        "close_right_target_frozen_after_contact": right_target_frozen_after_contact,
        "close_unilateral_progress_steps": unilateral_progress_steps,
        "close_unilateral_stall_steps": unilateral_stall_steps,
        "close_final_commanded_gap_m": desired_gap,
        "close_final_measured_gap_m": previous_measured_gap,
        "close_drive_force_limited": close_drive_force_limited,
        "close_bilateral_contact_observed": bool(bilateral_contact_max_consecutive_steps > 0),
        "close_bilateral_contact_max_consecutive_steps": bilateral_contact_max_consecutive_steps,
        "close_bilateral_contact_stable_steps": bilateral_contact_stable_steps,
        "close_contact_observed": bool(left_contact_path or right_contact_path),
        "close_failure_source": (
            None if bilateral_close_contact else
            "contact_sensor_initialization_pending" if not contact_sensor_ready_at_close_start else
            "no_persistent_bilateral_target_contact"
        ),
        "observed_contact_bodies_diagnostic": sorted(observed_contact_bodies),
        "bilateral_lift_hold_contact_fraction": bilateral_contact_fraction,
        "bilateral_lift_contact_fraction": lift_contact_fraction,
        "bilateral_hold_contact_fraction": hold_contact_fraction,
        "left_hold_contact_fraction": left_hold_contact_fraction,
        "right_hold_contact_fraction": right_hold_contact_fraction,
        "hold_ground_clearance_fraction": clearance["fraction"],
        "hold_ground_clearance_min_m": clearance["minimum"],
        "hold_ground_contact_fraction": clearance["ground_contact_fraction"],
        "required_ground_clearance_m": float(settings["grasp_ground_clearance_min_m"]),
        "ground_clearance_pass": clearance["pass"],
        "hold_relative_slip_m": hold_relative_slip_m,
        "hold_penetration_max_m": hold_penetration_max_m,
        "lift_penetration_max_m": lift_penetration_max_m,
        "hold_motion_pass": hold_motion_pass,
        "hold_penetration_pass": hold_penetration_pass,
        "hold_contact_report_diagnostic_only": False,
        "contact_persistence_pass": contact_persistence_pass,
        "failure_phase": failure_phase,
        "seating_contact_fraction": seating_contact_fraction,
        "seating_bilateral_contact_observed": any(bool(left) and bool(right) for left, right in seating_contacts),
        "seating_bilateral_contact_max_consecutive_steps": seating_bilateral_contact_max_consecutive_steps,
        "seating_pass": seating_pass,
        "close_gap_trace": close_gap_trace,
        "close_force_windows": close_force_windows,
        "seating_force_windows": seating_force_windows,
        "lift_force_windows": lift_force_windows,
        "hold_force_windows": hold_force_windows,
        "mechanical_travel_completed": mechanical_travel_completed,
        "object_mass_kg": object_mass,
        "grasp_force_target_n_per_finger": force_target_n,
        "max_left_contact_force_n": max_left_force_n,
        "max_right_contact_force_n": max_right_force_n,
        "max_left_force_command_n": max_left_force_command_n,
        "max_right_force_command_n": max_right_force_command_n,
        "lift_force_limit_n_per_finger": lift_force_limit_n,
        "lift_weight_feedforward_n_per_finger": lift_weight_feedforward_n,
        "max_left_lift_force_n": max_left_lift_force_n,
        "max_right_lift_force_n": max_right_lift_force_n,
        "lift_controller": lift_controller,
        "lift_controller_detail": (
            "force_limited_palm_cartesian_pd_with_prismatic_finger_closure"
            if lift_controller == "force_pd"
            else "world_anchored_d6_force_drive_with_prismatic_finger_closure"
        ),
        "max_gripper_lateral_drift_m": max_gripper_lateral_drift_m,
        "required_gripper_lateral_drift_max_m": float(settings["grasp_gripper_lateral_drift_max_m"]),
        "max_gripper_orientation_drift_rad": max_gripper_orientation_drift_rad,
        "palm_orientation_drift_rad": palm_orientation_drift_rad,
        "left_finger_orientation_drift_rad": left_finger_orientation_drift_rad,
        "right_finger_orientation_drift_rad": right_finger_orientation_drift_rad,
        "required_gripper_orientation_drift_max_rad": float(settings["grasp_gripper_orientation_drift_max_rad"]),
        "gripper_trajectory_pass": gripper_trajectory_pass,
        "gripper_lift_m": gripper_lift,
        "target_link_lift_m": target_link_lift,
        "articulation_root_lift_m": lifted,
        "joint_names": dof_names,
        "joint_max_displacement": dof_max_delta.tolist(),
        "maximum_joint_displacement": float(dof_max_delta.max()) if len(dof_max_delta) else 0.0,
        "left_object_static_friction": left_object_friction,
        "right_object_static_friction": right_object_friction,
        "left_friction_source": left_friction_source,
        "right_friction_source": right_friction_source,
        "gripper_static_friction": gripper_friction,
        "left_gripper_contact_static_friction": left_gripper_friction,
        "right_gripper_contact_static_friction": right_gripper_friction,
        "effective_friction_rule": "min_conservative",
        "contact_force_source": "target_raw_normal_impulse_windowed",
        "force_gate_mode": "diagnostic_force_window_plus_physical_proof_lift",
        "close_force_threshold_reached_diagnostic": force_threshold_reached,
        "bilateral_contact_ready": bilateral_contact_ready,
        "bilateral_force_fraction_diagnostic": bilateral_force_fraction_diagnostic,
        "lift_force_fraction_diagnostic": lift_force_fraction_diagnostic,
        "hold_force_fraction_diagnostic": hold_force_fraction_diagnostic,
        "force_control_mode": "symmetric_world_force_feedback",
        "force_application_point": "finger_center_world",
        "force_ramp_timeout_seconds": float(settings["grasp_force_ramp_timeout_seconds"]),
        "force_command_step_ratio": float(settings["grasp_force_command_step_ratio"]),
        "force_command_max_ratio": float(settings["grasp_force_command_max_ratio"]),
        "force_stable_steps": force_stable_steps,
        "force_stable_seconds": float(settings["grasp_force_stable_seconds"]),
        "force_threshold_reached": force_threshold_reached,
        "attachment_proxy": "physical_bilateral_contact_with_high_friction_fingers",
        "attachment_maintained": attachment_maintained,
        "object_lift_m": lifted,
        "required_object_lift_m": required_lift,
        "commanded_gripper_lift_m": commanded_lift,
        "object_gripper_slip_m": slip,
        "grasp_width_m": candidate["width"],
        "candidate_width_m": candidate.get("width"),
        "candidate_width_source": candidate.get("candidate_width_source"),
        "collision_swept_width_m": opening_geometry["collision_swept_width_m"],
        "opening_geometry_source": opening_geometry.get(
            "grasp_opening_source", "object_collision_union_projection"
        ),
        "opening_projection_links": collision_union["links"],
        "opening_projection_by_link": collision_union["projection_links"],
        "target_binding_source": candidate.get("target_binding_source"),
        "target_binding_status": candidate.get("target_binding_status"),
        "grasp_pose_source": candidate["annotation_diagnostics"].get("grasp_pose_source"),
        "authored_grasp_prim": candidate["annotation_diagnostics"].get("authored_grasp_prim"),
        "authored_grasp_local_transform": (
            candidate["annotation_diagnostics"].get("authored_grasp_local_transform").tolist()
            if hasattr(candidate["annotation_diagnostics"].get("authored_grasp_local_transform"), "tolist") else candidate["annotation_diagnostics"].get("authored_grasp_local_transform")
        ),
        "authored_grasp_world_transform": (
            candidate["annotation_diagnostics"].get("authored_grasp_world_transform").tolist()
            if hasattr(candidate["annotation_diagnostics"].get("authored_grasp_world_transform"), "tolist") else candidate["annotation_diagnostics"].get("authored_grasp_world_transform")
        ),
        "authored_grasp_axis_contract": candidate["annotation_diagnostics"].get("authored_grasp_axis_contract"),
        "authored_grasp_pose_valid": candidate["annotation_diagnostics"].get("authored_grasp_pose_valid"),
        "approach_distance_m": approach_distance,
        "pregrasp_center_world": pregrasp_center.tolist(),
        "approach_object_motion_m": approach_object_motion,
        "approach_object_motion_tolerance_m": approach_motion_tolerance,
        "approach_object_motion_failure_criterion": False,
        "approach_penetration_max_m": approach_max_penetration,
        "approach_penetration_tolerance_m": approach_penetration_tolerance,
        "grasp_center_world": candidate["center"].tolist(),
        "grasp_closing_world": candidate["closing"].tolist(),
        "grasp_approach_world": candidate["approach"].tolist(),
        "authored_closing_world": candidate.get("annotation_diagnostics", {}).get("authored_closing_world", candidate["closing"]).tolist() if hasattr(candidate.get("annotation_diagnostics", {}).get("authored_closing_world", candidate["closing"]), "tolist") else candidate.get("annotation_diagnostics", {}).get("authored_closing_world", candidate["closing"]),
        "authored_approach_world": candidate.get("annotation_diagnostics", {}).get("authored_approach_world", candidate["approach"]).tolist() if hasattr(candidate.get("annotation_diagnostics", {}).get("authored_approach_world", candidate["approach"]), "tolist") else candidate.get("annotation_diagnostics", {}).get("authored_approach_world", candidate["approach"]),
        "authored_closing_local": candidate.get("annotation_diagnostics", {}).get("authored_closing_local"),
        "authored_approach_local": candidate.get("annotation_diagnostics", {}).get("authored_approach_local"),
        "authored_attribute_frame": candidate.get("annotation_diagnostics", {}).get("authored_attribute_frame"),
        "gripper_closing_world": candidate["closing"].tolist(),
        "gripper_approach_world": gripper_approach.tolist(),
        "approach_motion_world": (final_translation / max(float(np.linalg.norm(final_translation)), 1e-8)).tolist(),
        "palm_closing_span_m": descriptor["palm_closing_span_m"],
        "rail_closing_span_m": descriptor["rail_closing_span_m"],
        "grasp_frame_convention": candidate.get(
            "grasp_frame_convention",
            "x=closing,y=approach,z=closing_cross_approach",
        ),
        "authored_grasp_pose_world": (
            candidate["authored_pose_world"].tolist()
            if candidate.get("authored_pose_world") is not None
            else None
        ),
        "evaluator_grasp_pose_world": frame_poses["evaluator_grasp_pose_world"].tolist(),
        "grasp_frame_to_palm_transform": frame_poses["grasp_frame_to_palm_transform"].tolist(),
        "commanded_palm_pose_world": frame_poses["commanded_palm_pose_world"].tolist(),
        "actual_palm_pose_world_at_arrival": {
            "position": [
                palm_arrival_pose.p.x,
                palm_arrival_pose.p.y,
                palm_arrival_pose.p.z,
            ],
            "orientation_xyzw": [
                palm_arrival_pose.r.x,
                palm_arrival_pose.r.y,
                palm_arrival_pose.r.z,
                palm_arrival_pose.r.w,
            ],
        },
        "grasp_arrival_position_error_m": palm_arrival_error["position_error_m"],
        "grasp_arrival_orientation_error_rad": palm_arrival_error["orientation_error_rad"],
        "arrival_finger_centers_world": [center.tolist() for center in arrival_finger_centers],
        "arrival_finger_center_error_m": arrival_finger_center_error,
        "candidate_validation": candidate["annotation_diagnostics"],
        "asset_bounds_world": [list(minimum), list(maximum)],
        "final_finger_centers_world": [
            left_after.tolist(),
            right_after.tolist(),
        ],
        "gripper_joint_axis_world": gripper["joint_axis_world"],
        "initial_finger_center_delta_world": gripper["initial_finger_center_delta_world"],
        "final_finger_center_delta_world": (right_after - left_after).tolist(),
        "closing_axis_alignment": float(
            np.dot(
                candidate["closing"],
                (right_after - left_after)
                / max(float(np.linalg.norm(right_after - left_after)), 1e-8),
            )
        ),
        "left_closing_motion_world": (left_after - left_before).tolist(),
        "right_closing_motion_world": (right_after - right_before).tolist(),
        "palm_to_finger_vector_world": gripper["palm_to_finger_vector_world"],
        "palm_behind_fingers_along_approach": bool(
            np.dot(np.asarray(gripper["palm_to_finger_vector_world"], dtype=float), gripper_approach) > 0.0
        ),
        "final_gripper_gap_m": max(
            0.0,
            float(np.linalg.norm(right_after - left_after)) - finger_thickness,
        ),
        "parameters_source": "paper_thresholds_plus_disclosed_reproduction_design_gripper_proxy",
    }


def grasp_lift(stage, check: dict, app, dataset: str, asset: dict) -> dict:
    pose_path = check["default_prim"] or check["rigid_bodies"][0]
    grasps = sorted(native_grasp_candidates(stage, pose_path), key=lambda item: item["prim_path"])
    if not grasps:
        return {
            "applicable": False,
            "pass": False,
            "reason": "missing_authored_grasppose",
            "attempt_count": 0,
            "attempts": [],
        }
    # Every authored grasp is part of the formal authored-pose evaluation.
    ranks = list(range(len(grasps)))
    scales = CONFIG["simulation"].get("grasp_finger_scales", [1.0])
    geometries = CONFIG["simulation"].get("grasp_geometry_modes", ["flat", "wrap"])
    source_path = Path(stage.GetRootLayer().realPath or stage.GetRootLayer().identifier)
    attempts = []
    total_attempts = len(ranks) * len(scales) * len(geometries)
    for geometry in geometries:
        for rank in ranks:
            for scale in scales:
                started = time.monotonic()
                print(
                    f"interaction grasp attempt={len(attempts) + 1}/{total_attempts} "
                    f"geometry={geometry} candidate={rank} finger_scale={float(scale):g} phase=start",
                    flush=True,
                )
                attempt_asset = dict(
                    asset,
                    _grasp_candidate_rank=rank,
                    _grasp_finger_scale=float(scale),
                    _grasp_geometry=geometry,
                )
                attempt_stage = stage if not attempts else open_stage(source_path, app)
                attempt_check = check if not attempts else precheck(attempt_stage)
                result = _grasp_lift_once(attempt_stage, attempt_check, app, dataset, attempt_asset)
                result["candidate_rank"] = rank
                result["finger_scale"] = float(scale)
                result["gripper_geometry"] = geometry
                result["attempt_score"] = grasp_score({"attempts": [result]})
                attempts.append(result)
                checkpoint_path = ensure_inside_root(
                    ROOT / "runs" / f"{RUN_LABEL}_grasp_checkpoint.json"
                )
                checkpoint_path.write_text(
                    json.dumps(
                        {
                            "run_label": RUN_LABEL,
                            "dataset": dataset,
                            "asset_id": asset["asset_id"],
                            "expected_attempts": total_attempts,
                            "completed_attempts": len(attempts),
                            "attempts": attempts,
                        },
                        ensure_ascii=False,
                        default=json_value,
                    ),
                    encoding="utf-8",
                )
                print(
                    f"interaction grasp attempt={len(attempts)}/{total_attempts} "
                    f"geometry={geometry} candidate={rank} finger_scale={float(scale):g} phase=end "
                    f"pass={bool(result.get('pass'))} elapsed_s={time.monotonic() - started:.1f}",
                    flush=True,
                )
    blocked_attempts = [
        item for item in attempts
        if item.get("status") == "evaluation_blocked"
        or item.get("reason") in {
            "evaluation_blocked",
            "runtime_grasp_handle_unavailable",
            "runtime_rigid_body_handle_unavailable",
            "runtime_rigid_body_properties_unavailable",
            "runtime_rigid_body_properties_exception",
            "authored_grasp_target_unresolved",
        }
    ]
    evaluated_attempts = [item for item in attempts if item.get("applicable") and item not in blocked_attempts]
    applicable = bool(evaluated_attempts)
    geometry_scores = common_grasp_geometry_score(attempts)
    reasons = [str(item.get("reason")) for item in attempts if item.get("reason")]
    binding_reasons = {
        "authored_grasp_target_unresolved",
        "runtime_grasp_handle_unavailable",
        "runtime_rigid_body_handle_unavailable",
        "runtime_rigid_body_properties_unavailable",
        "runtime_rigid_body_properties_exception",
    }
    if attempts and not applicable:
        reason = next((item for item in reasons if item in binding_reasons), "grasp_runtime_blocked")
        return {
            "applicable": False,
            "pass": False,
            "status": "evaluation_blocked",
            "reason": reason,
            "reason_class": "runtime" if any(item.startswith("runtime_") for item in reasons) else "evaluator",
            "attempt_count": len(attempts),
            "grasp_score_common": None,
            "grasp_score_by_geometry": geometry_scores,
            "blocked_attempt_count": len(blocked_attempts),
            "attempts": attempts,
        }
    reason = next((item for item in reasons if item in binding_reasons), None)
    if reason is None:
        reason = "all_grasp_candidates_failed" if applicable else "invalid_authored_grasppose"
    return {
        "applicable": applicable,
        "pass": all(
            any(item.get("pass") for item in attempts if item.get("gripper_geometry") == geometry)
            for geometry in geometries
        ),
        "reason": reason,
        "reason_class": "runtime" if any(item.startswith("runtime_") for item in reasons) else "evaluator" if reason in binding_reasons else "asset",
        "attempt_count": len(attempts),
        "blocked_attempt_count": len(blocked_attempts),
        "grasp_score_common": geometry_scores["common"],
        "grasp_score_by_geometry": geometry_scores,
        "attempts": attempts,
    }


def grasp_once_smoke(app, dataset: str, assets: list[dict], candidate_rank: int,
                     finger_scale: float, gripper_geometry: str) -> None:
    for index, asset in enumerate(assets, 1):
        path = resolved_asset_path(dataset, asset)
        if not path or not path.exists():
            raise FileNotFoundError(f"grasp smoke asset missing: {asset['asset_id']} {path}")
        stage = open_stage(path, app)
        check = precheck(stage)
        attempt_asset = dict(
            asset,
            _grasp_candidate_rank=candidate_rank,
            _grasp_finger_scale=finger_scale,
            _grasp_geometry=gripper_geometry,
        )
        try:
            result = _grasp_lift_once(stage, check, app, dataset, attempt_asset)
        finally:
            stop_timeline()
        payload = {
            "applicable": bool(result.get("applicable")),
            "pass": bool(result.get("pass")),
            "reason": result.get("reason"),
            "attempt_count": 1,
            "attempts": [{
                **result,
                "candidate_rank": candidate_rank,
                "finger_scale": finger_scale,
                "gripper_geometry": gripper_geometry,
            }],
        }
        append_result({
            **v5_base_row(dataset, asset, path),
            "metric": "grasp_lift",
            "grasp": payload,
            "grasp_score": grasp_score(payload) if payload["applicable"] else None,
        })
        print(
            f"[{index}/{len(assets)}] grasp-once asset={asset['asset_id']} "
            f"score={grasp_score(payload) if payload['applicable'] else None} "
            f"pass={payload['pass']}",
            flush=True,
        )


def grasp_candidate_ranks(total: int, count: int) -> list[int]:
    if total <= 0:
        return [0]
    count = min(max(1, count), total)
    return sorted(
        {int(round(index * (total - 1) / max(1, count - 1))) for index in range(count)}
    )


def matching_joint(stage, check: dict, dof_name: str):
    from pxr import UsdPhysics

    for path in check["joints"]:
        prim = stage.GetPrimAtPath(path)
        if prim.GetName() == dof_name:
            joint_type = (
                "rotation"
                if prim.IsA(UsdPhysics.RevoluteJoint)
                else "translation"
                if prim.IsA(UsdPhysics.PrismaticJoint)
                else "unknown"
            )
            return prim, joint_type
    return None, "unknown"


def joint_axis_world(stage, joint_prim):
    import numpy as np
    from pxr import Gf, UsdGeom, UsdPhysics

    token = str(joint_prim.GetAttribute("physics:axis").Get() or "X").upper()
    local_axis = {
        "X": Gf.Vec3d(1.0, 0.0, 0.0),
        "Y": Gf.Vec3d(0.0, 1.0, 0.0),
        "Z": Gf.Vec3d(0.0, 0.0, 1.0),
    }.get(token, Gf.Vec3d(1.0, 0.0, 0.0))
    parents = UsdPhysics.Joint(joint_prim).GetBody0Rel().GetTargets()
    if parents:
        parent = stage.GetPrimAtPath(parents[0])
        local_axis = UsdGeom.XformCache().GetLocalToWorldTransform(parent).TransformDir(local_axis)
    axis = np.asarray(tuple(local_axis), dtype=float)
    norm = float(np.linalg.norm(axis))
    return axis / norm if norm > 1e-8 else np.asarray((1.0, 0.0, 0.0))


def _legacy_task_actuation_v3(stage, check: dict, app, dataset: str, asset: dict) -> dict:
    import numpy as np
    from omni.isaac.dynamic_control import _dynamic_control
    from pxr import UsdPhysics

    if not check["joints"]:
        return {"applicable": False, "pass": False, "reason": "no_nonfixed_joint"}
    annotation = None
    if dataset == "robophyscan" and not (
        annotation
        and annotation.get("part_frames")
        and annotation.get("affordances")
        and annotation.get("motion_priors")
    ):
        return {"applicable": False, "pass": False, "reason": "missing_raw_task_annotation"}
    timeline, asset_minimum, asset_maximum = begin_physics(
        stage,
        check,
        app,
        settle_seconds=float(CONFIG["simulation"]["interaction_settle_seconds"]),
    )
    dc = _dynamic_control.acquire_dynamic_control_interface()
    articulation, articulation_path = dynamic_articulation(dc, check)
    if not articulation:
        timeline.stop()
        return {"applicable": True, "pass": False, "reason": "articulation_handle_unavailable"}
    selected = None
    for index in range(dc.get_articulation_dof_count(articulation)):
        dof = dc.get_articulation_dof(articulation, index)
        props = dc.get_dof_properties(dof)
        lower, upper = float(props.lower), float(props.upper)
        if all(map(math.isfinite, (lower, upper))) and upper - lower > 1e-8:
            candidate_name = dc.get_dof_name(dof)
            candidate_prim, candidate_type = matching_joint(stage, check, candidate_name)
            if candidate_prim is None or candidate_type == "unknown":
                continue
            targets = UsdPhysics.Joint(candidate_prim).GetBody1Rel().GetTargets()
            candidate_child = str(targets[0]) if targets else None
            if not candidate_child:
                continue
            selected = (
                dof,
                props,
                lower,
                upper,
                candidate_name,
                candidate_prim,
                candidate_type,
                candidate_child,
            )
            break
    if selected is None:
        timeline.stop()
        return {"applicable": False, "pass": False, "reason": "no_limited_nonfixed_dof"}
    (
        dof,
        props,
        lower,
        upper,
        dof_name,
        joint_prim,
        joint_type,
        child_path,
    ) = selected
    target_links = joint_descendants(stage, check["joints"], [child_path])
    valid_bounds = []
    for target_path in sorted(target_links):
        target_minimum, target_maximum = stage_bounds(stage, target_path)
        target_minimum = np.asarray(target_minimum, dtype=float)
        target_maximum = np.asarray(target_maximum, dtype=float)
        if (
            np.isfinite(target_minimum).all()
            and np.isfinite(target_maximum).all()
            and (target_maximum > target_minimum).any()
            and np.abs(target_minimum).max() < 1e6
            and np.abs(target_maximum).max() < 1e6
        ):
            valid_bounds.append((target_minimum, target_maximum))
    if not valid_bounds:
        timeline.stop()
        return {
            "applicable": False,
            "pass": False,
            "reason": "task_target_geometry_unavailable",
        }
    child_min = np.min([item[0] for item in valid_bounds], axis=0)
    child_max = np.max([item[1] for item in valid_bounds], axis=0)
    child_center = (child_min + child_max) * 0.5
    bounds_source = "joint_child_subtree_bounds"
    if dataset == "robophyscan" and annotation:
        prior = annotation["motion_priors"][0]
        frame = next(
            (
                item
                for item in annotation["part_frames"]
                if item["part_index"] == prior["part_index"]
            ),
            None,
        )
        if frame:
            local_center = np.asarray(frame["pose_matrix"], dtype=float)[:3, 3]
            child_center = np.asarray(
                asset_point_world(
                    stage,
                    check["rigid_bodies"][0],
                    local_center,
                )
            )
            bounds_source = "robophyscan_raw_part_frame_center_plus_asset_scale"
    child_extent = child_max - child_min
    diagonal = float(np.linalg.norm(child_extent))
    pusher_size = min(0.05, max(0.01, diagonal * 0.08))
    root_center = np.asarray(world_translation(stage, check["rigid_bodies"][0]))
    radial = child_center - root_center
    if np.linalg.norm(radial) < 1e-8:
        radial = np.asarray((1.0, 0.0, 0.0))
    radial /= np.linalg.norm(radial)
    axis = joint_axis_world(stage, joint_prim)
    planning_initial = float(
        dc.get_dof_state(dof, _dynamic_control.STATE_ALL).pos
    )
    desired_sign = (
        1.0
        if upper - planning_initial >= planning_initial - lower
        else -1.0
    )
    planned_target = upper if desired_sign > 0 else lower
    direction = (
        -axis * desired_sign
        if joint_type == "translation"
        else -np.cross(axis, radial) * desired_sign
    )
    if np.linalg.norm(direction) < 1e-8:
        direction = np.asarray((1.0, 0.0, 0.0))
    direction /= np.linalg.norm(direction)
    bounds_center = (child_min + child_max) * 0.5
    plane_point = child_center - direction * float(
        np.dot(child_center - bounds_center, direction)
    )
    plane_point = np.clip(plane_point, child_min, child_max)
    support_distance = float(np.dot(np.abs(direction), child_extent * 0.5))
    surface_point = plane_point + direction * support_distance
    start = surface_point + direction * (0.45 * pusher_size)
    travel = (
        min(0.5, max(pusher_size, abs(planned_target - planning_initial)))
        if joint_type == "translation"
        else min(0.25, max(0.05, diagonal))
    )
    end = start - direction * travel
    pusher_path = define_kinematic_box(
        stage,
        "/__raw_eval/TaskPusher",
        start,
        (pusher_size, pusher_size, pusher_size),
        kinematic=False,
    )
    try:
        contact_sensor = create_contact_sensor(pusher_path, "ContactSensor")
    except Exception:
        contact_sensor = None
    timeline.stop()
    app.update()
    timeline.play()
    for _ in range(3):
        app.update()
    if contact_sensor:
        contact_sensor.initialize()
    articulation, articulation_path = dynamic_articulation(dc, check)
    if not articulation:
        timeline.stop()
        return {"applicable": True, "pass": False, "reason": "articulation_reinitialize_failed"}
    for index in range(dc.get_articulation_dof_count(articulation)):
        candidate_dof = dc.get_articulation_dof(articulation, index)
        if dc.get_dof_name(candidate_dof) == dof_name:
            dof = candidate_dof
            break
    pusher = dc.get_rigid_body(pusher_path)
    if not pusher:
        timeline.stop()
        return {"applicable": True, "pass": False, "reason": "task_pusher_handle_unavailable"}
    dc.set_rigid_body_disable_gravity(pusher, True)
    initial = float(dc.get_dof_state(dof, _dynamic_control.STATE_ALL).pos)
    target_state = planned_target
    observed = [initial]
    contact_steps = 0
    active_steps = 0
    observed_contact_bodies = set()
    half = pusher_size * 0.5

    def actuate(step, steps):
        nonlocal contact_steps, active_steps
        alpha = (step + 1) / steps
        position = start * (1 - alpha) + end * alpha
        set_body_pose(dc, pusher, position, (0.0, 0.0, 0.0, 1.0))
        # 只移动物理操作器；关节由接触和资产自身的 joint drive 响应。
        # 不再用 position target 主动把 DOF 拉到目标，避免把控制器成功冒充成任务成功。
        dc.wake_up_articulation(articulation)
        observed.append(float(dc.get_dof_state(dof, _dynamic_control.STATE_ALL).pos))
        active_steps += 1
        frame = (
            contact_frame(contact_sensor, sorted(target_links))
            if contact_sensor
            else {"contact": False}
        )
        for contact in frame.get("contacts", []):
            observed_contact_bodies.update(
                str(contact.get(key, ""))
                for key in ("body0", "body1")
                if contact.get(key) not in (None, "")
            )
        if frame["contact"]:
            contact_steps += 1

    run_steps(app, float(CONFIG["simulation"]["task_segment_seconds"]), actuate)
    span = upper - lower
    state_completion = max(abs(value - initial) for value in observed) / span
    contact_fraction = contact_steps / active_steps if active_steps else 0.0
    final_state = float(dc.get_dof_state(dof, _dynamic_control.STATE_ALL).pos)
    finite_state = all(math.isfinite(value) for value in observed)
    limit_tolerance = max(
        float(CONFIG["simulation"].get("joint_limit_tolerance_min_m", 0.005)),
        float(CONFIG["simulation"].get("joint_limit_tolerance_ratio", 0.02)) * span,
    )
    limit_violation = (
        not finite_state
        or min(observed) < lower - limit_tolerance
        or max(observed) > upper + limit_tolerance
    )
    passed = (
        finite_state
        and not limit_violation
        and state_completion >= float(CONFIG["simulation"]["task_completion_min"])
    )
    timeline.stop()
    return {
        "applicable": True,
        "pass": passed,
        "articulation_path": articulation_path,
        "dof": dof_name,
        "joint_type": joint_type,
        "joint_axis_world": axis.tolist(),
        "planned_state_direction": desired_sign,
        "commanded_target_state": target_state,
        "interaction_bounds_source": bounds_source,
        "interaction_center_world": child_center.tolist(),
        "target_bounds_world": [child_min.tolist(), child_max.tolist()],
        "pusher_start_world": start.tolist(),
        "pusher_end_world": end.tolist(),
        "task": "slide" if joint_type == "translation" else "rotate",
        "initial_state": initial,
        "final_state": final_state,
        "state_completion": state_completion,
        "limit_violation": limit_violation,
        "contact_fraction": contact_fraction,
        "contact_fraction_role": "diagnostic_only",
        "observed_contact_bodies": sorted(observed_contact_bodies),
        "contact_measurement": (
            "physx_contact_sensor" if contact_sensor else "unavailable"
        ),
        "annotation_source": (
            "robophyscan_raw_part_frame_affordance_motion_prior"
            if dataset == "robophyscan"
            else "baseline_joint_and_child_geometry"
        ),
        "controller_proxy": "physical_bbox_pusher_without_active_dof_target",
        "parameters_source": "paper_success_rule_plus_disclosed_reproduction_design_controller_and_timing",
    }


def complete_task_annotations(annotation: dict | None) -> list[dict]:
    if not annotation:
        return []
    keys = ("part_frames", "affordances", "grasps", "motion_priors")
    by_key = {
        key: {item["part_index"]: item for item in annotation.get(key, [])}
        for key in keys
    }
    common = set.intersection(*(set(items) for items in by_key.values())) if by_key else set()
    return [
        {"part_index": index, **{key: by_key[key][index] for key in keys}}
        for index in sorted(common)
    ]


def point_in_expanded_bounds(point, minimum, maximum) -> bool:
    import numpy as np

    point = np.asarray(point, dtype=float)
    minimum = np.asarray(minimum, dtype=float)
    maximum = np.asarray(maximum, dtype=float)
    margin = max(0.02, 0.10 * float(np.linalg.norm(maximum - minimum)))
    return bool(np.all(point >= minimum - margin) and np.all(point <= maximum + margin))


def task_grasp_match(native: list[dict], joint_names: str, minimum, maximum):
    import numpy as np

    normalized_names = str(joint_names).lower().replace("_", " ")
    component_match = next(
        (
            grasp
            for grasp in native
            if grasp.get("component")
            and str(grasp["component"]).lower().replace("_", " ") in normalized_names
        ),
        None,
    )
    if component_match is not None:
        return component_match, "native_component_match"
    spatial_matches = [
        grasp
        for grasp in native
        if grasp.get("center") is not None
        and point_in_expanded_bounds(grasp["center"], minimum, maximum)
    ]
    if spatial_matches:
        midpoint = (np.asarray(minimum, dtype=float) + np.asarray(maximum, dtype=float)) * 0.5
        spatial_matches.sort(
            key=lambda grasp: float(np.linalg.norm(np.asarray(grasp["center"]) - midpoint))
        )
        return spatial_matches[0], "native_pose_in_child_bounds"
    return None, "no_native_grasp_match"


def task_binding(stage, check: dict, articulation, dataset: str, asset: dict):
    import numpy as np

    candidates = []
    native = native_grasp_candidates(stage, check["default_prim"] or check["rigid_bodies"][0])
    if not native:
        return None
    for index, dof_name in enumerate(articulation.dof_names):
        props = articulation.dof_properties[index]
        lower, upper = float(props["lower"]), float(props["upper"])
        if not bool(props["hasLimits"]) or not all(map(math.isfinite, (lower, upper))) or upper <= lower:
            continue
        geometry = joint_force_geometry(stage, check, dof_name)
        if geometry is None:
            continue
        bounds = [stage_bounds(stage, path) for path in geometry["related"]]
        minimum = np.min([item[0] for item in bounds], axis=0)
        maximum = np.max([item[1] for item in bounds], axis=0)
        names = f"{dof_name} {geometry['child']}"
        matching, matching_source = task_grasp_match(native, names, minimum, maximum)
        if matching is None:
            continue
        priority = -2.0 if matching_source == "native_component_match" else -1.0
        candidates.append((priority, index, dof_name, lower, upper, geometry, matching, minimum, maximum, matching_source))
    return min(candidates, key=lambda item: item[0]) if candidates else None


def task_grasp_candidate(stage, check: dict, group: dict | None, minimum, maximum):
    import numpy as np

    def geometry_width(direction) -> float:
        direction = np.asarray(direction, dtype=float)
        direction /= max(1e-8, float(np.linalg.norm(direction)))
        return float(np.dot(np.abs(direction), np.asarray(maximum) - np.asarray(minimum)))

    if group is None:
        return None
    if "center" in group:
        closing = np.asarray(group["closing"], dtype=float)
        approach = np.asarray(group.get("approach", group["rotation"][:, 1]), dtype=float)
        approach -= closing * float(np.dot(closing, approach))
        approach /= max(1e-8, float(np.linalg.norm(approach)))
        authored_width = group.get("width")
        return {
            "center": np.asarray(group["center"], dtype=float),
            "closing": closing,
            "approach": approach,
            "width": geometry_width(closing),
            "authored_width_m": authored_width,
            "width_source": "child_bounds_projection",
            "orientation": quaternion_xyzw(
                np.column_stack((closing, approach, np.cross(closing, approach)))
            ),
            "source": "native_usd_component_grasp",
            "authored_pose_world": group.get("authored_pose_world"),
        }
    grasp = group["grasps"]
    matrix = np.asarray(grasp["pose_matrix"], dtype=float)
    root = check["rigid_bodies"][0]
    closing = asset_direction_world(stage, root, matrix[:3, 0])
    approach = asset_direction_world(stage, root, matrix[:3, 1])
    rotation = np.column_stack(
        [asset_direction_world(stage, root, matrix[:3, axis]) for axis in range(3)]
    )
    center = np.asarray(asset_point_world(stage, root, matrix[:3, 3]))
    authored_width = float(grasp.get("grasp_width_m", grasp.get("gripper_width_ratio", 0.5) * 0.062))
    return {
        "center": center,
        "closing": closing,
        "approach": approach,
        "width": geometry_width(closing),
        "authored_width_m": authored_width,
        "width_source": "child_bounds_projection",
        "orientation": quaternion_xyzw(rotation),
        "source": "robophyscan_raw_grouped_grasp",
        "authored_pose_world": None,
    }


def task_contact_quality(asset_contacts, target_contacts, minimum_fraction: float) -> dict:
    count = len(asset_contacts)
    bilateral_fraction = (
        sum(left and right for left, right in asset_contacts) / count if count else 0.0
    )
    target_contact_fraction = (
        sum(left or right for left, right in target_contacts) / len(target_contacts)
        if target_contacts
        else 0.0
    )
    threshold = float(minimum_fraction)
    return {
        "bilateral_fraction": bilateral_fraction,
        "target_contact_fraction": target_contact_fraction,
        "pass": bilateral_fraction >= threshold and target_contact_fraction >= threshold,
    }


def task_drive_command(geometry: dict, gripper_center, direction_sign: float, mass: float, gravity: float) -> dict:
    import numpy as np

    force_base = 5.0 * max(1e-6, float(mass)) * abs(float(gravity))
    if geometry["type"] == "translation":
        direction = np.asarray(geometry["axis"], dtype=float) * float(direction_sign)
        direction /= max(1e-8, float(np.linalg.norm(direction)))
        return {
            "direction": direction,
            "force_n": min(100.0, max(1.0, force_base)),
            "torque_nm": None,
        }
    radial = np.asarray(gripper_center, dtype=float) - np.asarray(geometry["pivot"], dtype=float)
    radius = max(1e-4, float(np.linalg.norm(radial)))
    direction = np.cross(np.asarray(geometry["axis"], dtype=float), radial)
    direction *= float(direction_sign)
    direction /= max(1e-8, float(np.linalg.norm(direction)))
    torque = min(10.0, max(0.05, force_base * radius))
    return {"direction": direction, "force_n": torque / radius, "torque_nm": torque}


def link_relative_gripper_target(
    initial_link_position,
    initial_link_orientation,
    initial_gripper_position,
    initial_gripper_orientation,
    current_link_position,
    current_link_orientation,
) -> dict:
    import numpy as np
    from scipy.spatial.transform import Rotation

    initial_link_rotation = Rotation.from_quat(initial_link_orientation)
    relative_position = initial_link_rotation.inv().apply(
        np.asarray(initial_gripper_position, dtype=float)
        - np.asarray(initial_link_position, dtype=float)
    )
    relative_rotation = initial_link_rotation.inv() * Rotation.from_quat(
        initial_gripper_orientation
    )
    current_link_rotation = Rotation.from_quat(current_link_orientation)
    target_rotation = current_link_rotation * relative_rotation
    return {
        "position": np.asarray(current_link_position, dtype=float)
        + current_link_rotation.apply(relative_position),
        "orientation": target_rotation.as_quat(),
        "closing": target_rotation.apply((1.0, 0.0, 0.0)),
    }


def _task_actuation_once(
    stage, check: dict, app, dataset: str, asset: dict, direction_sign: float,
    finger_scale: float, gripper_geometry: str = "flat",
) -> dict:
    import numpy as np

    settings = CONFIG["simulation"]
    authored = sorted(native_grasp_candidates(
        stage, check["default_prim"] or check["rigid_bodies"][0]
    ), key=lambda item: item["prim_path"])
    if not authored:
        return {"applicable": False, "pass": False, "reason": "missing_authored_grasppose"}
    timeline, _minimum, _maximum, runtime_articulation = begin_physics(
        stage, check, app,
        settle_seconds=float(settings["interaction_settle_seconds"]),
        return_articulation=True, pause_after_settle=True,
    )
    articulation, articulation_path, unavailable_reason = runtime_articulation
    if articulation is None:
        timeline.stop()
        return unavailable_articulation_result(unavailable_reason)
    binding = task_binding(stage, check, articulation, dataset, asset)
    timeline.stop()
    # Match begin_physics' stopped-stage synchronization before grasp geometry.
    for _ in range(2):
        app.update()
    if binding is None:
        if not check.get("limited_nonfixed_joint_count", 0):
            return {"applicable": False, "pass": False, "reason": "no_limited_nonfixed_dof",
                    "reason_class": "not_applicable", "status": "not_applicable"}
        return {
            "applicable": False,
            "pass": False,
            "reason": "raw_task_annotation_binding_failed",
            "reason_class": "missing", "status": "missing",
        }
    _score, dof_index, dof_name, lower, upper, geometry, group, child_min, child_max, binding_source = binding
    rank = next((index for index, item in enumerate(authored)
                 if item["prim_path"] == group.get("prim_path")), None)
    if rank is None:
        return {"applicable": False, "pass": False, "reason": "authored_grasp_target_unresolved"}
    target_links = sorted(geometry["related"])
    acquisition = acquire_grasp(
        stage, check, app, dataset,
        {**asset, "_grasp_candidate_rank": rank, "_grasp_finger_scale": finger_scale,
         "_grasp_geometry": gripper_geometry},
        target_paths=target_links, prepared_timeline=timeline,
    )
    if "state" not in acquisition:
        return acquisition
    state = acquisition["state"]
    dc, timeline = state["dc"], state["timeline"]
    gripper, descriptor = state["gripper"], state["descriptor"]
    candidate, opening_geometry = state["candidate"], state["opening_geometry"]
    palm, left, right = state["palm"], state["left"], state["right"]
    left_sensor, right_sensor = state["left_sensor"], state["right_sensor"]
    center = np.asarray(candidate["center"], dtype=float)
    diagonal = float(np.linalg.norm(np.asarray(child_max) - np.asarray(child_min)))
    object_mass = state["object_mass"]
    pregrasp_center, approach_distance = state["pregrasp_center"], state["approach_distance"]
    approach_contacts = state["approach_contacts"]
    approach_contact_parts = state["approach_contact_parts"]
    approach_max_penetration = state["approach_max_penetration"]
    observed_contact_bodies = state["observed_contact_bodies"]
    left_any, right_any = state["left_any_contact"], state["right_any_contact"]
    seating_quality = task_contact_quality(
        state["seating_contacts"], state["seating_contacts"],
        settings["grasp_bilateral_contact_fraction_min"],
    )
    bilateral_initial = bool(acquisition["grasp_acquisition"]["pass"] and seating_quality["pass"])
    # Capture closure targets before releasing the temporary world hold.
    hold_targets = parallel_gripper_drive_targets(stage, gripper)
    if state["contact_hold"] is not None:
        stage.RemovePrim(state["contact_hold"]["path"])
    acquisition["grasp_acquisition"]["palm_hold_released_for_motion"] = True
    articulation, articulation_path, unavailable_reason = modern_articulation(check)
    if articulation is None:
        timeline.stop()
        return {**unavailable_articulation_result(unavailable_reason),
                "grasp_acquisition": acquisition["grasp_acquisition"]}
    dof_index = list(articulation.dof_names).index(dof_name)
    target_link = dc.get_rigid_body(geometry["child"])
    _, blocked = rigid_body_mass_or_blocked(dc, target_link, geometry["child"], "actuation_target")
    if blocked:
        timeline.stop()
        return {**blocked, "grasp_acquisition": acquisition["grasp_acquisition"]}
    initial = float(articulation.get_joint_positions()[dof_index])
    observed = [initial]
    max_velocity = 0.0
    segment_steps = duration_steps(settings["task_segment_seconds"], settings["dt"]) if bilateral_initial else 0
    motion_speed = max(0.02, min(0.15, diagonal / max(1.0, float(settings["task_segment_seconds"]))))
    drive_asset_contacts = []
    drive_target_contacts = []
    palm_mass = float(dc.get_rigid_body_properties(palm).mass)
    palm_inertia = palm_mass * sum(value * value for value in descriptor["palm_dimensions"]) / 12.0
    initial_link_pose = dc.get_rigid_body_pose(target_link)
    initial_palm_pose = dc.get_rigid_body_pose(palm)
    initial_link_position = (
        initial_link_pose.p.x,
        initial_link_pose.p.y,
        initial_link_pose.p.z,
    )
    initial_link_orientation = (
        initial_link_pose.r.x,
        initial_link_pose.r.y,
        initial_link_pose.r.z,
        initial_link_pose.r.w,
    )
    initial_palm_position = (
        initial_palm_pose.p.x,
        initial_palm_pose.p.y,
        initial_palm_pose.p.z,
    )
    initial_palm_orientation = (
        initial_palm_pose.r.x,
        initial_palm_pose.r.y,
        initial_palm_pose.r.z,
        initial_palm_pose.r.w,
    )
    max_drive_force = 0.0
    max_link_relative_position_error = 0.0
    max_link_relative_orientation_error = 0.0
    commanded_torque = None
    for _step in range(segment_steps):
        left_pose = dc.get_rigid_body_pose(left).p
        right_pose = dc.get_rigid_body_pose(right).p
        gripper_center = np.asarray(((left_pose.x + right_pose.x) * 0.5, (left_pose.y + right_pose.y) * 0.5, (left_pose.z + right_pose.z) * 0.5))
        command = task_drive_command(geometry, gripper_center, direction_sign, object_mass, settings["gravity"])
        commanded_torque = command["torque_nm"]
        palm_pose = dc.get_rigid_body_pose(palm)
        link_pose = dc.get_rigid_body_pose(target_link)
        target_pose = link_relative_gripper_target(
            initial_link_position,
            initial_link_orientation,
            initial_palm_position,
            initial_palm_orientation,
            (link_pose.p.x, link_pose.p.y, link_pose.p.z),
            (link_pose.r.x, link_pose.r.y, link_pose.r.z, link_pose.r.w),
        )
        palm_velocity = dc.get_rigid_body_linear_velocity(palm)
        velocity = np.asarray((palm_velocity.x, palm_velocity.y, palm_velocity.z))
        desired_velocity = command["direction"] * motion_speed
        palm_position = np.asarray((palm_pose.p.x, palm_pose.p.y, palm_pose.p.z))
        palm_force = command["direction"] * command["force_n"] + palm_mass * (
            float(settings["grasp_lift_servo_kp"]) * (target_pose["position"] - palm_position)
            + float(settings["grasp_lift_servo_kd"]) * (desired_velocity - velocity)
        )
        force_limit = max(1.0, 2.0 * float(command["force_n"]))
        magnitude = float(np.linalg.norm(palm_force))
        if magnitude > force_limit:
            palm_force *= force_limit / magnitude
        max_drive_force = max(max_drive_force, float(np.linalg.norm(palm_force)))
        apply_world_body_force(dc, palm, palm_force, (palm_pose.p.x, palm_pose.p.y, palm_pose.p.z))
        angular_velocity = dc.get_rigid_body_angular_velocity(palm)
        link_angular_velocity = dc.get_rigid_body_angular_velocity(target_link)
        relative_angular_velocity = (
            angular_velocity.x - link_angular_velocity.x,
            angular_velocity.y - link_angular_velocity.y,
            angular_velocity.z - link_angular_velocity.z,
        )
        torque = virtual_palm_torque(
            (palm_pose.r.x, palm_pose.r.y, palm_pose.r.z, palm_pose.r.w),
            target_pose["orientation"],
            relative_angular_velocity,
            palm_inertia,
            settings["grasp_palm_rotation_kp"],
            settings["grasp_palm_rotation_kd"],
            force_limit * max(descriptor["palm_dimensions"]) * 0.5,
        )
        dc.apply_body_torque(palm, tuple(torque), False)
        tracking_error = pose_arrival_error(
            target_pose["position"],
            target_pose["orientation"],
            palm_position,
            (palm_pose.r.x, palm_pose.r.y, palm_pose.r.z, palm_pose.r.w),
        )
        max_link_relative_position_error = max(
            max_link_relative_position_error,
            tracking_error["position_error_m"],
        )
        max_link_relative_orientation_error = max(
            max_link_relative_orientation_error,
            tracking_error["orientation_error_rad"],
        )
        set_parallel_gripper_drive_targets(stage, gripper, hold_targets)
        blocked = state["handle_blocked"]("actuation_drive_before_update")
        if blocked:
            return {**blocked, "grasp_acquisition": acquisition["grasp_acquisition"]}
        app.update()
        blocked = state["handle_blocked"]("actuation_drive_after_update")
        if blocked:
            return {**blocked, "grasp_acquisition": acquisition["grasp_acquisition"]}
        left_asset = contact_frame(left_sensor, check["rigid_bodies"])["contact"]
        right_asset = contact_frame(right_sensor, check["rigid_bodies"])["contact"]
        left_target = contact_frame(left_sensor, target_links)["contact"]
        right_target = contact_frame(right_sensor, target_links)["contact"]
        drive_asset_contacts.append((left_asset, right_asset))
        drive_target_contacts.append((left_target, right_target))
        position = float(articulation.get_joint_positions()[dof_index])
        joint_velocity = abs(float(articulation.get_joint_velocities()[dof_index]))
        observed.append(position)
        max_velocity = max(max_velocity, joint_velocity)
        signed_completion = direction_sign * (position - initial) / (upper - lower)
        if signed_completion >= float(settings["task_completion_min"]):
            break
    final_left_frame = contact_frame(left_sensor)
    final_right_frame = contact_frame(right_sensor)
    final_left = contact_frame(left_sensor, target_links)["contact"]
    final_right = contact_frame(right_sensor, target_links)["contact"]
    final = float(articulation.get_joint_positions()[dof_index])
    span = upper - lower
    completion = max(0.0, max(direction_sign * (value - initial) for value in observed) / span)
    tolerance = max(
        float(settings["joint_limit_tolerance_min_m"] if geometry["type"] == "translation" else settings["joint_limit_tolerance_min_rad"]),
        float(settings["joint_limit_tolerance_ratio"]) * span,
    )
    finite = all(math.isfinite(value) for value in observed)
    limit_violation = not finite or min(observed) < lower - tolerance or max(observed) > upper + tolerance
    velocity_limit = float(settings["joint_velocity_explosion_mps"] if geometry["type"] == "translation" else settings["joint_velocity_explosion_radps"])
    contact_quality = task_contact_quality(
        drive_asset_contacts,
        drive_target_contacts,
        settings["task_contact_fraction_min"],
    )
    passed = (
        opening_geometry["opening_supported"]
        and bilateral_initial
        and contact_quality["pass"]
        and completion >= float(settings["task_completion_min"])
        and not limit_violation
        and max_velocity <= velocity_limit
    )
    timeline.stop()
    return {
        "grasp_acquisition": acquisition["grasp_acquisition"],
        "applicable": True,
        "pass": passed,
        "articulation_path": articulation_path,
        "dof": dof_name,
        "joint_type": geometry["type"],
        "direction_sign": direction_sign,
        "finger_scale": finger_scale,
        "gripper_geometry": gripper_geometry,
        **opening_geometry,
        "gripper_structure": "dynamic_palm_mirrored_prismatic_fingers",
        "gripper_material_roles": gripper["materials"],
        "approach_collision_free": True,
        "approach_contact_fraction": sum(any(pair) for pair in approach_contacts) / max(1, len(approach_contacts)),
        **opening_geometry,
        "opening_reason": None if opening_geometry["opening_supported"] else "grasp_opening_exceeds_reference",
        "approach_contact_parts": sorted(approach_contact_parts),
        "approach_penetration_max_m": approach_max_penetration,
        "approach_penetration_tolerance_m": float(settings["penetration_max_m"]),
        "approach_object_motion_failure_criterion": False,
        "approach_distance_m": approach_distance,
        "pregrasp_center_world": pregrasp_center.tolist(),
        "grasp_center_world": center.tolist(),
        "grasp_width_m": float(candidate["width"]),
        "authored_grasp_width_m": candidate.get("width"),
        "grasp_width_source": candidate.get("candidate_width_source", candidate["source"]),
        "initial_bilateral_contact": bilateral_initial,
        "initial_bilateral_contact_fraction": seating_quality["bilateral_fraction"],
        "initial_target_contact_fraction": seating_quality["target_contact_fraction"],
        "initial_any_contact_both_fingers_diagnostic": left_any and right_any,
        "final_bilateral_contact": final_left and final_right,
        "final_contact_is_diagnostic_only": True,
        "final_any_contact_both_fingers_diagnostic": final_left_frame["contact"] and final_right_frame["contact"],
        "contact_persistence": contact_quality["bilateral_fraction"],
        "target_contact_persistence": contact_quality["target_contact_fraction"],
        "observed_contact_bodies_diagnostic": sorted(observed_contact_bodies),
        "state_completion": completion,
        "initial_state": initial,
        "final_state": final,
        "limit_violation": limit_violation,
        "max_velocity": max_velocity,
        "maximum_drive_force_n": max_drive_force,
        "maximum_link_relative_position_error_m": max_link_relative_position_error,
        "maximum_link_relative_orientation_error_rad": max_link_relative_orientation_error,
        "commanded_joint_torque_nm": commanded_torque,
        "target_links": target_links,
        "task_part_index": group.get("part_index") if group else None,
        "binding_source": binding_source,
        "grasp_source": candidate["source"],
        "controller_proxy": "force_feedback_parallel_gripper_with_force_limited_palm_joint_drive",
        "parameters_source": "shared_grasp_acquisition_r26_plus_force_limited_joint_actuation",
    }


def task_actuation(stage, check: dict, app, dataset: str, asset: dict) -> dict:
    source_path = Path(stage.GetRootLayer().realPath or stage.GetRootLayer().identifier)
    attempts = []
    geometries = CONFIG["simulation"].get("grasp_geometry_modes", ["flat"])
    scales = CONFIG["simulation"].get("grasp_finger_scales", [1.0])
    total_attempts = len(geometries) * len(scales) * 2
    for gripper_geometry in geometries:
        for finger_scale in scales:
            for direction_sign in (1.0, -1.0):
                started = time.monotonic()
                print(
                    f"interaction actuation attempt={len(attempts) + 1}/"
                    f"{total_attempts} geometry={gripper_geometry} "
                    f"direction={direction_sign:+g} finger_scale={float(finger_scale):g} phase=start",
                    flush=True,
                )
                attempt_stage = stage if not attempts else open_stage(source_path, app)
                attempt_check = check if not attempts else precheck(attempt_stage)
                result = _task_actuation_once(
                    attempt_stage, attempt_check, app, dataset, asset,
                    direction_sign, float(finger_scale), str(gripper_geometry),
                )
                attempts.append(result)
                print(
                    f"interaction actuation attempt={len(attempts)}/"
                    f"{total_attempts} geometry={gripper_geometry} "
                    f"direction={direction_sign:+g} finger_scale={float(finger_scale):g} phase=end "
                    f"pass={bool(result.get('pass'))} elapsed_s={time.monotonic() - started:.1f}",
                    flush=True,
                )
                if result.get("reason") == "no_limited_nonfixed_dof":
                    return {**result, "attempts": attempts}
                if result.get("reason") == "missing_authored_grasppose":
                    return {**result, "status": "missing", "reason_class": "missing", "attempts": attempts}
                if result.get("pass"):
                    return {**result, "attempts": attempts}
    if attempts and all(item.get("status") in {"evaluation_blocked", "missing"} for item in attempts):
        representative = next((item for item in attempts if item.get("status") == "evaluation_blocked"), attempts[0])
        return {**representative, "applicable": False, "attempts": attempts}
    return {
        "applicable": any(item.get("applicable") for item in attempts),
        "pass": False,
        "reason": next((item.get("reason") for item in attempts if item.get("reason")), "both_task_directions_failed"),
        "attempts": attempts,
    }


def mesh_triangles(stage):
    import numpy as np
    from pxr import Usd, UsdGeom

    visual_triangles = []
    collision_triangles = []
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for prim in stage_prims(stage, include_instance_proxies=True):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get() or []
        counts = mesh.GetFaceVertexCountsAttr().Get() or []
        indices = mesh.GetFaceVertexIndicesAttr().Get() or []
        if not points or not counts or not indices:
            continue
        matrix = cache.GetLocalToWorldTransform(prim)
        vertices = np.asarray([matrix.Transform(point) for point in points], dtype=float)
        triangles = []
        offset = 0
        for count in counts:
            face = indices[offset : offset + count]
            offset += count
            for index in range(1, count - 1):
                triangles.append(vertices[[face[0], face[index], face[index + 1]]])
        if not triangles:
            continue
        triangles = np.asarray(triangles)
        current = prim
        is_collision = False
        while current and current.IsValid():
            if collision_enabled(current):
                is_collision = True
                break
            current = current.GetParent()
        path = str(prim.GetPath()).lower()
        if "/visuals/" in path:
            visual_triangles.append(triangles)
        elif "/colliders/" in path or "/collisions/" in path:
            collision_triangles.append(triangles)
        elif path.startswith("/meshes/"):
            # Isaac URDF importer 的共享几何资源库，不是场景中的独立表面。
            continue
        elif is_collision:
            # 部分原生 USD 用同一个可见 mesh 同时承载碰撞。
            visual_triangles.append(triangles)
            collision_triangles.append(triangles)
        else:
            visual_triangles.append(triangles)
    return visual_triangles, collision_triangles


def sample_triangles(triangle_sets, count: int, rng):
    import numpy as np

    triangles = np.concatenate(triangle_sets)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1) * 0.5
    valid = areas > 1e-12
    triangles = triangles[valid]
    areas = areas[valid]
    if not len(triangles):
        return np.empty((0, 3))
    selected = rng.choice(len(triangles), size=count, replace=True, p=areas / areas.sum())
    picked = triangles[selected]
    u = np.sqrt(rng.random(count))
    v = rng.random(count)
    return (
        (1 - u)[:, None] * picked[:, 0]
        + (u * (1 - v))[:, None] * picked[:, 1]
        + (u * v)[:, None] * picked[:, 2]
    )


def partnet_source_physics(asset):
    import xml.etree.ElementTree as ET

    root = ET.parse(asset["source_asset"]).getroot()
    coverage, plausibility = [], []
    details = {"links": [], "joints": []}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        mass_node = inertial.find("mass") if inertial is not None else None
        inertia_node = inertial.find("inertia") if inertial is not None else None
        mass_authored = mass_node is not None and mass_node.get("value") is not None
        inertia_authored = inertia_node is not None and all(inertia_node.get(name) is not None for name in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"))
        coverage.extend((float(mass_authored), float(inertia_authored)))
        mass_valid = mass_authored and math.isfinite(float(mass_node.get("value"))) and float(mass_node.get("value")) > 0
        inertia_valid = False
        if inertia_authored:
            import numpy as np
            values = {name: float(inertia_node.get(name)) for name in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
            matrix = np.asarray([[values["ixx"], values["ixy"], values["ixz"]], [values["ixy"], values["iyy"], values["iyz"]], [values["ixz"], values["iyz"], values["izz"]]])
            eigenvalues = np.linalg.eigvalsh(matrix)
            inertia_valid = bool(np.all(np.isfinite(matrix)) and np.all(eigenvalues > 0) and eigenvalues[2] <= eigenvalues[0] + eigenvalues[1] + 1e-9)
        plausibility.extend((float(mass_valid), float(inertia_valid)))
        details["links"].append({"name": link.get("name"), "mass_authored": mass_authored, "mass_valid": mass_valid, "inertia_authored": inertia_authored, "inertia_valid": inertia_valid})
    for joint in root.findall("joint"):
        axis, limit, dynamics = joint.find("axis"), joint.find("limit"), joint.find("dynamics")
        joint_type = joint.get("type") or ""
        fields = {
            "axis": axis is not None and axis.get("xyz") is not None,
            "lower_limit": limit is not None and limit.get("lower") is not None,
            "upper_limit": limit is not None and limit.get("upper") is not None,
            "damping": dynamics is not None and dynamics.get("damping") is not None,
            "friction": dynamics is not None and dynamics.get("friction") is not None,
            "effort": limit is not None and limit.get("effort") is not None,
        }
        applicable = {
            "fixed": (),
            "continuous": ("axis", "damping", "friction", "effort"),
            "revolute": tuple(fields),
            "prismatic": tuple(fields),
        }.get(joint_type, ())
        coverage.extend(float(fields[name]) for name in applicable)
        numeric = []
        for node, name in ((limit, "lower"), (limit, "upper"), (limit, "effort"), (dynamics, "damping"), (dynamics, "friction")):
            if node is not None and node.get(name) is not None:
                numeric.append(float(node.get(name)))
        numeric_valid = all(math.isfinite(value) for value in numeric)
        if applicable:
            plausibility.append(float(numeric_valid))
        details["joints"].append({"name": joint.get("name"), "type": joint_type, "applicable_metadata": list(applicable), "authored": fields, "numeric_valid": numeric_valid})
    return {"metadata_coverage": sum(coverage) / len(coverage) if coverage else 0.0, "plausibility_score": sum(plausibility) / len(plausibility) if plausibility else 0.0, "details": details, "provenance": "source_urdf_not_importer_defaults"}


def asset_quality(stage, check: dict, source_physics=None, records=None) -> dict:
    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    records = records if records is not None else mesh_records(stage)
    collision_bounds_by_body = {}
    for row in records:
        if not row["collision"] or not row["rigid_body"]:
            continue
        vertices = row["vertices"]
        minimum, maximum = vertices.min(axis=0), vertices.max(axis=0)
        bounds = collision_bounds_by_body.setdefault(
            row["rigid_body"], [minimum.copy(), maximum.copy()]
        )
        bounds[0] = np.minimum(bounds[0], minimum)
        bounds[1] = np.maximum(bounds[1], maximum)
    rigid_rows = []
    plausible = []
    metadata = []
    rigid_masses = {}
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for path in check.get("rigid_bodies", []):
        prim = stage.GetPrimAtPath(path)
        values = {}
        for name in ("physics:mass", "physics:density", "physics:diagonalInertia", "physics:centerOfMass"):
            attribute = prim.GetAttribute(name)
            authored = bool(attribute and attribute.HasAuthoredValueOpinion())
            values[name] = {"authored": authored, "value": attribute.Get() if authored else None}
        mass = values["physics:mass"]["value"]
        density = values["physics:density"]["value"]
        rigid_masses[path] = mass
        metadata.extend((
            float(values["physics:mass"]["authored"] or values["physics:density"]["authored"]),
            float(values["physics:diagonalInertia"]["authored"]),
        ))
        scalar_valid = all(value is None or (math.isfinite(float(value)) and float(value) > 0) for value in (mass, density))
        inertia = values["physics:diagonalInertia"]["value"]
        inertia_valid = True
        if inertia is not None:
            principal = sorted(map(float, inertia))
            inertia_valid = all(math.isfinite(value) and value > 0 for value in principal) and principal[2] <= principal[0] + principal[1] + 1e-9
        com_valid = True
        com_checked = False
        com = values["physics:centerOfMass"]["value"]
        bounds = collision_bounds_by_body.get(path)
        if com is not None and bounds is not None:
            com_checked = True
            world_com = xform_cache.GetLocalToWorldTransform(prim).Transform(Gf.Vec3d(*map(float, com)))
            bounds_diagonal = distance(bounds[0], bounds[1])
            com_valid = all(bounds[0][axis] - 0.1 * bounds_diagonal <= float(world_com[axis]) <= bounds[1][axis] + 0.1 * bounds_diagonal for axis in range(3))
        diagonal = distance(bounds[0], bounds[1]) if bounds is not None else None
        scale_ratio = inertia_scale_ratio(mass, inertia, diagonal)
        inertia_scale_valid = scale_ratio is None or 1e-8 <= scale_ratio <= 10.0
        plausible.extend((float(scalar_valid), float(inertia_valid), float(com_valid), float(inertia_scale_valid)))
        rigid_rows.append({"path": path, "properties": values, "mass_or_density_authored": values["physics:mass"]["authored"] or values["physics:density"]["authored"], "mass_density_valid": scalar_valid, "inertia_valid": inertia_valid, "com_checked_against_collision": com_checked, "com_valid": com_valid, "collision_diagonal_m": diagonal, "inertia_mass_scale_ratio": scale_ratio, "inertia_scale_valid": inertia_scale_valid})
    joint_rows = []
    graph_edges = []
    bound_pairs = []
    for path in check.get("joints", []):
        prim = stage.GetPrimAtPath(path)
        applicable_names = joint_metadata_names(prim.GetTypeName())
        authored = {}
        for name in applicable_names:
            attribute = prim.GetAttribute(name)
            authored[name] = bool(attribute and attribute.HasAuthoredValueOpinion())
        metadata.extend(float(value) for value in authored.values())
        lower = prim.GetAttribute("physics:lowerLimit").Get()
        upper = prim.GetAttribute("physics:upperLimit").Get()
        limits_valid = lower is None or upper is None or (math.isfinite(float(lower)) and math.isfinite(float(upper)) and float(upper) >= float(lower))
        joint = UsdPhysics.Joint(prim)
        parents = [str(value) for value in joint.GetBody0Rel().GetTargets()]
        children = [str(value) for value in joint.GetBody1Rel().GetTargets()]
        link_mass_ratios = [mass_ratio(rigid_masses.get(parent), rigid_masses.get(child)) for parent in parents for child in children]
        link_mass_ratios = [value for value in link_mass_ratios if value is not None]
        mass_ratio_valid = all(value <= 1e6 for value in link_mass_ratios)
        graph_edges.extend((parent, child) for parent in parents for child in children)
        bound_pairs.extend((path, child) for child in children)
        state_values = []
        unresolved_state_attrs = []
        for name in ("state:angular:physics:position", "state:linear:physics:position"):
            attribute = prim.GetAttribute(name)
            if attribute and attribute.HasAuthoredValueOpinion():
                value = attribute.Get()
                if value is None:
                    unresolved_state_attrs.append(name)
                else:
                    state_values.append(float(value))
        initial_in_limits = not unresolved_state_attrs and all(math.isfinite(value) and (lower is None or value >= float(lower)) and (upper is None or value <= float(upper)) for value in state_values)
        plausible.extend((float(limits_valid), float(initial_in_limits), float(mass_ratio_valid)))
        joint_rows.append({"path": path, "joint_type": prim.GetTypeName(), "applicable_metadata": list(applicable_names), "authored": authored, "limits_valid": limits_valid, "initial_state": state_values, "unresolved_initial_state_attrs": unresolved_state_attrs, "initial_state_in_limits": initial_in_limits, "parents": parents, "children": children, "parent_child_mass_ratios": link_mass_ratios, "mass_ratio_valid": mass_ratio_valid})
    material_attrs = []
    for prim in stage_prims(stage, include_instance_proxies=True):
        row = {}
        for name in ("physics:staticFriction", "physics:dynamicFriction", "physics:restitution"):
            attribute = prim.GetAttribute(name)
            row[name] = bool(attribute and attribute.HasAuthoredValueOpinion())
            if row[name]:
                value = attribute.Get()
                plausible.append(float(value is not None and math.isfinite(float(value)) and float(value) >= 0 and (name != "physics:restitution" or float(value) <= 1)))
        if any(row.values()):
            material_attrs.append({"path": str(prim.GetPath()), **row})
    metadata.extend(float(any(row[name] for row in material_attrs)) for name in ("physics:staticFriction", "physics:dynamicFriction", "physics:restitution"))
    adjacency = defaultdict(set)
    for parent, child in graph_edges:
        adjacency[parent].add(child)
    visiting, visited = set(), set()
    def cyclic(node):
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        found = any(cyclic(child) for child in adjacency.get(node, ()))
        visiting.remove(node); visited.add(node)
        return found
    graph_cycle = any(cyclic(node) for node in list(adjacency))
    duplicate_child_binding = len(bound_pairs) != len(set(bound_pairs)) or len([child for _path, child in bound_pairs]) != len(set(child for _path, child in bound_pairs))
    graph_nodes = set(check.get("rigid_bodies", []))
    connected_nodes = {value for edge in graph_edges for value in edge}
    disconnected = sorted(graph_nodes - connected_nodes) if graph_edges else []
    graph_valid = not graph_cycle and not duplicate_child_binding and not check.get("missing_joint_bodies")
    plausible.append(float(graph_valid))
    readiness = float(check.get("load_pass", False))
    metadata_score = sum(metadata) / len(metadata) if metadata else 0.0
    plausibility_score = sum(plausible) / len(plausible) if plausible else 0.0
    if source_physics is not None:
        metadata_score = float(source_physics["metadata_coverage"])
        plausibility_score = (float(source_physics["plausibility_score"]) + plausibility_score) * 0.5
    score = readiness * (metadata_score + plausibility_score) * 0.5
    strict_pass = bool(readiness and all(plausible))
    return {
        "score": score,
        "applicable": True,
        "strict_pass": strict_pass,
        "diagnostics": {
            "readiness": check,
            "metadata_coverage": metadata_score,
            "plausibility_score": plausibility_score,
            "rigid_bodies": rigid_rows,
            "joints": joint_rows,
            "physics_materials": material_attrs,
            "articulation_graph": {"edges": graph_edges, "cycle": graph_cycle, "duplicate_child_binding": duplicate_child_binding, "disconnected_rigid_bodies": disconnected, "valid": graph_valid},
            "source_physics_metadata": source_physics,
        },
        "reason": None if strict_pass else ("physics_configuration_quality_threshold_not_met" if readiness else "asset_readiness_failed"),
    }


def mesh_records(stage):
    import numpy as np
    from pxr import Usd, UsdGeom

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    records = []
    for prim in stage_prims(stage, include_instance_proxies=True):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get() or []
        counts = mesh.GetFaceVertexCountsAttr().Get() or []
        indices = mesh.GetFaceVertexIndicesAttr().Get() or []
        if not points or not counts or not indices:
            continue
        vertices = np.asarray([cache.GetLocalToWorldTransform(prim).Transform(point) for point in points], dtype=float)
        faces, offset = [], 0
        for count in counts:
            face = indices[offset : offset + count]
            offset += count
            faces.extend((int(face[0]), int(face[index]), int(face[index + 1])) for index in range(1, count - 1))
        if not faces:
            continue
        current = prim
        collision = False
        rigid_body = None
        while current and current.IsValid():
            applied = schemas(current)
            collision |= collision_enabled(current)
            if rigid_body is None and "PhysicsRigidBodyAPI" in applied:
                rigid_body = str(current.GetPath())
            current = current.GetParent()
        path = str(prim.GetPath()).lower()
        shared_library = path.startswith("/meshes/")
        explicit_visual = "/visuals/" in path
        explicit_collision = "/colliders/" in path or "/collisions/" in path
        records.append({
            "path": str(prim.GetPath()),
            "vertices": vertices,
            "faces": np.asarray(faces, dtype=int),
            "collision": bool(collision or explicit_collision) and not shared_library,
            # A renderable mesh can also carry CollisionAPI. Only dedicated
            # collision namespaces are collision-only geometry.
            "visual": bool(explicit_visual or not explicit_collision) and not shared_library,
            "rigid_body": rigid_body,
        })
    return records


def initial_collision_overlap(stage, check, records, max_triangle_pairs=50_000):
    import numpy as np
    from v5_geometry import triangle_aabb_pairs, triangles_intersect

    collision = [row for row in records if row["collision"] and row["rigid_body"]]
    graph_edges = []
    for path in check.get("joints", []):
        from pxr import UsdPhysics
        joint = UsdPhysics.Joint(stage.GetPrimAtPath(path))
        for parent in joint.GetBody0Rel().GetTargets():
            for child in joint.GetBody1Rel().GetTargets():
                graph_edges.append((str(parent), str(child)))
    adjacent = collapsed_collision_adjacency(graph_edges, (row["rigid_body"] for row in collision))
    rows = []
    checked_pairs = 0
    truncated = False
    budget_exhausted = False
    witnessed_body_pairs = set()
    for index, first in enumerate(collision):
        first_triangles = first["vertices"][first["faces"]]
        for second in collision[index + 1 :]:
            if first["rigid_body"] == second["rigid_body"]:
                continue
            pair = frozenset((first["rigid_body"], second["rigid_body"]))
            if max_triangle_pairs is None and pair in witnessed_body_pairs:
                continue
            second_triangles = second["vertices"][second["faces"]]
            if not len(first_triangles) or not len(second_triangles):
                continue
            if any(first_triangles.max(axis=(0, 1)) < second_triangles.min(axis=(0, 1))) or any(second_triangles.max(axis=(0, 1)) < first_triangles.min(axis=(0, 1))):
                continue
            intersections = 0
            for left, right in triangle_aabb_pairs(first_triangles, second_triangles):
                if max_triangle_pairs is not None and checked_pairs >= int(max_triangle_pairs):
                    truncated = budget_exhausted = True
                    break
                checked_pairs += 1
                intersections += int(triangles_intersect(first_triangles[left], second_triangles[right]))
                if intersections and max_triangle_pairs is None:
                    witnessed_body_pairs.add(pair)
                    break
            if intersections:
                rows.append({"body0": first["rigid_body"], "body1": second["rigid_body"], "adjacent_joint_links": pair in adjacent, "triangle_intersection_count_capped": intersections})
            if budget_exhausted:
                break
        if budget_exhausted:
            break
    severe = any(not row["adjacent_joint_links"] for row in rows)
    return {
        "score": 0.0 if severe else (None if truncated else 0.5 if rows else 1.0),
        "nonadjacent_overlap": severe,
        "pairs": rows,
        "check_truncated": truncated,
        "checked_triangle_pairs": checked_pairs,
        "intersection_counts_are_lower_bounds": max_triangle_pairs is None,
    }


def collision_alignment(stage, check: dict, records=None) -> dict:
    import numpy as np
    from scipy.spatial import cKDTree
    from v5_geometry import audit_collision_mesh, bounds_metrics, complexity_metrics, mean_score

    records = records if records is not None else mesh_records(stage)
    visual_triangles = [row["vertices"][row["faces"]] for row in records if row["visual"]]
    collision_triangles = [row["vertices"][row["faces"]] for row in records if row["collision"]]
    if not visual_triangles or not collision_triangles:
        missing = "visual" if not visual_triangles else "collision"
        return {"score": 0.0, "applicable": True, "strict_pass": False, "status": "evaluated", "reason": f"missing_{missing}_mesh", "diagnostics": {"visual_mesh_count": len(visual_triangles), "collision_mesh_count": len(collision_triangles)}}
    visual_faces = np.concatenate(visual_triangles)
    collision_faces = np.concatenate(collision_triangles)
    exact_match = visual_faces.shape == collision_faces.shape and np.allclose(
        visual_faces, collision_faces, rtol=0.0, atol=1e-9
    )
    sample_count = int(CONFIG["simulation"]["collision_samples"])
    diagonal = float(
        np.linalg.norm(visual_faces.reshape(-1, 3).max(axis=0) - visual_faces.reshape(-1, 3).min(axis=0))
    )
    settings = CONFIG["simulation"]
    tolerance = min(
        float(settings["collision_tolerance_max_m"]),
        max(
            float(settings["collision_tolerance_min_m"]),
            float(settings["collision_tolerance_diagonal_ratio"]) * diagonal,
        ),
    )
    rng = np.random.default_rng(int(CONFIG["simulation"]["seed"]))
    visual = sample_triangles(visual_triangles, sample_count, rng)
    collision = sample_triangles(collision_triangles, sample_count, rng)
    if not len(visual) or not len(collision):
        return {"score": 0.0, "applicable": False, "strict_pass": False, "reason": "degenerate_visual_or_collision_mesh", "diagnostics": {}}
    visual_distances = np.zeros(len(visual)) if exact_match else cKDTree(collision).query(visual, workers=-1)[0]
    collision_distances = np.zeros(len(collision)) if exact_match else cKDTree(visual).query(collision, workers=-1)[0]
    visual_coverage = float(np.mean(visual_distances <= tolerance))
    reverse_coverage = float(np.mean(collision_distances <= tolerance))
    surface_fscore = 2 * visual_coverage * reverse_coverage / max(1e-12, visual_coverage + reverse_coverage)
    visual_vertices = visual_faces.reshape(-1, 3)
    collision_vertices = collision_faces.reshape(-1, 3)
    bounds = bounds_metrics(collision_vertices, visual_vertices)
    fidelity_score = mean_score(surface_fscore, bounds.get("scale_score"), bounds.get("center_alignment_score"))
    topology_rows = []
    for row in records:
        if row["collision"]:
            topology_rows.append({"path": row["path"], **audit_collision_mesh(
                row["vertices"],
                row["faces"],
                watertight_expected=bool(check.get("rigid_bodies")),
                # Exact streamed decisions; legacy count caps remain available to diagnostic callers.
                self_intersection_max_pairs=None,
                topology_max_triangles=None,
            )})
    topology_score = mean_score(*(row.get("score") for row in topology_rows))
    overlap = initial_collision_overlap(
        stage,
        check,
        records,
        max_triangle_pairs=None,
    )
    complexity = complexity_metrics(
        sum(len(row["faces"]) for row in records if row["collision"]),
        sum(len(row["vertices"]) for row in records if row["collision"]),
        sum(row["collision"] for row in records),
        len(check.get("collisions", [])),
        sum("convex" in row["path"].lower() for row in records if row["collision"]),
        sum(len(row["faces"]) for row in records if row["visual"]),
        dict(sorted((body, sum(row["collision"] and row["rigid_body"] == body for row in records)) for body in {row["rigid_body"] for row in records if row["collision"] and row["rigid_body"]})),
    )
    score = mean_score(fidelity_score, topology_score, bounds.get("score"), overlap["score"], complexity["score"])
    severe_topology = any(row.get("nonmanifold_edge_count", 0) or row.get("zero_area_face_count", 0) or row.get("self_intersection_count", 0) or (row.get("watertight_expected") and not row.get("source_watertight")) for row in topology_rows)
    budget_unknown = bool(overlap.get("check_truncated")) or any(
        row.get("self_intersection_check_truncated") or row.get("topology_check_truncated")
        for row in topology_rows
    )
    audit_complete = not overlap.get("check_truncated") and not any(
        row.get("self_intersection_check_truncated") or row.get("topology_check_truncated")
        for row in topology_rows
    )
    strict_pass = audit_complete and visual_coverage >= float(settings["collision_visual_coverage_min"]) and reverse_coverage >= float(settings["collision_visual_coverage_min"]) and not severe_topology and not overlap["nonadjacent_overlap"] and bounds.get("valid_extent") and bounds.get("center_offset_visual_diagonal", 0) <= 1.0
    return {
        "score": score,
        "applicable": True,
        "strict_pass": bool(strict_pass),
        "pass": bool(strict_pass),
        "status": "evaluated" if audit_complete else "partial",
        "diagnostics": {
            "fidelity": {"score": fidelity_score, "tolerance_m": tolerance, "sample_count_each_surface": sample_count, "visual_to_collision_coverage": visual_coverage, "collision_to_visual_coverage": reverse_coverage, "surface_fscore": surface_fscore, "visual_distance_p50_m": float(np.percentile(visual_distances, 50)), "visual_distance_p95_m": float(np.percentile(visual_distances, 95)), "visual_distance_max_m": float(visual_distances.max()), "collision_distance_p50_m": float(np.percentile(collision_distances, 50)), "collision_distance_p95_m": float(np.percentile(collision_distances, 95)), "collision_distance_max_m": float(collision_distances.max())},
            "topology": {"score": topology_score, "collision_meshes": topology_rows},
            "watertightness": [{key: row.get(key) for key in ("path", "watertight_expected", "source_watertight", "effective_collider_closed", "boundary_edge_ratio", "open_component_count", "closed_component_count", "orientation_reversed")} for row in topology_rows],
            "bounds": bounds,
            "initial_overlap": overlap,
            "complexity": complexity,
        },
        "reason": None if strict_pass else (
            "collision_audit_budget_exhausted"
            if budget_unknown
            else "collision_quality_threshold_not_met"
        ),
    }


def result_row(dataset: str, asset: dict, test: str, load_success: bool, payload: dict) -> dict:
    return {
        "run_id": time.strftime("%Y%m%d_%H%M%S"),
        "run_label": RUN_LABEL,
        "seed": int(CONFIG["simulation"]["seed"]),
        "dataset": dataset,
        "asset_id": asset["asset_id"],
        "category": asset.get("category"),
        "source_asset": asset["source_asset"],
        "selection": asset.get("selection"),
        "test": test,
        "load_success": load_success,
        **payload,
    }


def bounded_score(value, threshold, lower_is_better=True):
    if value is None:
        return 0.0
    value, threshold = float(value), max(1e-12, float(threshold))
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, 1.0 - value / threshold)) if lower_is_better else max(0.0, min(1.0, value / threshold))


def stability_quality(payload):
    if not payload.get("applicable", True):
        result = {
            "score": None,
            "applicable": False,
            "strict_pass": False,
            "diagnostics": payload,
            "reason": payload.get("reason"),
        }
        if payload.get("status"):
            result["status"] = payload["status"]
        return result
    settings = CONFIG["simulation"]
    support = payload.get("support", {})
    persistence = float(support.get("contact_persistence", 0.0))
    support_existence = 1.0 if support.get("support_exists") else (0.5 if persistence > 0 else 0.0)
    margin = support.get("com_signed_margin_m")
    margin_score = (
        0.5 if support_existence else 0.0
    ) if margin is None else max(0.0, min(1.0, 0.5 + float(margin) / 0.02))
    penetration_score = bounded_score(payload.get("maximum_ground_penetration_m", math.inf), settings["penetration_max_m"])
    support_score = sum((persistence, support_existence, margin_score, penetration_score)) / 4
    settling_parts = (
        bounded_score(payload.get("root_drift_m", math.inf), settings["root_drift_max_m"]),
        bounded_score(payload.get("final_linear_speed_mps", math.inf), settings["linear_speed_max_mps"]),
        bounded_score(payload.get("final_angular_speed_radps", math.inf), settings["angular_speed_max_radps"]),
        float(payload.get("kinetic_energy_decay_score", 0.0)),
        penetration_score,
    )
    settling_score = sum(settling_parts) / len(settling_parts)
    severe = bool(payload.get("severe_runtime_event"))
    score = 0.0 if severe else (support_score + settling_score) * 0.5
    return {"score": score, "applicable": payload.get("applicable", True), "strict_pass": bool(payload.get("pass")), "severe_runtime_event": severe, "diagnostics": {"support_score": support_score, "settling_score": settling_score, **payload}, "reason": payload.get("reason")}


def push_quality(payload):
    direction_scores = []
    for sample in payload.get("samples", []):
        acquisition = float(sample.get("contact", False))
        sustained = float(sample.get("contact_fraction", 0.0))
        penetration = bounded_score(sample.get("maximum_contact_penetration_m") or 0.0, CONFIG["simulation"]["push_penetration_max_m"])
        bounded = float(sample.get("motion_bounded", False))
        recovery = float(sample.get("recovery", {}).get("pass", False))
        direction_scores.append(sum((acquisition, sustained, penetration, bounded, recovery)) / 5)
    samples = payload.get("samples", [])
    severe = any(sample.get("recovery", {}).get("severe_runtime_event", False) for sample in samples)
    runtime_sample = any(
        str(sample.get("reason", "")).startswith(("runtime_", "push_runtime", "timeout"))
        or sample.get("reason_class") in {"runtime", "evaluator"}
        for sample in samples
    )
    reason_class = payload.get("reason_class") or ("runtime" if severe else "asset")
    if severe or runtime_sample or (payload.get("status") == "evaluation_blocked" and reason_class in {"runtime", "evaluator"}):
        return {"score": None, "applicable": False, "strict_pass": False, "status": "evaluation_blocked", "severe_runtime_event": severe, "diagnostics": {"direction_scores": direction_scores, **payload}, "reason": payload.get("reason") or "push_runtime_blocked", "reason_class": reason_class}
    return {"score": (sum(direction_scores) / len(direction_scores) if direction_scores else 0.0), "applicable": payload.get("applicable", True), "strict_pass": bool(payload.get("pass")), "severe_runtime_event": severe, "diagnostics": {"direction_scores": direction_scores, **payload}, "reason": payload.get("reason"), "reason_class": reason_class}


def articulation_quality(payload, check):
    if not payload.get("applicable"):
        result = {"score": None, "applicable": False, "strict_pass": False, "diagnostics": payload, "reason": payload.get("reason"), "reason_class": payload.get("reason_class")}
        if payload.get("status"):
            result["status"] = payload["status"]
        return result
    rows = payload.get("joints", [])
    scores = []
    blocked_joints = 0
    blocked_details = []
    for row in rows:
        if row.get("reason") in {"joint_child_geometry_unavailable", "joint_child_handle_unavailable", "joint_moving_mass_unavailable", "unsupported_authored_drive_protocol"}:
            blocked_joints += 1
            blocked_details.append({key: row.get(key) for key in (
                "joint_path", "joint_type", "dof_index", "lower_limit",
                "upper_limit", "child_body", "articulation_handle",
                "failure_phase", "reason",
            )})
            continue
        structure = float(not check.get("missing_joint_bodies") and row.get("axis_pivot_valid", False))
        forward = (max(0.0, min(1.0, float(row.get("upper_completion", 0.0)))) + float(row.get("forward_monotonicity", 0.0))) * 0.5
        returned = (max(0.0, min(1.0, float(row.get("return_completion", 0.0)))) + float(row.get("return_monotonicity", 0.0)) + bounded_score(row.get("normalized_hysteresis", math.inf), 0.05)) / 3
        limit = float(not row.get("limit_violation", True))
        link = float(not row.get("detached_child", True))
        isolation = bounded_score(row.get("unrelated_link_motion_m", math.inf), CONFIG["simulation"]["unrelated_link_motion_max_m"])
        row["quality_score"] = sum((structure, forward, returned, limit, link, isolation)) / 6
        scores.append(row["quality_score"])
    severe_reasons = {"nonfinite_joint_state", "joint_velocity_explosion", "nonfinite_actuation_vector"}
    catastrophic_joints = sum(row.get("reason") in severe_reasons for row in rows)
    if not scores:
        return {"score": None, "applicable": False, "strict_pass": False, "status": "evaluation_blocked" if blocked_joints else "not_applicable", "diagnostics": {"dof_coverage": 0.0, "blocked_joint_count": blocked_joints, "blocked_joints": blocked_details, "catastrophic_joint_count": catastrophic_joints, **payload}, "reason": "all_joint_trials_blocked" if blocked_joints else payload.get("reason"), "reason_class": "evaluator" if blocked_joints else "not_applicable"}
    return {"score": sum(scores) / len(scores), "applicable": True, "strict_pass": bool(payload.get("pass")) and catastrophic_joints == 0, "severe_runtime_event": catastrophic_joints > 0, "diagnostics": {"dof_coverage": len(scores) / max(1, len(check.get("joints", []))), "blocked_joint_count": blocked_joints, "catastrophic_joint_count": catastrophic_joints, **payload}, "reason": payload.get("reason"), "reason_class": "runtime" if catastrophic_joints else "asset"}


def grasp_score(payload):
    if payload.get("grasp_score_common") is not None:
        return max(0.0, min(1.0, float(payload["grasp_score_common"])))
    attempts = payload.get("attempts", [payload])
    scores = []
    for row in attempts:
        bilateral = float(row.get("bilateral_close_contact", False))
        hold = float(row.get("bilateral_hold_contact_fraction", 0.0)) * float(row.get("hold_penetration_pass", False))
        clearance = max(0.0, min(1.0, (row.get("hold_ground_clearance_min_m") or 0.0) / max(1e-12, row.get("required_ground_clearance_m", 1.0))))
        lift_ratio = max(0.0, min(1.0, float(row.get("object_lift_m", 0.0)) / max(1e-12, float(row.get("required_object_lift_m", 0.10)))))
        score = lift_ratio * sum((bilateral, clearance, bilateral * bounded_score(row.get("object_gripper_slip_m", math.inf), CONFIG["simulation"]["grasp_slip_max_m"]), hold)) / 4
        scores.append(score)
    return max(scores, default=0.0)


def actuation_score(payload):
    attempts = payload.get("attempts", [payload])
    scores = []
    for row in attempts:
        scores.append(sum((float(bool(row.get("target_links"))), float(row.get("initial_bilateral_contact", False)), float(row.get("final_bilateral_contact", False)), max(0.0, min(1.0, float(row.get("state_completion", 0.0)))), float(not row.get("limit_violation", True)))) / 5)
    return max(scores, default=0.0)


def grasp_quality(payload):
    if not payload.get("applicable", True):
        result = {"score": None, "applicable": False, "strict_pass": False, "diagnostics": payload, "reason": payload.get("reason"), "reason_class": payload.get("reason_class")}
        if payload.get("status"):
            result["status"] = payload["status"]
        return result
    return {
        "score": grasp_score(payload),
        "applicable": True,
        "strict_pass": bool(payload.get("pass")),
        "diagnostics": payload,
        "reason": payload.get("reason"),
        "reason_class": payload.get("reason_class") or ("runtime" if any(str(item.get("reason", "")).startswith("runtime_") for item in payload.get("attempts", [])) else "asset"),
    }


def actuation_quality(payload):
    if not payload.get("applicable", True):
        result = {"score": None, "applicable": False, "strict_pass": False, "diagnostics": payload, "reason": payload.get("reason"), "reason_class": payload.get("reason_class")}
        if payload.get("status"):
            result["status"] = payload["status"]
        return result
    return {
        "score": actuation_score(payload),
        "applicable": True,
        "strict_pass": bool(payload.get("pass")),
        "diagnostics": payload,
        "reason": payload.get("reason"),
        "reason_class": payload.get("reason_class") or ("runtime" if any(str(item.get("reason", "")).startswith("runtime_") for item in payload.get("attempts", [])) else "asset"),
    }


def v5_base_row(dataset, asset, path):
    selection = asset.get("selection") or {}
    return {"run_id": time.strftime("%Y%m%d_%H%M%S"), "run_label": RUN_LABEL, "seed": int(CONFIG["simulation"]["seed"]), "dataset": dataset, "asset_id": asset["asset_id"], "category": asset.get("category"), "anchor_category": selection.get("anchor_category"), "anchor_robophyscan_id": selection.get("anchor_robophyscan_id"), "source_asset": str(path) if path else None, "selection": selection}


def inspect_assets(app, dataset, assets):
    for index, asset in enumerate(assets, 1):
        started = time.monotonic()
        path = resolved_asset_path(dataset, asset)
        base = v5_base_row(dataset, asset, path)
        print(f"[{index}/{len(assets)}] inspect opening path={path}", flush=True)
        try:
            stage = open_stage(path, app) if path and path.exists() else None
        except BaseException as exc:
            blocked = {"score": None, "applicable": False, "strict_pass": False, "status": "evaluation_blocked", "reason_class": "evaluator", "diagnostics": {}, "reason": "stage_open_exception", "error": f"{type(exc).__name__}: {exc}"}
            append_result({**base, "elapsed_seconds": time.monotonic() - started, "metrics": {"physics_configuration_quality": blocked, "collision_quality": blocked}})
            print(f"[{index}/{len(assets)}] inspect open failed: {blocked['error']}", flush=True)
            continue
        if stage is None:
            blocked = {"score": None, "applicable": False, "strict_pass": False, "status": "evaluation_blocked", "reason_class": "evaluator", "diagnostics": {}, "reason": "stage_open_failed"}
            append_result({**base, "elapsed_seconds": time.monotonic() - started, "metrics": {"physics_configuration_quality": blocked, "collision_quality": blocked}})
            continue
        try:
            print(f"[{index}/{len(assets)}] inspect phase=precheck asset={asset['asset_id']}", flush=True)
            check = precheck(stage)
            source_physics = partnet_source_physics(asset) if dataset == "partnet_mobility" else None
            records = mesh_records(stage)
            print(f"[{index}/{len(assets)}] inspect phase=static_collision asset={asset['asset_id']}", flush=True)
            collision = collision_alignment(stage, check, records)
            if "initial_overlap" in collision.get("diagnostics", {}):
                print(f"[{index}/{len(assets)}] inspect phase=first_physx_step asset={asset['asset_id']}", flush=True)
                dynamic_stage = open_stage(path, app)
                first_step = first_physx_link_contacts(dynamic_stage, precheck(dynamic_stage), app)
                initial = collision["diagnostics"]["initial_overlap"]
                initial["first_physx_step"] = first_step
                if initial.get("score") is None:
                    collision["status"] = "partial"
                    collision["reason"] = "initial_overlap_budget_exhausted"
                else:
                    old_score = float(initial["score"])
                    first_step_score = 0.0 if first_step["severe"] else 1.0
                    initial["score"] = min(old_score, first_step_score)
                    if collision.get("score") is not None:
                        collision["score"] = max(0.0, float(collision["score"]) + (initial["score"] - old_score) / 5.0)
                    if first_step["severe"]:
                        collision["strict_pass"] = collision["pass"] = False
                        collision["reason"] = "first_physx_step_link_penetration"
            physics = asset_quality(stage, check, source_physics, records)
            if isinstance(physics, dict) and physics.get("status") is None and physics.get("score") is not None:
                physics["status"] = "evaluated"
            if isinstance(physics, dict) and physics.get("score") is not None:
                physics.setdefault("reason_class", "asset")
            if isinstance(collision, dict) and collision.get("status") is None and collision.get("score") is not None:
                collision["status"] = "evaluated"
                collision.setdefault("reason_class", "asset")
            if isinstance(collision, dict) and collision.get("status") == "partial":
                # Partial collision audits are terminal evaluator blocks, not
                # comparable static scores and not scheduler completion gaps.
                partial_score = collision.get("score")
                raw_reason = collision.get("reason") or "partial_collision_audit"
                collision = {
                    **collision,
                    "score": None,
                    "applicable": False,
                    "strict_pass": False,
                    "status": "evaluation_blocked",
                    "reason": raw_reason,
                    "reason_class": "evaluator",
                    "diagnostics": {
                        **collision.get("diagnostics", {}),
                        "partial_score": partial_score,
                        "blocked_at": "inspect_collision",
                    },
                }
            metrics = {"physics_configuration_quality": physics, "collision_quality": collision}
        except BaseException as exc:
            blocked = {"score": None, "applicable": False, "strict_pass": False, "status": "evaluation_blocked", "diagnostics": {}, "reason": "inspection_exception", "error": f"{type(exc).__name__}: {exc}"}
            blocked["traceback"] = traceback.format_exc()
            metrics = {"physics_configuration_quality": blocked, "collision_quality": blocked}
        append_result({**base, "elapsed_seconds": time.monotonic() - started, "metrics": metrics})
        print(f"[{index}/{len(assets)}] inspected {asset['asset_id']}", flush=True)


def stop_timeline() -> None:
    try:
        import omni.timeline

        omni.timeline.get_timeline_interface().stop()
    except Exception:
        pass


def run_dynamic_metric(app, asset_id: str, trial: int, metric_name: str, function):
    timeout = float(
        CONFIG["simulation"].get("metric_wall_timeout_seconds", {}).get(
            metric_name, 3600.0
        )
    )
    guarded_app = DeadlineApp(app, timeout, metric_name)
    print(
        f"simulate asset={asset_id} trial={trial} metric={metric_name} "
        f"phase=start timeout_s={timeout:g}",
        flush=True,
    )
    try:
        result = function(guarded_app)
        elapsed_seconds = time.monotonic() - guarded_app.started
        if isinstance(result, dict):
            result = {**result, "elapsed_seconds": elapsed_seconds}
        print(
            f"simulate asset={asset_id} trial={trial} metric={metric_name} "
            f"phase=end elapsed_s={elapsed_seconds:.1f}",
            flush=True,
        )
        return result
    except BaseException as exc:
        timed_out = isinstance(exc, EvaluationTimeout)
        reason = "evaluation_timeout" if timed_out else "evaluation_exception"
        error = f"{type(exc).__name__}: {exc}"
        trace = traceback.format_exc()
        print(
            f"simulate asset={asset_id} trial={trial} metric={metric_name} "
            f"phase=blocked reason={reason} error={error}\n{trace}",
            flush=True,
        )
        event = {
            "phase": "evaluation",
            "time_s": time.monotonic() - guarded_app.started,
            "body": None,
            "event": reason,
            "value": time.monotonic() - guarded_app.started,
            "threshold": timeout if timed_out else None,
            "severe": False,
        }
        return {
            "score": None,
            "applicable": False,
            "strict_pass": False,
            "status": "evaluation_blocked",
            "diagnostics": {},
            "reason": reason,
            "error": error,
            "traceback": trace,
            "runtime_events": [event],
            "elapsed_seconds": time.monotonic() - guarded_app.started,
        }
    finally:
        stop_timeline()


def simulation_job_in_shard(asset_index, trial, trials, shard_index, shard_count):
    return (asset_index * trials + trial) % shard_count == shard_index


def simulate_assets(app, dataset, assets, shard_index=0, shard_count=1):
    trials = int(CONFIG["simulation"].get("dynamic_trials", 3))
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError(f"invalid shard {shard_index}/{shard_count}")
    completed = 0
    assigned = sum(
        simulation_job_in_shard(asset_index, trial, trials, shard_index, shard_count)
        for asset_index in range(len(assets))
        for trial in range(trials)
    )
    for asset_index, asset in enumerate(assets):
        path = resolved_asset_path(dataset, asset)
        for trial in range(trials):
            if not simulation_job_in_shard(asset_index, trial, trials, shard_index, shard_count):
                continue
            base = {**v5_base_row(dataset, asset, path), "trial": trial}
            metrics = {}
            def checkpoint():
                append_result({
                    **base,
                    "metrics": dict(metrics),
                    "runtime_events": [],
                    "checkpoint": {"complete_metrics": list(metrics)},
                })
            if not path or not path.exists():
                failed = {"score": 0.0, "applicable": True, "strict_pass": False, "diagnostics": {}, "reason": "stage_open_failed"}
                metrics = {name: failed for name in ("settling_stability", "push_robustness", "articulation_quality", "grasp_quality", "actuation_quality")}
            else:
                def opened(metric_app):
                    current_stage = open_stage(path, metric_app)
                    return current_stage, precheck(current_stage)

                def settling_stability(metric_app):
                    current_stage, current_check = opened(metric_app)
                    return stability_quality(settle(current_stage, current_check, metric_app))

                def push(metric_app):
                    current_stage, current_check = opened(metric_app)
                    return push_quality(push_contact(current_stage, current_check, metric_app))

                articulation_check = {}

                def articulation(metric_app):
                    nonlocal articulation_check
                    current_stage, articulation_check = opened(metric_app)
                    return articulation_quality(
                        joint_sweep(current_stage, articulation_check, metric_app),
                        articulation_check,
                    )

                metrics["settling_stability"] = run_dynamic_metric(app, asset["asset_id"], trial, "settling_stability", settling_stability)
                checkpoint()
                metrics["push_robustness"] = run_dynamic_metric(app, asset["asset_id"], trial, "push_robustness", push)
                checkpoint()
                metrics["articulation_quality"] = run_dynamic_metric(app, asset["asset_id"], trial, "articulation_quality", articulation)
                checkpoint()

                def grasp(metric_app):
                    current_stage, current_check = opened(metric_app)
                    return grasp_lift(current_stage, current_check, metric_app, dataset, asset)

                def actuation(metric_app):
                    current_stage, current_check = opened(metric_app)
                    return task_actuation(current_stage, current_check, metric_app, dataset, asset)

                grasp_payload = run_dynamic_metric(app, asset["asset_id"], trial, "grasp_quality", grasp)
                metrics["grasp_quality"] = grasp_quality(grasp_payload)
                metrics["grasp_quality"]["elapsed_seconds"] = grasp_payload.get("elapsed_seconds")
                checkpoint()
                actuation_payload = run_dynamic_metric(app, asset["asset_id"], trial, "actuation_quality", actuation)
                metrics["actuation_quality"] = actuation_quality(actuation_payload)
                metrics["actuation_quality"]["elapsed_seconds"] = actuation_payload.get("elapsed_seconds")
                checkpoint()
            runtime_events = []
            def collect(value, metric_name):
                # Metric diagnostics can be deeply nested and may reuse a container.
                # Use an explicit stack so result collection cannot overflow Python's
                # recursion limit or loop forever on a cyclic diagnostic payload.
                pending = [value]
                visited = set()
                while pending:
                    current = pending.pop()
                    if isinstance(current, dict):
                        identity = id(current)
                        if identity in visited:
                            continue
                        visited.add(identity)
                        for event in current.get("runtime_events", []):
                            if not isinstance(event, dict):
                                continue
                            runtime_events.append({"dataset": dataset, "asset_id": asset["asset_id"], "trial": trial, "metric": metric_name, "phase": event.get("phase"), "time": event.get("time_s"), "body": event.get("body"), "event": event.get("event"), "value": event.get("value"), "threshold": event.get("threshold"), "severe": event.get("severe", False)})
                        pending.extend(
                            item
                            for key, item in current.items()
                            if key != "runtime_events"
                        )
                    elif isinstance(current, list):
                        identity = id(current)
                        if identity in visited:
                            continue
                        visited.add(identity)
                        pending.extend(current)
            for metric_name, value in metrics.items():
                collect(value, metric_name)
            append_result({**base, "metrics": metrics, "runtime_events": runtime_events})
            completed += 1
            print(
                f"[{completed}/{assigned}] simulated asset={asset['asset_id']} trial={trial} "
                f"shard={shard_index}/{shard_count}",
                flush=True,
            )


def interaction_failure_reason(kind, row):
    if row.get("reason"):
        return row["reason"]
    if kind == "grasp":
        if not row.get("bilateral_close_contact"):
            return "bilateral_contact_failed"
        if not row.get("ground_clearance_pass"):
            return "insufficient_ground_clearance"
        if not row.get("gripper_trajectory_pass", False):
            return "gripper_trajectory_drift"
        if row.get("object_gripper_slip_m", math.inf) >= CONFIG["simulation"]["grasp_slip_max_m"]:
            return "excessive_slip"
        if row.get("bilateral_lift_hold_contact_fraction", 0.0) < CONFIG["simulation"]["grasp_bilateral_contact_fraction_min"]:
            return "hold_contact_lost"
    else:
        if not row.get("initial_bilateral_contact"):
            return "initial_target_contact_failed"
        if row.get("limit_violation"):
            return "limit_violation"
        if row.get("state_completion", 0.0) < CONFIG["simulation"]["task_completion_min"]:
            return "state_completion_failed"
    return "physical_execution_failed"


def video_samples():
    rows = read_jsonl(ROOT / "runs" / "trials_v5.jsonl")
    selected = {}
    for row in sorted(rows, key=lambda item: (item["dataset"], item["asset_id"], item.get("trial", 0))):
        for kind, metric_name in (("grasp", "grasp_quality"), ("actuation", "actuation_quality")):
            payload = row.get("metrics", {}).get(metric_name, {}).get("diagnostics", {})
            if not isinstance(payload, dict):
                continue
            attempts = payload.get("attempts")
            if not isinstance(attempts, list):
                attempts = []
            attempts = [item for item in attempts if isinstance(item, dict)] or [payload]
            if payload.get("pass"):
                attempt = next((item for item in attempts if item.get("pass")), attempts[0] if attempts else {})
                reason = "success"
                key = (row["dataset"], kind, reason)
                selected.setdefault(key, {"row": row, "attempt": attempt})
            else:
                for attempt in attempts or [{}]:
                    reason = interaction_failure_reason(kind, attempt or payload)
                    key = (row["dataset"], kind, reason)
                    selected.setdefault(key, {"row": row, "attempt": attempt or payload})
    return [(key, value) for key, value in sorted(selected.items())]


def video_target_render_evidence(stage, target_path: str) -> dict:
    """Read authored target visual/material evidence without touching its layers."""
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    root = stage.GetPrimAtPath(target_path)
    root_layer = stage.GetRootLayer()
    layer_path = Path(root_layer.realPath) if root_layer.realPath else None
    meshes = 0
    visible_meshes = 0
    material_paths = set()
    texture_paths = []
    missing_texture_paths = []
    for prim in Usd.PrimRange(root) if root else ():
        imageable = UsdGeom.Imageable(prim)
        if imageable and prim != root:
            meshes += 1
            if imageable.ComputeVisibility() != UsdGeom.Tokens.invisible:
                visible_meshes += 1
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if material:
                material_paths.add(str(material.GetPath()))
        for attribute in prim.GetAttributes():
            if attribute.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            value = attribute.Get()
            asset_path = str(getattr(value, "resolvedPath", "") or getattr(value, "path", ""))
            if not asset_path:
                continue
            texture_paths.append(asset_path)
            candidate = Path(asset_path)
            if not candidate.is_absolute() and layer_path is not None:
                candidate = layer_path.parent / candidate
            if not candidate.exists():
                missing_texture_paths.append(asset_path)
    return {
        "target_prim": target_path,
        "target_prim_valid": bool(root),
        "visible_mesh_count": visible_meshes,
        "mesh_count": meshes,
        "material_binding_count": len(material_paths),
        "material_paths": sorted(material_paths),
        "texture_asset_paths": sorted(set(texture_paths)),
        "missing_texture_asset_paths": sorted(set(missing_texture_paths)),
        "root_layer_identifier": root_layer.identifier,
        "used_layer_identifiers": sorted(layer.identifier for layer in stage.GetUsedLayers()),
    }


class VideoRecordingApp:
    def __init__(self, app, stage, output_path, bounds):
        import cv2
        import numpy as np
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, Usd, UsdGeom, UsdLux

        context_stage = omni.usd.get_context().get_stage()
        if context_stage is None or context_stage.GetRootLayer().identifier != stage.GetRootLayer().identifier:
            raise RuntimeError("video_stage_context_mismatch")

        with Usd.EditContext(stage, stage.GetSessionLayer()):
            dome = UsdLux.DomeLight.Define(stage, "/__raw_eval/VideoDomeLight")
            dome.GetIntensityAttr().Set(550.0)
            key = UsdLux.DistantLight.Define(stage, "/__raw_eval/VideoKeyLight")
            key.GetIntensityAttr().Set(3300.0)
            UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 30.0, 0.0))

        minimum, maximum = np.asarray(bounds[0]), np.asarray(bounds[1])
        self.center = (minimum + maximum) * 0.5
        self.diagonal = max(0.1, float(np.linalg.norm(maximum - minimum)))
        default_view_direction = np.asarray((1.0, 1.0, 1.0)) / np.sqrt(3.0)
        self.camera_position = self.center + default_view_direction * self.diagonal * 2.2
        self.clipping_range = (
            max(0.0001, self.diagonal * 0.01),
            max(100.0, self.diagonal * 100.0),
        )
        self.stage = stage
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (640, 480))
        if not self.writer.isOpened():
            raise RuntimeError(f"video_writer_open_failed: {output_path}")
        self.app = app
        self.cv2 = cv2
        self.np = np
        self.rep = rep
        self.update_count = 0
        self.capture_stride = max(1, round((1.0 / float(CONFIG["simulation"]["dt"])) / 20.0))
        self.frame_count = 0
        self.rejected_frame_count = 0
        self.render_product = None
        self.annotator = None
        self._attach_replicator()

    def _attach_replicator(self):
        if self.annotator is not None:
            try:
                self.annotator.detach(self.render_product)
            except Exception:
                pass
        if self.render_product is not None:
            self.render_product.destroy()
        camera = self.rep.create.camera(
            position=tuple(self.camera_position),
            look_at=tuple(self.center),
            look_at_up_axis=(0.0, 0.0, 1.0),
            clipping_range=self.clipping_range,
        )
        self.render_product = self.rep.create.render_product(camera, (640, 480))
        self.annotator = self.rep.AnnotatorRegistry.get_annotator("rgb")
        self.annotator.attach(self.render_product)
        # Initialize Replicator before physics starts. Captures below only read
        # the RGB product already produced by the real SimulationApp step.
        self.rep.orchestrator.step(delta_time=0.0, pause_timeline=True)

    def _sync_gripper_visuals(self):
        from omni.isaac.dynamic_control import _dynamic_control
        from pxr import Gf, Usd, UsdGeom

        dc = _dynamic_control.acquire_dynamic_control_interface()
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            for name in ("GripperPalm", "GripperLeft", "GripperRight"):
                source_path = f"/__raw_eval/{name}"
                source_prim = self.stage.GetPrimAtPath(source_path)
                if not source_prim:
                    continue
                handle = dc.get_rigid_body(source_path)
                if not handle:
                    continue
                if source_prim.IsA(UsdGeom.Xform):
                    for visual_prim in Usd.PrimRange(source_prim):
                        imageable = UsdGeom.Imageable(visual_prim)
                        if imageable:
                            imageable.MakeVisible()
                    pose = dc.get_rigid_body_pose(handle)
                    xform = UsdGeom.Xformable(source_prim)
                    for op in xform.GetOrderedXformOps():
                        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                            op.Set(Gf.Vec3d(float(pose.p.x), float(pose.p.y), float(pose.p.z)))
                        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                            op.Set(
                                Gf.Quatf(
                                    float(pose.r.w),
                                    Gf.Vec3f(float(pose.r.x), float(pose.r.y), float(pose.r.z)),
                                )
                            )
                    continue
                raise RuntimeError(f"gripper_visual_tree_not_xform: {source_path}")

    def capture(self):
        self.update_count += 1
        if self.update_count % self.capture_stride:
            return
        image = self.np.asarray(self.annotator.get_data())
        if image.ndim == 3 and image.shape[0] == 480 and image.shape[1] == 640:
            rgb = image[:, :, :3]
            visible_fraction = float(self.np.count_nonzero(rgb > 2)) / float(rgb.size)
            if int(rgb.max()) > 2 and visible_fraction > 0.001:
                self.writer.write(self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR))
                self.frame_count += 1
            else:
                self.rejected_frame_count += 1

    def close(self):
        self.writer.release()
        try:
            self.annotator.detach(self.render_product)
        except Exception:
            pass
        if self.render_product is not None:
            self.render_product.destroy()


def grasp_video_smoke(
    app,
    dataset: str,
    assets: list[dict],
    candidate_rank: int | None = None,
    finger_scale: float | None = None,
    gripper_geometry: str = "wrap",
) -> None:
    output_root = ensure_inside_root(ROOT / "reports" / "grasp_video_smoke_v5" / RUN_LABEL)
    if output_root.exists():
        raise FileExistsError(f"录像输出已存在，请更换 RAW_EVAL_RUN_LABEL: {output_root}")
    scales = [float(value) for value in CONFIG["simulation"].get("grasp_finger_scales", [1.0])]
    if finger_scale is not None:
        scales = [float(finger_scale)]
    ranks = list(range(max(1, int(CONFIG["simulation"].get("grasp_candidate_count", 5)))))
    if candidate_rank is not None:
        ranks = [int(candidate_rank)]
    total = len(ranks) * len(scales)
    for asset_index, asset in enumerate(assets, 1):
        path = resolved_asset_path(dataset, asset)
        attempts = []
        if not path or not path.exists():
            payload = {"applicable": False, "pass": False, "reason": "stage_open_failed", "attempts": []}
        else:
            for rank in ranks:
                for scale in scales:
                    attempt_number = len(attempts) + 1
                    print(
                        f"[{asset_index}/{len(assets)}] grasp-video asset={asset['asset_id']} "
                        f"attempt={attempt_number}/{total} candidate={rank} finger_scale={scale:g} phase=start",
                        flush=True,
                    )
                    stage = open_stage_in_context(path, app)
                    check = precheck(stage)
                    bounds_path = check["default_prim"] or check["rigid_bodies"][0]
                    render_evidence = video_target_render_evidence(stage, bounds_path)
                    visual_proxy_count = 0
                    app.update()
                    scale_name = f"{scale:g}".replace(".", "p")
                    output = output_root / asset["asset_id"] / f"candidate_{rank:02d}_scale_{scale_name}.mp4"
                    recorder = VideoRecordingApp(app, stage, output, stage_bounds(stage, bounds_path))
                    try:
                        for _ in range(round(0.5 / float(CONFIG["simulation"]["dt"]))):
                            app.update()
                            recorder.capture()
                        attempt_asset = dict(
                            asset,
                            _grasp_candidate_rank=rank,
                            _grasp_finger_scale=scale,
                            _grasp_geometry=gripper_geometry,
                        )
                        result = _grasp_lift_once(stage, check, app, dataset, attempt_asset, recorder)
                    except BaseException as exc:
                        result = {
                            "applicable": False,
                            "pass": False,
                            "reason": "evaluation_exception",
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        }
                    finally:
                        stop_timeline()
                        recorder.close()
                    if recorder.frame_count == 0:
                        raise RuntimeError(
                            f"video_capture_empty: asset={asset['asset_id']} "
                            f"candidate={rank} finger_scale={scale:g}"
                        )
                    result.update({
                        "candidate_rank": rank,
                        "finger_scale": scale,
                        "video": str(output.relative_to(ROOT)),
                        "frame_count": recorder.frame_count,
                        "rejected_frame_count": recorder.rejected_frame_count,
                        "fps": 20,
                        "resolution": [640, 480],
                        "capture_delta_time": 0.0,
                        "camera_position": recorder.camera_position.tolist(),
                        "camera_look_at": recorder.center.tolist(),
                        "camera_clipping_range": list(recorder.clipping_range),
                        "video_visual_proxy_count": visual_proxy_count,
                        "render_backend": "replicator_rgb",
                        "stage_identifier": stage.GetRootLayer().identifier,
                        "material_evidence": render_evidence,
                        "video_capture_invalid": bool(
                            recorder.frame_count == 0
                            or render_evidence["visible_mesh_count"] == 0
                            or render_evidence["missing_texture_asset_paths"]
                        ),
                    })
                    attempts.append(result)
                    print(
                        f"[{asset_index}/{len(assets)}] grasp-video asset={asset['asset_id']} "
                        f"attempt={attempt_number}/{total} phase=end pass={bool(result.get('pass'))} "
                        f"failure_phase={result.get('failure_phase')} "
                        f"force_ready={bool(result.get('force_threshold_reached'))} "
                        f"seating_fraction={float(result.get('seating_contact_fraction', 0.0)):.3f} "
                        f"lift_m={float(result.get('object_lift_m', 0.0)):.4f} "
                        f"hold_fraction={float(result.get('bilateral_hold_contact_fraction', 0.0)):.3f} "
                        f"frames={recorder.frame_count} "
                        f"rejected_frames={recorder.rejected_frame_count}",
                        flush=True,
                    )
                    if result.get("pass"):
                        break
                    if attempts and attempts[-1].get("pass"):
                        break


            applicable = any(item.get("applicable") for item in attempts)
            passed = any(item.get("pass") for item in attempts)
            payload = {
                "applicable": applicable,
                "pass": passed,
                "reason": None if passed else (
                    "all_grasp_candidates_failed" if applicable else "evaluation_blocked"
                ),
                "attempt_count": len(attempts),
                "attempts": attempts,
            }
            append_result({
                **v5_base_row(dataset, asset, path),
                "metric": "grasp_lift",
                "grasp": payload,
                "grasp_score": grasp_score(payload) if payload.get("applicable") else None,
            })


def prismatic_frame_smoke(app) -> dict:
    """Exercise the shared prismatic frame with three dynamic test cubes."""
    import numpy as np
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema
    from omni.isaac.dynamic_control import _dynamic_control
    from isaacsim.core.simulation_manager import SimulationManager
    import omni.timeline

    stage = Usd.Stage.CreateInMemory(f"raw_eval_prismatic_frame_{RUN_LABEL}.usd")
    add_session_physics(stage, 0.0)
    stage.SetEditTarget(stage.GetSessionLayer())
    UsdGeom.Xform.Define(stage, "/__raw_eval/Smoke").GetPrim()
    palm_center = np.asarray((0.0, 0.0, 0.25))
    closing = np.asarray((0.0, 1.0, 0.0))
    approach = np.asarray((-1.0, 0.0, 0.0))
    finger_centers = (palm_center - closing * 0.05, palm_center + closing * 0.05)
    contract = parallel_gripper_joint_contract(palm_center, finger_centers, closing, approach)
    orientation_xyzw = quaternion_xyzw(contract["basis"])
    orientation = Gf.Quatf(orientation_xyzw[3], *orientation_xyzw[:3])

    def jsonable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {key: jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [jsonable(item) for item in value]
        return value

    def body(path, position, mass):
        xform = UsdGeom.Xform.Define(stage, path)
        xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, position)))
        xform.AddOrientOp().Set(orientation)
        rigid = UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
        rigid.CreateKinematicEnabledAttr(False)
        UsdPhysics.MassAPI.Apply(xform.GetPrim()).CreateMassAttr(float(mass))
        cube = UsdGeom.Cube.Define(stage, f"{path}/Collision")
        cube.CreateSizeAttr(1.0)
        cube.AddScaleOp().Set(Gf.Vec3f(0.02, 0.02, 0.02))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim()).CreateRestOffsetAttr(0.0)
        return xform

    palm_path = "/__raw_eval/Smoke/Palm"
    left_path = "/__raw_eval/Smoke/Left"
    right_path = "/__raw_eval/Smoke/Right"
    body(palm_path, palm_center, 1.0)
    body(left_path, finger_centers[0], 0.25)
    body(right_path, finger_centers[1], 0.25)
    joint_paths = {}
    for name, path, side, lower, upper in (
        ("Left", left_path, "left", 0.0, 0.02),
        ("Right", right_path, "right", -0.02, 0.0),
    ):
        joint = UsdPhysics.PrismaticJoint.Define(stage, f"/__raw_eval/Smoke/{name}Joint")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(palm_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(path)])
        joint.CreateAxisAttr("X")
        joint.CreateLowerLimitAttr(float(lower))
        joint.CreateUpperLimitAttr(float(upper))
        joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, contract["local_anchor"][side])))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateTargetPositionAttr(0.01 if side == "left" else -0.01)
        drive.CreateTargetVelocityAttr(0.0)
        drive.CreateStiffnessAttr(5000.0)
        drive.CreateDampingAttr(100.0)
        drive.CreateMaxForceAttr(2.5)
        joint_paths[side] = str(joint.GetPath())
    SimulationManager.set_physics_dt(float(CONFIG["simulation"]["dt"]))
    attach_stage(stage, app)
    app.update()
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    # Let PhysX register the newly-authored session-layer bodies before querying
    # Dynamic Control handles. A single update can race stage attachment.
    for _ in range(8):
        app.update()
    dc = _dynamic_control.acquire_dynamic_control_interface()
    handles = {name: dc.get_rigid_body(path) for name, path in (("palm", palm_path), ("left", left_path), ("right", right_path))}
    handles_valid = all(handle for handle in handles.values())
    if not handles_valid:
        timeline.stop()
        app.update()
        return {
            "metric": "parallel_gripper_frame_smoke",
            "frame_contract_status": "unobservable_invalid_body_handle",
            "handle_status": {name: bool(handle) for name, handle in handles.items()},
            "joint_axis_world": contract["joint_axis_world"].tolist(),
            "joint_contract": jsonable({key: value for key, value in contract.items() if key != "local_anchor"}),
            "joint_paths": joint_paths,
        }
    for handle in handles.values():
        dc.set_rigid_body_disable_gravity(handle, True)
    start = {}
    for name, handle in handles.items():
        pose = dc.get_rigid_body_pose(handle).p
        start[name] = np.asarray((pose.x, pose.y, pose.z), dtype=float)
    samples = []
    for step in range(120):
        pose = dc.get_rigid_body_pose(handles["palm"]).p
        dc.apply_body_force(handles["palm"], (0.0, 0.0, 4.0), (pose.x, pose.y, pose.z), True)
        app.update()
        positions = {}
        for name, handle in handles.items():
            current = dc.get_rigid_body_pose(handle).p
            positions[name] = np.asarray((current.x, current.y, current.z), dtype=float)
        left_relative = positions["left"] - positions["palm"]
        right_relative = positions["right"] - positions["palm"]
        samples.append({
            "step": step,
            "left_target_position": 0.01,
            "right_target_position": -0.01,
            "left_relative": left_relative.tolist(),
            "right_relative": right_relative.tolist(),
            "common_mode": ((left_relative + right_relative) * 0.5 - (finger_centers[0] + finger_centers[1]) * 0.5 + palm_center).tolist(),
        })
    timeline.stop()
    app.update()
    final = {name: dc.get_rigid_body_pose(handle).p for name, handle in handles.items()}
    final_positions = {
        name: np.asarray((pose.x, pose.y, pose.z), dtype=float)
        for name, pose in final.items()
    }
    left_delta = final_positions["left"] - start["left"]
    right_delta = final_positions["right"] - start["right"]
    palm_delta = final_positions["palm"] - start["palm"]
    relative_drift = {
        "left": (final_positions["left"] - final_positions["palm"] - (start["left"] - start["palm"])).tolist(),
        "right": (final_positions["right"] - final_positions["palm"] - (start["right"] - start["palm"])).tolist(),
    }
    common_drift = (np.asarray(relative_drift["left"]) + np.asarray(relative_drift["right"])) * 0.5
    return {
        "metric": "parallel_gripper_frame_smoke",
        "frame_contract_status": "pass" if np.linalg.norm(common_drift) < 0.005 else "fail",
        "joint_axis_world": contract["joint_axis_world"].tolist(),
        "left_target_to_world_displacement": left_delta.tolist(),
        "right_target_to_world_displacement": right_delta.tolist(),
        "palm_world_displacement": palm_delta.tolist(),
        "relative_finger_to_palm_drift": relative_drift,
        "common_mode_drift": common_drift.tolist(),
        "joint_contract": jsonable({key: value for key, value in contract.items() if key != "local_anchor"}),
        "joint_paths": joint_paths,
        "sample_count": len(samples),
        "trace_tail": samples[-5:],
    }
def sample_videos(app):
    os.environ["RAW_EVAL_VIDEO"] = "1"
    assets_by_dataset = {
        dataset: {asset["asset_id"]: asset for asset in selected_assets(dataset, None)}
        for dataset in ("robophyscan", "artvip", "partnet_mobility")
    }
    output_root = ensure_inside_root(ROOT / "reports" / "diagnostic_videos_v5")
    manifest = []
    for (dataset, kind, reason), selection in video_samples():
        source_row, attempt = selection["row"], selection["attempt"]
        asset = assets_by_dataset[dataset].get(source_row["asset_id"])
        path = resolved_asset_path(dataset, asset) if asset else None
        if not path or not path.exists():
            continue
        recorder = None
        try:
            stage = open_stage_in_context(path, app)
            check = precheck(stage)
            bounds_path = check["default_prim"] or check["rigid_bodies"][0]
            render_evidence = video_target_render_evidence(stage, bounds_path)
            safe_reason = re.sub(r"[^A-Za-z0-9_.-]+", "_", reason)
            output = output_root / f"{dataset}_{kind}_{safe_reason}_{asset['asset_id']}.mp4"
            recorder = VideoRecordingApp(app, stage, output, stage_bounds(stage, bounds_path))
            if kind == "grasp":
                replay_asset = dict(
                    asset,
                    _grasp_candidate_rank=int(attempt.get("candidate_rank", 0)),
                    _grasp_finger_scale=float(attempt.get("finger_scale", 1.0)),
                    _grasp_geometry=attempt.get("gripper_geometry", "flat"),
                )
                replay = _grasp_lift_once(stage, check, app, dataset, replay_asset, recorder)
            else:
                replay = _task_actuation_once(stage, check, app, dataset, asset, float(attempt.get("direction_sign", 1.0)), float(attempt.get("finger_scale", 1.0)))
            manifest.append({
                "dataset": dataset, "asset_id": asset["asset_id"], "metric": kind,
                "stratum": reason, "source_trial": source_row.get("trial"),
                "source_attempt": attempt, "status": "invalid" if (
                    recorder.frame_count == 0 or render_evidence["visible_mesh_count"] == 0
                    or render_evidence["missing_texture_asset_paths"]
                ) else "ok",
                "replay_pass": replay.get("pass"), "video": str(output.relative_to(ROOT)),
                "frame_count": recorder.frame_count, "fps": 20, "resolution": [640, 480],
                "capture_delta_time": 0.0, "render_backend": "replicator_rgb",
                "stage_identifier": stage.GetRootLayer().identifier,
                "target_prim": bounds_path, "material_evidence": render_evidence,
                "video_visual_proxy_count": 0,
            })
        except BaseException as exc:
            manifest.append({
                "dataset": dataset, "asset_id": asset["asset_id"], "metric": kind,
                "stratum": reason, "source_trial": source_row.get("trial"),
                "source_attempt": attempt, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
            })
        finally:
            if recorder is not None:
                recorder.close()
    manifest_path = ensure_inside_root(ROOT / "reports" / "diagnostic_video_manifest.jsonl")
    manifest_path.write_text(
        "".join(json.dumps(json_safe(row), ensure_ascii=False) + "\n" for row in manifest),
        encoding="utf-8",
    )


def resolved_asset_path(dataset: str, asset: dict) -> Path | None:
    if dataset == "partnet_mobility":
        output = ROOT / "derived_assets" / "partnet_mobility_v5" / asset["asset_id"]
        flattened = output / "evaluation_asset.usd"
        if flattened.exists():
            return flattened
        physics = output / "configuration" / "asset_physics.usd"
        return physics if physics.exists() else output / "asset.usd"
    if dataset == "artvip":
        source = asset.get("source_asset")
        if not source:
            return None
        annotated = (
            ROOT
            / "derived_assets"
            / "artvip_annotated"
            / asset["asset_id"]
            / source_filename(source)
        )
        return annotated if annotated.exists() else None
    if dataset == "robophyscan":
        return configured_source_asset(dataset, asset)
    source = asset.get("source_asset")
    if not source:
        return None
    source_path = Path(source)
    return source_path


def validate(app, dataset: str, assets: list[dict], tests: list[str]) -> None:
    print(f"validate dataset={dataset} assets={len(assets)} tests={','.join(tests)}", flush=True)
    for index, asset in enumerate(assets, start=1):
        path = resolved_asset_path(dataset, asset)
        exists = bool(path and path.exists())
        print(f"[{index}/{len(assets)}] opening exists={exists} path={path}", flush=True)
        stage = open_stage(path, app) if exists else None
        print(f"[{index}/{len(assets)}] opened={stage is not None}", flush=True)
        if stage is None:
            for test in tests:
                append_result(
                    result_row(
                        dataset,
                        asset,
                        test,
                        False,
                        {"applicable": True, "pass": False, "reason": "stage_open_failed"},
                    )
                )
            print(f"[{index}/{len(assets)}] load failed {asset['asset_id']}")
            continue
        try:
            check = precheck(stage)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"[{index}/{len(assets)}] precheck exception {error}", flush=True)
            for test in tests:
                append_result(
                    result_row(
                        dataset,
                        asset,
                        test,
                        False,
                        {
                            "applicable": test == "load_precheck",
                            "pass": False,
                            "reason": "load_precheck_exception",
                            "error": error,
                        },
                    )
                )
            continue
        print(f"[{index}/{len(assets)}] precheck_done load_pass={check['load_pass']}", flush=True)
        for test in tests:
            try:
                if test == "load_precheck":
                    payload = {
                        "applicable": True,
                        "pass": check["load_pass"],
                        **check,
                    }
                elif not check["load_pass"]:
                    payload = {
                        "applicable": False,
                        "pass": False,
                        "reason": "blocked_by_load_failure",
                    }
                elif test == "collision":
                    payload = collision_alignment(stage, check)
                elif test in {"settle", "push_contact", "joint_sweep", "grasp_lift", "task_actuation"}:
                    dynamic_stage = open_stage(path, app)
                    dynamic_check = precheck(dynamic_stage)
                    print(f"[{index}/{len(assets)}] dynamic_precheck_done test={test}", flush=True)
                    try:
                        if test == "settle":
                            payload = settle(dynamic_stage, dynamic_check, app)
                        elif test == "push_contact":
                            payload = push_contact(dynamic_stage, dynamic_check, app)
                        elif test == "joint_sweep":
                            payload = joint_sweep(dynamic_stage, dynamic_check, app)
                        elif test == "grasp_lift":
                            payload = grasp_lift(dynamic_stage, dynamic_check, app, dataset, asset)
                        else:
                            payload = task_actuation(dynamic_stage, dynamic_check, app, dataset, asset)
                    finally:
                        import omni.timeline

                        omni.timeline.get_timeline_interface().stop()
                        app.update()
                else:
                    payload = {
                        "applicable": False,
                        "pass": False,
                        "reason": "unknown_test",
                    }
            except BaseException as exc:
                payload = {
                    "applicable": True,
                    "pass": False,
                    "reason": "grasp_binding_error" if isinstance(exc, GraspBindingError) else "test_exception",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                if isinstance(exc, GraspBindingError):
                    payload.update(applicable=False, status="evaluation_blocked")
            append_result(result_row(dataset, asset, test, check["load_pass"], payload))
        print(f"[{index}/{len(assets)}] validated {asset['asset_id']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified Executability v5 Isaac Sim 入口")
    subparsers = parser.add_subparsers(dest="command", required=True)
    convert = subparsers.add_parser("convert-partnet")
    convert.add_argument("--limit", type=int)
    convert.add_argument("--start", type=int, default=0)
    for command in ("inspect", "simulate", "sample-videos", "grasp-video-smoke", "grasp-once-smoke"):
        item = subparsers.add_parser(command)
        item.add_argument("--dataset", choices=["robophyscan", "partnet_mobility", "artvip"], required=command != "sample-videos")
        item.add_argument("--limit", type=int)
        item.add_argument("--start", type=int, default=0)
        item.add_argument("--asset-id")
        if command == "simulate":
            item.add_argument("--shard-index", type=int, default=0)
            item.add_argument("--shard-count", type=int, default=1)
        if command in {"grasp-video-smoke", "grasp-once-smoke"}:
            item.add_argument("--candidate-rank", type=int)
            item.add_argument("--finger-scale", type=float)
            item.add_argument("--gripper-geometry", choices=["flat", "wrap"], default="wrap")
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument(
        "--dataset",
        choices=["robophyscan", "partnet_mobility", "artvip"],
        required=True,
    )
    validate_parser.add_argument("--limit", type=int)
    validate_parser.add_argument("--start", type=int, default=0)
    validate_parser.add_argument("--asset-id")
    validate_parser.add_argument(
        "--tests",
        default="load_precheck,settle,collision,push_contact,joint_sweep,grasp_lift,task_actuation",
        help="逗号分隔；可执行全部七项统一验证",
    )
    subparsers.add_parser("prismatic-frame-smoke")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command in {"sample-videos", "grasp-video-smoke"}:
        os.environ["RAW_EVAL_VIDEO"] = "1"
    portable_root = ensure_inside_root(
        Path(os.environ.get("RAW_EVAL_PORTABLE_ROOT", ROOT / "cache" / "kit_portable"))
    )
    portable_root.mkdir(parents=True, exist_ok=True)
    sys.argv.extend(["--portable-root", str(portable_root)])
    dataset = "partnet_mobility" if args.command == "convert-partnet" else getattr(args, "dataset", None)
    assets = selected_assets(
        dataset, args.limit, args.start, getattr(args, "asset_id", None)
    ) if dataset else []
    print(
        f"command={args.command} selected_assets={len(assets)} root={ROOT}",
        flush=True,
    )
    app = initialize_app()
    try:
        if args.command == "convert-partnet":
            convert_partnet(app, assets)
        elif args.command == "inspect":
            inspect_assets(app, args.dataset, assets)
        elif args.command == "simulate":
            simulate_assets(app, args.dataset, assets, args.shard_index, args.shard_count)
        elif args.command == "sample-videos":
            sample_videos(app)
        elif args.command == "grasp-video-smoke":
            grasp_video_smoke(
                app,
                args.dataset,
                assets,
                args.candidate_rank,
                args.finger_scale,
                args.gripper_geometry,
            )
        elif args.command == "grasp-once-smoke":
            if args.candidate_rank is None or args.finger_scale is None:
                raise ValueError("grasp-once-smoke requires --candidate-rank and --finger-scale")
            grasp_once_smoke(
                app, args.dataset, assets, args.candidate_rank,
                args.finger_scale, args.gripper_geometry,
            )
        elif args.command == "prismatic-frame-smoke":
            payload = prismatic_frame_smoke(app)
            append_result({
                "run_label": RUN_LABEL,
                "metric": "parallel_gripper_frame_smoke",
                "payload": payload,
            })
            print(json.dumps(payload, ensure_ascii=False), flush=True)
        else:
            validate(
                app,
                args.dataset,
                assets,
                [value.strip() for value in args.tests.split(",") if value.strip()],
            )
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
    finally:
        if os.environ.get("RAW_EVAL_SKIP_APP_CLOSE") != "1":
            app.close()
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException:
        if os.environ.get("RAW_EVAL_SKIP_APP_CLOSE") == "1":
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        raise
    if os.environ.get("RAW_EVAL_SKIP_APP_CLOSE") == "1":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
    raise SystemExit(exit_code)
