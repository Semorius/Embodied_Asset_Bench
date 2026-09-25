from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np


def mean_score(*values):
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(valid) / len(valid) if valid else None


def smooth_ratio(value):
    value = float(value)
    return math.exp(-abs(math.log(value))) if math.isfinite(value) and value > 0 else 0.0


def bounds_metrics(collision_vertices, visual_vertices=None):
    collision = np.asarray(collision_vertices, dtype=float).reshape(-1, 3)
    visual = None if visual_vertices is None else np.asarray(visual_vertices, dtype=float).reshape(-1, 3)
    finite = bool(len(collision) and np.all(np.isfinite(collision)))
    if not finite:
        return {"score": 0.0, "finite": False, "valid_extent": False, "reason": "nonfinite_or_empty_bounds"}
    cmin, cmax = collision.min(axis=0), collision.max(axis=0)
    extent = cmax - cmin
    diagonal = float(np.linalg.norm(extent))
    positive = bool(np.all(extent > 0) and diagonal > 0)
    aspect = float(extent.max() / max(1e-12, extent.min())) if positive else math.inf
    result = {
        "finite": True,
        "valid_extent": positive,
        "minimum": cmin.tolist(),
        "maximum": cmax.tolist(),
        "extent": extent.tolist(),
        "diagonal_m": diagonal,
        "aspect_ratio": aspect,
        "extreme_aspect": aspect > 1000.0,
    }
    if visual is None or not len(visual) or not np.all(np.isfinite(visual)):
        result["score"] = mean_score(1.0 if positive else 0.0, smooth_ratio(max(1.0, aspect) / 1000.0) if aspect > 1000 else 1.0)
        return result
    vmin, vmax = visual.min(axis=0), visual.max(axis=0)
    vextent = vmax - vmin
    vdiag = float(np.linalg.norm(vextent))
    axis_ratios = np.divide(extent, vextent, out=np.full(3, math.inf), where=vextent > 0)
    diagonal_ratio = diagonal / vdiag if vdiag > 0 else math.inf
    cvolume, vvolume = float(np.prod(extent)), float(np.prod(vextent))
    volume_ratio = cvolume / vvolume if vvolume > 0 else math.inf
    center_offset = float(np.linalg.norm((cmin + cmax - vmin - vmax) * 0.5))
    center_normalized = center_offset / vdiag if vdiag > 0 else math.inf
    scale_score = mean_score(*(smooth_ratio(value) for value in axis_ratios), smooth_ratio(diagonal_ratio), smooth_ratio(volume_ratio) ** (1 / 3))
    center_score = max(0.0, 1.0 - center_normalized)
    result.update({
        "visual_extent": vextent.tolist(),
        "axis_extent_ratios": axis_ratios.tolist(),
        "diagonal_ratio": diagonal_ratio,
        "volume_ratio": volume_ratio,
        "center_offset_m": center_offset,
        "center_offset_visual_diagonal": center_normalized,
        "scale_score": scale_score,
        "center_alignment_score": center_score,
        "score": mean_score(1.0 if positive else 0.0, scale_score, center_score),
    })
    return result


def _face_components(faces):
    vertex_faces = defaultdict(list)
    for face_index, face in enumerate(faces):
        for vertex in set(map(int, face)):
            vertex_faces[vertex].append(face_index)
    unseen = set(range(len(faces)))
    components = []
    while unseen:
        start = unseen.pop()
        component = [start]
        pending = [start]
        while pending:
            for vertex in set(map(int, faces[pending.pop()])):
                for neighbor in vertex_faces[vertex]:
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        component.append(neighbor)
                        pending.append(neighbor)
        components.append(component)
    return components


def _point_in_triangle(point, triangle, normal, eps):
    for index in range(3):
        edge = triangle[(index + 1) % 3] - triangle[index]
        side = point - triangle[index]
        if float(np.dot(np.cross(edge, side), normal)) < -eps:
            return False
    return True


def _segment_triangle(start, end, triangle, eps):
    direction = end - start
    edge1, edge2 = triangle[1] - triangle[0], triangle[2] - triangle[0]
    pvec = np.cross(direction, edge2)
    determinant = float(np.dot(edge1, pvec))
    if abs(determinant) <= eps:
        return False
    inverse = 1.0 / determinant
    tvec = start - triangle[0]
    u = float(np.dot(tvec, pvec)) * inverse
    if u < -eps or u > 1.0 + eps:
        return False
    qvec = np.cross(tvec, edge1)
    v = float(np.dot(direction, qvec)) * inverse
    if v < -eps or u + v > 1.0 + eps:
        return False
    t = float(np.dot(edge2, qvec)) * inverse
    return -eps <= t <= 1.0 + eps


def triangles_intersect(first, second, eps=1e-10):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if np.any(first.max(axis=0) < second.min(axis=0) - eps) or np.any(second.max(axis=0) < first.min(axis=0) - eps):
        return False
    for triangle, other in ((first, second), (second, first)):
        for index in range(3):
            if _segment_triangle(triangle[index], triangle[(index + 1) % 3], other, eps):
                return True
    n1 = np.cross(first[1] - first[0], first[2] - first[0])
    n2 = np.cross(second[1] - second[0], second[2] - second[0])
    if np.linalg.norm(np.cross(n1, n2)) <= eps and abs(float(np.dot(n1, second[0] - first[0]))) <= eps:
        return _point_in_triangle(first[0], second, n2, eps) or _point_in_triangle(second[0], first, n1, eps)
    return False


def triangle_aabb_pairs(first, second=None, eps=1e-10):
    """Stream spatial candidates without materializing the triangle Cartesian product."""
    same_mesh = second is None
    second = first if same_mesh else second
    if not len(first) or not len(second):
        return

    def build(triangles):
        minimum, maximum = triangles.min(axis=1), triangles.max(axis=1)
        return minimum, maximum, node(np.arange(len(triangles)), minimum, maximum)

    def node(indices, minimum, maximum):
        node_min = minimum[indices].min(axis=0)
        node_max = maximum[indices].max(axis=0)
        if len(indices) <= 8:
            return (node_min, node_max, indices, None, None, len(indices))
        axis = int(np.argmax(node_max - node_min))
        ordered = indices[np.argsort((minimum[indices, axis] + maximum[indices, axis]) * 0.5)]
        middle = len(ordered) // 2
        return (node_min, node_max, None, node(ordered[:middle], minimum, maximum), node(ordered[middle:], minimum, maximum), len(indices))

    amin, amax, aroot = build(first)
    bmin, bmax, broot = (amin, amax, aroot) if same_mesh else build(second)

    def overlap(first, second):
        return bool(np.all(first[0] <= second[1] + eps) and np.all(second[0] <= first[1] + eps))

    def compare(first, second, same=False):
        if not overlap(first, second):
            return
        first_leaf, second_leaf = first[2] is not None, second[2] is not None
        if first_leaf and second_leaf:
            for left in first[2]:
                for right in second[2]:
                    if same_mesh and ((same and right <= left) or left == right):
                        continue
                    if np.all(amin[left] <= bmax[right] + eps) and np.all(bmin[right] <= amax[left] + eps):
                        yield int(left), int(right)
            return
        if same:
            yield from compare(first[3], first[3], True)
            yield from compare(first[3], first[4], False)
            yield from compare(first[4], first[4], True)
        elif second_leaf or (not first_leaf and first[5] >= second[5]):
            yield from compare(first[3], second, False)
            yield from compare(first[4], second, False)
        else:
            yield from compare(first, second[3], False)
            yield from compare(first, second[4], False)

    yield from compare(aroot, broot, same_mesh)


def self_intersections(vertices, faces, max_pairs=50_000):
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    triangles = vertices[faces]
    if not len(triangles):
        return [], False
    diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    shared_vertex_tolerance = max(1e-12, diagonal * 1e-9)
    hits = []
    for count, (first, second) in enumerate(triangle_aabb_pairs(triangles)):
        if max_pairs is not None and count >= max_pairs:
            return hits, True
        if set(map(int, faces[first])) & set(map(int, faces[second])):
            continue
        if np.any(
            np.all(
                np.abs(triangles[first][:, None, :] - triangles[second][None, :, :])
                <= shared_vertex_tolerance,
                axis=2,
            )
        ):
            continue
        if triangles_intersect(triangles[first], triangles[second]):
            hits.append((first, second))
            if max_pairs is None:
                # The score is binary: one witness proves self-intersection.
                return hits, False
    return hits, False


def audit_collision_mesh(
    vertices,
    faces,
    watertight_expected=True,
    check_self_intersection=True,
    self_intersection_max_pairs=50_000,
    topology_max_triangles=None,
):
    vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
    faces = np.asarray(faces, dtype=int).reshape(-1, 3)
    valid_indices = bool(len(faces) and len(vertices) and np.all((faces >= 0) & (faces < len(vertices))))
    finite = bool(len(vertices) and np.all(np.isfinite(vertices)))
    if not valid_indices or not finite:
        return {"score": 0.0, "valid": False, "reason": "invalid_collision_mesh", "watertight_expected": bool(watertight_expected)}
    duplicate_index_faces = int(np.sum([len(set(map(int, face))) < 3 for face in faces]))
    canonical_faces = [tuple(sorted(map(int, face))) for face in faces]
    duplicate_faces = sum(count - 1 for count in Counter(canonical_faces).values() if count > 1)
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) * 0.5
    diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    tiny_threshold = 1e-8 * diagonal * diagonal
    zero_faces = int(np.sum(areas <= np.finfo(float).eps * max(1.0, diagonal * diagonal)))
    tiny_faces = int(np.sum((areas > 0) & (areas < tiny_threshold)))
    directed = Counter()
    undirected = Counter()
    for a, b, c in faces:
        for start, end in ((a, b), (b, c), (c, a)):
            directed[(int(start), int(end))] += 1
            undirected[tuple(sorted((int(start), int(end))))] += 1
    boundary = sum(count == 1 for count in undirected.values())
    nonmanifold = sum(count > 2 for count in undirected.values())
    winding_errors = sum(
        count == 2 and directed[(edge[0], edge[1])] != 1
        for edge, count in undirected.items()
    )
    topology_truncated = (
        topology_max_triangles is not None
        and len(faces) > int(topology_max_triangles)
    )
    components = [] if topology_truncated else _face_components(faces)
    component_rows = []
    closed_count = open_count = reversed_count = 0
    for component in components:
        component_edges = Counter()
        for face_index in component:
            a, b, c = faces[face_index]
            for edge in ((a, b), (b, c), (c, a)):
                component_edges[tuple(sorted(map(int, edge)))] += 1
        closed = bool(component_edges and all(value == 2 for value in component_edges.values()))
        volume = float(np.sum(np.einsum("ij,ij->i", triangles[component, 0], np.cross(triangles[component, 1], triangles[component, 2]))) / 6.0)
        volume_valid = math.isfinite(volume) and abs(volume) > max(1e-18, diagonal ** 3 * 1e-12)
        closed = closed and volume_valid
        closed_count += int(closed)
        open_count += int(not closed)
        reversed_count += int(closed and volume < 0)
        component_rows.append({"face_count": len(component), "closed": closed, "signed_volume_m3": volume})
    intersections, truncated = (
        self_intersections(vertices, faces, self_intersection_max_pairs)
        if check_self_intersection and len(faces) and not topology_truncated
        else ([], bool(topology_truncated))
    )
    nondegenerate_score = max(0.0, 1.0 - (duplicate_index_faces + duplicate_faces + zero_faces + tiny_faces) / max(1, len(faces)))
    manifold_score = max(0.0, 1.0 - (boundary + nonmanifold) / max(1, len(undirected)))
    winding_score = max(0.0, 1.0 - winding_errors / max(1, len(undirected)))
    source_watertight = boundary == 0 and nonmanifold == 0 and (
        topology_truncated or open_count == 0
    )
    watertight_score = float(source_watertight) if watertight_expected else None
    intersection_score = 0.0 if intersections else (None if truncated else 1.0)
    confirmed_failure = bool(intersections or nonmanifold or zero_faces or (watertight_expected and boundary))
    audit_status = "confirmed_failure" if confirmed_failure else ("unknown_due_to_budget" if (topology_truncated or truncated) else "pass")
    score = mean_score(nondegenerate_score, manifold_score, watertight_score, winding_score, intersection_score)
    return {
        "valid": True,
        "score": score,
        "vertex_count": int(len(vertices)),
        "triangle_count": int(len(faces)),
        "duplicate_vertex_index_face_count": duplicate_index_faces,
        "duplicate_triangle_count": int(duplicate_faces),
        "zero_area_face_count": zero_faces,
        "tiny_area_face_count": tiny_faces,
        "tiny_area_threshold_m2": tiny_threshold,
        "boundary_edge_count": boundary,
        "boundary_edge_ratio": boundary / max(1, len(undirected)),
        "nonmanifold_edge_count": nonmanifold,
        "winding_error_edge_count": winding_errors,
        "component_count": len(components),
        "isolated_triangle_count": sum(len(component) == 1 for component in components),
        "tiny_component_count": sum(len(component) <= max(2, int(len(faces) * 0.001)) for component in components),
        "watertight_expected": bool(watertight_expected),
        "source_watertight": source_watertight,
        "effective_collider_closed": None,
        "open_component_count": open_count,
        "closed_component_count": closed_count,
        "orientation_reversed": reversed_count > 0,
        "components": component_rows,
        "self_intersection_count": len(intersections),
        "self_intersection_count_is_lower_bound": self_intersection_max_pairs is None and bool(intersections),
        "self_intersection_pairs": intersections[:100],
        "self_intersection_check_truncated": truncated,
        "topology_check_truncated": topology_truncated,
        "nondegenerate_score": nondegenerate_score,
        "manifold_score": manifold_score,
        "watertightness_score": watertight_score,
        "winding_score": winding_score,
        "self_intersection_free_score": intersection_score,
        "audit_status": audit_status,
    }


def complexity_metrics(collision_triangles, collision_vertices, mesh_prims, colliders, convex_pieces, visual_triangles=None, colliders_per_link=None):
    ratio = collision_triangles / visual_triangles if visual_triangles else None
    warning = collision_triangles > 50_000 or (ratio is not None and ratio > 0.5 and collision_triangles > 10_000) or convex_pieces > 64
    penalties = [collision_triangles / 50_000, convex_pieces / 64]
    if ratio is not None and collision_triangles > 10_000:
        penalties.append(ratio / 0.5)
    score = 1.0 / max(1.0, max(penalties, default=1.0))
    return {
        "score": score,
        "warning": warning,
        "collision_triangle_count": int(collision_triangles),
        "collision_vertex_count": int(collision_vertices),
        "collision_mesh_prim_count": int(mesh_prims),
        "collider_count": int(colliders),
        "convex_piece_count": int(convex_pieces),
        "collision_visual_triangle_ratio": ratio,
        "colliders_per_link": colliders_per_link or {},
    }


def support_patch(points, com_xy=None, scale_diagonal=1.0):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if not len(points):
        return {"support_exists": False, "point_count": 0, "patch_area_m2": None, "normalized_patch_area": None, "com_signed_margin_m": None}
    unique = sorted(set(map(tuple, points.tolist())))
    if len(unique) < 3:
        return {"support_exists": True, "point_count": len(unique), "patch_area_m2": None, "normalized_patch_area": None, "com_signed_margin_m": None, "support_dimension": len(unique) - 1}
    points = np.asarray(unique)
    def cross2(first, second):
        return float(first[0] * second[1] - first[1] * second[0])

    lower = []
    for point in points:
        while len(lower) >= 2 and cross2(lower[-1] - lower[-2], point - lower[-1]) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross2(upper[-1] - upper[-2], point - upper[-1]) <= 0:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1])
    area = abs(float(np.dot(hull[:, 0], np.roll(hull[:, 1], -1)) - np.dot(hull[:, 1], np.roll(hull[:, 0], -1)))) * 0.5
    margin = None
    if com_xy is not None:
        com = np.asarray(com_xy, dtype=float)
        signed = []
        for index, start in enumerate(hull):
            edge = hull[(index + 1) % len(hull)] - start
            signed.append(cross2(edge, com - start) / max(1e-12, float(np.linalg.norm(edge))))
        margin = min(signed)
    return {
        "support_exists": True,
        "point_count": len(unique),
        "support_dimension": 2,
        "hull": hull.tolist(),
        "patch_area_m2": area,
        "normalized_patch_area": area / max(1e-12, float(scale_diagonal) ** 2),
        "com_signed_margin_m": margin,
    }


def aabb_overlap(first_min, first_max, second_min, second_max):
    depth = np.minimum(first_max, second_max) - np.maximum(first_min, second_min)
    return {"intersects": bool(np.all(depth > 0)), "axis_depth_m": depth.tolist(), "penetration_depth_m": float(depth.min()) if np.all(depth > 0) else 0.0}
