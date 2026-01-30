#============ phân loại object = pedestrian, vehicle, unknown ============
import numpy as np

def classify_from_obb_and_plane(obb, plane_model):
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)

    # Footprint from OBB
    w, l, h = obb.extent
    footprint_max = max(w, l)
    footprint_min = min(w, l)
    ratio = footprint_max / footprint_min

    # Highest point elevation
    top_point = obb.get_max_bound()
    top_elevation = abs(np.dot(normal, top_point) + d) / np.linalg.norm(normal)

    # Heuristics (tune to your data)
    # Pedestrian: small footprint, tall above road
    if footprint_max < 1.2 and top_elevation > 1.2:
        return "p"

    # Vehicle: large footprint, not extremely tall
    if footprint_max > 1.3 and top_elevation < 3.0:
        return "v"

    #Use footprint ratio as a backup
    if ratio < 1.4 and top_elevation > 1.0:
        return "p"
    if ratio > 2.0 and footprint_max > 2.0:
        return "v"

    return "u"