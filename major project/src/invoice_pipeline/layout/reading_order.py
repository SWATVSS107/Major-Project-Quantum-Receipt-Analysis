import numpy as np


def build_spatial_graph(tokens, image_width=1000, image_height=1400):
    """
    Constructs a 2D Reading Graph from OCR tokens with dynamic radius based on average text height.
    Nodes represent text tokens, edges represent spatial relationships (RIGHT, LEFT, BELOW, ABOVE).
    """
    if not tokens:
        return {"nodes": [], "edges": []}

    # Calculate average token height for adaptive radius
    heights = [t["bbox"][3] - t["bbox"][1] for t in tokens if (t["bbox"][3] - t["bbox"][1]) > 0]
    avg_height = float(np.median(heights)) if heights else 15.0

    dynamic_radius = max(avg_height * 8.0, 100.0)
    row_tolerance = max(avg_height * 0.6, 8.0)

    nodes = []
    for i, tok in enumerate(tokens):
        x0, y0, x1, y1 = tok["bbox"]
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        nodes.append({
            "id": i,
            "text": tok["text"],
            "confidence": tok["confidence"],
            "bbox": tok["bbox"],
            "cx": cx,
            "cy": cy,
            "height": y1 - y0,
            "width": x1 - x0
        })

    # Sort in reading order: top-to-bottom, left-to-right
    nodes_sorted = sorted(nodes, key=lambda n: (round(n["cy"] / row_tolerance), n["cx"]))

    edges = []
    for i, u in enumerate(nodes_sorted):
        for j, v in enumerate(nodes_sorted):
            if i == j:
                continue
            dx = v["cx"] - u["cx"]
            dy = v["cy"] - u["cy"]
            dist = float(np.sqrt(dx * dx + dy * dy))

            if dist <= dynamic_radius:
                rel = None
                if abs(dy) <= row_tolerance and dx > 0:
                    rel = "RIGHT"
                elif abs(dy) <= row_tolerance and dx < 0:
                    rel = "LEFT"
                elif dy > row_tolerance and abs(dx) <= dynamic_radius * 0.5:
                    rel = "BELOW"
                elif dy < -row_tolerance and abs(dx) <= dynamic_radius * 0.5:
                    rel = "ABOVE"

                if rel:
                    edges.append({
                        "source": u["id"],
                        "target": v["id"],
                        "relation": rel,
                        "distance": dist
                    })

    return {
        "nodes": nodes_sorted,
        "edges": edges,
        "avg_height": avg_height,
        "radius": dynamic_radius
    }


def find_right_neighbor(graph, node_id):
    """Find closest token to the right of node_id on same row."""
    candidates = []
    for edge in graph["edges"]:
        if edge["source"] == node_id and edge["relation"] == "RIGHT":
            target_node = next((n for n in graph["nodes"] if n["id"] == edge["target"]), None)
            if target_node:
                candidates.append((target_node, edge["distance"]))
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0] if candidates else None


def find_below_neighbor(graph, node_id):
    """Find closest token directly below node_id."""
    candidates = []
    for edge in graph["edges"]:
        if edge["source"] == node_id and edge["relation"] == "BELOW":
            target_node = next((n for n in graph["nodes"] if n["id"] == edge["target"]), None)
            if target_node:
                candidates.append((target_node, edge["distance"]))
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0] if candidates else None
