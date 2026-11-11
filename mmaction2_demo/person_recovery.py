import numpy as np


def overlap(box, boxes):
    lower = np.maximum(box[:2], boxes[:, :2])
    upper = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.maximum(upper - lower, 0).prod(axis=1)
    area = np.maximum(box[2:] - box[:2], 0).prod()
    areas = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(axis=1)
    return intersection / np.maximum(area + areas - intersection, 1e-6)


def duplicate(box, boxes):
    boxes = np.asarray(boxes)
    iou = overlap(box, boxes)
    distance = np.linalg.norm((boxes[:, :2] + boxes[:, 2:] - box[:2] - box[2:]) / 2, axis=1)
    scale = np.minimum(np.linalg.norm(boxes[:, 2:] - boxes[:, :2], axis=1),
                       np.linalg.norm(box[2:] - box[:2]))
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    different_extent = np.maximum(area, areas) / np.maximum(np.minimum(area, areas), 1e-6) > 1.25
    return bool(((iou > 0.8) | ((iou > 0.5) & different_extent & (distance < 0.15 * scale))).any())


def select_people(frame, detect, threshold=0.6, limit=6):
    height, width = frame.shape[:2]
    boxes, scores = detect(frame)
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32)
    valid = np.isfinite(boxes).all(axis=1) & np.isfinite(scores)
    boxes, scores = boxes[valid], scores[valid]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    sizes = boxes[:, 2:] - boxes[:, :2]
    valid = (sizes >= 8).all(axis=1)
    boxes, scores, sizes = boxes[valid], scores[valid], sizes[valid]
    selected_boxes = list(boxes[scores >= threshold])
    selected_scores = list(scores[scores >= threshold])
    if len(selected_boxes) >= 2:
        areas = np.asarray([(box[2] - box[0]) * (box[3] - box[1]) for box in selected_boxes])
        largest = selected_boxes[int(areas.argmax())]
        if areas.max() >= 2 * np.median(np.delete(areas, areas.argmax())):
            box_width, box_height = largest[2:] - largest[:2]
            left = max(0, int(largest[0] - 0.3 * box_width))
            top = max(0, int(largest[1] - 0.15 * box_height))
            right = min(width, int(largest[2] + 0.3 * box_width))
            bottom = min(height, int(largest[3] + 0.15 * box_height))
            if (right - left) * (bottom - top) < width * height * 0.65:
                recovered, confidence = detect(np.ascontiguousarray(frame[top:bottom, left:right]))
                for candidate, score in sorted(zip(recovered, confidence), key=lambda item: -item[1]):
                    candidate = np.asarray(candidate, dtype=np.float32) + [left, top, left, top]
                    candidate[[0, 2]] = candidate[[0, 2]].clip(0, width)
                    candidate[[1, 3]] = candidate[[1, 3]].clip(0, height)
                    if not np.isfinite(candidate).all() or not np.isfinite(score) or score < 0.85:
                        continue
                    if (candidate[2:] - candidate[:2] < 8).any():
                        continue
                    separation = np.linalg.norm((candidate[:2] + candidate[2:] - largest[:2] - largest[2:]) / 2)
                    if (not duplicate(candidate, selected_boxes)
                            and overlap(candidate, np.asarray([largest]))[0] >= 0.3
                            and separation >= 0.12 * min(box_height, candidate[3] - candidate[1])):
                        selected_boxes.append(candidate)
                        selected_scores.append(float(score))
                        break
    weak = boxes[(scores >= 0.12) & (scores < threshold) & (sizes[:, 0] >= 1.5 * sizes[:, 1])]
    if len(weak):
        if selected_boxes:
            weak = np.asarray([box for box in weak if overlap(box, np.asarray(selected_boxes)).max() < 0.35], dtype=np.float32).reshape(-1, 4)
        for turns in (3, 1) if len(weak) else ():
            rotated = np.ascontiguousarray(np.rot90(frame, turns))
            recovered, confidence = detect(rotated)
            for box, score in zip(recovered, confidence):
                if not np.isfinite(box).all() or not np.isfinite(score) or score < 0.8:
                    continue
                left, top, right, bottom = box
                mapped = np.array([top, height - right, bottom, height - left] if turns == 3
                                  else [width - bottom, left, width - top, right], dtype=np.float32)
                mapped[[0, 2]] = mapped[[0, 2]].clip(0, width)
                mapped[[1, 3]] = mapped[[1, 3]].clip(0, height)
                if (mapped[2:] - mapped[:2] >= 8).all() and overlap(mapped, weak).max() >= 0.25:
                    selected_boxes.append(mapped)
                    selected_scores.append(float(score))
            if selected_boxes:
                weak = np.asarray([box for box in weak if overlap(box, np.asarray(selected_boxes)).max() < 0.25], dtype=np.float32).reshape(-1, 4)
                if not len(weak):
                    break
    result = []
    for index in np.argsort(-np.asarray(selected_scores)):
        box = selected_boxes[index]
        if result and duplicate(box, result):
            continue
        result.append(box)
        if len(result) >= limit:
            break
    return np.asarray(result, dtype=np.float32).reshape(-1, 4)
