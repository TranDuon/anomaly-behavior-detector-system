"""Explain limited camera framing without suppressing positive fight evidence."""

import numpy as np


def assess_view(person_boxes, frame_shape):
    height, width = frame_shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Positive frame dimensions required")
    boxes = np.asarray(person_boxes, dtype=np.float64).reshape(-1, 4)
    if not np.isfinite(boxes).all():
        raise ValueError("Finite person boxes required")
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / (width * height)
    edge = (
        (boxes[:, 0] <= width * 0.02)
        | (boxes[:, 1] <= height * 0.02)
        | (boxes[:, 2] >= width * 0.98)
        | (boxes[:, 3] >= height * 0.98)
    )
    clipped = edge & (areas >= 0.35) & ((boxes[:, 3] - boxes[:, 1]) >= height * 0.65)
    reasons = []
    if not len(boxes):
        reasons.append("no_current_person")
    if clipped.any():
        reasons.append("dominant_person_at_frame_edge")
        if len(boxes) == 1:
            reasons.append("interaction_partner_not_visible")
    return {
        "limited": bool(reasons),
        "reasons": reasons,
        "visible_people": len(boxes),
        "largest_person_area_fraction": float(areas.max()) if len(areas) else 0.0,
        "clipped_dominant_people": int(clipped.sum()),
        "scope": "Framing diagnostic only; not a fight classifier or a requirement for two fully visible people",
    }


def explain_view(quality):
    if "no_current_person" in quality["reasons"]:
        return "Không thấy người ở khung mới nhất; chưa thể xác minh hành vi."
    if "interaction_partner_not_visible" in quality["reasons"]:
        return "Chỉ thấy rõ một người sát mép hình; người tương tác có thể bị khuất. Chưa xác minh được, không phải kết luận an toàn."
    if quality["limited"]:
        return "Người chiếm phần lớn hình và chạm mép khung; có thể thiếu vùng tiếp xúc. Chưa xác minh được, không phải kết luận an toàn."
    return ""
