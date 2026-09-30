import cv2


def draw_detections(image, detections, color=(0, 255, 0)):
    result = image.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.get("bbox", [0, 0, 0, 0])]
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        cv2.putText(result, f"{det.get('confidence', 0):.2f}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return result
