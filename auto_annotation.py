import os
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics import SAM


IMAGE_DIR = "images"
LABEL_DIR = "labels"
MASK_DIR = "masks"
VIS_DIR = "visualizations"

YOLO_MODEL = "weights/yolo.pt"
SAM_MODEL = "weights/sam3.pt"

os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)


# ----------------------------------------------------------
# Load models
# ----------------------------------------------------------

detector = YOLO(YOLO_MODEL)
segmenter = SAM(SAM_MODEL)


def normalize_polygon(contour, w, h):
    contour = contour.reshape(-1, 2)

    pts = []
    for x, y in contour:
        pts.append(x / w)
        pts.append(y / h)

    return pts


for image_name in os.listdir(IMAGE_DIR):

    if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(IMAGE_DIR, image_name)

    image = cv2.imread(image_path)

    H, W = image.shape[:2]

    print(f"Processing {image_name}")

    detections = detector.predict(
        image,
        verbose=False,
        conf=0.25
    )[0]

    label_file = os.path.join(
        LABEL_DIR,
        os.path.splitext(image_name)[0] + ".txt"
    )

    vis = image.copy()

    with open(label_file, "w") as f:

        if detections.boxes is None:
            continue

        boxes = detections.boxes.xyxy.cpu().numpy()
        classes = detections.boxes.cls.cpu().numpy().astype(int)

        for box, cls in zip(boxes, classes):

            x1, y1, x2, y2 = box

            sam_result = segmenter.predict(
                image,
                bboxes=[box],
                verbose=False
            )[0]

            if sam_result.masks is None:
                continue

            mask = sam_result.masks.data[0].cpu().numpy()

            mask = (mask * 255).astype(np.uint8)

            mask_name = (
                os.path.splitext(image_name)[0]
                + f"_{cls}.png"
            )

            cv2.imwrite(
                os.path.join(MASK_DIR, mask_name),
                mask
            )

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if len(contours) == 0:
                continue

            contour = max(contours, key=cv2.contourArea)

            if len(contour) < 3:
                continue

            polygon = normalize_polygon(contour, W, H)

            line = str(cls)

            for p in polygon:
                line += f" {p:.6f}"

            f.write(line + "\n")

            color = (
                np.random.randint(255),
                np.random.randint(255),
                np.random.randint(255),
            )

            cv2.drawContours(
                vis,
                [contour],
                -1,
                color,
                2
            )

            cv2.rectangle(
                vis,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                2,
            )

            cv2.putText(
                vis,
                str(cls),
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

    cv2.imwrite(
        os.path.join(VIS_DIR, image_name),
        vis,
    )

print("Done.")