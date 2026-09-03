"""Tile recognition from the overhead camera (K230 module or any USB cam).

Pipeline:
  1. capture()           - grab a frame (OpenCV VideoCapture)
  2. find_tiles(frame)   - threshold + contour detection of tile-sized white
                           rectangles inside a region of interest
  3. classify(crop)      - identify the tile face. Two backends:
       a. TemplateClassifier: normalized-correlation match against a folder
          of reference crops (assets/templates/<kind_id>.png) - zero
          training, good enough under fixed lighting.
       b. CNNClassifier: a small conv net (train with scripts in a later
          milestone, or run on the K230 itself via nncase and read results
          over serial instead of doing host-side vision).

Regions of interest (pixel boxes) are calibrated once per table setup in
robot/table_calibration.json:  {"hand": [x,y,w,h], "discard": [...],
"wall": [...], "melds": [...]}.
"""
import json
import os

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

CALIB_FILE = "robot/table_calibration.json"
TEMPLATE_DIR = "robot/assets/templates"
TILE_MIN_AREA = 900          # px^2, tune for camera height
TILE_ASPECT = (0.6, 0.95)    # w/h range of an upright tile face


class Camera:
    def __init__(self, index=0):
        if cv2 is None:
            raise RuntimeError("opencv not installed: pip install opencv-python")
        self.cap = cv2.VideoCapture(index)
        self.rois = {}
        if os.path.exists(CALIB_FILE):
            with open(CALIB_FILE) as f:
                self.rois = json.load(f)

    def capture(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        return frame

    def roi(self, frame, name):
        x, y, w, h = self.rois[name]
        return frame[y:y + h, x:x + w]


def find_tiles(region):
    """Return list of (x, y, crop) for tile faces found in a region."""
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    tiles = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < TILE_MIN_AREA or h == 0:
            continue
        if not (TILE_ASPECT[0] <= w / h <= TILE_ASPECT[1]):
            continue
        tiles.append((x, y, region[y:y + h, x:x + w]))
    tiles.sort(key=lambda t: (t[1] // 40, t[0]))  # row-major reading order
    return tiles


class TemplateClassifier:
    """Nearest-template matching over the 34 tile kinds."""

    def __init__(self, template_dir=TEMPLATE_DIR, size=(48, 64)):
        self.size = size
        self.templates = {}
        if os.path.isdir(template_dir):
            for fn in os.listdir(template_dir):
                kid = int(os.path.splitext(fn)[0])
                img = cv2.imread(os.path.join(template_dir, fn),
                                 cv2.IMREAD_GRAYSCALE)
                self.templates[kid] = cv2.resize(img, size)

    def classify(self, crop):
        """Return (kind_id, confidence). kind_id per hk_mahjong.tiles."""
        if not self.templates:
            raise RuntimeError(f"no templates in {TEMPLATE_DIR}; capture "
                               "reference crops first (see docs)")
        g = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), self.size)
        best_kid, best_score = -1, -2.0
        for kid, tmpl in self.templates.items():
            score = cv2.matchTemplate(g, tmpl, cv2.TM_CCOEFF_NORMED)[0][0]
            if score > best_score:
                best_kid, best_score = kid, float(score)
        return best_kid, best_score


def _build_name_map():
    """Map annotation class names -> kind_id, tolerant of naming styles.

    Accepted per suit (case/space/underscore-insensitive):
      dots:       1p, 1dot, 1dots, 1circle    characters: 1m, 1man, 1char,
      bamboo:     1s, 1sou, 1bam, 1bamboo                 1character, 1crak
      honors:     east, south, west, north, white, green, red
                  (an optional 'dragon'/'wind' suffix is fine)
    Name your Roboflow classes accordingly and detection maps automatically.
    """
    m = {}
    suits = {0: ["p", "dot", "dots", "circle"],
             1: ["m", "man", "char", "character", "crak"],
             2: ["s", "sou", "bam", "bamboo"]}
    for t, names in suits.items():
        for i in range(1, 10):
            for n in names:
                m[f"{i}{n}"] = t * 9 + i - 1
    honors = {"east": 1, "south": 2, "west": 3, "north": 4,
              "white": 5, "green": 6, "red": 7}
    for name, idx in honors.items():
        for suffix in ("", "wind", "dragon"):
            m[name + suffix] = 27 + idx - 1
    return m


NAME_TO_KIND = _build_name_map()


class RoboflowDetector:
    """Tile detection with a Roboflow-trained YOLO model (preferred backend).

    Train and annotate on Roboflow, then export the model weights
    (YOLOv8/YOLO11 format) and save them as robot/assets/tiles_yolo.pt.
    Runs fully offline via the ultralytics package:
        pip install ultralytics
    Detection + classification in one pass - no contour step needed.
    """

    def __init__(self, weights="robot/assets/tiles_yolo.pt", conf=0.5):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf

    def detect(self, region):
        """[(x, y, kind_id, confidence)] in reading order for a region."""
        res = self.model.predict(region, conf=self.conf, verbose=False)[0]
        out = []
        for b in res.boxes:
            raw = res.names[int(b.cls)]
            name = raw.lower().replace(" ", "").replace("_", "").replace("-", "")
            kid = NAME_TO_KIND.get(name)
            if kid is None:
                continue  # unknown class name: check _build_name_map docstring
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            out.append((x1, y1, kid, float(b.conf)))
        out.sort(key=lambda t: (t[1] // 40, t[0]))  # row-major reading order
        return out


def detect_tiles(region, backend):
    """[(x, y, kind_id)] via either backend: RoboflowDetector (detect())
    or TemplateClassifier (contours + classify())."""
    if hasattr(backend, "detect"):
        return [(x, y, kid) for x, y, kid, _ in backend.detect(region)]
    return [(x, y, backend.classify(crop)[0])
            for x, y, crop in find_tiles(region)]


def make_classifier(prefer_yolo=True):
    """Best available backend: Roboflow YOLO if weights exist, else templates."""
    if prefer_yolo and os.path.exists("robot/assets/tiles_yolo.pt"):
        try:
            return RoboflowDetector()
        except ImportError:
            print("ultralytics not installed - falling back to templates "
                  "(pip install ultralytics to use the Roboflow model)")
    return TemplateClassifier()


def read_hand(camera, classifier):
    """Kind-ids of our rack tiles, left to right (slot order)."""
    frame = camera.capture()
    tiles = detect_tiles(camera.roi(frame, "hand"), classifier)
    return [kid for _, _, kid in tiles]


def read_last_discard(camera, classifier, prev_count):
    """If a new tile appeared in the discard area, return its kind_id."""
    frame = camera.capture()
    tiles = detect_tiles(camera.roi(frame, "discard"), classifier)
    if len(tiles) <= prev_count:
        return None, len(tiles)
    return tiles[-1][2], len(tiles)
