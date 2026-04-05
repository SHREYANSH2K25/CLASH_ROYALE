import cv2
from ultralytics import YOLO

model = YOLO('models/card.pt')

for i in range(1, 5):
    path = f'screenshots/card_{i}.png'
    img = cv2.imread(path)
    if img is None:
        continue
    
    print(f"\n--- Testing {path} ---")
    
    # Test 1: Native YOLO inference with low confidence threshold
    res1 = model(img, conf=0.01, verbose=False)[0]
    print("Native YOLO (conf=0.01):")
    if not res1.boxes:
        print("  -> None")
    else:
        for b in res1.boxes:
            print(f"  -> {model.names[int(b.cls[0])]:20s} conf={float(b.conf[0]):.3f}")

    # Test 2: Old manual resize to 640x640 with low conf
    resized = cv2.resize(img, (640, 640))
    res2 = model(resized, conf=0.01, verbose=False)[0]
    print("Manual Resize 640x640 (conf=0.01):")
    if not res2.boxes:
        print("  -> None")
    else:
        for b in res2.boxes:
            print(f"  -> {model.names[int(b.cls[0])]:20s} conf={float(b.conf[0]):.3f}")
