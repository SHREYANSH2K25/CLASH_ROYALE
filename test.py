# import cv2
# from ultralytics import YOLO

# # 🔹 Load model
# model = YOLO("models/troop.pt")  # adjust path if needed

# # 🔹 Load image
# img_path = "images/scr.jpg"   # <-- put your screenshot path here
# img = cv2.imread(img_path)

# if img is None:
#     print("❌ Failed to load image. Check path.")
#     exit()

# # 🔹 Run inference
# results = model(img)[0]

# # 🔹 Print summary
# print("Total detections:", len(results.boxes))

# # 🔹 Print each detection
# for box in results.boxes:
#     conf = float(box.conf[0])
#     cls_id = int(box.cls[0])
#     cls_name = results.names[cls_id]
#     print(f"Detected: {cls_name} | Confidence: {conf:.2f}")

# # 🔹 Show annotated image
# annotated = results.plot()
# cv2.imshow("YOLO Detection", annotated)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2
from ultralytics import YOLO

# ===== LOAD MODEL =====
model = YOLO("models/card.pt")   # adjust path if needed

# ===== LOAD IMAGE =====
img = cv2.imread("screenshots\card_4.png")  # change to your image path

if img is None:
    print("Error: Image not found")
    exit()

# ===== RUN PREDICTION =====
results = model(img)
best_class = ""
confidence = 0.1
# ===== PROCESS RESULT =====
for r in results:
    probs = r.probs.data.tolist()   # probabilities
    names = model.names             # class names

    # get best prediction
    max_idx = probs.index(max(probs))
    best_class = names[max_idx]
    confidence = probs[max_idx]

print(f"Prediction: {best_class[P]} ({confidence:.4f})")

# ===== OPTIONAL: SHOW IMAGE =====
cv2.imshow("Input", img)
cv2.waitKey(0)
cv2.destroyAllWindows()