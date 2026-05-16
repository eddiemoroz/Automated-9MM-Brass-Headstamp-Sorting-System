import cv2
import numpy as np
import os

# File paths
inputImagePath =
outputImagePath =

os.makedirs(os.path.dirname(outputImagePath), exist_ok=True)

cropWidth = 750
cropHeight = 750

centerX = 1690
centerY = 681
primerRadius = 180
outerMaskRadius = 360   # masks 3D printed holder / casing texture

contrast = 1.0

def crop_center(img, cx, cy, w, h):
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = x1 + w
    y2 = y1 + h
    return img[max(0, y1):min(img.shape[0], y2),
               max(0, x1):min(img.shape[1], x2)]

def mask_regions(img, innerRadius, outerRadius):
    h, w = img.shape
    center = (w // 2, h // 2)

    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.circle(mask, center, outerRadius, 255, -1)
    cv2.circle(mask, center, innerRadius, 0, -1)

    return cv2.bitwise_and(img, mask)

def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def morphological_gradient(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

image = cv2.imread(inputImagePath)

cropped = crop_center(image, centerX, centerY, cropWidth, cropHeight)
gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

masked = mask_regions(gray, primerRadius, outerMaskRadius)
enhanced = apply_clahe(masked)
edges = morphological_gradient(enhanced)

# Adjust contrast if needed
edges = cv2.convertScaleAbs(edges, alpha=contrast, beta=0)

# Blend the CLAHE image + edge emphasis together
final = cv2.addWeighted(
    enhanced, 0.6,
    edges, 0.8,
    0
)

cv2.imwrite(outputImagePath, final)

print("Done.")
print("Saved processed image to:", outputImagePath)

# Show all the images in the pipeline and how each filter affected it for demonstration purposes
cv2.imshow("Cropped", cropped)
cv2.imshow("Enhanced (CLAHE)", enhanced)
cv2.imshow("Edges (Morph Gradient)", edges)
cv2.imshow("Final Output", final)

cv2.waitKey(0)
cv2.destroyAllWindows()
