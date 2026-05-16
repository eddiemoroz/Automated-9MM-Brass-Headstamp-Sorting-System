import cv2
import numpy as np
import os

headstampNum = 9
inputImagePath =
outputDir =

os.makedirs(outputDir, exist_ok=True)

cropWidth = 750
cropHeight = 750
centerX = 1690
centerY = 681
primerRadius = 180
outerMaskRadius = 360 # masks 3D printed holder / casing texture
rotationStep = 22.5
numRotations = int(360 / rotationStep)

contrastFactors = [0.9, 1.0, 1.1]
blurValues = [0, 3]
image = cv2.imread(inputImagePath)

def crop_center(img, cx, cy, w, h):
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = x1 + w
    y2 = y1 + h
    return img[max(0,y1):min(img.shape[0],y2), max(0,x1):min(img.shape[1],x2)]

def rotate_image(img, angle, cx, cy):
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    return cv2.warpAffine(
        img, M, (img.shape[1], img.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT)

def mask_regions(img, inner_radius, outer_radius):
    h, w = img.shape
    center = (w // 2, h // 2)

    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.circle(mask, center, outer_radius, 255, -1)
    cv2.circle(mask, center, inner_radius, 0, -1)

    return cv2.bitwise_and(img, mask)

def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def morphological_gradient(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

count = 0

for i in range(numRotations):
    angle = i * rotationStep

    rotated = rotate_image(image, angle, centerX, centerY)
    cropped = crop_center(rotated, centerX, centerY, cropWidth, cropHeight)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) # Mask primer + outer casing

    masked = mask_regions(gray, primerRadius, outerMaskRadius)

    # CLAHE Contrast-Limited Adaptive Histogram Equalization
    enhanced = apply_clahe(masked)

    # Morphological gradient to emphasize letters
    edges = morphological_gradient(enhanced)

    for blur in blurValues:
            if blur == 0:
                processed = edges.copy()
            else:
                processed = cv2.GaussianBlur(edges, (blur, blur), 0)

            for contrast in contrastFactors:

                final = cv2.convertScaleAbs(processed, alpha=contrast, beta=0)

                final = cv2.addWeighted(
                    enhanced, # CLAHE image
                    0.6,
                    edges, # morphological gradient
                    0.8,
                    0)

                filename = (
                    f"rot{angle:.1f}_" 
                    f"blur{blur}_" 
                    f"ctr{contrast}_" 
                    f"num{headstampNum}.jpg")

                cv2.imwrite(os.path.join(outputDir, filename), final)
                count += 1

print(f"Done. Generated {count} images.")