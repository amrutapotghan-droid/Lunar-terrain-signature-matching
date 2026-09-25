import cv2
import numpy as np
import math
import os


# ============================================================
# LUNAR TERRAIN SIGNATURE MATCHING (LTSM)
# ============================================================

# -----------------------------
# 1. FILE SETTINGS
# -----------------------------

IMAGE_A_PATH = "image_A.png"
IMAGE_B_PATH = "image_B.png"

OUTPUT_FOLDER = "output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# -----------------------------
# 2. READ IMAGES
# -----------------------------

print("\n======================================")
print("   LUNAR TERRAIN SIGNATURE MATCHING")
print("======================================\n")

print("Step 1: Reading images...")

image_A = cv2.imread(IMAGE_A_PATH)
image_B = cv2.imread(IMAGE_B_PATH)

if image_A is None:
    print("ERROR: image_A.png was not found.")
    print("Make sure image_A.png is inside the project folder.")
    exit()

if image_B is None:
    print("ERROR: image_B.png was not found.")
    print("Make sure image_B.png is inside the project folder.")
    exit()

print("Image A loaded:", image_A.shape)
print("Image B loaded:", image_B.shape)


# -----------------------------
# 3. CONVERT TO GRAYSCALE
# -----------------------------

print("\nStep 2: Converting images to grayscale...")

gray_A = cv2.cvtColor(image_A, cv2.COLOR_BGR2GRAY)
gray_B = cv2.cvtColor(image_B, cv2.COLOR_BGR2GRAY)

print("Grayscale conversion completed.")


# -----------------------------
# 4. RESIZE IMAGE B
# -----------------------------
# For the first prototype, we resize
# Image B to Image A dimensions.
#
# IMPORTANT:
# This is only a prototype.
# Later we will handle real scale differences
# using scale-invariant matching.

print("\nStep 3: Resizing Image B...")

height_A, width_A = gray_A.shape

gray_B = cv2.resize(
    gray_B,
    (width_A, height_A)
)

print("Image B resized to:", gray_B.shape)


# -----------------------------
# 5. CLAHE NORMALIZATION
# -----------------------------
# CLAHE improves local contrast and
# reduces some illumination differences.

print("\nStep 4: Applying illumination normalization...")

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

normalized_A = clahe.apply(gray_A)
normalized_B = clahe.apply(gray_B)

cv2.imwrite(
    os.path.join(OUTPUT_FOLDER, "01_normalized_A.png"),
    normalized_A
)

cv2.imwrite(
    os.path.join(OUTPUT_FOLDER, "02_normalized_B.png"),
    normalized_B
)

print("Normalization completed.")


# -----------------------------
# 6. CALCULATE GRADIENT
# -----------------------------
# Gradient shows structural changes
# such as crater edges, ridges and
# geological boundaries.

print("\nStep 5: Calculating terrain gradients...")

gx_A = cv2.Sobel(
    normalized_A,
    cv2.CV_32F,
    1,
    0,
    ksize=3
)

gy_A = cv2.Sobel(
    normalized_A,
    cv2.CV_32F,
    0,
    1,
    ksize=3
)

gx_B = cv2.Sobel(
    normalized_B,
    cv2.CV_32F,
    1,
    0,
    ksize=3
)

gy_B = cv2.Sobel(
    normalized_B,
    cv2.CV_32F,
    0,
    1,
    ksize=3
)


gradient_A = cv2.magnitude(
    gx_A,
    gy_A
)

gradient_B = cv2.magnitude(
    gx_B,
    gy_B
)

print("Gradient calculation completed.")


# ============================================================
# 7. SIFT KEYPOINT DETECTION
# ============================================================
# SIFT is used ONLY to find candidate points.
#
# LTSM will create the actual terrain descriptor.

print("\nStep 6: Detecting candidate keypoints using SIFT...")

sift = cv2.SIFT_create(
    nfeatures=3000,
    contrastThreshold=0.02
)

keypoints_A, _ = sift.detectAndCompute(
    normalized_A,
    None
)

keypoints_B, _ = sift.detectAndCompute(
    normalized_B,
    None
)

print("Keypoints in Image A:", len(keypoints_A))
print("Keypoints in Image B:", len(keypoints_B))


# ============================================================
# 8. LTSM TERRAIN SIGNATURE FUNCTION
# ============================================================

def create_terrain_signature(
    gradient,
    x,
    y
):
    """
    Creates a 24-value terrain signature.

    3 rings × 8 angular sectors = 24 values.

    Each value represents the average
    gradient strength in that region.
    """

    # Three spatial rings
    rings = [
        (1, 4),
        (4, 8),
        (8, 16)
    ]

    # Each ring has 8 angular sectors
    angle_bins = 8

    signature = []

    height, width = gradient.shape

    # Loop through each ring
    for r_min, r_max in rings:

        # Loop through 8 angular sectors
        for sector in range(angle_bins):

            start_angle = (
                2 * math.pi *
                sector /
                angle_bins
            )

            end_angle = (
                2 * math.pi *
                (sector + 1) /
                angle_bins
            )

            values = []

            # Sample points inside the ring
            for radius in range(
                r_min,
                r_max
            ):

                # Sample several angles
                for angle in np.linspace(
                    start_angle,
                    end_angle,
                    5
                ):

                    px = int(
                        x +
                        radius *
                        math.cos(angle)
                    )

                    py = int(
                        y +
                        radius *
                        math.sin(angle)
                    )

                    # Check image boundaries
                    if (
                        0 <= px < width
                        and
                        0 <= py < height
                    ):

                        values.append(
                            gradient[py, px]
                        )

            # Calculate average gradient
            if len(values) > 0:

                mean_gradient = np.mean(
                    values
                )

            else:

                mean_gradient = 0

            signature.append(
                mean_gradient
            )

    # Convert to NumPy array
    signature = np.array(
        signature,
        dtype=np.float32
    )

    # Normalize signature
    norm = np.linalg.norm(
        signature
    )

    if norm > 0:

        signature = (
            signature /
            norm
        )

    return signature


# ============================================================
# 9. CREATE LTSM DESCRIPTORS FOR IMAGE A
# ============================================================

print("\nStep 7: Creating LTSM terrain signatures for Image A...")

descriptors_A = []
valid_keypoints_A = []

image_height, image_width = gradient_A.shape

border = 16

for kp in keypoints_A:

    x, y = kp.pt

    # Make sure enough surrounding area exists
    if (
        x >= border
        and
        y >= border
        and
        x < image_width - border
        and
        y < image_height - border
    ):

        signature = create_terrain_signature(
            gradient_A,
            x,
            y
        )

        descriptors_A.append(
            signature
        )

        valid_keypoints_A.append(
            kp
        )


if len(descriptors_A) == 0:

    print("ERROR: No valid descriptors found in Image A.")
    exit()


descriptors_A = np.array(
    descriptors_A,
    dtype=np.float32
)

print(
    "Image A descriptors:",
    descriptors_A.shape
)


# ============================================================
# 10. CREATE LTSM DESCRIPTORS FOR IMAGE B
# ============================================================

print("\nStep 8: Creating LTSM terrain signatures for Image B...")

descriptors_B = []
valid_keypoints_B = []

image_height, image_width = gradient_B.shape

for kp in keypoints_B:

    x, y = kp.pt

    if (
        x >= border
        and
        y >= border
        and
        x < image_width - border
        and
        y < image_height - border
    ):

        signature = create_terrain_signature(
            gradient_B,
            x,
            y
        )

        descriptors_B.append(
            signature
        )

        valid_keypoints_B.append(
            kp
        )


if len(descriptors_B) == 0:

    print("ERROR: No valid descriptors found in Image B.")
    exit()


descriptors_B = np.array(
    descriptors_B,
    dtype=np.float32
)

print(
    "Image B descriptors:",
    descriptors_B.shape
)


# ============================================================
# 11. MATCH LTSM DESCRIPTORS
# ============================================================

print("\nStep 9: Matching terrain signatures...")

# L2 distance is suitable for our normalized
# numerical descriptors.

matcher = cv2.BFMatcher(
    cv2.NORM_L2
)

matches = matcher.knnMatch(
    descriptors_A,
    descriptors_B,
    k=2
)

print(
    "Potential descriptor matches:",
    len(matches)
)


# ============================================================
# 12. LOWE RATIO TEST
# ============================================================

print("\nStep 10: Applying Lowe ratio test...")

good_matches = []

ratio_threshold = 0.75

for pair in matches:

    # We need two neighbours
    if len(pair) < 2:
        continue

    m, n = pair

    if (
        m.distance <
        ratio_threshold *
        n.distance
    ):

        good_matches.append(m)


print(
    "Good matches after ratio test:",
    len(good_matches)
)


# ============================================================
# 13. RANSAC GEOMETRIC VERIFICATION
# ============================================================

print("\nStep 11: Applying RANSAC...")

inlier_matches = []

H = None

if len(good_matches) >= 4:

    points_A = np.float32([
        valid_keypoints_A[
            match.queryIdx
        ].pt
        for match in good_matches
    ]).reshape(-1, 1, 2)

    points_B = np.float32([
        valid_keypoints_B[
            match.trainIdx
        ].pt
        for match in good_matches
    ]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        points_A,
        points_B,
        cv2.RANSAC,
        4.0,
        maxIters=5000,
        confidence=0.995
    )

    if mask is not None:

        mask = mask.ravel()

        for match, keep in zip(
            good_matches,
            mask
        ):

            if keep:

                inlier_matches.append(
                    match
                )

else:

    print(
        "Not enough matches for RANSAC."
    )


print(
    "Final RANSAC inliers:",
    len(inlier_matches)
)


# ============================================================
# 14. CALCULATE BASIC INLIER RATIO
# ============================================================

if len(good_matches) > 0:

    inlier_ratio = (
        len(inlier_matches) /
        len(good_matches)
    ) * 100

else:

    inlier_ratio = 0


print(
    f"Inlier ratio: {inlier_ratio:.2f}%"
)


# ============================================================
# 15. DRAW FINAL MATCHES
# ============================================================

print("\nStep 12: Creating final correspondence image...")

result = cv2.drawMatches(
    normalized_A,
    valid_keypoints_A,
    normalized_B,
    valid_keypoints_B,
    inlier_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

result_path = os.path.join(
    OUTPUT_FOLDER,
    "03_final_LTSM_matches.png"
)

cv2.imwrite(
    result_path,
    result
)

print(
    "Final match image saved to:",
    result_path
)


# ============================================================
# 16. SAVE CORRESPONDENCES TO CSV
# ============================================================

print("\nStep 13: Saving correspondence coordinates...")

csv_path = os.path.join(
    OUTPUT_FOLDER,
    "04_correspondences.csv"
)

with open(
    csv_path,
    "w"
) as file:

    file.write(
        "ImageA_X,ImageA_Y,ImageB_X,ImageB_Y,Distance\n"
    )

    for match in inlier_matches:

        x1, y1 = (
            valid_keypoints_A[
                match.queryIdx
            ].pt
        )

        x2, y2 = (
            valid_keypoints_B[
                match.trainIdx
            ].pt
        )

        file.write(
            f"{x1:.3f},"
            f"{y1:.3f},"
            f"{x2:.3f},"
            f"{y2:.3f},"
            f"{match.distance:.4f}\n"
        )


print(
    "Correspondence CSV saved to:",
    csv_path
)


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print("\n======================================")
print("           LTSM COMPLETE")
print("======================================")

print(
    "Image A keypoints:",
    len(keypoints_A)
)

print(
    "Image B keypoints:",
    len(keypoints_B)
)

print(
    "Good matches:",
    len(good_matches)
)

print(
    "Final RANSAC inliers:",
    len(inlier_matches)
)

print(
    f"Inlier ratio: {inlier_ratio:.2f}%"
)

print("\nOutput files:")
print("1. output/01_normalized_A.png")
print("2. output/02_normalized_B.png")
print("3. output/03_final_LTSM_matches.png")
print("4. output/04_correspondences.csv")

print("\nDone!")

