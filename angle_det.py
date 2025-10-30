import cv2
import numpy as np
import sys


def resize_for_display(image, max_width=1000):
    """
    Resizes an image to a maximum width while maintaining aspect ratio.
    """
    h, w = image.shape[:2]
    if w > max_width:
        ratio = max_width / float(w)
        new_h = int(h * ratio)
        return cv2.resize(image, (max_width, new_h), interpolation=cv2.INTER_AREA)
    return image


def find_and_draw_coil_orientation(image_path):
    """
    Loads an image, finds ALL large rectangular objects, and draws
    their center, orientation, and angle.
    """
    # --- 1. Load and Pre-process Image ---
    # This part is unchanged. We create the 'processed_img' once.

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    output_image = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    canny = cv2.Canny(blur, 50, 150)
    kernel = np.ones((9, 9), np.uint8)
    processed_img = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel, iterations=2)

    # --- 2. Find ALL Contours ---

    contours, _ = cv2.findContours(
        processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        print("Error: No contours found. Adjust Canny thresholds or kernel size.")
        debug_display_img = resize_for_display(processed_img, max_width=400)
        cv2.imshow("Debug: Canny + Close", debug_display_img)
        cv2.waitKey(0)
        return

    print(f"Found {len(contours)} potential objects...")

    # --- 3. Loop Through Each Detected Object ---
    # This is the main change. We loop through 'contours' instead of
    # just using the 'largest_contour'.

    object_count = 0
    for cnt in contours:
        # --- 3a. Filter out small noise contours ---
        # We check the area of the contour. If it's too small, we skip it.
        # *** YOU WILL NEED TO TUNE THIS VALUE! ***
        area = cv2.contourArea(cnt)
        if area < 1000:  # e.g., ignore anything smaller than 1000 pixels
            continue

        object_count += 1
        print(f"\n--- Processing Object {object_count} ---")

        # --- 3b. Get Rotated Rectangle (same as before) ---
        # All logic from here on is just moved inside the loop
        rect = cv2.minAreaRect(cnt)  # Use 'cnt' (current contour)
        center, (width, height), angle = rect

        # --- 3c. Normalize Angle (same as before) ---
        if width < height:
            final_angle_degrees = 90 + angle
            long_axis_length = height
        else:
            final_angle_degrees = angle
            long_axis_length = width

        if final_angle_degrees < 0:
            final_angle_degrees += 180

        # --- 3d. Print Results (same as before) ---
        print(f"Center Coordinates: ({int(center[0])}, {int(center[1])})")
        print(f"Dimensions (W, H):  ({int(width):.0f}, {int(height):.0f})")
        print(f"Raw OpenCV Angle:   {angle:.2f} degrees")
        print(f"Normalized Long Axis Angle (0-180): {final_angle_degrees:.2f} degrees")

        # --- 3e. Draw Visualizations (same as before) ---
        # All drawing commands are now drawing on the SAME 'output_image'

        # A. Draw the rotated bounding box (Blue)
        box = cv2.boxPoints(rect)
        box = np.int64(box)
        cv2.drawContours(output_image, [box], 0, (255, 0, 0), 2)

        # B. Draw the line through the center (Green)
        rad_long_axis = np.deg2rad(final_angle_degrees)
        line_draw_length = max(width, height) * 0.6

        p1 = (
            int(center[0] - line_draw_length * np.cos(rad_long_axis)),
            int(center[1] - line_draw_length * np.sin(rad_long_axis)),
        )
        p2 = (
            int(center[0] + line_draw_length * np.cos(rad_long_axis)),
            int(center[1] + line_draw_length * np.sin(rad_long_axis)),
        )
        cv2.line(output_image, p1, p2, (0, 255, 0), 2)

        # C. Draw the measured angle visualization
        ref_radius = int(min(width, height) * 0.8)
        if ref_radius < 20:
            ref_radius = 20

        ref_p_end = (int(center[0] + ref_radius), int(center[1]))
        cv2.line(
            output_image, (int(center[0]), int(center[1])), ref_p_end, (0, 0, 255), 2
        )

        cv2.ellipse(
            output_image,
            (int(center[0]), int(center[1])),
            (ref_radius, ref_radius),
            0,
            0,
            final_angle_degrees,
            (0, 255, 255),
            2,
        )

        text = f"{final_angle_degrees:.1f} deg"
        text_offset_angle_rad = np.deg2rad(final_angle_degrees / 2.0)
        text_x = int(center[0] + (ref_radius + 20) * np.cos(text_offset_angle_rad))
        text_y = int(center[1] + (ref_radius + 20) * np.sin(text_offset_angle_rad))

        cv2.putText(
            output_image,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    # --- 4. Display the Final Image ---
    # This section is outside the loop. It shows the final 'output_image'
    # after all objects have been drawn on it.

    print(f"\nFound and drew {object_count} objects.")

    display_img = resize_for_display(output_image, max_width=400)
    debug_display_img = resize_for_display(processed_img, max_width=400)

    cv2.imshow("Detected Object Orientation", display_img)
    cv2.imshow("Debug: Canny + Close", debug_display_img)

    print("\nDisplaying image. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --- Main execution ---
if __name__ == "__main__":
    # Make sure this image path has *multiple* objects
    image_file = r"C:\Users\SHIV\Desktop\pen.jpg"

    find_and_draw_coil_orientation(image_file)
