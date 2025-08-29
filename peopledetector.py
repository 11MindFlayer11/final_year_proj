import cv2
import argparse
from ultralytics import YOLO


def count_persons_in_image(image_path: str, output_path: str):
    """
    Detects and counts persons in an image using YOLOv8, draws bounding boxes,
    and saves the result.

    Args:
        image_path (str): Path to the input image file.
        output_path (str): Path to save the output image with detections.
    """
    try:
        # 1. Load the pre-trained YOLOv8 model
        # 'yolov8n.pt' is a small and fast model, suitable for general use.
        model = YOLO("yolov8n.pt")

        # 2. Read the input image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image from path: {image_path}")
            return

        # 3. Run inference on the image
        results = model(frame)

        person_count = 0

        # 4. Process the detection results
        # The model returns a list of results, one for each image (we only have one)
        for result in results:
            boxes = result.boxes  # Get the bounding box object
            for box in boxes:
                # Get the class ID for each detected object
                class_id = int(box.cls[0])

                # The model.names dictionary maps class IDs to class names
                # We check if the detected object's class name is 'person'
                if model.names[class_id] == "person":
                    person_count += 1

                    # Get the coordinates of the bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Draw a green rectangle around the detected person
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Add a label for clarity
                    label = "Person"
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

        # 5. Display the final count on the image
        count_text = f"Total Persons Detected: {person_count}"
        print(count_text)
        cv2.putText(
            frame, count_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )

        # 6. Save the output image
        cv2.imwrite(output_path, frame)
        print(f"Output image saved to: {output_path}")

        # 7. Optionally, display the image in a window
        cv2.imshow("YOLO Person Detection", frame)
        print("Press any key to close the image window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Set up argument parser to accept image path from the command line
    parser = argparse.ArgumentParser(
        description="Detect and count persons in an image using YOLOv8."
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--output", default="output.jpg", help="Path to save the output image."
    )

    args = parser.parse_args()

    count_persons_in_image(args.image, args.output)
