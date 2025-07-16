import os
import sys
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Check if FER is installed, if not install it
try:
    from fer import FER
    from fer.utils import draw_annotations
except ImportError:
    import subprocess

    print("Installing FER library...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fer"])
    from fer import FER
    from fer.utils import draw_annotations


def clear_terminal():
    """Clear the terminal screen based on OS"""
    os.system("cls" if os.name == "nt" else "clear")


def emotion_detection():
    """
    Real-time emotion detection using FER library.
    Detects faces and emotions from webcam feed.
    """
    print("Starting real-time emotion detection with FER...")
    print("Press 'q' to quit")
    time.sleep(2)  # Give time to read instructions

    # Initialize FER detector
    # Using MTCNN for better face detection accuracy
    detector = FER(mtcnn=True)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    # For frame rate calculation
    prev_time = 0
    # Process every nth frame to reduce CPU usage
    process_every_n_frames = 10
    frame_count = 0

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Only process every nth frame
        frame_count += 1
        if frame_count % process_every_n_frames != 0:
            # Still show the frame but don't process it
            cv2.imshow("FER Emotion Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Calculate FPS for processed frames
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        try:
            # Detect emotions
            result = detector.detect_emotions(frame)

            # Clear terminal for better readability
            clear_terminal()

            # Print header with FPS
            print(f"===== FER EMOTION DETECTION =====")
            print(f"FPS: {fps:.2f} (processing every {process_every_n_frames} frames)")
            print(f"Press 'q' in the video window to quit")
            print("=======================================\n")

            # Process results
            if result:
                # Draw boxes and emotions on the frame
                annotated_frame = frame.copy()

                for face in result:
                    # Get face box
                    box = face["box"]
                    x, y, w, h = box

                    # Draw rectangle around face
                    cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Get emotions
                    emotions = face["emotions"]

                    # Find dominant emotion
                    dominant_emotion = max(emotions, key=emotions.get)
                    dominant_score = emotions[dominant_emotion]

                    # Add text to the image
                    text = f"{dominant_emotion}: {dominant_score:.2f}"
                    cv2.putText(
                        annotated_frame,
                        text,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                    # Print emotion results in terminal with nice formatting
                    print(f"Face at position {box}:")
                    print(f"Dominant emotion: {dominant_emotion.upper()} ({dominant_score:.2f})")

                    # Sort emotions by score for better display
                    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)

                    # Print emotion bars in terminal
                    for emotion, score in sorted_emotions:
                        # Create a visual bar representation
                        bar_length = int(score * 50)  # Scale to reasonable length
                        bar = "█" * bar_length
                        print(f"{emotion.ljust(10)}: {bar} {score:.2f}")

                    print()  # Add space between faces

                frame = annotated_frame
            else:
                print("No faces detected")

        except Exception as e:
            print(f"Error: {str(e)}")

        # Display FPS on frame
        cv2.putText(
            frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )

        # Display the frame
        cv2.imshow("FER Emotion Detection", frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


def process_image(image_path):
    """
    Process a single image and display the results
    """
    # Initialize FER detector
    detector = FER(mtcnn=True)

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image at {image_path}")
        return

    # Detect emotions
    result = detector.detect_emotions(img)

    if result:
        # Create a copy for drawing
        annotated_img = img.copy()

        for face in result:
            # Get face box
            box = face["box"]
            x, y, w, h = box

            # Draw rectangle around face
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Get emotions
            emotions = face["emotions"]

            # Find dominant emotion
            dominant_emotion = max(emotions, key=emotions.get)
            dominant_score = emotions[dominant_emotion]

            # Add text to the image
            text = f"{dominant_emotion}: {dominant_score:.2f}"
            cv2.putText(
                annotated_img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

            # Print emotion results
            print(f"Face at position {box}:")
            print(f"Dominant emotion: {dominant_emotion.upper()} ({dominant_score:.2f})")

            # Print all emotions
            for emotion, score in emotions.items():
                print(f"{emotion}: {score:.2f}")

            print()

        # Display the result
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Emotion Detection Results")
        plt.show()

        # Save the result
        result_path = "fer_result.jpg"
        cv2.imwrite(result_path, annotated_img)
        print(f"Result saved to {result_path}")
    else:
        print("No faces detected in the image")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # If image path is provided, process the image
        process_image(sys.argv[1])
    else:
        # Otherwise, use webcam
        emotion_detection()
