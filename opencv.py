import cv2
import numpy as np

def open_camera(camera_index=0):
    """
    Opens the camera and checks if it's successfully opened.
    Returns the VideoCapture object if successful, None otherwise.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {camera_index}.")
        return None
    return cap

def preprocess_frame(frame):
    """
    Prepares a frame for the ML model.

    - If the image is grayscale/black-and-white, convert it to RGB.
    - If the image is a color image, convert OpenCV's BGR format to RGB.
    - Resize to 64x64.
    - Normalize pixel values to 0-1.
    - Add batch dimension -> (1, 64, 64, 3).
    """

    if frame is None or frame.size == 0:
        raise ValueError("Invalid or empty frame.")

    if len(frame.shape) == 2:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif len(frame.shape) == 3 and frame.shape[2] == 3:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported image format: {frame.shape}")

    resized_frame = cv2.resize(rgb_frame, (64, 64))
    normalized_frame = resized_frame.astype(np.float32) / 255.0
    input_data = np.expand_dims(normalized_frame, axis=0)

    if input_data.shape != (1, 64, 64, 3):
        raise ValueError(
            f"Unexpected model input shape: {input_data.shape}"
        )

    return input_data

def display_frame(frame, input_shape, model_status="NOT LOADED"):
    """
    Overlays useful information on the frame and displays the live feed.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    green = (0, 255, 0)
    red = (0, 0, 255)
    
    display_img = frame.copy()
    cv2.putText(display_img, "Camera: OK", (10, 30), font, font_scale, green, thickness)
    cv2.putText(display_img, "Preprocessing: OK", (10, 60), font, font_scale, green, thickness)
    
    cv2.putText(display_img, f"Model Input: {input_shape}", (10, 90), font, font_scale, green, thickness)
    cv2.putText(display_img, f"Model Status: {model_status}", (10, 120), font, font_scale, red, thickness)
    cv2.imshow("Live Camera Feed", display_img)

def main():
    print("Initializing computer vision pipeline...")
    
    cap = open_camera()
    if cap is None:
        return

    print("Camera opened successfully. Press 'q' to exit the video feed.")

    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture image from camera.")
                break
                
            input_data = preprocess_frame(frame)
            
            #MODEL SECTION
            # prediction = model.predict(input_data, verbose=0)
            
            display_frame(frame, input_shape=input_data.shape, model_status="NOT LOADED")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exit requested by user.")
                break
    finally:
        print("Releasing resources...")
        cap.release()
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == "__main__":
    main()
