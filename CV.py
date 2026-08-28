import cv2
import numpy as np
import tensorflow as tf

# model goes here (update this later)
# model = tf.keras.models.load_model('model.h5')
model = None 

# Open the webcam (0 is usually the default laptop camera)
cap = cv2.VideoCapture(0)
print("Starting camera... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not connect to camera.")
        break
        
    # resize and prep a copy of the frame for the model
    img_resized = cv2.resize(frame, (64, 64))
    img_data = np.expand_dims(img_resized, axis=0)

    if model is not None:
        # get predictions (verbose=0 stops it from spamming the console)
        preds = model.predict(img_data, verbose=0)
        
        labels = ["Unripe", "Ripe", "Overripe"]
        res = labels[np.argmax(preds[0])]
        
        # Put the prediction text directly on the camera feed
        cv2.putText(frame, f"Prediction: {res}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Waiting for model...", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the live computer vision feed
    cv2.imshow('Fruit Ripeness Vision', frame)

    # Stop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up when done
cap.release()
cv2.destroyAllWindows()
