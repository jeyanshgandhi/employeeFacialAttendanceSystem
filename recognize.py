import cv2
import numpy as np
from pymongo import MongoClient
from keras_facenet import FaceNet
from datetime import datetime
from cryptography.fernet import Fernet
import pickle

# Initialize FaceNet embedder
embedder = FaceNet()

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["attendance_db"]
collection = db["employees"]
attendance_log = db["attendance_log"]

# Decrypt stored face embedding
def decrypt_face_vector(encrypted_data, key):
    fernet = Fernet(key)
    decrypted = fernet.decrypt(encrypted_data)
    return pickle.loads(decrypted)

# Compare embeddings
def is_match(stored_vector, new_vector, threshold=0.7):
    diff = np.linalg.norm(np.array(stored_vector) - np.array(new_vector))
    return diff < threshold

# Check if already marked today
def already_marked(name):
    today = datetime.now().strftime("%Y-%m-%d")
    return attendance_log.find_one({"name": name, "date": today}) is not None

# Save attendance
def mark_attendance(name):
    attendance_log.insert_one({
        "name": name,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M:%S")
    })
    print(f"[✓] Attendance marked for {name}")

def recognize():
    cap = cv2.VideoCapture(0)  # Use your IP/CCTV stream URL here if needed
    print("Camera started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error.")
            break

        # Detect and get embedding
        results = embedder.extract(frame, threshold=0.95)

        if results:
            embedding = results[0]['embedding']
            x, y, w, h = results[0]['box']

            for result in results:
                embedding = result['embedding']
                x, y, w, h = result['box']

                for record in collection.find():
                    decrypted_vector = decrypt_face_vector(record['embedding'], record['key'])

                    if is_match(decrypted_vector, embedding):
                        name = record["name"]

                        if not already_marked(name):
                            mark_attendance(name)
                        else:
                            print(f"[i] {name} already marked today.")

                        # Draw box and label
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, name, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        break  # Stop checking once matched


        # Show camera
        cv2.imshow("Entrance Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting camera...")
            break

    cap.release()
    cv2.destroyAllWindows()

# Run recognition
recognize()
