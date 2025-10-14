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

# Load all decrypted embeddings once
def load_all_embeddings():
    print("🔃 Loading employee embeddings from DB...")
    employees = []
    for record in collection.find():
        decrypted_vector = decrypt_face_vector(record['embedding'], record['key'])
        employees.append((record['name'], decrypted_vector))
    print(f"✅ Loaded {len(employees)} employee embeddings.")
    return employees

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

# Main recognition function
def recognize():
    known_faces = load_all_embeddings()
    cap = cv2.VideoCapture(0)  # Use your IP/CCTV stream URL here if needed
    print("📷 Camera started. Press 'q' to quit.")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera error.")
            break

        frame_count += 1
        if frame_count % 5 != 0:
            continue  # Process every 5th frame only

        # Resize frame to half for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Detect faces
        results = embedder.extract(small_frame, threshold=0.90)

        for result in results:
            embedding = result['embedding']
            x, y, w, h = result['box']
            # Scale back coordinates to original frame size
            x, y, w, h = [int(coord * 2) for coord in [x, y, w, h]]

            for name, decrypted_vector in known_faces:
                if is_match(decrypted_vector, embedding):
                    if not already_marked(name):
                        mark_attendance(name)
                    else:
                        print(f"[i] {name} already marked today.")

                    # Draw box and label
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    break  # Stop checking after match

        # Show camera frame
        cv2.imshow("Entrance Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🚪 Exiting camera...")
            break

    cap.release()
    cv2.destroyAllWindows()

# Run recognition
recognize()
