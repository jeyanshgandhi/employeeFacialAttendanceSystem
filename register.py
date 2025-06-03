import cv2
import numpy as np
from cryptography.fernet import Fernet
from pymongo import MongoClient
from keras_facenet import FaceNet
import pickle

# Initialize FaceNet embedder
embedder = FaceNet()

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["attendance_db"]
collection = db["employees"]

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def encrypt_face_vector(vector, key):
    fernet = Fernet(key)
    face_bytes = pickle.dumps(vector)
    return fernet.encrypt(face_bytes)

# Helper to extract and embed face
def get_face_embedding(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        embeddings = embedder.embeddings([face_rgb])
        # if embeddings[0] != None:
        return embeddings[0]
    return None

cap = cv2.VideoCapture(0)
print("Opening camera. Press 's' to capture.")

embedding = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error.")
        break

    # Display the frame
    cv2.imshow("Face Capture - Press 's' to save", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # If 's' is pressed
        embedding = get_face_embedding(frame)
        if embedding is not None:
            print("Face captured successfully.")
        else:
            print("No face detected. Try again.")
        break
    elif key == ord('q'):
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()

if embedding is None:
    print("No embedding found. Exiting.")
    exit()    

found = False
for record in collection.find():
    encrypted_data = record["embedding"]
    key = record["key"]

    fernet = Fernet(key)
    decrypted_data = fernet.decrypt(encrypted_data)
    stored_embedding = pickle.loads(decrypted_data)

    dist = np.linalg.norm(embedding - stored_embedding)


    if dist < 0.9:  # You can tune this threshold
        # print(f"Face already Registered!!")
        found = True
        break

if not found:
    print("Face not recognized.")
    name = input("Enter your name: ")
    email = input("Enter your email: ")

    key = Fernet.generate_key()
    encrypted_vector = encrypt_face_vector(embedding, key)

    collection.insert_one({
        "name": name,
        "email": email,
        "embedding": encrypted_vector,
        "key": key,
    })
    print(f"{name} registered successfully.")

else:
    print(f"Face already Registered!!")
