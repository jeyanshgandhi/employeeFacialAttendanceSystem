from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, send_file, jsonify
from pymongo import MongoClient
from cryptography.fernet import Fernet
from keras_facenet import FaceNet
import pickle
import numpy as np
import pandas as pd
import base64
import cv2, os, re, datetime
from werkzeug.utils import secure_filename 
from io import BytesIO
from xlsxwriter import Workbook

app = Flask(__name__)

# Initialize MongoDB
try:
    client = MongoClient("mongodb://localhost:27017/")
    db = client["attendance_db"]
    collection = db["employees"]
    project_coll = db["projects"]
    attendance_collection = db["attendance"]
    print("✅ Connected to MongoDB")
except Exception as e:
    print(f"❌ Database connection failed: {e}")
    collection = None

# Initialize FaceNet and Haar Cascade
embedder = FaceNet()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Helper: encrypt face embedding
def encrypt_face_vector(vector, key):
    fernet = Fernet(key)
    face_bytes = pickle.dumps(vector)
    return fernet.encrypt(face_bytes)

# Helper: extract face embedding from image
def get_face_embedding(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        embeddings = embedder.embeddings([face_rgb])
        return embeddings[0]
    return None

# Decrypt function
def decrypt_face_vector(encrypted_data, key):
    fernet = Fernet(key)
    decrypted = fernet.decrypt(encrypted_data)
    return pickle.loads(decrypted)

# Match function
def is_match(stored_vector, new_vector, threshold=0.8):
    diff = np.linalg.norm(np.array(stored_vector) - np.array(new_vector))
    return diff < threshold


@app.route('/', methods=['GET', 'POST'])
def home():
    if 'email' not in session:
        return render_template('home.html', redirect_to_login=True)
    
        # DELETE action if POST request is made
    if request.method == 'POST' and 'delete_employee_id' in request.form:
        emp_id = request.form['delete_employee_id']
        employee = collection.find_one({"employee_id": emp_id})
        
        if not employee:
            return jsonify({'success': False, 'message': 'Employee not found'}), 404
        
        # Delete image if exists
        image_path = employee.get('captured_image_path')
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
        
        # Delete employee from collection
        result = collection.delete_one({"employee_id": emp_id})
        if result.deleted_count == 1:
            return jsonify({'success': True, 'message': 'Employee deleted successfully'})
        
        return jsonify({'success': False, 'message': 'Deletion failed'}), 500
    
    employees_data = []
    for emp in collection.find():
        image_path = emp.get('captured_image_path')
        base64_image = None
        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                mime_type = 'image/jpeg'  # Assuming JPEG, adjust if needed
                base64_image = f'data:{mime_type};base64,{encoded_string}'
        employees_data.append({
            'employee_name': emp.get('employee_name'),
            'position': emp.get('position'), # Or 'position
            'department': emp.get('department'),
            'base64_image': base64_image,
            "employee_id": emp.get('employee_id')
        })
    projects = project_coll.find()
    completed_project = []
    ongoing_project = []
    for pro in projects:
        if pro['status'] == 'Completed':
            completed_project.append(
                {
                'project_name': pro.get('project_name'),
                'company_name': pro.get('company_name'),
                'start_date': pro.get('start_date'),  
                'end_date': pro.get('end_date'),
                'budget': pro.get('budget'),
                'project_type': pro.get('project_type'),
                'description': pro.get('description'),
                'status': pro.get('status'),
                }
        )
        if pro['status'] == 'Ongoing':
            ongoing_project.append(
                {
                'project_name': pro.get('project_name'),
                'company_name': pro.get('company_name'),
                'start_date': pro.get('start_date'),  
                'end_date': pro.get('end_date'),
                'budget': pro.get('budget'),
                'project_type': pro.get('project_type'),
                'description': pro.get('description'),
                'status': pro.get('status'),
                }
        )
    
    
    # Suppose you aggregated your top employees like this:
    pipeline = [
        {"$group": {"_id": "$employee_name", "attendance_count": {"$sum": 1}}},
        {"$sort": {"attendance_count": -1}},
        {"$limit": 10}
    ]
    top_attendance_cursor = attendance_collection.aggregate(pipeline)

    top_employees = []
    for record in top_attendance_cursor:
        employee_name = record['_id']
        attendance_count = record['attendance_count']
        # Lookup employee details from your 'collection'
        emp = collection.find_one({"employee_name": employee_name})
        if emp:
            # Process the image (using the same method as in your sidebar)
            image_path = emp.get('captured_image_path')
            base64_image = None
            if image_path and os.path.exists(image_path):
                with open(image_path, 'rb') as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    mime_type = 'image/jpeg'  # Adjust if necessary
                    base64_image = f'data:{mime_type};base64,{encoded_string}'
            
            top_employees.append({
                'employee_name': emp.get('employee_name'),
                'employee_id': emp.get('employee_id'),
                'attendance_count': attendance_count,
                'base64_image': base64_image
            })
        
    return render_template("home.html", employees=employees_data, completed_projects = completed_project, ongoing_projects=ongoing_project, top_employees=top_employees)

@app.route('/delete_employee/<emp_id>', methods=['GET'])
def delete_employee(emp_id):
    print(f"Trying to delete employee with ID: {emp_id}")  # Debug log

    employee = collection.find_one({"employee_id": emp_id})
    if not employee:
        print("Employee not found.")
        return jsonify({'success': False, 'message': 'Employee not found'}), 404

    image_path = employee.get('captured_image_path')
    if image_path and os.path.exists(image_path):
        os.remove(image_path)

    result = collection.delete_one({"employee_id": emp_id})
    if result.deleted_count == 1:
        print("Employee deleted successfully.")
        return jsonify({'success': True, 'message': 'Employee deleted successfully'})
    else:
        print("Deletion failed.")
        return jsonify({'success': False, 'message': 'Deletion failed'}), 500



today = datetime.datetime.now().date()


@app.route('/recognize', methods=['POST'])
def recognize_face():
    data = request.get_json()
    base64_image = data['image']

    # Convert base64 to OpenCV format
    img_data = base64.b64decode(base64_image)
    np_array = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))

    if len(faces) == 0:
        return jsonify({"success": False, "message": "No faces detected"})

    recognized_faces = []

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        embedding = get_face_embedding(face_roi)

        if embedding is None:
            continue

        for record in collection.find():
            encrypted_data = record["embedding"]
            key = record["key"]
            fernet = Fernet(key)
            stored_embedding = pickle.loads(fernet.decrypt(encrypted_data))

            dist = np.linalg.norm(embedding - stored_embedding)
            if dist < 0.7:
                name = record["employee_name"]
                recognized_faces.append({
                    "name": name,
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h)
                })

                existing_entry = attendance_collection.find_one({
                    "employee_name": name,
                    "timestamp": {
                        "$gte": datetime.datetime.combine(today, datetime.datetime.min.time()),
                        "$lte": datetime.datetime.combine(today, datetime.datetime.max.time())
                    }
                })

                if not existing_entry:
                    # ⏰ Record attendance
                    attendance_collection.insert_one({
                        "employee_name": name,
                        "timestamp": datetime.datetime.now(),
                        "attendance": "present"
                    })

                break  # Stop checking once a match is found for this face


    if recognized_faces:
        return jsonify({"success": True, "faces": recognized_faces})
    else:
        return jsonify({"success": False, "message": "No match found"})


@app.route('/login_recognition')
def login_page():
    if 'email' not in session:
        return render_template('login_recognition.html')
    else:
        return render_template('home.html')


@app.route('/login_recognition', methods=['GET', 'POST'])
def login_recognition():
    if request.method == 'GET':
        return render_template('login_recognition.html')

    if request.method == 'POST':
        data = request.get_json()
        image_data = data['image']

        header, encoded = image_data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        embedding = get_face_embedding(frame)

        if embedding is None:
            return jsonify({"success": False, "message": "Face not detected"}), 400

        # Search in database for match
        for record in collection.find():
            encrypted_data = record["embedding"]
            key = record["key"]
            position = record['position']
            fernet = Fernet(key)
            stored_embedding = pickle.loads(fernet.decrypt(encrypted_data))
            dist = np.linalg.norm(embedding - stored_embedding)
            if dist < 0.9:
                if position.lower() == "hr":
                    session['email'] = record['email']
                    return jsonify({"success": True}), 200
                else:
                    return jsonify({"success": False, "message": "Access denied: Not an HR"}), 403

        return jsonify({"success": False, "message": "Face not recognized"}), 401

@app.route('/add_project')
def add_project():    
    return render_template('add_project.html')

@app.route('/submit_project', methods=['POST'])
def submit_project():
    project_name = request.form.get('project_name')
    company_name = request.form.get('company_name')
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    budget = request.form.get('budget')
    project_type = request.form.get('project_type')
    description = request.form.get('description')
    status = request.form.get('status')  # This will be 'on' if checked, None if not

    # Prepare document
    project_data = {
        'project_name': project_name,
        'company_name': company_name,
        'start_date': datetime.datetime.strptime(start_date, '%Y-%m-%d') if start_date else None,
        'end_date': datetime.datetime.strptime(end_date, '%Y-%m-%d') if end_date else None,
        'budget': float(budget) if budget else 0.0,
        'project_type': project_type,
        'description': description,
        'status': 'Completed' if status == 'on' else 'Ongoing',
    }

    # Insert into MongoDB
    project_coll.insert_one(project_data)

    return redirect(url_for('home'))

@app.route('/employees')
def employees():
    return render_template('emp_list.html')

@app.route('/export-excel')
def export_excel():
    range_type = request.args.get('range', 'month')  # default is month

    now = datetime.datetime.now()

    # Define start date based on range
    if range_type == 'week':
        start_date = now - datetime.timedelta(days=7)
    elif range_type == 'month':
        start_date = now.replace(day=1)
    elif range_type == 'year':
        start_date = now.replace(month=1, day=1)
    else:
        return "Invalid range", 400

    # Query attendance from MongoDB
    records = attendance_collection.find({
        "timestamp": {"$gte": start_date},
        "attendance": "present"
    })

    # Convert to DataFrame
    data = []
    for record in records:
        data.append({
            "Employee Name": record.get("employee_name"),
            "Date": record.get("timestamp").strftime("%Y-%m-%d"),
            "Time": record.get("timestamp").strftime("%H:%M:%S"),
            "Status": record.get("attendance")
        })

    if not data:
        return "No attendance records found for selected range.", 404

    df = pd.DataFrame(data)

    # Export to Excel in memory
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Attendance')
    output.seek(0)

    # Return Excel file
    filename = f"attendance_{range_type}_{now.strftime('%Y%m%d')}.xlsx"
    return send_file(output, download_name=filename, as_attachment=True, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

@app.route('/completed')
def completed():
    return "<h1>Completed Projects Page</h1>"

@app.route('/logout')
def logout():
    session.pop('email', None)   # Remove email from session
    return redirect(url_for('login_recognition'))  # Go back to login page


# File upload settings
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'jpg', 'png'}
MAX_CONTENT_LENGTH = 1 * 1024 * 1024  # 1 MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/register_emp')
def register():
    return render_template('register_emp.html')

@app.route('/register_emp', methods=['POST'])
def register_emp():
    if request.method == 'POST':
        try:
            # Retrieve form data
            name = request.form.get('employee_name')
            employee_id = request.form.get('employee_id')
            email = request.form.get('email')
            phone = request.form.get('phone')
            address = request.form.get('address')
            position = request.form.get('position')
            department = request.form.get('department')
            dob = request.form.get('dob')
            doj = request.form.get('doj')

            resume = request.files.get('resume')
            offer_letter = request.files.get('offer_letter')
            bond_document = request.files.get('bond_document')
            captured_image = request.files.get('captured_image')

            if not all([name, employee_id, email, phone, address, position, department, dob, doj, resume, offer_letter, captured_image]):
                return jsonify({"success": False, "message": "All required fields must be filled."}), 400

            # Validate email prefix (alphanumeric only)
            if not re.match(r'^[a-zA-Z0-9]+$', email):
                return jsonify({"success": False, "message": "Invalid email prefix. Only alphanumeric characters are allowed."}), 400

            # Construct the full email address
            email = f"{email}@company.in"

            # Check if employee_id or email already exists
            existing_employee = collection.find_one({ "$or": [{"employee_id": employee_id}, {"email": email}] })
            if existing_employee:
                return jsonify({"success": False, "message": "Employee with this ID or email already exists."}), 409 # 409 Conflict

            # Process captured image
            file_data = captured_image.read()
            nparr = np.frombuffer(file_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return jsonify({"success": False, "message": "Could not decode captured image."}), 400

            embedding = get_face_embedding(frame)

            if embedding is None:
                return jsonify({"success": False, "message": "No face detected in the captured image."}), 400

            # Check for similar face embedding by decrypting and comparing
            for emp in collection.find():
                if "embedding" in emp and "key" in emp:
                    try:
                        stored_embedding_encrypted = emp["embedding"]
                        key_db = emp["key"].encode()
                        stored_embedding = decrypt_face_vector(stored_embedding_encrypted, key_db)
                        if is_match(stored_embedding, embedding):
                            return jsonify({"success": False, "message": "A similar face is already registered."}), 409 # 409 Conflict
                    except Exception as e:
                        print(f"Error comparing embeddings: {e}")
                        continue # Ignore errors during comparison

            # Save uploaded files
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])

            resume_filename = secure_filename(resume.filename)
            resume.save(os.path.join(app.config['UPLOAD_FOLDER'], resume_filename))

            offer_letter_filename = secure_filename(offer_letter.filename)
            offer_letter.save(os.path.join(app.config['UPLOAD_FOLDER'], offer_letter_filename))

            bond_document_filename = None
            if bond_document:
                bond_document_filename = secure_filename(bond_document.filename)
                bond_document.save(os.path.join(app.config['UPLOAD_FOLDER'], bond_document_filename))

            # Save captured image
            captured_image_filename = f"{employee_id}.jpg"
            cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], captured_image_filename), frame)

            # Encrypt face embedding
            key = Fernet.generate_key()
            encrypted_embedding = encrypt_face_vector(embedding, key)

            # Save employee data to the database
            employee_data = {
                "employee_name": name,
                "employee_id": employee_id,
                "email": email,
                "phone": phone,
                "address": address,
                "position": position,
                "department": department,
                "dob": dob,
                "doj": doj,
                "resume": resume_filename,
                "offer_letter": offer_letter_filename,
                "bond_document": bond_document_filename,
                "captured_image_path": os.path.join(app.config['UPLOAD_FOLDER'], captured_image_filename),
                "embedding": encrypted_embedding,
                "key": key.decode()
            }
            try:
                collection.insert_one(employee_data)
                return jsonify({"success": True, "message": "Employee registered successfully"}), 200
            except Exception as e:
                return jsonify({"success": False, "message": f"Database error during insertion: {str(e)}"}), 500

        except Exception as e:
            print(f"Error during registration: {str(e)}")
            return jsonify({"success": False, "message": f"An error occurred during registration: {str(e)}"}), 500

    return jsonify({"success": False, "message": "Invalid request method"}), 405

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Error Routes
@app.errorhandler(404)
def not_found(e):
    return "404: Page not found.", 404

@app.errorhandler(500)
def server_error(e):
    return "500: Internal server error.", 500

if __name__ == '__main__':
    app.secret_key = 'FACIALrECORD'
    app.run(debug=True)
