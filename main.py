import os
import time
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, Response
from werkzeug.utils import secure_filename
import cv2
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('best.pt')

UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

detected_objects = []
processed_filename = None


@app.route('/', methods=['GET', 'POST'])
def home():
    global detected_objects, processed_filename

    if request.method == 'POST':
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return 'No selected file', 400

            detected_objects = []
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            results = model.predict(source=filepath)[0]

            img = cv2.imread(filepath)
            img_height, img_width, _ = img.shape

            for result in results.boxes:
                class_id = int(result.cls)
                score = result.conf.item()

                x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
                center_x, center_y, w, h = result.xywh[0].tolist()

                center_x /= img_width
                center_y /= img_height
                w /= img_width
                h /= img_height

                detected_objects.append({
                    'name': model.names[class_id],
                    'confidence': score * 100,
                    'class_id': class_id,
                })

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'{model.names[class_id]} {score:.2f}'
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            processed_filename = filename
            processed_image_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
            cv2.imwrite(processed_image_path, img)
            return render_template('home.html', filename=processed_filename, detected_objects=detected_objects)

    return render_template('home.html', filename=processed_filename, detected_objects=detected_objects)

@app.route('/webcam', methods=['GET', 'POST'])
def stream_webcam():
    def generate():
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame)[0]
            for result in results.boxes:
                class_id = int(result.cls)
                score = result.conf.item()
                bbox_xyxy = result.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, bbox_xyxy)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'{model.names[class_id]} {score:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
            time.sleep(0.03)

        cap.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/uploads/<filename>')
def uploaded_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def processed_image(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['PROCESSED_FOLDER']):
        os.makedirs(app.config['PROCESSED_FOLDER'])
    app.run(debug=True)
