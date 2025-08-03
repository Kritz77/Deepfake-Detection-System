import cv2
import os

# Load OpenCV's pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Input and output directories
input_dirs = ["dataset/real", "dataset/fake"]
output_dir = "cropped_faces"

# Create output folder if it doesn’t exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process images in both real and fake folders
for input_dir in input_dirs:
    category = input_dir.split("/")[-1]  # Get folder name (real or fake)
    save_path = os.path.join(output_dir, category)
    os.makedirs(save_path, exist_ok=True)

    for filename in os.listdir(input_dir):
        img_path = os.path.join(input_dir, filename)
        
        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Convert to grayscale for better detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for i, (x, y, w, h) in enumerate(faces):
            face = img[y:y+h, x:x+w]  # Crop face
            face_filename = f"{filename.split('.')[0]}_face{i}.jpg"
            cv2.imwrite(os.path.join(save_path, face_filename), face)

print("Face extraction completed!")
