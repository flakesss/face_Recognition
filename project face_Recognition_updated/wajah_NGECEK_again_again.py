import cv2
import face_recognition
import numpy as np
import os
import pickle
import sys
import logging
import matplotlib.pyplot as plt
import glob

ENCODINGS_PATH = "encodings.pickle"
REGISTERED_FACES_DIR = "registered_faces"

# Setup logging
logging.basicConfig(level=logging.INFO)

def load_encodings():
    if os.path.exists(ENCODINGS_PATH):
        with open(ENCODINGS_PATH, "rb") as f:
            data = pickle.load(f)
        return data["encodings"], data["names"], data["departments"]
    else:
        return [], [], []

def save_encodings(encodings, names, departments):
    data = {"encodings": encodings, "names": names, "departments": departments}
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(data, f)

def check_dataset():
    known_encodings, known_names, known_departments = load_encodings()
    if len(known_encodings) != len(known_names) or len(known_names) != len(known_departments):
        print("Data tidak konsisten! Periksa file encodings.pickle Anda.")
    else:
        print(f"Jumlah wajah terdaftar: {len(known_encodings)}")
        for i in range(len(known_names)):
            print(f"{i+1}. Nama: {known_names[i]}, Jurusan: {known_departments[i]}")

def show_registered_faces():
    if not os.path.exists(REGISTERED_FACES_DIR):
        print("Tidak ada wajah yang terdaftar.")
        return

    images = glob.glob(os.path.join(REGISTERED_FACES_DIR, "*.jpg"))
    if not images:
        print("Tidak ada wajah yang terdaftar.")
        return

    plt.figure(figsize=(10, 5))
    for idx, img_path in enumerate(images):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, len(images), idx+1)
        plt.imshow(img)
        plt.title(os.path.basename(img_path).split(".")[0])
        plt.axis('off')
    plt.show()

def register_face(video_capture):
    print("Masuk ke mode pendaftaran wajah.")

    if not os.path.exists(REGISTERED_FACES_DIR):
        os.makedirs(REGISTERED_FACES_DIR)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Tidak dapat mengambil frame dari kamera.")
            break

        # Flip frame horizontally
        frame = cv2.flip(frame, 1)

        # Display instructions
        cv2.putText(frame, "Fokus ke kamera lalu tekan 'c' untuk capture, 'q' untuk batal.", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow('Pendaftaran Wajah', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            # Capture frame for face detection
            captured_frame = frame.copy()

            small_frame = cv2.resize(captured_frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Detect face
            face_locations = face_recognition.face_locations(rgb_small_frame)
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                face_encoding = face_encodings[0]
                top, right, bottom, left = [v * 4 for v in face_locations[0]]
                cv2.rectangle(captured_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.imshow('Pendaftaran Wajah', captured_frame)

                print("Wajah terdeteksi. Tekan 's' untuk simpan, atau 'q' untuk batal.")
                key = cv2.waitKey(0) & 0xFF
                if key == ord('s'):
                    cv2.destroyAllWindows()
                    name = input("Masukkan nama lengkap Anda: ").strip()
                    while not name:
                        print("Nama tidak boleh kosong.")
                        name = input("Masukkan nama lengkap Anda: ").strip()

                    department = input("Masukkan jurusan Anda: ").strip()
                    while not department:
                        print("Jurusan tidak boleh kosong.")
                        department = input("Masukkan jurusan Anda: ").strip()

                    known_encodings, known_names, known_departments = load_encodings()
                    known_encodings.append(face_encoding)
                    known_names.append(name)
                    known_departments.append(department)
                    save_encodings(known_encodings, known_names, known_departments)

                    # Simpan gambar wajah
                    face_image = captured_frame[top:bottom, left:right]
                    face_image_path = os.path.join(REGISTERED_FACES_DIR, f"{name}.jpg")
                    cv2.imwrite(face_image_path, face_image)

                    print(f"Wajah {name} telah didaftarkan.")
                    break
                elif key == ord('q'):
                    print("Pendaftaran dibatalkan.")
                    break
            else:
                print("Wajah tidak terdeteksi. Silakan coba lagi.")
        elif key == ord('q'):
            print("Pendaftaran dibatalkan.")
            break

    cv2.destroyAllWindows()

def main():
    # Initialize camera
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Tidak dapat membuka kamera.")
        sys.exit()

    # Load face encodings
    known_encodings, known_names, known_departments = load_encodings()
    check_dataset()

    process_this_frame = True  # To process every other frame for performance

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Tidak dapat mengambil frame dari kamera.")
            break

        # Flip frame horizontally
        frame = cv2.flip(frame, 1)

        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            face_departments = []

            for face_encoding in face_encodings:
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                name = "Tidak Dikenali"
                department = ""
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    logging.debug(f"Face distances: {face_distances}")
                    logging.debug(f"Best match index: {best_match_index}")
                    if face_distances[best_match_index] < 0.6:  # Adjust threshold as needed
                        name = known_names[best_match_index]
                        department = known_departments[best_match_index]

                face_names.append(name)
                face_departments.append(department)

        process_this_frame = not process_this_frame  # Toggle to process every other frame

        for (top, right, bottom, left), name, department in zip(face_locations, face_names, face_departments):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            if name == "Tidak Dikenali":
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name + " - Tekan 'r' untuk daftar", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 215, 0), 2)
                cv2.putText(frame, name, (left, top - 35),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 215, 0), 2)
                cv2.putText(frame, department, (left, top - 10),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 215, 0), 1)

        cv2.imshow('Pengenalan Wajah', frame)

        key = cv2.waitKey(1) & 0xFF
        # Press 'q' to exit
        if key == ord('q'):
            break
        # Press 'r' to register
        elif key == ord('r'):
            cv2.destroyAllWindows()
            register_face(video_capture)
            known_encodings, known_names, known_departments = load_encodings()
            check_dataset()
        # Press 'd' to display registered faces
        elif key == ord('d'):
            show_registered_faces()

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
