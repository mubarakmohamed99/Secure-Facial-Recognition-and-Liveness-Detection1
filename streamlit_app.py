import os
import pickle

import cv2
import numpy as np
import streamlit as st
from insightface.app import FaceAnalysis


SIMILARITY_THRESHOLD = 0.45
EMBEDDINGS_DIR = "face_embeddings"


def load_recognizer():
    """Load FaceAnalysis and known embeddings once and cache in session state."""
    if "face_app" not in st.session_state:
        app = FaceAnalysis(
            name="buffalo_s",
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        st.session_state["face_app"] = app

    if "known_faces" not in st.session_state:
        known_faces: dict[str, np.ndarray] = {}
        if os.path.exists(EMBEDDINGS_DIR):
            for f in os.listdir(EMBEDDINGS_DIR):
                if f.endswith(".pkl"):
                    name = os.path.splitext(f)[0].replace("_", " ")
                    with open(os.path.join(EMBEDDINGS_DIR, f), "rb") as fh:
                        known_faces[name] = pickle.load(fh)
        st.session_state["known_faces"] = known_faces

    return st.session_state["face_app"], st.session_state["known_faces"]


def opencv_image_from_upload(uploaded_file) -> np.ndarray | None:
    """Convert a Streamlit UploadedFile (image) into an OpenCV BGR image."""
    if uploaded_file is None:
        return None

    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def verify_image(frame_bgr: np.ndarray):
    """Run recognition on a single BGR frame and display results in the UI."""
    app, known_faces = load_recognizer()

    # Resize similar to main app to speed up processing
    scale_factor = 0.5
    small_frame = cv2.resize(frame_bgr, (0, 0), fx=scale_factor, fy=scale_factor)

    faces = app.get(small_frame)
    if not faces:
        st.error("No face detected in the image.")
        return

    # Use the largest face
    face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))

    best_name = None
    best_score = 0.0

    for name, ref_emb in known_faces.items():
        sim = float(np.dot(face.embedding, ref_emb) / (np.linalg.norm(face.embedding) * np.linalg.norm(ref_emb)))
        if sim > best_score:
            best_score = sim
            best_name = name

    st.subheader("Result")
    if best_name is not None and best_score >= SIMILARITY_THRESHOLD:
        st.success(f"Identity verified: {best_name} (confidence: {best_score * 100:.1f}%)")
    else:
        st.error(f"No known identity matched. Best similarity: {best_score * 100:.1f}%")


def main():
    st.set_page_config(page_title="Secure Facial Recognition", page_icon="🧑‍💻")
    st.title("Secure Facial Recognition & Liveness Detection")
    st.write(
        "This Streamlit app uses the existing InsightFace embeddings from the project "
        "to perform identity verification from a single image. "
        "For full active liveness (head turns, mouth open, etc.), use the OpenCV UI via main.py."
    )
    _, known_faces = load_recognizer()
    if not known_faces:
        st.warning(
            "No embeddings found in 'face_embeddings/'. "
            "Run extract_embeddings.py after populating faces_db/ to register users."
        )

    tab_camera, tab_upload = st.tabs(["Camera", "Upload Image"])

    with tab_camera:
        st.subheader("Verify with Camera Snapshot")
        camera_image = st.camera_input("Take a picture for verification")

        if camera_image is not None:
            # Convert to OpenCV BGR image
            frame_bgr = opencv_image_from_upload(camera_image)
            if frame_bgr is not None:
                st.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), caption="Captured image", use_column_width=True)
                if st.button("Verify from Camera Image"):
                    verify_image(frame_bgr)

    with tab_upload:
        st.subheader("Verify from Uploaded Image")
        uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            frame_bgr = opencv_image_from_upload(uploaded_file)
            if frame_bgr is not None:
                st.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), caption="Uploaded image", use_column_width=True)
                if st.button("Verify from Uploaded Image"):
                    verify_image(frame_bgr)


if __name__ == "__main__":
    main()
