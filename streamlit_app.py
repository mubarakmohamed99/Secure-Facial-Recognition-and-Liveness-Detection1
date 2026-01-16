import cv2
import numpy as np
import streamlit as st
from main import IdentitySystem


def load_system() -> IdentitySystem:
    """Singleton-style loader so the FaceAnalysis model is created only once."""
    if "identity_system" not in st.session_state:
        st.session_state["identity_system"] = IdentitySystem()
    return st.session_state["identity_system"]


def opencv_image_from_upload(uploaded_file) -> np.ndarray | None:
    """Convert a Streamlit UploadedFile (image) into an OpenCV BGR image."""
    if uploaded_file is None:
        return None

    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def verify_image(frame_bgr: np.ndarray):
    """Run recognition on a single BGR frame and display results in the UI."""
    system = load_system()

    # Run recognition using the existing logic from IdentitySystem
    name, score = system.recognize(frame_bgr)

    st.subheader("Result")
    if name is not None:
        st.success(f"Identity verified: {name} (confidence: {score * 100:.1f}%)")
    else:
        st.error(f"No known identity matched. Best similarity: {score * 100:.1f}%")


def main():
    st.set_page_config(page_title="Secure Facial Recognition", page_icon="🧑‍💻")
    st.title("Secure Facial Recognition & Liveness Detection")
    st.write(
        "This Streamlit app uses the existing InsightFace embeddings from the project "
        "to perform identity verification from a single image. "
        "For full active liveness (head turns, mouth open, etc.), use the OpenCV UI via main.py."
    )

    system = load_system()
    if not system.known_faces:
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
