# app.py
import os
import json
import hashlib
import secrets
import time
from datetime import datetime

import streamlit as st
import tensorflow as tf
import numpy as np
from keras.models import load_model

# -------------------- CONFIG --------------------
st.set_page_config(
page_title="AI ECG Analyzer | HeartCare Diagnostics",
page_icon="🫀",
layout="centered"
)

USERS_FILE = "users.json"
BASE_UPLOAD_DIR = "upload"  # base folder; per-user subfolders created
MODEL_PATH = "ECG_Recog_Model.keras"

# -------------------- HELPER: USERS DB --------------------
def ensure_files():
    os.makedirs(BASE_UPLOAD_DIR, exist_ok=True)
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump({"users": {}}, f)

def load_users():
    ensure_files()
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(data):
    with open(USERS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def hash_password(password: str, salt: str) -> str:
    # SHA256(salt + password). For demo only.
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()

def create_user(username: str, password: str, full_name: str = ""):
    data = load_users()
    if username in data["users"]:
        return False, "Username already exists."
    salt = secrets.token_hex(8)
    pwd_hash = hash_password(password, salt)
    data["users"][username] = {
        "full_name": full_name,
        "salt": salt,
        "pwd_hash": pwd_hash,
        "created_at": datetime.now().isoformat(),
        "history": []  # each entry: {filename, timestamp, label, confidence}
    }
    save_users(data)
    # create user upload dir
    os.makedirs(os.path.join(BASE_UPLOAD_DIR, username), exist_ok=True)
    return True, "User created."

def verify_user(username: str, password: str):
    data = load_users()
    user = data["users"].get(username)
    if not user:
        return False, "Username not found."
    salt = user["salt"]
    expected = user["pwd_hash"]
    if hash_password(password, salt) == expected:
        return True, "Login successful."
    else:
        return False, "Incorrect password."

def add_history_for_user(username: str, item: dict):
    data = load_users()
    if username in data["users"]:
        data["users"][username]["history"].insert(0, item)  # newest first
        # keep only last 50 entries
        data["users"][username]["history"] = data["users"][username]["history"][:50]
        save_users(data)

# -------------------- CUSTOM CSS (kept from your file with small additions) --------------------
st.markdown("""
    <style>
            /* --- Light Green Login Block Styles --- */

.login-container {
    background: linear-gradient(#e9ffe9, #ccffcc);
    padding: 30px;
    border-radius: 18px;
    width: 420px;
    margin: auto;
    margin-top: 40px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.20);
    animation: fadeSlide 0.7s ease-in-out;
}

@keyframes fadeSlide {
    from {opacity: 0; transform: translateY(-20px);}
    to {opacity: 1; transform: translateY(0);}
}

.login-title {
    text-align: center;
    font-size: 1.8em;
    font-weight: bold;
    color: #004d00;
    margin-bottom: 15px;
}

.login-label {
    font-size: 1.1em;
    color: #003300;
    font-weight: 600;
}

.login-button {
    background-color: #009900 !important;
    color: white !important;
    padding: 10px 0;
    font-size: 1.1em;
    border-radius: 10px;
    width: 100%;
}

    </style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown('<div class="main-header">🫀 HeartCare AI ECG Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced deep-learning ECG interpretation — inspired by real hospital diagnostics</div>', unsafe_allow_html=True)

# -------------------- SESSION STATE --------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None

# -------------------- MODEL LOADING (lazy) --------------------
@st.cache_resource(show_spinner=False)
def load_ecg_model(path):
    try:
        return load_model(path)
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None

model = load_ecg_model(MODEL_PATH)

ECG_NAMES = [
    'Myocardial Infarction ECG',
    'History of MI ECG',
    'Abnormal Heartbeat ECG',
    'Normal ECG'
]

# -------------------- PREDICTION FUNCTION --------------------
def classify_images(image_path):
    if model is None:
        return "ModelNotLoaded", 0.0
    input_image = tf.keras.utils.load_img(image_path, target_size=(180,180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)
    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    label = ECG_NAMES[np.argmax(result)]
    confidence = float(np.max(result) * 100)
    return label, confidence

# -------------------- AUTH UI COMPONENTS --------------------
def show_signup():
    st.subheader("Create account")
    with st.form("signup_form", clear_on_submit=True):
        new_username = st.text_input("Username (no spaces)")
        full_name = st.text_input("Full name (optional)")
        new_password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Sign up")
        if submitted:
            if not new_username or not new_password:
                st.warning("Please enter a username and password.")
            elif " " in new_username:
                st.warning("Username cannot contain spaces.")
            elif new_password != confirm_password:
                st.warning("Passwords do not match.")
            else:
                ok, msg = create_user(new_username.strip(), new_password, full_name.strip())
                if ok:
                    st.success("Account created. Please log in below.")
                else:
                    st.error(msg)

def show_login():
    st.subheader("Sign in")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        remember = st.checkbox("Remember me (persists until logout)", value=True)
        submitted = st.form_submit_button("Log in")
        if submitted:
            ok, msg = verify_user(username.strip(), password)
            if ok:
                st.session_state.authenticated = True
                st.session_state.username = username.strip()
                st.session_state.remember = bool(remember)
                st.success(f"Logged in as {username}")
                time.sleep(0.6)
                st.rerun()

            else:
                st.error(msg)

# -------------------- APP: AUTH GATE --------------------
def auth_gate():
    # Two-column layout: login/signup side-by-side
    c1, c2 = st.columns([1, 1])
    with c1:
        show_login()
    with c2:
        show_signup()
    st.info("Demo auth: credentials stored locally in users.json. Use real auth in production.")

# -------------------- UPLOAD + PREDICTION UI --------------------
def user_upload_ui(username):
    st.markdown(f"### Welcome, **{username}**")
    st.markdown("Use the uploader below to analyze an ECG image.")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # build per-user upload dir
        user_dir = os.path.join(BASE_UPLOAD_DIR, username)
        os.makedirs(user_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = os.path.splitext(uploaded_file.name)[1] or ".png"
        safe_name = f"ECG_{timestamp}{ext}"
        file_path = os.path.join(user_dir, safe_name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.markdown(f'<p class="upload-label">📎 Uploaded ECG Image: <b>{safe_name}</b></p>', unsafe_allow_html=True)
        st.image(uploaded_file, width=300, caption="Uploaded ECG Image", use_container_width=False, output_format="auto")

        with st.spinner("🧠 Analyzing ECG Image..."):
            time.sleep(1.0)  # UX pause
            label, confidence = classify_images(file_path)

        # Save result to user's history
        history_item = {
            "filename": safe_name,
            "filepath": file_path,
            "timestamp": datetime.now().isoformat(),
            "label": label,
            "confidence": round(confidence, 2)
        }
        add_history_for_user(username, history_item)

        # Show popup result
        st.markdown(f"""
            <div class="popup">
                <h4>🩺 Diagnosis Result</h4>
                <p>Detected Type: {label}</p>
                <p>Confidence: {confidence:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)

# -------------------- USER PROFILE / HISTORY --------------------
def user_profile(username):
    st.markdown("#### Profile & Upload History")
    data = load_users()
    user = data["users"].get(username, {})
    st.write(f"Full name: **{user.get('full_name','-')}**")
    st.write(f"Account created: **{user.get('created_at','-')}**")
    history = user.get("history", [])
    if not history:
        st.info("No uploads yet. Try uploading an ECG image above.")
        return
    # show last 10 entries
    for item in history[:10]:
        ts = item.get("timestamp", "")
        fn = item.get("filename", "")
        lbl = item.get("label", "")
        conf = item.get("confidence", 0)
        col1, col2 = st.columns([3,1])
        with col1:
            st.markdown(f"**{fn}** — {lbl} ({conf}%)")
            st.caption(ts)
            # small preview if file exists
            if os.path.exists(item.get("filepath","")):
                st.image(item.get("filepath"), width=200)
        with col2:
            if st.button("Re-analyze", key=f"rean_{fn}"):
                label, confidence = classify_images(item.get("filepath"))
                st.success(f"Re-analysis: {label} ({confidence:.2f}%)")
                # update history record
                new_item = {
                    "filename": fn,
                    "filepath": item.get("filepath"),
                    "timestamp": datetime.now().isoformat(),
                    "label": label,
                    "confidence": round(float(confidence),2)
                }
                add_history_for_user(username, new_item)
                st.rerun()


# -------------------- NAVBAR / MAIN --------------------
def main_app():
    username = st.session_state.username
    # simple sidebar nav
    st.sidebar.markdown(f"**Signed in as:** {username}")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.experimental_rerun()

    menu = st.sidebar.radio("Menu", ["Analyze ECG", "Profile & History", "About"])
    if menu == "Analyze ECG":
        user_upload_ui(username)
    elif menu == "Profile & History":
        user_profile(username)
    else:
        st.markdown("### About")
        st.write("HeartCare Diagnostics — demo application for ECG classification.")
        st.write("Built with TensorFlow, Keras & Streamlit.")
        st.write("⚕️ For educational and research use only.")

# -------------------- ENTRY POINT --------------------
def run():
    ensure_files()
    if not st.session_state.authenticated:
        auth_gate()
    else:
        main_app()

    # footer
    st.markdown("""
    <div class="footer">
        © 2025 HeartCare Diagnostics Pvt. Ltd.<br>
        Built with ❤️ using TensorFlow, Keras & Streamlit.<br>
        ⚕️ For educational and research use only.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    run()