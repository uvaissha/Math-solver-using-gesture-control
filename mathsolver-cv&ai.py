import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st

st.set_page_config(layout="wide")

# Streamlit UI setup
col1, col2 = st.columns([3,2])
with col1:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])

with col2:
    st.title("Answer")
    output_text_area = st.subheader("")

# Initialize Gemini AI model
genai.configure(api_key="AIzaSyA6NUBiv09wDFy1a8QyHVQh-kTMP6PIf7w")
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

if not cap.isOpened():
    st.error("Error: Could not access webcam. Please check your camera connection.")
    st.stop()

# Initialize HandDetector
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

def getHandInfo(img):
    """Detect hands and return finger states & landmarks."""
    if img is None:
        return None

    hands, img = detector.findHands(img, draw=False, flipType=True)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None

def draw(info, prev_pos, canvas):
    """Draw lines on canvas based on hand gestures."""
    fingers, lmList = info
    current_pos = None

    if fingers == [0, 1, 0, 0, 0]:  # Index finger up -> Draw
        current_pos = lmList[8][:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, tuple(current_pos), tuple(prev_pos), (255, 0, 255), 10)
    elif fingers == [1, 0, 0, 0, 0]:  # Thumb up -> Clear
        canvas = np.zeros_like(canvas)

    return current_pos, canvas

def sendToAI(model, canvas, fingers):
    """Send the drawn equation to AI for solving."""
    if fingers == [1, 1, 1, 1, 0]:  # 4 fingers up -> Solve
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this math problem", pil_image])
        return response.text
    return ""

prev_pos = None
canvas = None
image_combined = None
output_text = ""

# Main loop
while run:
    success, img = cap.read()
    if not success or img is None:
        st.error("Error: Unable to read from camera.")
        break

    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandInfo(img)
    if info:
        prev_pos, canvas = draw(info, prev_pos, canvas)
        output_text = sendToAI(model, canvas, info[0])

    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(image_combined, channels="BGR")

    if output_text:
        output_text_area.text(output_text)

    cv2.waitKey(1)
