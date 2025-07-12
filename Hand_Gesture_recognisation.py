import streamlit as st
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
from collections import defaultdict
import heapq
import os
import hashlib
from PIL import Image

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Gesture labels (A-Z + Space + Erase + Completed)
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
    17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
    25: 'Z', 26: 'Space', 27: 'Erase', 28: 'Completed'
}

# User authentication functions
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

def create_user_data():
    if not os.path.exists('user_data.pkl'):
        with open('user_data.pkl', 'wb') as f:
            pickle.dump({}, f)

def register_user(username, password):
    with open('user_data.pkl', 'rb') as f:
        user_data = pickle.load(f)
    
    if username in user_data:
        return False
    
    user_data[username] = make_hashes(password)
    with open('user_data.pkl', 'wb') as f:
        pickle.dump(user_data, f)
    return True

def login_user(username, password):
    with open('user_data.pkl', 'rb') as f:
        user_data = pickle.load(f)
    
    if username in user_data:
        return check_hashes(password, user_data[username])
    return False

# Initialize the app
def main():
    st.title("Gesture Typing System with Authentication")
    
    # Create user data file if it doesn't exist
    create_user_data()
    
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'confirmed_text' not in st.session_state:
        st.session_state.confirmed_text = ""
    if 'current_word' not in st.session_state:
        st.session_state.current_word = ""
    if 'suggestions' not in st.session_state:
        st.session_state.suggestions = []
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None
    if 'prediction_time' not in st.session_state:
        st.session_state.prediction_time = None
    if 'sentence_complete' not in st.session_state:
        st.session_state.sentence_complete = False
    
    # Authentication
    if not st.session_state.authenticated:
        menu = ["Login", "Register"]
        choice = st.sidebar.selectbox("Menu", menu)
        
        if choice == "Login":
            st.subheader("Login Section")
            username = st.sidebar.text_input("User Name")
            password = st.sidebar.text_input("Password", type='password')
            
            if st.sidebar.button("Login"):
                if login_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success(f"Logged In as {username}")
                else:
                    st.error("Incorrect Username/Password")
        
        elif choice == "Register":
            st.subheader("Create New Account")
            new_user = st.text_input("Username")
            new_password = st.text_input("Password", type='password')
            
            if st.button("Signup"):
                if register_user(new_user, new_password):
                    st.success("You have successfully created an account")
                    st.info("Go to Login Menu to login")
                else:
                    st.error("Username already exists")
    
    # Main application
    else:
        st.sidebar.success(f"Logged in as: {st.session_state.username}")
        if st.sidebar.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.confirmed_text = ""
            st.session_state.current_word = ""
            st.session_state.suggestions = []
            st.experimental_rerun()
        
        # Load the gesture recognition model
        model = pickle.load(open('./model.p', 'rb'))
        
        # Simple word frequency dictionary
        word_freq = defaultdict(int)
        load_basic_dictionary(word_freq)
        
        # Main app layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("Camera Feed")
            FRAME_WINDOW = st.image([])
            camera = cv2.VideoCapture(0)
        
        with col2:
            st.header("Recognized Text")
            text_output = st.empty()
            text_output.text_area("Text", value=st.session_state.confirmed_text, height=200, key="output_text")
            
            st.header("Word Suggestions")
            suggestion_buttons = st.columns(3)
            for i in range(3):
                with suggestion_buttons[i]:
                    if i < len(st.session_state.suggestions):
                        if st.button(st.session_state.suggestions[i], key=f"sugg_{i}"):
                            use_suggestion(i, model, word_freq)
                    else:
                        st.button("", disabled=True, key=f"empty_{i}")
            
            st.header("Controls")
            if st.button("Clear Text"):
                clear_text()
            if st.button("Speak"):
                speak_text(st.session_state.confirmed_text)
        
        # Process frames
        process_frames(camera, model, word_freq, FRAME_WINDOW)
        
        # Release camera when done
        if not st.session_state.authenticated:
            camera.release()

def load_basic_dictionary(word_freq):
    """Load a basic word dictionary for suggestions"""
    common_words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
        "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
        "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
        "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
        "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
        "hello", "world", "computer", "science", "python", "programming", "gesture", "typing"
    ]

    # Assign frequencies (higher for more common words)
    for i, word in enumerate(common_words):
        word_freq[word] = len(common_words) - i

def process_frames(camera, model, word_freq, frame_window):
    """Process video frames and update the app"""
    while st.session_state.authenticated:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to capture video")
            break
        
        # Process frame for hand landmarks
        processed_frame, current_prediction = process_frame(frame, model)
        
        # Display the processed frame
        frame_window.image(processed_frame, channels="BGR")
        
        # Handle prediction confirmation
        handle_prediction(current_prediction, word_freq)
        
        # Small delay to prevent high CPU usage
        time.sleep(0.1)

def process_frame(frame, model):
    """Process a single frame and return it with predictions"""
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    current_prediction = None
    data_aux = []
    x_ = []
    y_ = []

    if results.multi_hand_landmarks:
        # Find the hand with the largest bounding box (dominant hand)
        max_area = 0
        dominant_hand = None
        dominant_x_ = []
        dominant_y_ = []

        for hand_landmarks in results.multi_hand_landmarks:
            x_ = [landmark.x for landmark in hand_landmarks.landmark]
            y_ = [landmark.y for landmark in hand_landmarks.landmark]

            x1, y1 = int(min(x_) * W), int(min(y_) * H)
            x2, y2 = int(max(x_) * W), int(max(y_) * H)

            # Calculate bounding box area
            area = (x2 - x1) * (y2 - y1)

            if area > max_area:
                max_area = area
                dominant_hand = hand_landmarks
                dominant_x_ = x_
                dominant_y_ = y_

        if dominant_hand is not None:
            # Draw landmarks for the dominant hand
            mp_drawing.draw_landmarks(
                frame,
                dominant_hand,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Prepare data for prediction (normalized coordinates)
            data_aux = []
            min_x, min_y = min(dominant_x_), min(dominant_y_)

            for x, y in zip(dominant_x_, dominant_y_):
                data_aux.append(x - min_x)
                data_aux.append(y - min_y)

            # Get bounding box coordinates
            x1 = int(min(dominant_x_) * W) - 10
            y1 = int(min(dominant_y_) * H) - 10
            x2 = int(max(dominant_x_) * W) - 10
            y2 = int(max(dominant_y_) * H) - 10

            # Pad data_aux to exactly 100 features
            if len(data_aux) < 100:
                data_aux.extend([0] * (100 - len(data_aux)))
            elif len(data_aux) > 100:
                data_aux = data_aux[:100]

            # Make prediction with padded features
            prediction = model.predict([np.asarray(data_aux)])
            current_prediction = labels_dict[int(prediction[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, current_prediction, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            # Display confirmation timer
            if current_prediction == st.session_state.last_prediction and st.session_state.prediction_time:
                elapsed = time.time() - st.session_state.prediction_time
                remaining = max(0, 2 - elapsed)  # 2 seconds confirmation delay
                cv2.putText(frame, f"Confirming: {remaining:.1f}s",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame, current_prediction

def handle_prediction(current_prediction, word_freq):
    """Handle gesture predictions and update text"""
    if current_prediction is None:
        return

    # If prediction changed, reset the timer
    if current_prediction != st.session_state.last_prediction:
        st.session_state.prediction_time = time.time()
        st.session_state.last_prediction = current_prediction

    # Check if confirmation time has elapsed
    if (st.session_state.prediction_time and
            (time.time() - st.session_state.prediction_time) >= 2):  # 2 seconds confirmation delay

        if current_prediction == "Space":
            st.session_state.confirmed_text += " "
            st.session_state.current_word = ""  # Reset current word
            update_suggestions("", word_freq)  # Clear suggestions after space
        elif current_prediction == "Erase":
            st.session_state.confirmed_text = st.session_state.confirmed_text[:-1]  # Remove last character
            # Update current word being typed
            if st.session_state.confirmed_text and st.session_state.confirmed_text[-1] != " ":
                st.session_state.current_word = st.session_state.current_word[:-1]
            else:
                st.session_state.current_word = ""
            update_suggestions(st.session_state.current_word, word_freq)
        elif current_prediction == "Completed":
            speak_text(st.session_state.confirmed_text)
            st.session_state.sentence_complete = True
        else:
            st.session_state.confirmed_text += current_prediction.lower()  # Add lowercase letter
            st.session_state.current_word += current_prediction.lower()
            update_suggestions(st.session_state.current_word, word_freq)
            st.session_state.sentence_complete = False

        # Reset for next prediction
        st.session_state.prediction_time = None
        st.session_state.last_prediction = None

        # Update the display
        st.experimental_rerun()

def update_suggestions(current_input, word_freq):
    """Update the word suggestions based on current input"""
    if not current_input:
        st.session_state.suggestions = []
        return

    # Find words that start with the current input
    matches = [(word, freq) for word, freq in word_freq.items()
               if word.startswith(current_input)]

    # Get top 3 suggestions by frequency
    top_matches = heapq.nlargest(3, matches, key=lambda x: x[1])
    st.session_state.suggestions = [word for word, freq in top_matches]

def use_suggestion(index, model, word_freq):
    """Use the selected suggestion"""
    if index < len(st.session_state.suggestions):
        # Replace current word with suggestion
        if st.session_state.current_word:
            # Remove the partially typed word
            st.session_state.confirmed_text = st.session_state.confirmed_text[:-len(st.session_state.current_word)]

        # Add the suggested word and a space
        suggested_word = st.session_state.suggestions[index]
        st.session_state.confirmed_text += suggested_word + " "

        # Reset current word and suggestions
        st.session_state.current_word = ""
        update_suggestions("", word_freq)

        # Update the display
        st.experimental_rerun()

def speak_text(text):
    """Convert the current text to speech"""
    if text.strip():  # Only speak if there's text to speak
        engine = pyttsx3.init()
        engine.setProperty('rate', 90)
        engine.say(text)
        engine.runAndWait()

def clear_text():
    """Clear the current text"""
    st.session_state.confirmed_text = ""
    st.session_state.current_word = ""
    st.session_state.sentence_complete = False
    update_suggestions("", {})  # Clear suggestions
    st.experimental_rerun()

if __name__ == "__main__":
    main()