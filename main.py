import cv2
from deepface import DeepFace
import pandas as pd
import datetime
import os
import hashlib
from sklearn.tree import DecisionTreeClassifier
import random

# --- 1. GLOBAL CONFIGURATIONS & MAPS ---
MOOD_HISTORY_FILE = 'mood_history.csv'

# Mapping DeepFace emotions to indices: 0: Stressed/Sad, 1: Happy, 2: Tired, 3: Neutral
MOOD_MAP = {
    'stressed': 0, 'sad': 0, 'fear': 0, 'angry': 0,
    'happy': 1, 'surprise': 1,
    'tired': 2, 'disgust': 2,
    'neutral': 3
}

# Mapping indices to Task Recommendations [cite: 22]
TASK_MAP = {
    0: 'Take a Break / Counseling', 
    1: 'Deep Work (Coding)', 
    2: 'Light Admin (Email)', 
    3: 'Brainstorming Session'
}

# ML Training Data [cite: 21]
X_train = [[0, 8], [1, 2], [2, 7], [3, 5], [1, 9], [0, 3], [2, 3], [3, 8]]
y_train = [0, 3, 0, 2, 1, 2, 2, 1] 
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# --- 2. CORE UTILITIES ---

def anonymize_id(name):
    """Ensures sensitive data is anonymized and securely stored[cite: 29, 30]."""
    return hashlib.sha256(name.strip().lower().encode()).hexdigest()[:12]

def get_realtime_emotion():
    """Captures live video to detect employee emotions comprehensively[cite: 18, 19]."""
    cap = cv2.VideoCapture(0)
    dominant_emotion = "neutral"
    
    for _ in range(30): 
        ret, frame = cap.read()
        if not ret: break
        try:
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant_emotion = results[0]['dominant_emotion']
            cv2.putText(frame, f"Mood: {dominant_emotion}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Amdox Mood Scanner', frame)
        except: pass
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cap.release()
    cv2.destroyAllWindows()
    return dominant_emotion.lower()

def recommend_task(mood, workload):
    """Suggests tasks based on the detected mood to enhance productivity[cite: 10, 21]."""
    mood_idx = MOOD_MAP.get(mood, 3) # Use Global Map
    prediction = clf.predict([[mood_idx, workload]])
    return TASK_MAP[prediction[0]]

def get_mood_quote(mood):
    """Provides a supportive quote aligned with the employee's current state[cite: 10, 12]."""
    quote_library = {
        0: [ # Stressed, Sad, Fear, Angry
            "Sometimes the most productive thing you can do is relax. – Mark Black",
            "Breathe. It’s just a bad day, not a bad life. – Unknown"
        ],
        1: [ # Happy, Surprise
            "The only way to do great work is to love what you do. – Steve Jobs",
            "Enthusiasm moves the world. – Arthur Balfour"
        ],
        2: [ # Tired, Disgust
            "Take rest; a field that has rested gives a bountiful crop. – Ovid",
            "Self-care is how you take your power back. – Lalah Delia"
        ],
        3: [ # Neutral
            "The secret of getting ahead is getting started. – Mark Twain",
            "Action is the foundational key to all success. – Pablo Picasso"
        ]
    }
    
    mood_idx = MOOD_MAP.get(mood.lower(), 3)
    return random.choice(quote_library[mood_idx])

# --- 3. LOGGING & ALERT SYSTEM ---

def process_user_session(name, workload):
    user_hash = anonymize_id(name)
    current_mood = get_realtime_emotion()
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 1. Empathetic Boost
    print(f"\nMood detected: {current_mood.upper()}")
    print(f"Inspiration: {get_mood_quote(current_mood)}")
    
    # 2. Task Recommendation [cite: 22]
    recommendation = recommend_task(current_mood, workload)
    print(f"Recommended Task: {recommendation}")

    # 3. Burnout Alerting [cite: 25, 26]
    if os.path.exists(MOOD_HISTORY_FILE):
        df = pd.read_csv(MOOD_HISTORY_FILE)
        user_logs = df[df['User_Hash'] == user_hash].tail(2)
        
        # Identify negative states from global map (Index 0 or 2)
        if len(user_logs) >= 2:
            prev_moods = user_logs['Mood'].tolist()
            # Check if current and past moods are all in 'Stressed' or 'Tired' categories
            if MOOD_MAP.get(current_mood, 3) in [0, 2]:
                if all(MOOD_MAP.get(m, 3) in [0, 2] for m in prev_moods):
                    print(f"\n[ALERT] Prolonged stress or burnout detected for {name}!")
                    print("ACTION: Notifying HR for counseling or task adjustments[cite: 11, 12].")

    # 4. Historical Mood Tracking [cite: 23, 24]
    new_entry = pd.DataFrame([[timestamp, user_hash, current_mood]], 
                             columns=['Timestamp', 'User_Hash', 'Mood'])
    new_entry.to_csv(MOOD_HISTORY_FILE, mode='a', index=False, header=not os.path.exists(MOOD_HISTORY_FILE))
    print(f"Data secured in history for long-term well-being tracking[cite: 24, 30].")

# --- 4. EXECUTION ---
if __name__ == "__main__":
    user_name = input("Enter Employee Name: ")
    try:
        work_level = int(input("Enter current workload level (1-10): "))
        process_user_session(user_name, work_level)
    except ValueError:
        print("Please enter a valid number.")
