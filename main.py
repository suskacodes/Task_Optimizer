import cv2
from deepface import DeepFace
import pandas as pd
import datetime
import os
import hashlib
from sklearn.tree import DecisionTreeClassifier
import random

MOOD_HISTORY_FILE = 'mood_history.csv'

MOOD_MAP = {'stressed': 0, 'sad': 0, 'fear': 0, 'angry': 0,'happy': 1, 'surprise': 1, 'tired': 2, 'disgust': 2, 'neutral': 3}

TASK_MAP = {0: 'Take a Break / Counseling', 1: 'Deep Work ', 2: 'Listen to Music', 3: 'Brainstorming Session'}
X_train = [[0, 8], [1, 2], [2, 7], [3, 5], [1, 9], [0, 3], [2, 3], [3, 8]]
y_train = [0, 3, 0, 2, 1, 2, 2, 1] 
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

def anonymize_id(name):
    return hashlib.sha256(name.strip().lower().encode()).hexdigest()[:12]

def emotion():
    cap = cv2.VideoCapture(0)
    dominant_emotion = "neutral"
    
    for _ in range(30): 
        ret, frame = cap.read()
        if not ret:
            break
        try:
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant_emotion = results[0]['dominant_emotion']
            cv2.putText(frame, f"Mood: {dominant_emotion}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Analyzing Mood', frame)
        except:
            pass
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    return dominant_emotion.lower()

def recommend_task(mood, workload):
    mood_idx = MOOD_MAP.get(mood, 3) 
    prediction = clf.predict([[mood_idx, workload]])
    return TASK_MAP[prediction[0]]

def get_mood_quote(mood):
    quote_library = {
        0: [ 
            "Sometimes the most productive thing you can do is relax. – Mark Black",
            "Breathe. It’s just a bad day, not a bad life. – Unknown"
        ],
        1: [ 
            "The only way to do great work is to love what you do. – Steve Jobs",
            "Enthusiasm moves the world. – Arthur Balfour"
        ],
        2: [ 
            "Take rest; a field that has rested gives a bountiful crop. – Ovid",
            "Self-care is how you take your power back. – Lalah Delia"
        ],
        3: [ 
            "The secret of getting ahead is getting started. – Mark Twain",
            "Action is the foundational key to all success. – Pablo Picasso"
        ]
    }
    
    mood_idx = MOOD_MAP.get(mood.lower(), 3)
    return random.choice(quote_library[mood_idx])


def userinput(name, workload):
    user_hash = anonymize_id(name)
    current_mood = emotion()
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 1. Empathetic Boost
    print(f"\nMood detected: {current_mood.upper()}")
    print(f"Inspirational quote: {get_mood_quote(current_mood)}")
    
    recommendation = recommend_task(current_mood, workload)
    print(f"Recommended Task: {recommendation}")

    if os.path.exists(MOOD_HISTORY_FILE):
        df = pd.read_csv(MOOD_HISTORY_FILE)
        user_logs = df[df['User_Hash'] == user_hash].tail(2)
        
        if len(user_logs) >= 2:
            prev_moods = user_logs['Mood'].tolist()
            if MOOD_MAP.get(current_mood, 3) in [0, 2]:
                if all(MOOD_MAP.get(m, 3) in [0, 2] for m in prev_moods):
                    print(f"\n[ALERT] Prolonged stress or burnout detected for {name}!")
                    print("ACTION: Notifying HR for counseling or task adjustments")

    new_entry = pd.DataFrame([[timestamp, user_hash, current_mood]], columns=['Timestamp', 'User_Hash', 'Mood'])
    new_entry.to_csv(MOOD_HISTORY_FILE, mode='a', index=False, header=not os.path.exists(MOOD_HISTORY_FILE))
    print(f"Data stored in file")

if __name__ == "__main__":
    user_name = input("Enter Employee Name: ")
    try:
        work_level = int(input("Enter current workload level (1-10): "))
        userinput(user_name, work_level)
    except ValueError:
        print("Please enter a valid number.")
