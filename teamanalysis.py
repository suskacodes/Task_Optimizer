import pandas as pd
import matplotlib.pyplot as plt
import os

def team_mood():
    file_path = 'mood_history.csv'
    
    if not os.path.exists(file_path):
        print("No data found.")
        return

    df = pd.read_csv(file_path)
    mood_counts = df['Mood'].value_counts()
    
    plt.figure(figsize=(10, 6))
    colors = ['#4CAF50', '#FFC107', '#F44336', '#2196F3', '#9C27B0', '#FF9800', '#9E9E9E']
    mood_counts.plot(kind='bar', color=colors[:len(mood_counts)])
    
    plt.title('Team Mood Analytics', fontsize=15)
    plt.xlabel('Mood Category', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    print("\n--- Team Mood Report ---")
    total_entries = len(df)
    negative_states = (mood_counts.get('stressed', 0) + mood_counts.get('sad', 0) + mood_counts.get('fear', 0) + mood_counts.get('angry', 0) + mood_counts.get('tired', 0) + mood_counts.get('disgust', 0))
    
    print("Total neg entries: ", negative_states)
    neg_morale = (negative_states / total_entries) * 100
    print(f"Total Observations: {total_entries}")
    print(f"Negative Morale: {neg_morale}%")
    
    if neg_morale > 30:
        print("Recommendation: Checkup on employees and review workloads.")
    else:
        print("Recommendation: Morale is stable.")

if __name__ == "__main__":
    team_mood()
