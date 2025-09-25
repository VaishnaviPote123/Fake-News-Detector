import tkinter as tk
from tkinter import messagebox, ttk
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

# Check if model exists, else train it

MODEL_PATH = "fake_news_model.pkl"

def train_and_save_model():
    # For demo purpose, we'll create a small dummy dataset inline
    data = {

        'text': [

            'The economy is doing well and jobs are increasing',

            'Aliens have landed on Earth and taken over the government',

            'New vaccine proves to be 99% effective',

            'Scientists found cure for cancer last year',

            'Politician involved in scandal denies all allegations',

            'Fake news about celebrity death spreads on social media',

            'NASA confirms discovery of water on Mars',

            'Miracle diet pill causes instant weight loss',

            'Government announces new education reforms',

            'Conspiracy theories about moon landing exposed'
        ],

        'label': ['REAL', 'FAKE', 'REAL', 'REAL', 'REAL', 'FAKE', 'REAL', 'FAKE', 'REAL', 'FAKE']
    }
    df = pd.DataFrame(data)
    x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

    x_train_vec = vectorizer.fit_transform(x_train)

    model = PassiveAggressiveClassifier(max_iter=50)

    model.fit(x_train_vec, y_train)

    # Save both vectorizer and model as tuple

    joblib.dump((vectorizer, model), MODEL_PATH)

    print("Model trained and saved.")


if not os.path.exists(MODEL_PATH):
    train_and_save_model()

# Load model and vectorizer

vectorizer, model = joblib.load(MODEL_PATH)

# GUI Setup

app = tk.Tk()

app.title("📰 Fake News Detector - Vaishnavi Pote")

app.geometry("700x500")

app.configure(bg="#1e1e2f")

# Styles

style = ttk.Style()

style.theme_use('clam')

style.configure("TButton", foreground="white", background="#6c63ff", font=("Helvetica", 12, "bold"), padding=10)

style.map("TButton", background=[('active', '#5146d8')])

# Title Label

tk.Label(app, text="🧠 Fake News Detector", font=("Helvetica", 22, "bold"), bg="#1e1e2f", fg="#6c63ff").pack(pady=20)

# Textbox for news input

text_input = tk.Text(app, height=10, width=80, font=("Helvetica", 12), bg="#2d2d44", fg="white",
                     insertbackground="white", wrap="word", borderwidth=2, relief="groove")

text_input.pack(pady=10)

# Result Label

result_label = tk.Label(app, text="", font=("Helvetica", 16, "bold"), bg="#1e1e2f", fg="white")

result_label.pack(pady=10)


# Detect Button Function

def detect_news():
    news = text_input.get("1.0", tk.END).strip()

    if news:

        vec_input = vectorizer.transform([news])

        prediction = model.predict(vec_input)[0]

        color = "#00e676" if prediction == "REAL" else "#ff1744"

        result_label.config(text=f"🔍 News is: {prediction}", fg=color)

    else:

        messagebox.showwarning("Input Needed", "Please enter some news text!")


# Detect Button

ttk.Button(app, text="DETECT FAKE NEWS", command=detect_news).pack(pady=20)

# Footer

tk.Label(app, text="Developed by Vaishnavi Pote ", font=("Helvetica", 10), bg="#1e1e2f", fg="#888").pack(side="bottom",
                                                                                                         pady=5)

app.mainloop()