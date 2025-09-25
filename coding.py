# app.py
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

MODEL_PATH = "fake_news_model.pkl"

# ---- CONFIG ----
# If you want to test quickly without downloading dataset, set FAST_TRAIN = True.
# FAST_TRAIN creates a small artificial dataset and trains quickly (good for testing UI).
# Set to False to use True.csv + Fake.csv (recommended for real accuracy).
FAST_TRAIN = True

# If using real dataset but want faster training on a smaller subset, set SAMPLE_FRAC (0.0 < frac <= 1.0)
SAMPLE_FRAC = 0.2  # use 20% of full dataset to train faster (set to 1.0 to use all)


# Globals to hold loaded model/vectorizer
vectorizer = None
model = None


def read_and_prepare_dataset(true_path="True.csv", fake_path="Fake.csv", sample_frac=1.0):
    # Try to robustly load the text column from CSVs
    def load_csv_to_text(path):
        df = pd.read_csv(path)
        # find likely text columns
        if "text" in df.columns:
            txt = df["text"].astype(str)
        elif "content" in df.columns:
            txt = df["content"].astype(str)
        elif "article" in df.columns:
            txt = df["article"].astype(str)
        elif "title" in df.columns and "text" in df.columns:
            txt = (df["title"].astype(str) + ". " + df["text"].astype(str))
        else:
            # fallback: join all object columns
            text_cols = [c for c in df.columns if df[c].dtype == "object"]
            if not text_cols:
                raise ValueError(f"No text-like column found in {path}. Columns: {df.columns.tolist()}")
            txt = df[text_cols].agg(" ".join, axis=1).astype(str)
        return txt

    true_text = load_csv_to_text(true_path)
    fake_text = load_csv_to_text(fake_path)

    df_true = pd.DataFrame({"text": true_text, "label": "REAL"})
    df_fake = pd.DataFrame({"text": fake_text, "label": "FAKE"})

    df = pd.concat([df_real := df_true, df_fake], axis=0).reset_index(drop=True)

    # Optional sampling for faster runs
    if 0 < sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

    # Basic cleaning
    df["text"] = df["text"].astype(str).str.replace("\n", " ").str.strip()
    df = df[df["text"].str.len() > 10]  # keep meaningful rows
    return df


def create_small_demo_dataset():
    # Small synthetic dataset for fast testing (not for real accuracy)
    real_texts = [
        "The economy is growing and unemployment is down.",
        "Scientists published a peer reviewed study on climate change.",
        "The city council announced new parks for the community.",
        "The new vaccine passed phase 3 trials with positive results."
    ]
    fake_texts = [
        "Alien spacecraft landed in the town center last night.",
        "Miracle pill removes fat while you sleep, doctors stunned.",
        "Celebrities die hoax: the star was never born.",
        "Government secretly replaced leaders with robots overnight."
    ]
    rows = []
    for i in range(50):
        rows.append({"text": real_texts[i % len(real_texts)], "label": "REAL"})
        rows.append({"text": fake_texts[i % len(fake_texts)], "label": "FAKE"})
    return pd.DataFrame(rows)


def train_model_thread(callback_on_done=None):
    global vectorizer, model
    try:
        # Choose dataset
        if FAST_TRAIN:
            df = create_small_demo_dataset()
        else:
            if not (os.path.exists("True.csv") and os.path.exists("Fake.csv")):
                raise FileNotFoundError("True.csv and Fake.csv not found in the script folder.")
            df = read_and_prepare_dataset("True.csv", "Fake.csv", sample_frac=SAMPLE_FRAC)

        X = df["text"]
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        vectorizer_local = TfidfVectorizer(stop_words="english", max_df=0.7)
        X_train_vec = vectorizer_local.fit_transform(X_train)
        X_test_vec = vectorizer_local.transform(X_test)

        model_local = PassiveAggressiveClassifier(max_iter=100, class_weight="balanced")
        model_local.fit(X_train_vec, y_train)

        acc = model_local.score(X_test_vec, y_test)

        # Save
        joblib.dump((vectorizer_local, model_local), MODEL_PATH)

        # Set globals
        vectorizer, model = vectorizer_local, model_local

        if callback_on_done:
            callback_on_done(success=True, accuracy=acc)
    except Exception as e:
        if callback_on_done:
            callback_on_done(success=False, error=str(e))


def start_training_from_ui(status_label, train_button):
    # Launch training on background thread and update UI when done
    def on_done(success, accuracy=None, error=None):
        # This runs in background thread; use app.after to update UI in main thread
        def finish():
            train_button.config(state="normal", text="🛠 Train Model")
            if success:
                status_label.config(text=f"Training complete. Accuracy: {accuracy:.2%}")
                messagebox.showinfo("Training Complete", f"Model trained and saved as {MODEL_PATH}\nAccuracy: {accuracy:.2%}")
            else:
                status_label.config(text="Training failed.")
                messagebox.showerror("Training Error", f"Training failed:\n{error}")
        app.after(1, finish)

    # disable button and start
    train_button.config(state="disabled", text="Training... (this may take a while)")
    status_label.config(text="Training in progress...")
    t = threading.Thread(target=train_model_thread, args=(on_done,), daemon=True)
    t.start()


def load_model_if_exists():
    global vectorizer, model
    if os.path.exists(MODEL_PATH):
        try:
            vectorizer, model = joblib.load(MODEL_PATH)
            return True
        except Exception:
            return False
    return False


# ------------------ UI ------------------
app = tk.Tk()
app.title("📰 Fake News Chatbot - Trainer + Chat")
app.geometry("700x650")
app.configure(bg="#1e1e2f")

# Top area: Train button + status
top_frame = tk.Frame(app, bg="#1e1e2f")
top_frame.pack(fill="x", pady=(8, 0))

status_label = tk.Label(top_frame, text="Model not loaded.", bg="#1e1e2f", fg="white")
status_label.pack(side="right", padx=10)

train_button = tk.Button(top_frame, text="🛠 Train Model", bg="#ff5722", fg="white",
                         font=("Helvetica", 11, "bold"),
                         command=lambda: start_training_from_ui(status_label, train_button))
train_button.pack(side="left", padx=10)

# If model exists, load it now
if load_model_if_exists():
    status_label.config(text="Model loaded from disk. Ready.")
else:
    status_label.config(text="Model not found. Click 'Train Model' to train.")

# Chat area (scrollable canvas)
chat_frame = tk.Frame(app, bg="#1e1e2f")
chat_frame.pack(fill="both", expand=True, padx=8, pady=6)

canvas = tk.Canvas(chat_frame, bg="#1e1e2f", highlightthickness=0)
scrollbar = ttk.Scrollbar(chat_frame, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas, bg="#1e1e2f")
scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# bottom entry
entry_frame = tk.Frame(app, bg="#1e1e2f")
entry_frame.pack(fill="x", side="bottom")

user_entry = tk.Entry(entry_frame, font=("Helvetica", 12), bg="#2d2d44", fg="white", insertbackground="white")
user_entry.pack(side="left", padx=10, pady=10, fill="x", expand=True)

def add_message_bubble(sender, message, align="left", avatar="🤖", bg_color="#6c63ff"):
    row_frame = tk.Frame(scrollable_frame, bg="#1e1e2f")
    if align == "left":
        avatar_lbl = tk.Label(row_frame, text=avatar, bg="#1e1e2f", fg="white", font=("Helvetica", 16))
        avatar_lbl.pack(side="left", padx=4)
        msg_lbl = tk.Label(row_frame, text=message, wraplength=420, justify="left", bg=bg_color,
                            fg="white", font=("Helvetica", 11), padx=10, pady=6, bd=1, relief="ridge")
        msg_lbl.pack(side="left", padx=4)
        row_frame.pack(anchor="w", pady=6, padx=6, fill="x")
    else:
        msg_lbl = tk.Label(row_frame, text=message, wraplength=420, justify="left", bg=bg_color,
                            fg="white", font=("Helvetica", 11), padx=10, pady=6, bd=1, relief="ridge")
        msg_lbl.pack(side="right", padx=4)
        avatar_lbl = tk.Label(row_frame, text=avatar, bg="#1e1e2f", fg="white", font=("Helvetica", 16))
        avatar_lbl.pack(side="right", padx=4)
        row_frame.pack(anchor="e", pady=6, padx=6, fill="x")
    app.update_idletasks()
    canvas.yview_moveto(1.0)

def on_send(event=None):
    global vectorizer, model
    text = user_entry.get().strip()
    if not text:
        return
    add_message_bubble("You", text, align="right", avatar="🧑", bg_color="#0078ff")
    user_entry.delete(0, tk.END)

    if vectorizer is None or model is None:
        add_message_bubble("Bot", "⚠️ Model not ready. Click 'Train Model' first.", align="left", avatar="🤖", bg_color="#ff9800")
        return

    try:
        x = vectorizer.transform([text])
        pred = model.predict(x)[0]
        if pred == "REAL":
            add_message_bubble("Bot", "✅ This news seems REAL.", align="left", avatar="🤖", bg_color="#00c853")
        else:
            add_message_bubble("Bot", "❌ This news seems FAKE.", align="left", avatar="🤖", bg_color="#d50000")
    except Exception as e:
        add_message_bubble("Bot", f"Error during prediction: {e}", align="left", avatar="🤖", bg_color="#ff5722")

send_btn = tk.Button(entry_frame, text="Send", command=on_send, bg="#6c63ff", fg="white", font=("Helvetica", 11, "bold"))
send_btn.pack(side="right", padx=8)

user_entry.bind("<Return>", on_send)

# welcome
add_message_bubble("Bot", "👋 Hi — paste/type news and I'll check REAL/FAKE.\nIf first time: click 'Train Model'.\nTip: for testing without dataset, set FAST_TRAIN=True in script.", align="left", avatar="🤖", bg_color="#6c63ff")

app.mainloop()
