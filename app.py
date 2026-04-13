from flask import Flask, render_template, request, redirect, session, jsonify
import sqlite3
import pickle
import re
import string
from datetime import datetime

app = Flask(__name__)
app.secret_key = "secret123"

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS comments(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT,
        sentiment TEXT,
        time TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()

# ---------------- HOME ----------------
@app.route('/')
def home():
    return redirect('/login')

# ---------------- REGISTER ----------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        try:
            conn = sqlite3.connect("users.db")
            c = conn.cursor()
            c.execute("INSERT INTO users(username,password) VALUES(?,?)",
                      (username, password))
            conn.commit()
            conn.close()
            return redirect('/login')
        except:
            return "User already exists!"

    return render_template('register.html')

# ---------------- LOGIN ----------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?",
                  (username, password))
        user = c.fetchone()
        conn.close()

        if user:
            session['user'] = username
            return redirect('/dashboard')
        else:
            return "Invalid Username or Password!"

    return render_template('login.html')

# ---------------- DASHBOARD ----------------
@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect('/login')
    return render_template('dashboard.html', user=session['user'])

# ---------------- ANALYZE (MAIN LOGIC) ----------------
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    text = request.form['text']

    # Clean text
    cleaned_text = clean_text(text)

    # Vectorize
    vector = vectorizer.transform([cleaned_text])

    # Predict
    prediction = model.predict(vector)[0]

    # Save to DB
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("INSERT INTO comments(text, sentiment, time) VALUES(?,?,?)",
              (text, prediction, str(datetime.now())))
    conn.commit()
    conn.close()

    return jsonify({"sentiment": prediction})

# ---------------- ADMIN DASHBOARD ----------------
@app.route('/admin')
def admin():
    if 'user' not in session:
        return redirect('/login')

    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT * FROM comments ORDER BY id DESC")
    data = c.fetchall()
    conn.close()

    return render_template('admin.html', data=data)

# ---------------- LOGOUT ----------------
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)