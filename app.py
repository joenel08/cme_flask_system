from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response
import mysql.connector
import hashlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import csv
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import io
import base64
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag

# LDA imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.lda_model

import joblib
from scipy.sparse import hstack, csr_matrix

from datetime import datetime

from flask_cors import CORS
# ✅ Import functions and models from sample_prediction.py
from sentiment_analysis import predict_sentiment_label_ann, loaded_ann_model, loaded_tfidf_vectorizer

import pymysql


app = Flask(__name__)
CORS(app)
app.secret_key = "secret123"

# Load CSV file
df = pd.read_csv('excel_files/VOC_DATA.csv', encoding='latin1')
df_cme = pd.read_csv('excel_files/CME_DATA.csv', encoding='latin1')


# Drop empty values (NaN and empty strings)
df = df.dropna(subset=['Feedback'])  # Drop rows where Feedback is NaN
df = df[df['Feedback'].str.strip() != '']  # Drop rows where Feedback is empty after stripping spaces

def final_remove_noise(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', str(text))
    return text.lower()

def final_tokenize(text):
    return word_tokenize(text)

def final_normalize_characters(text):
    return text.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")

def final_pos_tagging(tokens):
    return pos_tag(tokens)

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

nltk.download('wordnet')

# Preprocess data
df['Feedback'] = df['Feedback'].apply(final_remove_noise)
df['Feedback'] = df['Feedback'].apply(final_normalize_characters)
df['Tokenized'] = df['Feedback'].apply(final_tokenize)
df['POS_Tags'] = df['Tokenized'].apply(final_pos_tagging)

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    score = analyzer.polarity_scores(str(text))
    compound = score['compound']
    return 'Positive' if compound >= 0.05 else 'Negative' if compound <= -0.05 else 'Neutral'

df['Sentiment'] = df['Feedback'].apply(analyze_sentiment)

# Compute sentiment ratios
total_feedback = len(df)
sentiment_counts = df['Sentiment'].value_counts()
ratios = {
    "positive": int(sentiment_counts.get('Positive', 0)),
    "negative": int(sentiment_counts.get('Negative', 0)),
    "neutral": int(sentiment_counts.get('Neutral', 0)),
    "total": total_feedback
}

def generate_wordcloud(sentiment):
    text = ' '.join(df[df['Sentiment'] == sentiment]['Feedback'])
    wordcloud = WordCloud(width=1200, height=600, background_color='white').generate(text)
    img = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(img, format='png', bbox_inches='tight')
    
    
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

DB_CONFIG = {"host": "localhost", "user": "root", "password": "", "database": "cme_db"}

def md5_hash(password):
    return hashlib.md5(password.encode()).hexdigest()

def get_db_connection():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Exception as e:
        print(f"Error connecting to MySQL: {e}")
        return None
    


@app.route("/")
def home():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        remember_me = "remember_me" in request.form
        hashed_password = md5_hash(password)
        try:
            connection = get_db_connection()
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE email = %s AND password = %s", (email, hashed_password))
            user = cursor.fetchone()
            cursor.close()
            connection.close()
        except Exception as e:
            return jsonify({"status": "error", "message": "Database connection error"}), 500
        if user:
            session["user_id"] = user["id"]
            session["email"] = user['email']
            session["fullname"] = user["firstname"] + " " + user["lastname"]
            if remember_me:
                session.permanent = True
            return jsonify({"status": "success", "message": "Login successful", "redirect": url_for("dashboard")})
        return jsonify({"status": "error", "message": "Invalid email or password"}), 401
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))
    email = session.get("email", "Guest")
    fullname = session.get("fullname", "Guest")
    return render_template("index.html", email=email, fullname=fullname)



@app.route('/api/sentiment_ratios')
def sentiment_ratios():
    return jsonify(ratios)

@app.route('/api/wordcloud/<sentiment>')
def wordcloud(sentiment):
    if sentiment in ["Positive", "Negative", "Neutral"]:
        return jsonify({"wordcloud": generate_wordcloud(sentiment)})
    return jsonify({"error": "Invalid sentiment category"}), 400



# voc
@app.route("/voc")
def voc():
    if "user_id" not in session:
        return redirect(url_for("login"))
    email = session.get("email", "Guest")
    fullname = session.get("fullname", "Guest")
    return render_template("voc.html", email=email, fullname=fullname)

@app.route('/api/wordclouds')
def wordclouds_by_category():
    categories = df['Category'].unique()
    wordclouds = {}
    
    for category in categories:
        text = ' '.join(df[df['Category'] == category]['Feedback'])
        wordcloud = WordCloud(width=1200, height=400, background_color='white').generate(text)
        img = io.BytesIO()
        plt.figure(figsize=(10, 4))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.savefig(img, format='png', bbox_inches='tight')
        plt.close()
        img.seek(0)
        wordclouds[category] = base64.b64encode(img.getvalue()).decode()
    
    return jsonify(wordclouds)


@app.route('/api/sentiment_category')
def sentiment_per_category():
    sentiment_counts = df.groupby(['Category', 'Sentiment']).size().unstack(fill_value=0)
    sentiment_counts = sentiment_counts.reset_index().to_dict(orient='records')
    return jsonify(sentiment_counts)


@app.route('/api/sentiment_department')
def sentiment_per_department():
    sentiment_counts = df.groupby(['Department', 'Sentiment']).size().unstack(fill_value=0)
    sentiment_counts = sentiment_counts.reset_index().to_dict(orient='records')
    return jsonify(sentiment_counts)
#end voc

#CME page

@app.route("/cme")
def cme():
    if "user_id" not in session:
        return redirect(url_for("login"))

    email = session.get("email", "Guest")
    fullname = session.get("fullname", "Guest")

    return render_template(
        "cme.html",
        email=email,
        fullname=fullname,
   
    )

#end cme page

#LDA Page

@app.route("/lda")
def lda():
    if "user_id" not in session:
        return redirect(url_for("login"))

    email = session.get("email", "Guest")
    fullname = session.get("fullname", "Guest")
    
    """Load the LDA page immediately with a loader."""
    return render_template("lda.html",email=email,
        fullname=fullname,)


# LDA Visualization cache
lda_vis_html = None

# ================================
# LDA PAGE
# ================================\
@app.route("/lda_visualization")
def lda_visualization():
    # Load the CSV file
    file_path = 'excel_files/final_data_AMA.csv'  # Make sure the file is in the same directory
    data = pd.read_csv(file_path)

    # Preprocessing
    if 'Cleaned Feedback' not in data.columns:
        return "Column 'Cleaned Feedback' not found in CSV file."

    data['Cleaned Feedback'].dropna(inplace=True)
    data['Cleaned Feedback'] = data['Cleaned Feedback'].astype(str)

    # Convert text data into a document-term matrix
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(data['Cleaned Feedback'])

    # Train LDA Model
    num_topics = 3  # Adjust the number of topics as needed
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(dtm)

    # Prepare LDA visualization
    vis = pyLDAvis.lda_model.prepare(lda_model, dtm, vectorizer)


  # Save the visualization to an external HTML file
     # Save visualization to HTML
    vis_html = io.StringIO()
    pyLDAvis.save_html(vis, vis_html)
    vis_html.seek(0)

    # Return the visualization content
    return vis_html.getvalue()
#end LDA Page



#sentiment page
@app.route("/sentiment")
def sentiment():
    if "user_id" not in session:
        return redirect(url_for("login"))

    email = session.get("email", "Guest")
    fullname = session.get("fullname", "Guest")

    # Connect to MySQL
    connection = get_db_connection()
    data = []
    ratio = {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0}

    try:
        with connection.cursor() as cursor:
            # Fetch sentiment data
            cursor.execute("SELECT * FROM `new_sentiment_data`")
            data = cursor.fetchall()

            # Fetch sentiment ratio
            cursor.execute("SELECT COUNT(*) FROM new_sentiment_data WHERE Predicted_Sentiment = 'Positive'")
            positive = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM new_sentiment_data WHERE Predicted_Sentiment = 'Negative'")
            negative = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM new_sentiment_data WHERE Predicted_Sentiment = 'Neutral'")
            neutral = cursor.fetchone()[0]

            # Calculate the total count
            total = positive + negative + neutral

            # Store the ratio
            ratio = {
                'positive': positive,
                'negative': negative,
                'neutral': neutral,
                'total': total
            }

    except Exception as e:
        print(f"Error fetching data: {e}")
    finally:
        connection.close()

    # Pass the ratio along with the data to the template
    return render_template("sentiment_analysis.html", email=email, fullname=fullname, data=data, ratio=ratio)


#end sentiment page


@app.route("/predict_sentiment", methods=["POST"])
def predict_sentiment():
    try:
        # Validate JSON request
        if not request.is_json:
            return jsonify({'error': 'Invalid JSON format'}), 400

        # Extract text from JSON payload
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Use the ANN prediction function
        predicted_sentiment = predict_sentiment_label_ann(text, loaded_ann_model, loaded_tfidf_vectorizer)

        # ✅ Save prediction to MySQL
        connection = get_db_connection()

        if connection:
            cursor = connection.cursor()

            # Get current date and time
            current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # SQL query to insert data
            query = """
            INSERT INTO new_sentiment_data (Date, new_reviews, Predicted_Sentiment)
            VALUES (%s, %s, %s)
            """

            cursor.execute(query, (current_date, text, predicted_sentiment))
            connection.commit()

            cursor.close()
            connection.close()

        # Return the predicted sentiment
        return jsonify({'sentiment': predicted_sentiment})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# Route to display the table
@app.route('/get_sentiment_ratio')
def get_sentiment_ratio():
    try:
        # Connect to the MySQL database
        connection = get_db_connection()
        cursor = connection.cursor()

        # Fetch sentiment ratio
        cursor.execute("SELECT COUNT(*) FROM new_sentiment_data WHERE Predicted_Sentiment = 'Positive'")
        positive = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM new_sentiment_data WHERE Predicted_Sentiment = 'Negative'")
        negative = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM new_sentiment_data WHERE Predicted_Sentiment = 'Neutral'")
        neutral = cursor.fetchone()[0]

        # Calculate total
        total = positive + negative + neutral

        # Fetch the latest table data
        cursor.execute("SELECT * FROM new_sentiment_data ORDER BY s_id DESC")
        table_data = cursor.fetchall()

        # Prepare response data
        response = {
            'ratio': {
                'positive': positive,
                'negative': negative,
                'neutral': neutral,
                'total': total
            },
            'table': [
                {
                    'id': row[0],
                    'date': row[1],
                    'review': row[2],
                    'sentiment': row[3]
                }
                for row in table_data
            ]
        }

       
        # Close connections
        cursor.close()
        connection.close()

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get_table_data', methods=['GET'])
def get_table_data():
    try:
        # Connect to your database
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Fetch the latest data from your table
        cursor.execute("SELECT * FROM new_sentiment_data ORDER BY s_id DESC")
        rows = cursor.fetchall()

        # Format the data for DataTables
        data = []
        for row in rows:
            data.append({
                "id": row['s_id'],
                "date": row['Date'],
                "review": row['new_reviews'],
                "sentiment": row['Predicted_Sentiment']
            })

        # Return the data in JSON format
        return jsonify({
            "data": data
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        conn.close()
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# @app.route("/user_form")
# def user_form():
#     return render_template("user_form.html")

@app.route('/user_form')
def user_form():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT q_id, category, question_text FROM questions WHERE category IN ('Course Content', 'Mentorship')")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    # Group questions by category name (text)
    questions_by_category = {'Course Content': [], 'Mentorship': []}
    for row in rows:
        questions_by_category[row['category']].append(row)

    return render_template('user_form.html', questions_by_category=questions_by_category)

# Save form data route
@app.route('/save_feedback', methods=['POST'])
def save_feedback():
    try:
        data = request.json

        course_code = data.get('course_code')
        insights = data.get('insights')
        mentor_feedback_text = data.get('mentor_feedback_text')

        # Connect to DB
        conn = get_db_connection()
        cursor = conn.cursor()

        # Insert main feedback info (course code + text feedback)
        cursor.execute(
            "INSERT INTO user_feedback (course_code, insights, mentor_feedback_text) VALUES (%s, %s, %s)",
            (course_code, insights, mentor_feedback_text)
        )
        feedback_id = cursor.lastrowid

        # Now save each question answer
        for key, value in data.items():
            if key.startswith('question_'):
                question_id = int(key.split('_')[1])  # get q_id from key 'question_123'
                answer = int(value)
                cursor.execute(
                    "INSERT INTO user_feedback_answers (feedback_id, question_id, answer) VALUES (%s, %s, %s)",
                    (feedback_id, question_id, answer)
                )

        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({'status': 'success', 'message': 'Feedback saved successfully!'})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route("/feedback")
def feedback():
    if "user_id" not in session:
        return redirect(url_for("login"))

    email = session.get("email", "Guest")
    fullname = session.get("fullname", "Guest")

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Fetch all questions
    cursor.execute("SELECT q_id, question_text FROM questions ORDER BY q_id")
    questions = cursor.fetchall()

    # Fetch feedback entries
    cursor.execute("""
        SELECT 
            uf.id AS feedback_id,
            uf.course_code,
            uf.insights,
            uf.mentor_feedback_text,
            uf.created_at
        FROM user_feedback uf
        ORDER BY uf.created_at DESC
    """)
    feedbacks = cursor.fetchall()

    # Fetch all answers (we'll match them later)
    cursor.execute("""
        SELECT feedback_id, question_id, answer
        FROM user_feedback_answers
    """)
    all_answers = cursor.fetchall()

    # Organize answers per feedback
    answers_by_feedback = {}
    for ans in all_answers:
        fid = ans['feedback_id']
        if fid not in answers_by_feedback:
            answers_by_feedback[fid] = {}
        answers_by_feedback[fid][ans['question_id']] = ans['answer']

    # Combine feedback + answers
    feedback_data = []
    for fb in feedbacks:
        row = {
            "id": fb["feedback_id"],
            "course_code": fb["course_code"],
            "insights": fb["insights"],
            "mentor_feedback_text": fb["mentor_feedback_text"],
            "created_at": fb["created_at"],
            "answers": []
        }

        for q in questions:
            q_id = q["q_id"]
            row["answers"].append(answers_by_feedback.get(fb["feedback_id"], {}).get(q_id, "N/A"))
        
        feedback_data.append(row)

    cursor.close()
    conn.close()

    return render_template(
        "edit_feedback.html",
        data=feedback_data,
        questions=questions,
        email=email,
        fullname=fullname,
    )


   
#admin users
@app.route("/admin-users")
def admin_users():
    if "user_id" not in session:
        return redirect(url_for("login"))

    email = session.get("email", "Guest")
    fullname = session.get("fullname", "Guest")

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Fetch all admin users (assuming userType = 1 means admin)
    cursor.execute("""
        SELECT id, firstname, lastname, email, avatar, date_created
        FROM users
        ORDER BY date_created DESC
    """)
    admin_users = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template(
        "admin_users.html",
        admin_users=admin_users,
        email=email,
        fullname=fullname
    )


@app.route("/admin-users/save", methods=["POST"])
def save_admin_user():
    if "user_id" not in session:
        return jsonify(success=False, error="Unauthorized"), 403

    data = request.get_json()

    user_id = data.get("id")
    firstname = data.get("firstname")
    lastname = data.get("lastname")
    email = data.get("email")
    password = data.get("password")  # Only hashed if new or updated
    hashed_password = hashlib.md5(password.encode()).hexdigest()

    conn = get_db_connection()
    cursor = conn.cursor()

    if user_id:  # UPDATE
        if password:
            cursor.execute("""
                UPDATE users SET firstname=%s, lastname=%s, email=%s, password=SHA2(%s, 256)
                WHERE id=%s 
            """, (firstname, lastname, email, password, user_id))
        else:
            cursor.execute("""
                UPDATE users SET firstname=%s, lastname=%s, email=%s
                WHERE id=%s
            """, (firstname, lastname, email, user_id))
    else:  # ADD
        cursor.execute("""
            INSERT INTO users (firstname, lastname, email, password, date_created)
            VALUES (%s, %s, %s, %s, 1, NOW())
        """, (firstname, lastname, email, hashed_password))

    conn.commit()
    cursor.close()
    conn.close()
    return jsonify(success=True)

@app.route("/admin-users/delete/<int:user_id>", methods=["DELETE"])
def delete_admin_user(user_id):
    if "user_id" not in session:
        return jsonify(success=False, error="Unauthorized"), 403

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify(success=True)

#end_users
 
# Load data
# Load your CSV file
CMEdata = "excel_files/CME_DATA.csv"  
df_CME = pd.read_csv(CMEdata)

# Data Cleaning
df_CME['COURSE CONTENT'].fillna('No Comment', inplace=True)
df_CME['MENTOR'].fillna('No Comment', inplace=True)
df_CME['COURSE CODE'].fillna('Not Specified', inplace=True)
df_CME = df_CME.drop('ID', axis=1, errors='ignore')
df_CME.drop_duplicates(inplace=True)

def analyze_sentiment(text):
    scores = analyzer.polarity_scores(str(text))
    return scores['compound']

df_CME['Sentiment Compound'] = df_CME['COURSE CONTENT'].apply(analyze_sentiment)
df_CME['Sentiment Label'] = df_CME['Sentiment Compound'].apply(
    lambda x: 'Positive' if x >= 0.05 else 'Negative' if x <= -0.05 else 'Neutral'
)

# Word Clouds
def generate_wordcloud_CME(text):
    wc = WordCloud(width=1200, height=600, background_color='white').generate(text)
    img = io.BytesIO()
    plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)

    # Convert image to Base64 string
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

# Prepare word cloud data
wordcloud_data = {
    sentiment: generate_wordcloud_CME(" ".join(df_CME[df_CME['Sentiment Label'] == sentiment]['COURSE CONTENT']))
    for sentiment in df_CME['Sentiment Label'].unique()
}



# Prepare chart data
course_counts = df_CME['COURSE CODE'].value_counts().nlargest(100).reset_index()
course_counts.columns = ['Course Code', 'Count']

@app.route('/CME-chart-data')
def CME_chart_data():
    chart_data = {
        'labels': course_counts['Course Code'].tolist(),
        'data': course_counts['Count'].tolist()
    }
    return jsonify(chart_data)

@app.route('/CME-wordcloud-data')
def CME_wordcloud():
    sentiment_order = ['Positive', 'Neutral', 'Negative']  # Define the desired order

    # Generate word clouds in the specified order
    wordcloud_data = {
        sentiment: generate_wordcloud_CME(" ".join(df_CME[df_CME['Sentiment Label'] == sentiment]['COURSE CONTENT']))
        for sentiment in sentiment_order if sentiment in df_CME['Sentiment Label'].unique()
    }
    
    return jsonify(wordcloud_data)


#download csv
@app.route('/download-csv')
def download_csv():
    # Query data from your database
    query = """
        SELECT 
            id, course_code, syllabus, course_goals, readings, delivery_format, overall_content, 
            insights, mentor_feedback, mentor_response, mentor_availability, 
            mentor_overall, mentor_feedback_text, created_at
        FROM user_feedback
    """
    # Connect to MySQL
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query)
    
    data = cursor.fetchall()

    # Prepare CSV data
    output = io.StringIO()
    writer = csv.writer(output)

    # Write the header row
    writer.writerow([
        'ID', 'Course Code', 'Syllabus', 'Course Goals', 'Readings', 'Delivery Format', 'Overall Content',
        'Insights', 'Mentor Feedback', 'Mentor Response', 'Mentor Availability',
        'Mentor Overall', 'Mentor Feedback Text', 'Date of Feedback'
    ])

    # Write the rows
    for row in data:
        writer.writerow(row)

    output.seek(0)

    # Return CSV file
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=feedback_data.csv"}
    )

@app.route('/questions')
def questions():

    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM questions ORDER BY category, q_id")
    questions = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('questions_list.html', questions=questions)
@app.route('/questions/save', methods=['POST'])
def save_question():
    data = request.json
    qid = data.get('id')
    category = data['category']
    question_text = data['question_text']

    conn = get_db_connection()
    cursor = conn.cursor()

    if qid:  # Update existing
        cursor.execute("UPDATE questions SET category=%s, question_text=%s WHERE q_id=%s",
                       (category, question_text, qid))
    else:  # Insert new
        cursor.execute("INSERT INTO questions (category, question_text) VALUES (%s, %s)",
                       (category, question_text))
        qid = cursor.lastrowid

    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({'success': True, 'q_id': qid})

@app.route('/questions/delete/<int:qid>', methods=['DELETE'])
def delete_question(qid):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM questions WHERE q_id=%s", (qid,))
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({'success': True})



if __name__ == "__main__":
    app.run(debug=True)