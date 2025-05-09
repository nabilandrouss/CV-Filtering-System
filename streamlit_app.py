
# Import necessary libraries
import streamlit as st
import joblib
import pdfplumber
import docx
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the trained model and TF-IDF vectorizer
model = joblib.load('model_v3.1.pkl')
vectorizer = joblib.load('vectorizer_v3.1.pkl')


# Set the title and description of the app
st.title("CV Classifier - Computer Science ")
st.write("Upload your CVs (PDF or DOCX), set your keywords and threshold, and classify CVs based on relevance!")

# ------------------ Upload CV Files Section ------------------
st.header("Upload CV Files")
uploaded_files = st.file_uploader(
    "Upload multiple CVs (PDF or DOCX)", 
    type=["pdf", "docx"], 
    accept_multiple_files=True
)

if not uploaded_files:
    st.warning("Please upload at least two file to continue.")

# ------------------ Helper Functions ------------------
# Extract text from PDF or DOCX
def extract_text(file):
    text = ""
    if file.type == "application/pdf":
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text.strip()

# Define keyword weights and bonus phrases
keywords_weights = {
    'programming': 1.5, 'algorithms': 2, 'data structures': 2, 'object-oriented programming': 2,
    'oop': 2, 'functional programming': 2, 'recursion': 1.5, 'complexity analysis': 2,
    'time complexity': 1.5, 'python': 3, 'java': 2.5, 'c++': 2.5, 'c': 2, 'c#': 2.5, 'javascript': 2.5,
    'typescript': 2, 'go': 2, 'rust': 2.5, 'dart': 2, 'swift': 2, 'html': 2, 'css': 2, 'sass': 1.5,
    'tailwind': 1.5, 'react': 2.5, 'next.js': 2, 'vue': 2, 'angular': 2, 'node.js': 2.5, 'express.js': 2,
    'flutter': 2, 'react native': 2, 'rest api': 2.5, 'graphql': 2, 'jwt': 2, 'authentication': 2,
    'authorization': 2, 'microservices': 2, 'websockets': 1.5, 'sql': 2, 'mysql': 2, 'postgresql': 2,
    'sqlite': 1.5, 'mongodb': 2, 'nosql': 1.5, 'firebase': 1.5, 'redis': 2, 'git': 1.5, 'github': 1.5,
    'gitlab': 1.5, 'ci/cd': 2, 'jenkins': 1.5, 'docker': 2, 'docker-compose': 2, 'kubernetes': 2,
    'linux': 2, 'bash': 1.5, 'unit testing': 1.5, 'integration testing': 1.5, 'jest': 1.5,
    'pytest': 1.5, 'test automation': 2, 'aws': 3, 'azure': 3, 'gcp': 2.5, 'cloud functions': 2,
    'serverless': 2, 'cloud deployment': 2, 'data science': 3, 'machine learning': 3,
    'deep learning': 2.5, 'nlp': 2, 'computer vision': 2, 'tensorflow': 2.5, 'keras': 2,
    'scikit-learn': 2.5, 'pandas': 2, 'numpy': 2, 'matplotlib': 1.5, 'seaborn': 1.5,
    'data preprocessing': 2, 'software engineering': 2.5, 'design patterns': 2, 'system design': 2.5,
    'uml': 1.5, 'cybersecurity': 2, 'ethical hacking': 2, 'penetration testing': 2, 'encryption': 2,
    'network security': 2, 'operating systems': 2, 'concurrent programming': 2, 'threading': 1.5,
    'process scheduling': 1.5, 'compiler design': 1.5, 'automata theory': 1.5,
    'computational theory': 1.5, 'discrete mathematics': 1.5, 'computer architecture': 2,
    'parallel programming': 2, 'version control': 1.5, 'debugging': 1.5, 'agile': 1, 'scrum': 1,
    'kanban': 1, 'technical documentation': 1.5, 'api design': 2
}

bonus_phrases = [
    "developed a full stack web application",
    "built a real-time chat app using websockets",
    "designed and implemented restful apis",
    "used git and github for version control",
    "collaborated with developers in agile sprints",
    "created responsive interfaces with react",
    "built and deployed containerized apps with docker",
    "implemented user authentication with jwt",
    "connected frontend and backend using api calls",
    "cleaned and analyzed datasets using pandas",
    "trained a classification model using scikit-learn",
    "deployed model to flask api endpoint",
    "worked with sql and nosql databases",
    "ran unit and integration tests with pytest",
    "used cloud services for hosting and storage",
    "documented backend routes using swagger",
    "developed flutter app for mobile platforms",
    "secured user data with encryption techniques",
    "used state management in react apps",
    "monitored application performance in production"
]



# Compute keyword-based score
def compute_keyword_score(text, keywords_dict, phrases):
    text_lower = text.lower()
    total_score = 0
    word_count = len(text_lower.split())

    for keyword, weight in keywords_dict.items():
        count = text_lower.count(keyword)
        total_score += count * weight

    for phrase in phrases:
        if phrase in text_lower:
            total_score += 5

    return total_score / max(word_count, 1)

# Check if all or any keywords exist, based on selected logic
def contains_keywords(text, keywords, mode="AND"):
    text = text.lower()
    if mode == "AND":
        return all(keyword in text for keyword in keywords)
    else:
        return any(keyword in text for keyword in keywords)


# ------------------ Process Uploaded Files ------------------
cv_texts = []
file_names = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        text = extract_text(uploaded_file)
        if text:
            cv_texts.append(text)
            file_names.append(uploaded_file.name)

# ------------------ Classification and Scoring ------------------
if cv_texts:
    X_new = vectorizer.transform(cv_texts)
    probabilities = model.predict_proba(X_new)[:, 1]
    predictions = model.predict(X_new)

    # Compute keyword score for each CV
    keyword_scores = [compute_keyword_score(text, keywords_weights, bonus_phrases) for text in cv_texts]

    # Normalize model and keyword scores
    scaler = MinMaxScaler()
    normalized_model = scaler.fit_transform(probabilities.reshape(-1, 1))
    normalized_keyword = scaler.fit_transform(pd.DataFrame(keyword_scores).values)

    # Combine both into hybrid score
    hybrid_scores = 0.7 * normalized_model + 0.3 * normalized_keyword
    hybrid_scores = hybrid_scores.flatten()

    # Create dataframe
    results_df = pd.DataFrame({
        'File Name': file_names,
        'Model Score': normalized_model.flatten(),
        'Keyword Score': normalized_keyword.flatten(),
        'Hybrid Score': hybrid_scores,
        'Prediction': predictions
    })

    results_df['Prediction'] = results_df['Prediction'].map({0: "Not Relevant", 1: "Relevant"})

    # ------------------ Filter by Relevance Score ------------------
    st.header("Filter by Relevance Score")

    threshold = st.slider(
        "Select minimum score to display:",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01
    )
    st.caption("Higher thresholds show only the most relevant CVs based on both model and keyword scores.")

    filtered_results = results_df[results_df['Hybrid Score'] >= threshold]

# ------------------ Filter by Keywords (Optional) ------------------
st.header("Filter by Keywords (Optional)")

keywords_input = st.text_input(
    "Enter keywords separated by commas (e.g., Python, AWS, Machine Learning):",
    ""
)

# Let the user choose whether to match all keywords or any
filter_logic = st.radio(
    "Keyword match logic:",
    ["Match ALL keywords (AND)", "Match ANY keyword (OR)"],
    index=0
)

st.caption("If keywords are provided, only CVs containing them will be shown. You can choose whether to match all keywords (AND) or any (OR).")


if 'filtered_results' in locals() and not filtered_results.empty:

        if keywords_input.strip():
            keywords = [kw.strip().lower() for kw in keywords_input.split(",") if kw.strip()]
            keyword_matches = [  contains_keywords(cv_texts[i], keywords, mode="AND" if filter_logic.startswith("Match ALL") else "OR")
              for i in filtered_results.index
    ]
            filtered_results["Keyword Match"] = keyword_matches
            filtered_results = filtered_results[filtered_results["Keyword Match"] == True]

            st.subheader(f"CVs matching keywords: {', '.join(keywords)}")
            if filtered_results.empty:
                st.warning("No CVs matched the keywords provided.")
            else:
                st.dataframe(filtered_results.sort_values(by='Hybrid Score', ascending=False))
        else:
            st.subheader("CVs filtered by score only (no keywords applied).")
            st.caption("""

Each CV is evaluated using a hybrid scoring approach:
- **Model Score**: Based on a trained XGBoost classifier.
- **Keyword Score**: Based on how frequently technical keywords and phrases appear in the CV.

The final **Hybrid Score** is a weighted average of both:
`Hybrid Score = 0.7 * Model Score + 0.3 * Keyword Score`

""")

            st.dataframe(filtered_results.sort_values(by='Hybrid Score', ascending=False))

        # ------------------ View CV Content ------------------
        st.header("View CV Content")
        for idx, row in filtered_results.iterrows():
            file_name = row['File Name']
            hybrid_score = row['Hybrid Score']
            prediction = row['Prediction']
            with st.expander(f"{file_name} (Hybrid Score: {hybrid_score:.2f}, Prediction: {prediction})"):
                st.markdown(cv_texts[idx].replace("\n", "\n\n"))
else:
          st.info("No CVs available to display. Adjust your filters or keywords.")



