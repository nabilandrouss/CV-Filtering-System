# CV Filtering System – Computer Science Focused

This is a hybrid machine learning application designed to classify CVs (resumés) as **relevant** or **not relevant** within the field of **Computer Science**. It combines traditional ML techniques with a custom keyword scoring system, all deployed through a user-friendly Streamlit interface.

## Live Demo

https://cv-filtering-system.streamlit.app

## Features

- Upload CVs in `.pdf` or `.docx` format
- Calculates three relevance scores:
  - Model Score (XGBoost classifier)
  - Keyword Score (based on weighted technical terms and phrases)
  - Hybrid Score (70% model, 30% keyword)
- Filter by minimum relevance threshold
- Optional keyword filtering using AND / OR logic
- View extracted CV content and prediction results

## How it Works

- CV text is extracted and preprocessed.
- A TF-IDF vectorizer feeds the content into a trained XGBoost model.
- Simultaneously, a keyword matching engine scores the CV based on relevant terms and phrases.
- Both scores are combined into a hybrid score.
- The system classifies each CV as Relevant or Not Relevant based on a threshold and keyword match logic.

## Important Note

This system is specifically trained to detect relevance in CVs related to **Computer Science**, including:

- Software Engineering  
- Data Science / AI  
- DevOps / Cloud Computing  
- Web and App Development

Uploading CVs from unrelated fields (e.g., healthcare, education, construction) may result in low or inaccurate scores.
