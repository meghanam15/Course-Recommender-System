# Course Recommender System Based on Job Trends

A web application that analyzes real job market data to recommend relevant online courses based on in-demand skills.

## What It Does

- User enters a skill (e.g. "Python", "Machine Learning", "SQL")
- System searches 80,000+ job postings to find matching companies and job levels
- Recommends top 5 courses using a hybrid ML approach (TF-IDF + KNN + SVD)
- Dashboard displays visual insights from job market analysis

## How It Works

**Content-Based Filtering** — TF-IDF vectorizes course descriptions, KNN finds the most similar courses to the skill query.

**Collaborative Filtering** — TruncatedSVD (matrix factorization) predicts course ratings based on user-item interactions.

**Hybrid Output** — Both approaches are combined and ranked by similarity score and rating.

## Tech Stack

- **Backend:** Python, Flask
- **ML:** scikit-learn (TF-IDF, KNN, TruncatedSVD)
- **Data Processing:** Pandas, NumPy
- **Frontend:** HTML, CSS
- **Dataset:** 89,380 LinkedIn job postings, 98,104 Udemy courses

## Project Structure

```
Course-Recommender-System/
├── finalapp.py          # Main Flask application
├── courserec.py         # ML recommendation engine
├── requirements.txt     # Dependencies
├── data/
│   ├── company_cluster.csv   # Processed job postings with clusters
│   └── ufinal.csv            # Processed course dataset
├── templates/
│   ├── index.html
│   ├── search.html
│   └── dashboard.html
└── static/
    ├── styles.css
    └── images/
```

## Setup & Run

```bash
# 1. Clone the repository
git clone https://github.com/meghanam15/Course-Recommender-System.git
cd Course-Recommender-System

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add datasets to data/ folder
# Download datasets from Kaggle (links below) and place in data/

# 4. Run the app
python finalapp.py
```

Open browser at `http://localhost:5000`

## Datasets

Due to file size, datasets are not included in the repo. Download from Kaggle:
- [LinkedIn Job Postings Dataset](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)
- [Udemy Courses Dataset](https://www.kaggle.com/datasets/andrewmvd/udemy-courses)

## Features

- Skill-based job market search
- Hybrid course recommendations (content + collaborative filtering)
- Handles edge cases (C++ vs C vs C#, AI/ML keywords)
- Visual dashboard with job trend insights

## Author

Meghana M — [LinkedIn](https://linkedin.com/in/your-profile) | [GitHub](https://github.com/meghanam15)
