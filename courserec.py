import pandas as pd
import numpy as np
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD

# Relative path - works on any machine
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'data', 'ufinal.csv')

df = pd.read_csv(csv_path)

def preprocess_data(df):
    df['description'] = (df['title'] + " " + df['objectives'] + " " + df['curriculum']).fillna("").str.lower()
    df['description'] = df['description'].replace(r'\bc\+\+\b', 'c plus plus', regex=True)
    df['description'] = df['description'].replace(r'\bc\b', 'c programming', regex=True)
    df['normalized_title'] = df['title'].str.lower()
    return df

df = preprocess_data(df)

def custom_tokenizer(text):
    return re.findall(r'\b\w+\b|\bc plus plus\b|\bc programming\b|\bc\+\+\b|\bc\+\+ programming\b', text)

def compute_tfidf(df):
    vectorizer = TfidfVectorizer(
        tokenizer=custom_tokenizer,
        token_pattern=None,
        max_features=5000
    )
    tfidf_matrix = vectorizer.fit_transform(df['description'])
    return tfidf_matrix, vectorizer

tfidf_matrix, tfidf_vectorizer = compute_tfidf(df)

def preprocess_skill_query(query):
    query = query.lower()
    if query == "c++":
        query = "c plus plus"
    elif query == "c":
        query = "c programming"
    return query

def get_nearest_courses(skill_query, tfidf_matrix, tfidf_vectorizer):
    query_vector = tfidf_vectorizer.transform([skill_query])
    nn = NearestNeighbors(n_neighbors=20, metric='cosine')
    nn.fit(tfidf_matrix)
    distances, indices = nn.kneighbors(query_vector)
    return distances[0], indices[0]

def filter_recommendations(skill_query, recommendations):
    filtered_courses = []
    programming_keywords = [
        'programming', 'crash course', 'training course', 'coding', 'development',
        'learn', 'tutorial', 'course', 'developer', 'beginner', 'advanced',
        'c#', 'c sharp', 'c plus plus', 'c++ programming', 'c++', 'c++ training'
    ]
    ai_keywords = [
        'artificial intelligence', 'ai', 'machine learning', 'deep learning', 'neural networks'
    ]
    irrelevant_courses = [
        "Art History Renaissance to 20th Century",
        "ManageEngine OPManager Plus Network Monitoring course & Lab",
        "How to AI with ChatGPT Plus and ChatGPT 4 Course",
    ]
    skill_query = skill_query.strip().lower()

    if skill_query in ["c plus plus", "c++"]:
        default_courses = [
            {"title": "C++ Training Crash Course for C++ Beginners", "score": 0.9, "rating": 4.5,
             "url": "https://www.udemy.com/course/c-training-crash-course-2022/"},
            {"title": "C++ Complete Training Course for Beginners All In One", "score": 0.83, "rating": 4.3,
             "url": "https://www.udemy.com/course/c-complete-training-course-for-beginners-2022/"}
        ]
        filtered_courses.extend(default_courses)

    for course in recommendations:
        course_title = course['title'].lower()
        course_description = course.get('description', '').lower()

        if 'score' not in course or isinstance(course['score'], str):
            course['score'] = float(course.get('score', 0))
        if 'rating' not in course or course['rating'] == 'N/A':
            course['rating'] = 0.0
        else:
            course['rating'] = float(course['rating'])

        if any(ic.lower() in course_title for ic in irrelevant_courses):
            continue

        if skill_query in ["c plus plus", "c++"]:
            if "c programming" in course_title or "c#" in course_title:
                continue
        elif skill_query == "c programming":
            if "c++" in course_title or "c plus plus" in course_title or "c#" in course_title:
                continue
        elif skill_query in ["c#", "c sharp"]:
            if "c programming" in course_title or "c++" in course_title:
                continue
            if "c#" in course_title or "c sharp" in course_title:
                filtered_courses.append(course)
                continue
        elif skill_query in ["ai", "artificial intelligence", "machine learning", "deep learning"]:
            if not any(kw in course_title or kw in course_description for kw in ai_keywords):
                continue

        if not any(kw in course_title or kw in course_description for kw in programming_keywords + ai_keywords):
            continue

        filtered_courses.append(course)

    filtered_courses.sort(key=lambda x: (x.get('score', 0), x.get('rating', 0.0)), reverse=True)
    return filtered_courses[:5]

def collaborative_filtering(user_item_matrix, num_components=2):
    svd = TruncatedSVD(n_components=num_components)
    svd_matrix = svd.fit_transform(user_item_matrix)
    reconstructed_matrix = np.dot(svd_matrix, svd.components_)
    predicted_ratings = reconstructed_matrix[0]
    course_ratings = list(zip(df['id'], predicted_ratings))
    sorted_courses = sorted(course_ratings, key=lambda x: x[1], reverse=True)
    return [course[0] for course in sorted_courses[1:6]]

def hybrid_filtering(skill_query, user_item_matrix):
    skill_query = preprocess_skill_query(skill_query)
    distances, indices = get_nearest_courses(skill_query, tfidf_matrix, tfidf_vectorizer)
    recommendations = []

    for idx, distance in zip(indices, distances):
        if idx < len(df):
            course = df.iloc[idx]
            similarity_score = 1 - distance
            recommendations.append({
                "title": course['title'],
                "url": course.get('url', 'N/A'),
                "rating": course.get('rating', 0),
                "score": f"{similarity_score:.2f}"
            })

    recommendations = filter_recommendations(skill_query, recommendations)

    collab_recs = collaborative_filtering(user_item_matrix)
    for course_id in collab_recs:
        matches = df[df['id'] == course_id]
        if matches.empty:
            continue
        course_row = matches.iloc[0]
        if course_row['title'] not in [c['title'] for c in recommendations]:
            recommendations.append({
                "title": course_row['title'],
                "url": course_row['url'],
                "rating": course_row['rating'],
                "score": f"{course_row['rating']:.2f}"
            })

    return recommendations[:5]
