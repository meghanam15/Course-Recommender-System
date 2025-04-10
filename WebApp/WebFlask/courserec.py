import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD

# Load dataset
df = pd.read_csv(r"D:\MCA Final Demo\WebApp\ufinal.csv")

# Preprocess dataset
def preprocess_data(df):
    df['description'] = (df['title'] + " " + df['objectives'] + " " + df['curriculum']).fillna("").str.lower()
    df['description'] = df['description'].replace(r'\bc\+\+\b', 'c plus plus', regex=True)
    df['description'] = df['description'].replace(r'\bc\b', 'c programming', regex=True)
    df['normalized_title'] = df['title'].str.lower()
    return df

df = preprocess_data(df)

# Tokenizer for TF-IDF
def custom_tokenizer(text):
    return re.findall(r'\b\w+\b|\bc plus plus\b|\bc programming\b|\bc\+\+\b|\bc\+\+ programming\b', text)

# Precompute TF-IDF matrix
def compute_tfidf(df):
    vectorizer = TfidfVectorizer(
        tokenizer=custom_tokenizer,
        token_pattern=None,
        max_features=5000
    )
    tfidf_matrix = vectorizer.fit_transform(df['description'])
    return tfidf_matrix, vectorizer

tfidf_matrix, tfidf_vectorizer = compute_tfidf(df)

# Preprocess skill query
def preprocess_skill_query(query):
    query = query.lower()
    if query == "c++":
        query = "c plus plus"
    elif query == "c":
        query = "c programming"
    return query

# Get nearest courses
def get_nearest_courses(skill_query, tfidf_matrix, tfidf_vectorizer):
    query_vector = tfidf_vectorizer.transform([skill_query])
    nn = NearestNeighbors(n_neighbors=20, metric='cosine')
    nn.fit(tfidf_matrix)
    distances, indices = nn.kneighbors(query_vector)
    return distances[0], indices[0]

def filter_recommendations(skill_query, recommendations):
    filtered_courses = []
    programming_keywords = [
        'programming', 'crash course', 'training course', 'coding', 'development', 'learn', 
        'tutorial', 'course', 'developer', 'beginner', 'advanced',
        'c#', 'c sharp', 'c plus plus', 'c++ programming', 'c++', 'c++ training'
    ]
    ai_keywords = ['artificial intelligence', 'ai', 'machine learning', 'deep learning', 'neural networks']
    irrelevant_courses = [
        "Art History Renaissance to 20th Century",
        "ManageEngine OPManager Plus Network Monitoring course & Lab",
        "How to AI with ChatGPT Plus and ChatGPT 4 Course",
    ]
    skill_query = skill_query.strip().lower()
    if skill_query in ["c plus plus", "c++"]:
        # Add default C++ courses with URLs
        default_cplusplus_courses = [
            {"title": "C++ Training Crash Course for C++ Beginners", "score": 0.9, "rating": 4.5, "url": "https://www.udemy.com/course/c-training-crash-course-2022/"},
            {"title": "C++ Complete Training Course for C++ Beginners All In One", "score": 0.83, "rating": 4.3, "url": "https://www.udemy.com/course/c-complete-training-course-for-beginners-2022/"}
        ]
        for default_course in default_cplusplus_courses:
            filtered_courses.append(default_course)

    for course in recommendations:
        course_title = course['title'].lower()
        course_description = course.get('description', '').lower()
        if 'score' not in course or isinstance(course['score'], str):
            course['score'] = float(course.get('score', 0))
        if 'rating' not in course or course['rating'] == 'N/A':
            course['rating'] = 0.0
        else:
            course['rating'] = float(course['rating'])

        if any(irrelevant_course.lower() in course_title for irrelevant_course in irrelevant_courses):
            continue
        if skill_query in ["c plus plus", "c++"]:
            if "c programming" in course_title or "c#" in course_title:
                continue
            if "c programming" in course_description or "c#" in course_description:
                continue
        elif skill_query == "c programming":
            if "c++" in course_title or "c plus plus" in course_title or "c#" in course_title:
                continue
            if "c++" in course_description or "c plus plus" in course_description or "c#" in course_description:
                continue
        elif skill_query in ["c#", "c sharp"]:
            if "c programming" in course_title or "c programming" in course_description:
                continue
            if "c++" in course_title or "c plus plus" in course_title:
                continue
            if "c#" in course_title or "c sharp" in course_title or "c# programming" in course_description:
                filtered_courses.append(course)
                continue
        elif skill_query in ["ai", "AI", "artificial intelligence", "machine learning", "deep learning", "neural networks"]:
            if not any(ai_keyword in course_title or ai_keyword in course_description for ai_keyword in ai_keywords):
                continue
        if not any(keyword in course_title or keyword in course_description for keyword in programming_keywords + ai_keywords):
            continue
        filtered_courses.append(course)

    filtered_courses.sort(key=lambda x: (x.get('score', 0), x.get('rating', 0.0)), reverse=True)
    return filtered_courses[:5]

# Collaborative Filtering using SVD
def collaborative_filtering(user_item_matrix, num_components=2):
    svd = TruncatedSVD(n_components=num_components)
    svd_matrix = svd.fit_transform(user_item_matrix)
    reconstructed_matrix = np.dot(svd_matrix, svd.components_)
    predicted_ratings = reconstructed_matrix[0]
    course_ratings = list(zip(df['id'], predicted_ratings))
    sorted_courses = sorted(course_ratings, key=lambda x: x[1], reverse=True)
    recommendations = [course[0] for course in sorted_courses[1:6]]
    return recommendations

# Generate hybrid recommendations (content-based + collaborative)
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
    
    collaborative_recommendations = collaborative_filtering(user_item_matrix)
    for course_id in collaborative_recommendations:
        course_row = df[df['id'] == course_id].iloc[0]
        if course_row['title'] not in [course['title'] for course in recommendations]:
            recommendations.append({
                "title": course_row['title'],
                "url": course_row['url'],
                "rating": course_row['rating'],
                "score": f"{course_row['rating']:.2f}"
            })

    return recommendations[:5]
