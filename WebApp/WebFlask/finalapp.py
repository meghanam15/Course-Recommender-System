from flask import Flask, render_template, request
import webbrowser
import threading
import pandas as pd
import numpy as np
from courserec import hybrid_filtering  
from courserec import * 

app = Flask(__name__)

csv_file = r'D:\MCA Final Demo\WebApp\company_cluster.csv'

company_cluster_df = None

def load_dataset():
    
    global company_cluster_df
    try:
        company_cluster_df = pd.read_csv(csv_file)
        print(f"DataFrame loaded successfully with {len(company_cluster_df)} rows.")
    except Exception as e:
        print(f"Error loading DataFrame: {e}")
        company_cluster_df = pd.DataFrame()  

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/search", methods=["GET", "POST"])
def search():
    search_result = None
    error_message = None
    recommended_courses = []
    skill_input = None

    if request.method == "POST":
        skill_input = request.form.get("search_query", "").strip()

        if not skill_input:
            error_message = "Please enter a skill to search for."
        else:
           
            skills = [skill.strip() for skill in skill_input.split(",") if skill.strip()]
            skill_regex = "|".join(skills)  

            if not company_cluster_df.empty:
               
                matched_rows = company_cluster_df[
                    company_cluster_df['job_skills'].str.contains(skill_regex, case=False, na=False)
                ]

                if not matched_rows.empty:
                  
                    search_result = matched_rows[['company', 'cluster_name', 'job level']].head(5).to_dict(orient='records')
                else:
                    error_message = f"No matches found for '{skill_input}'."
            else:
                error_message = "Error with the dataset."

           
            if skills:
                
                user_item_matrix = np.random.rand(100, 5)  

               
                recommended_courses = hybrid_filtering(skill_input, user_item_matrix)

                
                recommended_courses = recommended_courses[:5]

                if not recommended_courses:
                    error_message = f"No recommendations found for '{skill_input}'."

    return render_template(
        "search.html",
        search_result=search_result,
        error_message=error_message,
        skill_input=skill_input,
        recommended_courses=recommended_courses
    )


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

def open_browser():
    """Automatically open the browser."""
    webbrowser.open_new('http://localhost:5000')

if __name__ == "__main__":
    load_dataset()
    threading.Timer(0.1, open_browser).start() 
    app.run(debug=True, use_reloader=False)  
    