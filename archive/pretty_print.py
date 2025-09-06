import pandas as pd
import json
import os
from sklearn.preprocessing import LabelEncoder

df = pd.DataFrame()
dir_name = "answers_free_text"
for paper in os.listdir(dir_name):
    if not paper.endswith(".json"):
        continue
    results = json.load(open(f"{dir_name}/{paper}"))
    answers = results["answers"]  # list of dicts
    answers_df = pd.DataFrame(answers)
    answers_df["paper"] = results["paper"]
    df = pd.concat([df, answers_df])
df["paper_id"] = LabelEncoder().fit_transform(df["paper"])
os.makedirs("answers", exist_ok=True)
df.to_csv("answers/all_answers.csv", index=False)
