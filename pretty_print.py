import pandas as pd
import json
import os

df = pd.DataFrame()
for paper in os.listdir("answers"):
    if not paper.endswith(".json"):
        continue
    results = json.load(open(f"answers/{paper}"))
    answers = results["answers"]  # list of dicts
    answers_df = pd.DataFrame(answers)
    answers_df["paper"] = results["paper"]
    df = pd.concat([df, answers_df])

df.to_csv("answers/all_answers.csv", index=False)
