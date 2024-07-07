import pdfplumber
import os
import pandas as pd
import collections
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from scipy.stats import poisson
import scipy
import numpy as np

def preprocess_pdfs():
    lines = []
    for source_file in os.listdir("source_files"):
        with pdfplumber.open(f"source_files/{source_file}") as pdf:
            for page in pdf.pages:
                tables = page.extract_table(table_settings={
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                    "text_x_tolerance":1,
                })
                parsed = 0
                for line in tables:
                    line = [cell.replace("\u2010", "-") for cell in line]
                    if len(line) == 13:
                        line = line[:11]+line[12:]
                    # print("PA‐" == "PA-")
                    if "PA-" not in "".join(line):
                        print("Bad", line)
                        continue
                    # print(line)
                    lines.append(line)
                    parsed += 1
                print(source_file, page, len(tables), parsed)
                # print(tables)
                    
    print(lines[-1])
    print(collections.Counter([len(line) for line in lines]))
    df = pd.DataFrame(lines, columns=["_", "Date", "Model", "Year", "TIS", "FSH_Left", "FSH_Right", "LF", "LA", "RF", "RA", "Pictures"])
    # Drop column _
    df = df.drop(columns="_")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["TIS"] = pd.to_numeric(df["TIS"], errors="coerce")
    df["FSH_Left"] = pd.to_numeric(df["FSH_Left"].str.replace(" ", "."), errors="coerce")
    df["FSH_Right"] = pd.to_numeric(df["FSH_Right"].str.replace(" ", "."), errors="coerce")
    df.to_parquet("aircraft.parquet")
    print(df)

X_AXES = ["FSH", "TIS", "CSH"]

def data():
    df = pd.read_parquet("aircraft.parquet")
    for column in ["LF", "LA", "RF", "RA"]:
        df[column] = df[column].map({'Acee ted':True, 'Unknown':pd.NA, 'Rejected': False, 'Accepted': True, 'NA': pd.NA})
    print(df)
    return df

def denorm_data():
    df = data()
    frames = []
    for fsh, outcome in [("FSH_Left", "LF"),("FSH_Left", "LA"), ("FSH_Right", "RF"), ("FSH_Right", "RA")]:
        subdf = df[["Date", "Model", "Year", "TIS", fsh, outcome]].rename(columns={fsh: "FSH", outcome: "Outcome"})
        subdf["Side"] = outcome
        frames.append(subdf)
    df = pd.concat(frames)
    df["inspections"] = ((17*df["FSH"])-df["TIS"])/1600
    df["CSH"] = (df["TIS"]+df["inspections"]*100)/2
    df["FSH_Ratio"] = df["FSH"]/df["TIS"]
    df["Failure_Prob"] = 1-np.exp(-df["FSH"]/374901)
    return df

def failure_cdf():
    for X_AXIS in X_AXES:
        plt.clf()
        df = denorm_data()
        df = df.dropna()
        # df = df[df[X_AXIS] >= 5000]
        sns.ecdfplot(data=df, x=X_AXIS, hue="Outcome")
        plt.savefig(f"failure_cdf_{X_AXIS}.pdf", bbox_inches="tight")

def failure_cdf_with_hypothetical():
    for X_AXIS in ["FSH"]:
        plt.clf()
        df = denorm_data()
        df = df.dropna()
        # df = df[df[X_AXIS] >= 5000]
        sns.ecdfplot(data=df, x=X_AXIS, hue="Outcome")
        sns.ecdfplot(data=df, x=X_AXIS, weights="Failure_Prob", label="Hypothetical", color="red")
        plt.savefig(f"failure_cdf_{X_AXIS}_with_hypothetical.pdf", bbox_inches="tight")

def failure_rate():
    for X_AXIS in X_AXES:
        plt.clf()
        df = denorm_data()
        df = df.dropna()
        df = df[df[X_AXIS] < 15000]
        df["Bucket"] = df[X_AXIS] // 2500 * 2500
        df["Threshold"] = df[X_AXIS] >= 12000
        df["Failed"] = (1-df["Outcome"]).astype(bool)
        # grouped = df.groupby("Bucket").agg({"Outcome": ["sum", "count"]}).reset_index()
        # grouped["ci"] = grouped.apply(lambda x: binomtest(x[("Outcome","sum")], x[("Outcome","count")]).proportion_ci()[0], axis=1)
        # grouped["rate"] = 1-(grouped[("Outcome","sum")]/grouped[("Outcome","count")])
        grid = [[len(df[(df["Threshold"] == threshold) & (df["Failed"] == failed)]) for failed in (True, False)] for threshold in [False, True]]
        print(scipy.stats.chi2_contingency(grid))
        print(grid)
        # return
        def ci(x):
            print(x.sum(), len(x), x.sum()/len(x),binomtest(int(x.sum()), len(x)).proportion_ci())
            return binomtest(int(x.sum()), len(x)).proportion_ci()
        sns.barplot(data=df, x="Bucket", y="Failed", estimator="mean", errorbar=ci)
        plt.xticks(rotation=45)
        plt.xlabel(X_AXIS)
        plt.savefig(f"failure_rate_{X_AXIS}.pdf", bbox_inches="tight")

def optimize_lambda():
    X_AXIS="FSH"
    # Determine the optimal lambda for the poisson distribution of failure rates
    df = denorm_data()
    df = df.dropna()
    def log_likelihood(test_mean):
        test_lambda = 1/test_mean
        df["failure_prob"] = 1-np.exp(-test_lambda*df[X_AXIS])
        df["likelihood"] = np.abs(df["Outcome"]-df["failure_prob"])
        return np.log(df["likelihood"].astype(float)).sum()
    # Maximize log_likelihood in the range 1 to 15000
    best_likelihood = -1000000000
    best_likelihood_mean = 0
    for test_mean in range(1, 1000000, 100):
        print(test_mean)
        log_likelihood_test = log_likelihood(test_mean)
        if log_likelihood_test > best_likelihood:
            best_likelihood = log_likelihood_test
            best_likelihood_mean = test_mean
    print(best_likelihood, best_likelihood_mean)
    # print(df)

def counts():
    for X_AXIS in X_AXES:
        for outcome in [True, False]:
            plt.clf()
            df = denorm_data()
            df = df.dropna()
            df = df[df[X_AXIS] < 15000]
            df["Bucket"] = df[X_AXIS] // 1000 * 1000
            if not outcome:
                df = df[df.Outcome == False]
            grouped = df.groupby(["Bucket", "Outcome"]).Model.count().reset_index()
            sns.barplot(data=grouped, x="Bucket", y="Model", hue="Outcome")
            plt.xticks(rotation=45)
            plt.xlabel(X_AXIS)
            plt.savefig(f"counts_{X_AXIS}_{outcome}.pdf", bbox_inches="tight")
    

if __name__ == "__main__":
    optimize_lambda()
    # print(denorm_data())
    # # preprocess_pdfs()
    failure_rate()
    counts()
    failure_cdf()
    failure_cdf_with_hypothetical()
    # print(denorm_data())