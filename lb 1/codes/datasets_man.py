import pandas as pd
import matplotlib.pyplot as plt

datasets_df = pd.read_csv("datasets.csv")
datasets_df.head()

datasets_df["name_len"] = datasets_df['Dataset name'].astype(str).str.len()
datasets_df[["Dataset name","author","usabilities", "name_len"]].head()

datasets_clean = datasets_df.dropna(subset=["usabilities"]).copy()
datasets_clean.isna().sum()
datasets_clean[["usabilities", "name_len"]].describe()

stats_datasets = {
    "usabilities_mean": datasets_clean["usabilities"].mean(),
    "usabilities_median": datasets_clean["usabilities"].median(),
    "usabilities_mode": datasets_clean["usabilities"].mode().iloc[0],
    "usabilities_std": datasets_clean["usabilities"].std(),

    "name_len_mean": datasets_clean["name_len"].mean(),
    "name_len_median": datasets_clean["name_len"].median(),
    "name_len_mode": datasets_clean["name_len"].mode().iloc[0],
    "name_len_std": datasets_clean["name_len"].std(),
}
stats_datasets



plt.figure()
plt.boxplot(datasets_df["usabilities"].dropna())
plt.title("Datasets: Usability (Box Plot)")
plt.ylabel("Usability")
plt.show()

author_counts = datasets_df["author"].value_counts().head(10)
plt.figure()
plt.bar(author_counts.index, author_counts.values)
plt.title("Top 10 Authors by Number of Datasets")
plt.xlabel("Author")
plt.ylabel("Number of Datasets")
plt.xticks(rotation=45)
plt.show()

plt.figure()
plt.scatter(datasets_df["usabilities"], datasets_df["upvotes"])
plt.title("Usability vs Upvotes")
plt.xlabel("Usability")
plt.ylabel("Upvotes")
plt.show()
