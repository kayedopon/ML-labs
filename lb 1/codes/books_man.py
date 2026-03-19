import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

books_df = pd.read_csv("books.csv")
books_df.head()

books_df["price_num"] = (books_df["price"].astype(str).str.replace(r"[^\d.]", "", regex=True))
books_df["price_num"] = pd.to_numeric(books_df["price_num"], errors="coerce")

rating_map = {"One":1, "Two":2, "Three":3, "Four":4, "Five":5}
books_df["rating_num"] = books_df["rating"].map(rating_map)

books_clean = books_df.dropna(subset=["price_num", "rating_num"]).copy()
books_clean.isna().sum()

books_clean[["price_num", "rating_num"]].describe()
print(books_df)

stats_datasets = {
    "price_mean": books_clean["price_num"].mean(),
    "price_median": books_clean["price_num"].median(),
    "price_mode": books_clean["price_num"].mode().iloc[0] if not books_clean["price_num"].mode().empty else np.nan,
    "price_std": books_clean["price_num"].std(),

    "rating_mean": books_clean["rating_num"].mean(),
    "rating_median": books_clean["rating_num"].median(),
    "rating_mode": books_clean["rating_num"].mode().iloc[0] if not books_clean["rating_num"].mode().empty else np.nan,
    "rating_std": books_clean["rating_num"].std(),
}
stats_datasets

plt.figure()
plt.boxplot(books_df["price_num"].dropna())
plt.title("Books: Price discribution (Box Plot)")
plt.ylabel("Price")
plt.show()

rating_counts = books_df["rating_num"].value_counts()
plt.figure()
plt.bar(rating_counts.index, rating_counts.values)
plt.title("Books: Rating distribution")
plt.xlabel("Rating")
plt.ylabel("Number")
plt.show()

plt.figure()
plt.scatter(books_df["price_num"], books_df["rating_num"])
plt.title("Books: Prices vs Ratings")
plt.xlabel("Price")
plt.ylabel("Rating")
plt.show()