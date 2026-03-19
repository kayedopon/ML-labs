from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import csv
import pandas as pd

driver_path = "C:/studies/uni/semester 4/Machine Learning/lab works/lb 1/chromedriver-win64/chromedriver.exe"
service = webdriver.ChromeService(executable_path=driver_path)
driver = webdriver.Chrome(service=service)
KAGGLE_URL = "https://www.kaggle.com/datasets?tags=12107-Computer+Science"


def parse_dataset(dataset):
    name = dataset.find_element(By.XPATH, './/div[@class="sc-kCuUfV sc-bjxVRI bXQAUF dEeLno"]')
    author = dataset.find_element(By.XPATH, './/a[@class="sc-kjwdDK fDGKWq"]')
    upvotes = dataset.find_element(By.XPATH, './/span[@class="sc-fVHBlr sc-kJNXNN gHmzEf rvJZo"]')
    try:
        usability = dataset.find_element(By.XPATH, './/span[@class="sc-kMemMU SkXLN"]')
    except:
        usability = dataset.find_element(By.XPATH, './/span[@class="sc-kMemMU duSZa-d"]')

    return [name.text, author.text, usability.text, upvotes.text]

def scrape_datasets():
    result = [
        ["Dataset name", "author", "usabilities", "upvotes"],
    ]

    for i in range(1, 26):
        driver.get(KAGGLE_URL + f"&page={i}")
        time.sleep(3)
        datasets = driver.find_elements(By.XPATH, '//div[@class="sc-eXVaYZ idRORJ km-listitem--large"]')
        for dataset in datasets:
            result.append(parse_dataset(dataset))

    driver.quit()
    return result

if __name__ == "__main__":
    path = 'datasets.csv'
    data = scrape_datasets()
    with open(path, 'w', newline='', encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(data)

    