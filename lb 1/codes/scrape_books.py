from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from pathlib import Path
import time
import csv
# import pandas as pd

driver_path = "C:/studies/uni/semester 4/Machine Learning/lab works/lb 1/chromedriver-win64/chromedriver.exe"
service = webdriver.ChromeService(executable_path=driver_path)
driver = webdriver.Chrome(service=service)

URL = "https://books.toscrape.com/"
CATALOGUE_URL = URL + "catalogue/"

def parse_book(book):
    title = book.find_element(By.XPATH, './/h3')
    rating = book.find_element(By.XPATH, './/p[contains(@class,"star-rating")]').get_attribute("class").split()[1]
    price = book.find_element(By.XPATH, './/p[@class="price_color"]')

    return [title.text, price.text, rating]

def scrape_books():
    driver.get(URL)
    results = [
        ["title", "price", "rating"],
    ]
    while len(results) <= 501:
        books = driver.find_elements(By.XPATH, '//article[@class="product_pod"]')
        for book in books:
            results.append(parse_book(book))
            if len(results) >= 501:
                break
        driver.find_element(By.XPATH, '//a[text()="next"]').click()

    driver.quit()
    return results

if __name__ == "__main__":
    data = scrape_books()
    path = 'books.csv'
    with open(path, 'w', newline='', encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(data)