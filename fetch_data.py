from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import csv
import os
import numpy as np
import time

NUM_RUNS = 1
OUTPUT_FILE = "opacity_data.csv"
USE_CHROME_DEV = True
CHROME_VERSION = "143.0.7475.7"


def fill_form(driver, mixture, temp_min, temp_max, rho_min, rho_max, num_rho):
    # Fill mixture field (e.g. 0.5 h 0.5 he)
    mixture_field = driver.find_element(By.ID, "user-spec-mix-value")
    mixture_field.clear()
    mixture_field.send_keys(mixture)

    # Set temperature range (keV)
    tlow_select = Select(driver.find_element(By.ID, "temperature-low"))
    tlow_select.select_by_visible_text(f" {temp_min}")

    tup_select = Select(driver.find_element(By.ID, "temperature-high"))
    tup_select.select_by_visible_text(f" {temp_max}")

    # Set density range (g/cm^3)
    driver.find_element(By.NAME, "rlow").clear()
    driver.find_element(By.NAME, "rlow").send_keys(rho_min)

    driver.find_element(By.NAME, "rup").clear()
    driver.find_element(By.NAME, "rup").send_keys(rho_max)

    driver.find_element(By.NAME, "nr").clear()
    driver.find_element(By.NAME, "nr").send_keys(num_rho)

    # Press submit
    driver.find_element(By.CSS_SELECTOR, "button[value='Submit']").click()


def click_column_data(driver):
    driver.find_element(
        By.XPATH, "//input[./*[contains(text(), 'Numerical by Columns')]]"
    ).click()


def extract_to_csv(driver, filename, mix_fractions):
    # Get the output text content
    content = driver.find_element(By.CSS_SELECTOR, "code.text-muted").get_attribute(
        "innerHTML"
    )
    lines = [line.strip() for line in content.split("<br>")]

    data = []
    temp = None

    # Parse the data
    for line in lines:
        if not line:
            continue

        # Update the temperature
        if "T= " in line:
            parts = line.split()
            for i in range(len(parts)):
                if parts[i] == "T=" and i + 1 < len(parts):
                    temp = parts[i + 1]
                    break
            continue

        # Get data lines
        if line[0].isdigit() and temp:
            parts = line.split()
            density = parts[0]
            rosseland = parts[1]
            planck = parts[2]

            data.append(
                [
                    mix_fractions[0],
                    mix_fractions[1],
                    mix_fractions[2],
                    temp,
                    density,
                    rosseland,
                    planck,
                ]
            )

    # Update CSV file
    mode = "a" if os.path.exists(filename) else "w"
    with open(filename, mode, newline="") as file:
        writer = csv.writer(file)
        if mode == "w":
            # Set CSV headers
            writer.writerow(
                [
                    "Mix_H",
                    "Mix_He",
                    "Mix_Al",
                    "Temperature",
                    "Density",
                    "Rosseland_opacity",
                    "Planck_opacity",
                ]
            )
        writer.writerows(data)

    return len(data)


def main():
    print(f"=== Fetching Data from LANL TOPS ===")

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    # Start browser

    chrome_options = Options()
    if USE_CHROME_DEV:
        chrome_options.binary_location = (
            "/Applications/Google Chrome Dev.app/Contents/MacOS/Google Chrome Dev"
        )

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager(driver_version=CHROME_VERSION).install()),
        options=chrome_options,
    )

    try:
        for run in range(1, NUM_RUNS + 1):
            print(f"\n Fetching run {run}/{NUM_RUNS}")

            # Navigate to site
            print("\tNavigating to https://aphysics2.lanl.gov/")
            driver.get("https://aphysics2.lanl.gov/")
            time.sleep(3)

            # Get random element mixture
            fractions = np.random.dirichlet([1, 1, 1])
            mixture = f"{fractions[0]} h {fractions[1]} he {fractions[2]} al"

            # Page 1: Fill form (make sure options are valid on site)
            print("\tFilling in form")
            fill_form(
                driver,
                mixture,
                temp_min="0.0005",
                temp_max="40",
                rho_min="0.00001",
                rho_max="100.",
                num_rho="20",
            )
            print("\tForm submitted!")
            time.sleep(3)

            # Page 2: Click button
            click_column_data(driver)
            print("\tColumn data selected")
            time.sleep(3)

            # Page 3: Extract data
            print("\tExtracting column data")
            num_extracted = extract_to_csv(driver, OUTPUT_FILE, fractions)
            print(f"\tExtracted {num_extracted} data points!")

            print(f"Fetch complete for mixture {mixture}")

        print(f"All runs complete! Check {OUTPUT_FILE}")

    except Exception as e:
        print(f"\nError (run {run}): {e}")

    finally:
        time.sleep(2)
        driver.quit()


if __name__ == "__main__":
    main()
