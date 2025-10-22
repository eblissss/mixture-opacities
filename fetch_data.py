from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
import numpy as np
import time

def fill_form(driver, mixture, temp_min, temp_max, rho_min, rho_max, num_rho):
    print("Filling in data...")
    
    # Fill mixture field (e.g. 0.5 h 0.5 he)
    mixture_field = driver.find_element(By.ID, "user-spec-mix-value")
    mixture_field.clear()
    mixture_field.send_keys(mixture)
    
    # Set temperature range (keV)
    tlow_select = Select(driver.find_element(By.ID, "temperature-low"))
    tlow_select.select_by_value(temp_min)

    tup_select = Select(driver.find_element(By.ID, "temperature-high"))
    tup_select.select_by_value(temp_max)
    
    # Set density range (g/cm^3)
    driver.find_element(By.NAME, "rlow").clear()
    driver.find_element(By.NAME, "rlow").send_keys(rho_min)

    driver.find_element(By.NAME, "rup").clear()
    driver.find_element(By.NAME, "rup").send_keys(rho_max)

    driver.find_element(By.NAME, "nr").clear()
    driver.find_element(By.NAME, "nr").send_keys(num_rho)
    
    # Press submit
    driver.find_element(By.CSS_SELECTOR, "button[value='Submit']").click()
    print("Form submitted!")


def main():
    print("Fetching Data from LANL TOPS")
    
    # Start browser
    driver = webdriver.Chrome()
    
    try:
        # Navigate to site
        print("\nNavigating to https://aphysics2.lanl.gov/")
        driver.get("https://aphysics2.lanl.gov/")
        time.sleep(2)

        # Get random element mixture
        fractions = np.random.dirichlet([1, 1, 1])
        mixture = f"{fractions[0]} h {fractions[1]} he {fractions[2]} al"
        
        # Page 1: Fill form
        fill_form(driver, mixture, temp_min=2e-5, temp_max=50, rho_min=1e-5, rho_max=100, num_rho=20)
        time.sleep(3)

        # Page 2: Click button
        
        # Page 3: Extract data
        
        print(f"\nFetch Complete for mixture {mixture}")
        
    except Exception as e:
        print(f"\nError: {e}")
        
    finally:
        time.sleep(2)
        driver.quit()

if __name__ == "__main__":
    main()