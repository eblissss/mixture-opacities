from selenium import webdriver
import numpy as np
import time

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