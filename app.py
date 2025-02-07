import streamlit as st
import gspread
import pandas as pd
import requests
from oauth2client.service_account import ServiceAccountCredentials
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from skimage.metrics import structural_similarity as ssim
import cv2
import os
import time
import numpy as np

# Directories for temporary files
TEMP_DIR = "./temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)

def save_code_to_file(html_code, css_code, file_index):
    """Saves HTML and CSS code to local files."""
    html_path = os.path.join(TEMP_DIR, f"page_{file_index}.html")
    css_path = os.path.join(TEMP_DIR, f"style_{file_index}.css")

    with open(html_path, "w", encoding="utf-8") as html_file:
        html_file.write(f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Test Page</title>
            <link rel="stylesheet" href="style_{file_index}.css">
        </head>
        <body>
        {html_code}
        </body>
        </html>
        """)

    with open(css_path, "w", encoding="utf-8") as css_file:
        css_file.write(css_code)

    return html_path, css_path

def capture_screenshot_from_file(html_file, output_file, viewport_size):
    """Captures a screenshot of the HTML file."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument(f"--window-size={viewport_size['width']},{viewport_size['height']}")
    chrome_options.add_argument("--log-level=3")  # Suppress logs
    chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    try:
        driver.get(f"file:///{os.path.abspath(html_file)}")
        time.sleep(2)  # Wait for the page to load fully

        driver.execute_script("document.body.style.overflow = 'hidden';")  # Hide scrollbar

        # Get the total page height
        total_height = driver.execute_script("return document.body.scrollHeight")
        driver.set_window_size(viewport_size['width'], total_height)

        # Take the screenshot
        driver.save_screenshot(output_file)
        print(f"Screenshot saved: {output_file}")
    except Exception as e:
        print(f"Error capturing screenshot: {e}")
    finally:
        driver.quit()

def pixel_match(reference_path, test_path):
    """Compares images and returns pixel match percentage."""
    ref_image = cv2.imread(reference_path)
    test_image = cv2.imread(test_path)

    if ref_image is None:
        print(f"Error: Could not load reference image: {reference_path}")
        return 0.0
    if test_image is None:
        print(f"Error: Could not load test image: {test_path}")
        return 0.0

    # Resize test image to match reference image dimensions
    test_image = cv2.resize(test_image, (ref_image.shape[1], ref_image.shape[0]))

    # Convert to grayscale
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

    # Calculate Structural Similarity Index (SSIM)
    similarity_index, _ = ssim(ref_gray, test_gray, full=True)
    match_percentage = round(similarity_index * 100, 2)
    print(f"Match Percentage: {match_percentage}% for {reference_path} and {test_path}")
    return match_percentage

def process_google_sheet(sheet_url, credentials_path, local_excel_path):
    """Processes the Google Sheet to capture screenshots and compare images."""
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
    client = gspread.authorize(creds)

    sheet = client.open_by_url(sheet_url)
    worksheet = sheet.get_worksheet(0)
    data = worksheet.get_all_records()
    df = pd.DataFrame(data)

    for col in ['Desktop Match %', 'Mobile Match %']:
        if col not in df.columns:
            df[col] = None

    for index, row in df.iterrows():
        html_code = row.get('HTML Code', '').strip()
        css_code = row.get('CSS Code', '').strip()
        desktop_ref_image_url = row.get('Desktop Reference Image', '').strip()
        mobile_ref_image_url = row.get('Mobile Reference Image', '').strip()

        if not html_code or not css_code:
            print(f"Skipping row {index}: Missing HTML or CSS code.")
            continue
        if not desktop_ref_image_url or not mobile_ref_image_url:
            print(f"Skipping row {index}: Missing reference image URLs.")
            continue

        # Save HTML and CSS code to files
        html_file, css_file = save_code_to_file(html_code, css_code, index)

        # Paths for screenshots
        desktop_screenshot = os.path.join(TEMP_DIR, f"desktop_{index}.png")
        mobile_screenshot = os.path.join(TEMP_DIR, f"mobile_{index}.png")

        # Download reference images
        desktop_ref_image_path = os.path.join(TEMP_DIR, f"desktop_ref_{index}.png")
        mobile_ref_image_path = os.path.join(TEMP_DIR, f"mobile_ref_{index}.png")

        try:
            if not download_image(desktop_ref_image_url, desktop_ref_image_path):
                print(f"Skipping row {index}: Failed to download desktop reference image.")
                continue
            if not download_image(mobile_ref_image_url, mobile_ref_image_path):
                print(f"Skipping row {index}: Failed to download mobile reference image.")
                continue

            # Capture screenshots
            capture_screenshot_from_file(html_file, desktop_screenshot, {'width': 1920, 'height': 1080})
            capture_screenshot_from_file(html_file, mobile_screenshot, {'width': 420, 'height': 812})

            # Compare images
            desktop_match = pixel_match(desktop_ref_image_path, desktop_screenshot)
            mobile_match = pixel_match(mobile_ref_image_path, mobile_screenshot)

            # Update DataFrame
            df.at[index, 'Desktop Match %'] = desktop_match
            df.at[index, 'Mobile Match %'] = mobile_match
        except Exception as e:
            print(f"Error processing row {index}: {e}")

    # Update Google Sheet with results
    try:
        worksheet.update([df.columns.values.tolist()] + df.values.tolist())
        print("Google Sheet updated successfully.")
    except Exception as e:
        print(f"Error updating Google Sheet: {e}")

    # Save results locally
    df.to_excel(local_excel_path, index=False)
    print(f"Results saved to {local_excel_path}")

def download_image(url, output_path):
    """Downloads an image from a URL and saves it to the specified path."""
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(output_path, 'wb') as file:
                file.write(response.content)
            print(f"Image downloaded: {output_path}")
            return True
        else:
            print(f"Failed to download image from {url}. HTTP Status Code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return False

# Streamlit Interface
st.title("Google Sheet Image Comparison Web App")

sheet_url = st.text_input("Enter Google Sheet URL:")
credentials_path = r"C:\Users\umama\Downloads\WebPer\stunning-flight-450012-t8-05694950ecd2.json"
local_excel_path = st.text_input("Enter path to save the results (Excel file):", "results.xlsx")

if st.button("Process Sheet"):
    if not sheet_url or not credentials_path:
        st.error("Please provide both the Google Sheet URL and credentials file path.")
    else:
        try:
            process_google_sheet(sheet_url, credentials_path, local_excel_path)
            st.success(f"Processing complete. Results saved to {local_excel_path}.")
        except Exception as e:
            st.error(f"An error occurred: {e}")