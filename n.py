from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time

# --- ตั้งค่าตรงนี้ ---
FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSdhE56BkZzFRMiuEwrxWMe1ptGD_Pkv2uKYbqgL0-OlvGq-ew/viewform?pli=1&pli=1"
SUBMIT_XPATH = '//*[@id="mG61Hd"]/div[2]/div/div[3]/div[1]/div[1]/div'
# ------------------

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

try:
    # รอบแรก: เปิดมาเพื่อให้คุณ Login หรือเช็คความเรียบร้อย
    driver.get(FORM_URL)
    print("กรุณา Login และรอให้หน้าฟอร์มพร้อมส่ง (รอ 20 วินาที)...")
    time.sleep(20) 

    for i in range(1, 31):
        # ถ้าไม่ใช่รอบแรก ให้เปิดหน้าฟอร์มใหม่
        if i > 1:
            driver.get(FORM_URL)
            time.sleep(2) # รอหน้าเว็บโหลด

        # หาปุ่มส่งแล้วกดทันที
        submit_btn = driver.find_element(By.XPATH, SUBMIT_XPATH)
        submit_btn.click()

        print(f"ส่งรอบที่ {i} เรียบร้อย!")
        time.sleep(1.5) # พักแป๊บนึงป้องกันโดนบล็อก

finally:
    print("ทำงานครบ 30 รอบแล้ว")
    driver.quit()