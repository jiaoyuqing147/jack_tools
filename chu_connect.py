# -*- coding: utf-8 -*-
"""
chu_connect.py

功能：
    在服务器上用 selenium + chromedriver 自动登录学校 SRUN 校园网，
    然后访问 http://www.baidu.com 判断是否真正已联网。

使用方法：
    1. 确认 chromedriver 在 /usr/bin/chromedriver（你已经是这样）。
    2. 修改 USERNAME / PASSWORD 为你的学号和密码。
    3. 运行：python chu_connect.py

对了，记得在虚拟环境中装pip install selenium==4.15.2  #selenium 是唯一核心依赖。
系统依赖：Chrome/Chromium + chromedriver
"""

import time
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import sys

# ============= 需要你修改的两项 =============
USERNAME = "054059"
PASSWORD = "jiaoge147"
# ===========================================

LOGIN_PAGE = "http://210.45.92.67/srun_portal_pc?ac_id=3&theme=pro"
PORTAL_HOST = "210.45.92.67"          # 认证门户服务器
CHROMEDRIVER_PATH = "/usr/bin/chromedriver"

HEADLESS = True                       # 想看浏览器画面就改成 False
PAGELOAD_TIMEOUT = 8                  # 页面加载超时（秒）
TEST_URL = "http://www.baidu.com"     # 用来测试是否已联网的外网网址


def start_driver():
    options = Options()
    if HEADLESS:
        try:
            options.add_argument("--headless=new")
        except Exception:
            options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")

    service = Service(CHROMEDRIVER_PATH)
    try:
        driver = webdriver.Chrome(service=service, options=options)
    except WebDriverException as e:
        print("❌ 启动 ChromeDriver 失败：", e)
        sys.exit(2)

    # 防止页面一直卡住
    try:
        driver.set_page_load_timeout(PAGELOAD_TIMEOUT)
    except Exception:
        pass

    return driver


def find_and_fill(driver):
    """找到用户名/密码输入框并点击登录按钮"""
    wait = WebDriverWait(driver, 10)
    try:
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    except TimeoutException:
        print("⚠️ 登录页加载较慢，继续尝试查找输入框...")

    username_el = None
    password_el = None

    # 优先按常见 name/id 查找
    user_candidates = [
        (By.NAME, "DDDDD"),
        (By.NAME, "username"),
        (By.ID, "DDDDD"),
        (By.ID, "username"),
    ]
    pass_candidates = [
        (By.NAME, "upass"),
        (By.NAME, "password"),
        (By.ID, "upass"),
        (By.ID, "password"),
    ]

    for sel in user_candidates:
        try:
            username_el = driver.find_element(*sel)
            break
        except NoSuchElementException:
            continue

    for sel in pass_candidates:
        try:
            password_el = driver.find_element(*sel)
            break
        except NoSuchElementException:
            continue

    # 兜底：第一个 text / password 输入框
    if not username_el:
        try:
            username_el = driver.find_element(By.CSS_SELECTOR, "input[type='text']")
        except NoSuchElementException:
            pass

    if not password_el:
        try:
            password_el = driver.find_element(By.CSS_SELECTOR, "input[type='password']")
        except NoSuchElementException:
            pass

    if not username_el or not password_el:
        print("❌ 找不到用户名或密码输入框，可能页面结构变了，需要手动调整脚本。")
        return False

    # 填入账号密码
    username_el.clear()
    username_el.send_keys(USERNAME)
    password_el.clear()
    password_el.send_keys(PASSWORD)
    time.sleep(0.3)

    # 点击登录按钮（优先找带“登录”字样的按钮）
    btn_selectors = [
        (By.XPATH, "//button[contains(text(),'登录') or contains(text(),'登 录') or contains(text(),'登陆')]"),
        (By.CSS_SELECTOR, "button[type='submit']"),
        (By.CSS_SELECTOR, "input[type='submit']"),
        (By.CSS_SELECTOR, "button"),
    ]
    for sel in btn_selectors:
        try:
            btn = driver.find_element(*sel)
            if not btn.is_displayed():
                continue
            btn.click()
            print("✅ 已点击登录按钮（选择器：{}）".format(sel))
            return True
        except Exception:
            continue

    print("❌ 未找到可点击的登录按钮。")
    return False


def check_online(driver):
    """
    访问 TEST_URL，看是否还会被重定向回认证门户。
    返回 True/False，打印清晰提示。
    """
    print(f"🌐 正在访问测试网址：{TEST_URL} ...")
    try:
        driver.get(TEST_URL)
    except TimeoutException:
        print(f"⚠️ 访问 {TEST_URL} 超时（可能是站点慢 or 被屏蔽），继续根据当前 URL 判断是否已放行。")

    time.sleep(1)
    cur = driver.current_url
    host = urlparse(cur).netloc
    print("🔎 测试网址最终访问到：", cur)

    if PORTAL_HOST in host:
        print("❌ 测试网址被重定向回认证门户，说明服务器仍被拦截（未真正联网）。")
        return False
    else:
        print("✅ 测试网址没有被重定向回认证门户，说明外网访问已经放行，服务器已联网。")
        return True


def main():
    print("🚀 启动 webdriver ...")
    driver = start_driver()
    try:
        print("➡️ 打开登录页：", LOGIN_PAGE)
        driver.get(LOGIN_PAGE)

        clicked = find_and_fill(driver)
        if not clicked:
            print("⚠️ 未点击到登录按钮，先等几秒看看页面是否会自动认证...")
            time.sleep(5)
        else:
            time.sleep(3)

        cur = driver.current_url
        print("ℹ️ 登录后当前 URL:", cur)
        page_text = driver.page_source
        if "srun_portal_success" in cur or "网络准入认证" in page_text:
            print("✅ 页面显示为成功页，初步判断认证已通过。")
        else:
            print("⚠️ 没有明显看到成功页，但可能仍已登录，继续用外网测试判断。")

        ok = check_online(driver)
        if ok:
            print("\n🎉 最终结果：登录成功，服务器已联网 ✅")
        else:
            print("\n❌ 最终结果：登录流程执行了，但服务器仍未真正联网（可能是策略或绑定问题）。")

    except Exception as e:
        print("💥 运行过程中发生异常：", e)
    finally:
        try:
            driver.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
