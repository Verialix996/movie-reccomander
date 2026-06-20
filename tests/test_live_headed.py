"""
Live headed tests against the deployed Render app.
One browser window for all tests — uses Start over between sessions.
Takes a screenshot of the final recommendations for each test.
Usage: python tests/test_live_headed.py
"""
import os
import time
from playwright.sync_api import sync_playwright

LIVE_URL = "https://cyber-threat-prioritization-agent.onrender.com"
CHROME_PATH = "/usr/bin/google-chrome"
SLOW_MO = 700
SCREENSHOTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test results", "screenshots"))
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

TESTS = [
    {
        "name": "1_english_comedy",
        "label": "1 · English comedy",
        "answers": ["happy and relaxed", "comedy", "any era is fine", "I loved Home Alone", "light and fun"],
    },
    {
        "name": "2_hebrew_horror",
        "label": "2 · Hebrew horror",
        "answers": ["אני רוצה סרט מפחיד", "אימה ומתח", "לא משנה", "דלג", "עמוק ומפחיד"],
    },
    {
        "name": "3_90s_thriller",
        "label": "3 · 90s thriller + era filter",
        "answers": ["tense and on edge", "thriller suspense", "90s movies", "Hitchcock style", "deep and intense"],
    },
    {
        "name": "4_failure_nonsense",
        "label": "4 · FAILURE CASE — nonsense input",
        "answers": ["xzqwerty123!!", "!@#$%^", "ñoñoño", "???", "..."],
        "expect_failure": True,
    },
    {
        "name": "5_mixed_hebrew_english",
        "label": "5 · Mixed Hebrew + English",
        "answers": ["קומדיה", "comedy funny", "90s", "Seinfeld", "light"],
    },
]


# ── Helpers ─────────────────────────────────────────────────────────────────────

def wait_for_app(page):
    page.wait_for_selector("[data-testid='stChatMessage']", timeout=60000)


def send_answer(page, answer: str):
    inp = page.locator("[data-testid='stChatInputTextArea']")
    inp.wait_for(state="visible", timeout=20000)
    inp.click()
    inp.fill(answer)
    inp.press("Enter")


def get_message_count(page) -> int:
    return page.locator("[data-testid='stChatMessage']").count()


def wait_for_next_message(page, current_count: int, timeout: int = 90000):
    page.wait_for_function(
        "(count => document.querySelectorAll('[data-testid=\"stChatMessage\"]').length > count)",
        arg=current_count,
        timeout=timeout,
    )


def scroll_to_recommendations(page):
    """Scroll the Streamlit chat container so the Start over button is in view."""
    # Scroll the main Streamlit block container (inner scrollable div)
    page.evaluate("""
        const containers = document.querySelectorAll('[data-testid="stAppViewBlockContainer"], .main, section[data-testid="stMain"]');
        containers.forEach(el => { el.scrollTop = el.scrollHeight; });
        window.scrollTo(0, document.body.scrollHeight);
    """)
    time.sleep(0.5)
    # Also scroll the Start over button into view
    btn = page.get_by_role("button", name="Start over")
    btn.scroll_into_view_if_needed()
    time.sleep(0.5)


def take_screenshot(page, name: str):
    scroll_to_recommendations(page)
    time.sleep(0.5)
    path = os.path.join(SCREENSHOTS_DIR, f"{name}.png")
    page.screenshot(path=path)  # viewport screenshot after scrolling to recs
    print(f"  screenshot saved → {path}")


def print_recommendations(page):
    messages = page.locator("[data-testid='stChatMessage']").all()
    print("  --- Recommendations ---")
    found = False
    for msg in messages:
        text = msg.inner_text().strip()
        if any(m in text for m in ["Great answers", "Let me find"]):
            found = True
            continue
        if found and text:
            lines = [l.strip() for l in text.splitlines() if l.strip()][:4]
            print("  •", " | ".join(lines))
    print("  -----------------------")


def click_start_over(page):
    btn = page.get_by_role("button", name="Start over")
    btn.wait_for(state="visible", timeout=30000)
    btn.click()
    # Wait for chat to reset (back to just the first question)
    page.wait_for_function(
        "(n => document.querySelectorAll('[data-testid=\"stChatMessage\"]').length <= n)",
        arg=2,
        timeout=15000,
    )
    time.sleep(0.5)
    print("  chat reset via Start over.")


def wait_for_recommendations(page):
    """Wait until the Start over button appears — it only renders after all rec cards are on screen."""
    print("  waiting for recommendations to render...")
    page.get_by_role("button", name="Start over").wait_for(state="visible", timeout=120000)


def run_chat(page, answers: list[str]):
    wait_for_app(page)
    for i, answer in enumerate(answers):
        count_before = get_message_count(page)
        print(f"  answer {i+1}/{len(answers)}: {answer!r}")
        send_answer(page, answer)
        timeout = 90000 if i == len(answers) - 1 else 30000
        wait_for_next_message(page, count_before, timeout=timeout)
        time.sleep(0.6)
    # After last answer, wait for rec cards to fully render
    wait_for_recommendations(page)


# ── Main runner ──────────────────────────────────────────────────────────────────

def main():
    print(f"Target: {LIVE_URL}")
    print(f"Screenshots → {SCREENSHOTS_DIR}\n")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            executable_path=CHROME_PATH,
            headless=False,
            slow_mo=SLOW_MO,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        page = context.new_page()
        page.goto(LIVE_URL)

        for i, test in enumerate(TESTS):
            print(f"\n{'='*55}")
            print(f"TEST: {test['label']}")
            print(f"{'='*55}")

            try:
                run_chat(page, test["answers"])

                if test.get("expect_failure"):
                    body = page.locator("body").inner_text()
                    assert "Traceback" not in body, "App crashed with a Python traceback!"
                    print("  no crash — graceful fallback confirmed")

                print_recommendations(page)
                take_screenshot(page, test["name"])
                print(f"  PASS")

            except Exception as e:
                print(f"  FAIL — {e}")
                page.screenshot(path=os.path.join(SCREENSHOTS_DIR, f"{test['name']}_FAIL.png"), full_page=True)

            # Use Start over for all tests except the last
            if i < len(TESTS) - 1:
                time.sleep(2)
                click_start_over(page)

        print(f"\n{'='*55}")
        print("All tests done. Closing browser in 5s...")
        time.sleep(5)
        context.close()
        browser.close()


if __name__ == "__main__":
    main()
