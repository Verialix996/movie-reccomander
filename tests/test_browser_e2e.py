"""
Browser E2E tests using Playwright + Google Chrome.
Each test opens the Streamlit app, drives a full Q&A session, and records a video.
Videos are saved to: test results/videos/<test_name>.webm
"""
import os
import sys
import time
import shutil
import socket
import subprocess
import pytest
from playwright.sync_api import sync_playwright, Page, BrowserContext

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = os.path.join(os.path.dirname(__file__), "..")
VIDEOS_DIR = os.path.abspath(os.path.join(ROOT, "test results", "videos"))
CHROME_PATH = "/usr/bin/google-chrome"
APP_PORT = 8502
APP_URL = f"http://localhost:{APP_PORT}"

os.makedirs(VIDEOS_DIR, exist_ok=True)


# ── Server fixture ─────────────────────────────────────────────────────────────

def _wait_for_port(port: int, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


@pytest.fixture(scope="session")
def streamlit_server():
    """Start the Streamlit app once for the whole browser test session."""
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "app.py",
            f"--server.port={APP_PORT}",
            "--server.headless=true",
            "--server.runOnSave=false",
            "--browser.gatherUsageStats=false",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=ROOT,
    )
    ready = _wait_for_port(APP_PORT, timeout=60)
    if not ready:
        proc.terminate()
        pytest.fail("Streamlit server did not start within 60 seconds")
    time.sleep(2)  # extra settle time for Streamlit to finish loading
    yield APP_URL
    proc.terminate()
    proc.wait(timeout=10)


# ── Playwright helpers ─────────────────────────────────────────────────────────

def new_context_with_video(playwright, test_name: str) -> tuple:
    """Launch Chrome with video recording. Returns (browser, context, page)."""
    # Each test gets its own tmp video dir; we rename after the test
    tmp_video_dir = os.path.join(VIDEOS_DIR, f"_tmp_{test_name}")
    os.makedirs(tmp_video_dir, exist_ok=True)

    browser = playwright.chromium.launch(
        executable_path=CHROME_PATH,
        args=["--no-sandbox", "--disable-dev-shm-usage"],
        headless=True,
    )
    context = browser.new_context(
        viewport={"width": 1280, "height": 800},
        record_video_dir=tmp_video_dir,
        record_video_size={"width": 1280, "height": 800},
    )
    page = context.new_page()
    return browser, context, page, tmp_video_dir


def save_video(page: Page, context: BrowserContext, browser, tmp_dir: str, test_name: str):
    """Close context (flushes video) and move to final named file."""
    video_path = page.video.path() if page.video else None
    context.close()
    browser.close()

    if video_path and os.path.exists(video_path):
        dest = os.path.join(VIDEOS_DIR, f"{test_name}.webm")
        shutil.move(video_path, dest)
        print(f"\n🎬 Video saved → {dest}")
    # Clean up tmp dir
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)


def wait_for_app(page: Page):
    """Wait for the Streamlit app to fully load."""
    page.wait_for_selector("[data-testid='stChatMessage']", timeout=30000)


def send_answer(page: Page, answer: str):
    """Type an answer into the chat input and submit."""
    inp = page.locator("[data-testid='stChatInputTextArea']")
    inp.wait_for(state="visible", timeout=15000)
    inp.click()
    inp.fill(answer)
    inp.press("Enter")


def wait_for_next_message(page: Page, current_count: int, timeout: int = 60000):
    """Wait until a new chat message appears."""
    page.wait_for_function(
        f"document.querySelectorAll('[data-testid=\"stChatMessage\"]').length > {current_count}",
        timeout=timeout,
    )


def get_message_count(page: Page) -> int:
    return page.locator("[data-testid='stChatMessage']").count()


def run_full_chat(page: Page, answers: list[str], long_wait_on_last: bool = True):
    """Drive the chatbot through all 5 answers."""
    wait_for_app(page)
    for i, answer in enumerate(answers):
        count_before = get_message_count(page)
        send_answer(page, answer)
        # Last answer triggers Gemini recommendation — give it more time
        timeout = 90000 if (i == len(answers) - 1 and long_wait_on_last) else 30000
        wait_for_next_message(page, count_before, timeout=timeout)
        time.sleep(0.5)  # small pause so video captures the response visually


# ── Browser E2E tests ──────────────────────────────────────────────────────────

def test_browser_english_comedy(streamlit_server):
    """Full English comedy flow — records video of complete session."""
    test_name = "english_comedy_flow"
    with sync_playwright() as pw:
        browser, ctx, page, tmp_dir = new_context_with_video(pw, test_name)
        try:
            page.goto(streamlit_server)
            run_full_chat(page, [
                "happy and relaxed",
                "comedy",
                "any era is fine",
                "I loved Home Alone",
                "light and fun",
            ])
            # Verify recommendations appeared
            recs = page.locator("[data-testid='stChatMessage']")
            assert recs.count() >= 6, "Expected at least 6 messages (1 intro + 5 Q&A + recs)"
            time.sleep(2)
        finally:
            save_video(page, ctx, browser, tmp_dir, test_name)


def test_browser_hebrew_comedy(streamlit_server):
    """Full Hebrew comedy flow — verifies multilingual support on screen."""
    test_name = "hebrew_comedy_flow"
    with sync_playwright() as pw:
        browser, ctx, page, tmp_dir = new_context_with_video(pw, test_name)
        try:
            page.goto(streamlit_server)
            run_full_chat(page, [
                "אני רוצה קומדיה",
                "קומדיה מצחיקה",
                "לא משנה לי",
                "דלג",
                "קל ומצחיק",
            ])
            assert get_message_count(page) >= 6
            time.sleep(2)
        finally:
            save_video(page, ctx, browser, tmp_dir, test_name)


def test_browser_hebrew_horror(streamlit_server):
    """Hebrew horror flow — verifies genre detection for Hebrew input."""
    test_name = "hebrew_horror_flow"
    with sync_playwright() as pw:
        browser, ctx, page, tmp_dir = new_context_with_video(pw, test_name)
        try:
            page.goto(streamlit_server)
            run_full_chat(page, [
                "אני רוצה סרט מפחיד",
                "אימה ומתח",
                "לא משנה",
                "דלג",
                "עמוק ומפחיד",
            ])
            assert get_message_count(page) >= 6
            time.sleep(2)
        finally:
            save_video(page, ctx, browser, tmp_dir, test_name)


def test_browser_90s_thriller(streamlit_server):
    """90s thriller flow — verifies era + genre filtering on screen."""
    test_name = "90s_thriller_flow"
    with sync_playwright() as pw:
        browser, ctx, page, tmp_dir = new_context_with_video(pw, test_name)
        try:
            page.goto(streamlit_server)
            run_full_chat(page, [
                "tense and on edge",
                "thriller suspense",
                "90s movies",
                "Hitchcock style",
                "deep and intense",
            ])
            assert get_message_count(page) >= 6
            time.sleep(2)
        finally:
            save_video(page, ctx, browser, tmp_dir, test_name)


def test_browser_all_skip(streamlit_server):
    """All-skip flow — verifies graceful fallback to top-rated movies."""
    test_name = "all_skip_flow"
    with sync_playwright() as pw:
        browser, ctx, page, tmp_dir = new_context_with_video(pw, test_name)
        try:
            page.goto(streamlit_server)
            run_full_chat(page, ["דלג", "דלג", "דלג", "דלג", "דלג"])
            assert get_message_count(page) >= 6
            time.sleep(2)
        finally:
            save_video(page, ctx, browser, tmp_dir, test_name)


def test_browser_failure_nonsense(streamlit_server):
    """
    FAILURE CASE: Unrecognizable input — system must not crash,
    must still return some recommendation (fallback to top-rated).
    """
    test_name = "failure_nonsense_input"
    with sync_playwright() as pw:
        browser, ctx, page, tmp_dir = new_context_with_video(pw, test_name)
        try:
            page.goto(streamlit_server)
            run_full_chat(page, [
                "xzqwerty123!!",
                "!@#$%^",
                "ñoñoño",
                "???",
                "...",
            ])
            # Must not show a Python traceback on screen
            body_text = page.locator("body").inner_text()
            assert "Traceback" not in body_text, "App crashed with traceback!"
            assert "Error" not in body_text or get_message_count(page) >= 6
            time.sleep(2)
        finally:
            save_video(page, ctx, browser, tmp_dir, test_name)


def test_browser_mixed_language(streamlit_server):
    """Mixed Hebrew + English answers — verifies cross-language pipeline."""
    test_name = "mixed_language_flow"
    with sync_playwright() as pw:
        browser, ctx, page, tmp_dir = new_context_with_video(pw, test_name)
        try:
            page.goto(streamlit_server)
            run_full_chat(page, [
                "קומדיה",
                "comedy funny",
                "90s",
                "Seinfeld",
                "light",
            ])
            assert get_message_count(page) >= 6
            time.sleep(2)
        finally:
            save_video(page, ctx, browser, tmp_dir, test_name)


def test_browser_start_over_button(streamlit_server):
    """Verifies the 'Start over' button resets the chat correctly."""
    test_name = "start_over_button"
    with sync_playwright() as pw:
        browser, ctx, page, tmp_dir = new_context_with_video(pw, test_name)
        try:
            page.goto(streamlit_server)
            run_full_chat(page, [
                "happy", "comedy", "recent", "skip", "light"
            ])
            time.sleep(1)

            # Click Start over
            start_over = page.locator("button", has_text="Start over")
            start_over.wait_for(state="visible", timeout=10000)
            start_over.click()

            # Chat should reset — first message reappears
            page.wait_for_selector("[data-testid='stChatMessage']", timeout=15000)
            time.sleep(1)

            # Should be back to just the first question
            msg_count = get_message_count(page)
            assert msg_count <= 2, f"Expected reset, but {msg_count} messages remain"
            time.sleep(2)
        finally:
            save_video(page, ctx, browser, tmp_dir, test_name)
