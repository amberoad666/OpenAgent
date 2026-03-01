"""Automated test: make OpenAgent build a tech store website.

Runs OpenAgent programmatically in yolo mode, sends a task prompt,
then after each round checks the generated files for quality.
Gives specific feedback to fix problems and loops until the site is good.

Usage:
    cd <Qwen-Code dir>
    set OPEN_AGENT_DEBUG=1
    python test_techstore.py
"""
import os
import sys
import re
import shutil
import time
import py_compile
import tempfile

# ---------------------------------------------------------------------------
# Setup: work in a temp directory so we don't clobber anything
# ---------------------------------------------------------------------------

WORK_DIR = os.path.join(tempfile.gettempdir(), "openagent_techstore_test")

def setup_workdir():
    """Create (or clean) a fresh working directory for the test."""
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR, ignore_errors=True)
    os.makedirs(WORK_DIR, exist_ok=True)
    os.chdir(WORK_DIR)
    print(f"[test] Working directory: {WORK_DIR}")


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------

class QualityReport:
    """Collect issues found in the generated site."""

    def __init__(self):
        self.errors = []    # Must fix — site is broken
        self.warnings = []  # Should fix — bad practice / ugly

    def error(self, msg):
        self.errors.append(msg)
        print(f"  [ERROR] {msg}")

    def warn(self, msg):
        self.warnings.append(msg)
        print(f"  [WARN]  {msg}")

    @property
    def ok(self):
        return not self.errors

    def summary(self):
        lines = []
        if self.errors:
            lines.append(f"ERRORS ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"  - {e}")
        if self.warnings:
            lines.append(f"WARNINGS ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"  - {w}")
        if not self.errors and not self.warnings:
            lines.append("ALL CHECKS PASSED")
        return "\n".join(lines)


def find_files():
    """Find all generated files in the work directory."""
    result = {}
    for root, dirs, files in os.walk(WORK_DIR):
        # Skip __pycache__, node_modules, .git
        dirs[:] = [d for d in dirs if d not in {"__pycache__", "node_modules", ".git"}]
        for f in files:
            fpath = os.path.join(root, f)
            relpath = os.path.relpath(fpath, WORK_DIR).replace("\\", "/")
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                    result[relpath] = fh.read()
            except Exception:
                pass
    return result


def check_quality(files):
    """Run all quality checks on the generated files. Returns QualityReport."""
    report = QualityReport()

    # --- 1. Must have an HTML file ---
    all_html = {k: v for k, v in files.items() if k.endswith(".html")}
    if not all_html:
        report.error("No HTML file found")
        return report

    # If both static/ and templates/ HTML exist, only check the one Flask serves
    html_files = {}
    if any('templates/' in k for k in all_html):
        # Flask uses render_template -> check templates/ version
        html_files = {k: v for k, v in all_html.items() if 'templates/' in k}
    else:
        html_files = all_html

    # --- 2. Must have a Python backend ---
    py_files = {k: v for k, v in files.items() if k.endswith(".py")}
    if not py_files:
        report.error("No Python backend file found")
        return report

    # --- 3. Python syntax check ---
    for path, content in py_files.items():
        full = os.path.join(WORK_DIR, path)
        try:
            py_compile.compile(full, doraise=True)
        except py_compile.PyCompileError as e:
            report.error(f"Python syntax error in {path}: {e}")

    # --- 4. Python duplicate detection ---
    for path, content in py_files.items():
        # Multiple Flask() initializations
        flask_inits = len(re.findall(r'Flask\s*\(', content))
        if flask_inits > 1:
            report.error(f"{path}: Flask() called {flask_inits} times (should be 1)")

        # Duplicate route definitions
        routes = re.findall(r"@\w+\.route\(['\"]([^'\"]+)['\"]", content)
        seen_routes = {}
        for r in routes:
            seen_routes[r] = seen_routes.get(r, 0) + 1
        for r, count in seen_routes.items():
            if count > 1:
                report.error(f"{path}: route '{r}' defined {count} times")

        # Duplicate function definitions
        funcs = re.findall(r'^def\s+(\w+)\s*\(', content, re.MULTILINE)
        seen_funcs = {}
        for f in funcs:
            seen_funcs[f] = seen_funcs.get(f, 0) + 1
        for f, count in seen_funcs.items():
            if count > 1:
                report.error(f"{path}: function '{f}' defined {count} times")

    # --- 5. HTML quality checks ---
    for path, html in html_files.items():
        html_lower = html.lower()

        # Viewport meta tag
        if '<meta' not in html_lower or 'viewport' not in html_lower:
            report.error(f"{path}: missing <meta name='viewport'> tag")

        # DOCTYPE
        if '<!doctype' not in html_lower:
            report.warn(f"{path}: missing <!DOCTYPE html>")

        # Tailwind CDN — must be a valid <script> tag, NOT <link>
        if 'tailwindcss' not in html_lower and 'tailwind' not in html_lower:
            report.error(f"{path}: not using Tailwind CSS")
        elif re.search(r'<link[^>]*tailwindcss[^>]*>', html, re.IGNORECASE):
            report.error(f"{path}: Tailwind CDN loaded as <link> stylesheet — it must be "
                         '<script src="https://cdn.tailwindcss.com"></script>. '
                         "Tailwind CDN is a JavaScript file, not CSS!")
        else:
            # Verify the script tag is valid (not broken with ... or missing closing tag)
            tw_script = re.search(r'<script[^>]*tailwindcss[^>]*>', html, re.IGNORECASE)
            if tw_script:
                tag = tw_script.group(0)
                if '...' in tag:
                    report.error(f"{path}: Tailwind script tag is broken: {tag!r}. "
                                 'Must be exactly: <script src="https://cdn.tailwindcss.com"></script>')
                elif '</script>' not in html[tw_script.start():tw_script.start()+200]:
                    report.error(f"{path}: Tailwind <script> tag is not closed with </script>")

        # Has product cards (grid layout)
        if 'grid' not in html_lower:
            report.warn(f"{path}: no grid layout for product cards")

        # Has images or image placeholders
        if '<img' not in html_lower and 'image' not in html_lower:
            report.warn(f"{path}: no product images")

        # Has prices (dollar/ruble sign or 'price' in text)
        if 'price' not in html_lower and '$' not in html and '₽' not in html and 'руб' not in html_lower:
            report.warn(f"{path}: no visible prices")

        # Dark theme check: body should have dark background
        # Look for bg-gray-900, bg-gray-950, bg-slate-900 etc on body or root
        if not re.search(r'bg-(?:gray|slate|zinc|neutral)-(?:900|950)', html):
            report.warn(f"{path}: no dark background (expected bg-gray-900/950)")

        # TEXT CONTRAST: body with light text + white cards needs text color override
        body_has_light_text = bool(re.search(r'<body[^>]*class="[^"]*text-(?:white|gray-(?:100|200|300))', html))
        has_light_cards = bool(re.search(r'bg-white|bg-gray-(?:50|100|200)', html))
        if has_light_cards and body_has_light_text:
            # Check if text-gray-800/900 appears near card content (within 500 chars of bg-white)
            has_dark_text_on_cards = False
            for m in re.finditer(r'bg-white', html):
                after = html[m.start():m.start()+500]
                if re.search(r'text-(?:gray|slate|zinc)-(?:[7-9]00)', after):
                    has_dark_text_on_cards = True
                    break
            if not has_dark_text_on_cards:
                report.error(f"{path}: body has light text and cards have bg-white but no dark text color. "
                             "Add text-gray-800 to card elements.")

        # Gradient text should be on h1/span, not on header/section
        gradient_text_patterns = re.findall(r'(bg-gradient-to-[a-z]+\s+from-\w+\s+to-\w+[^>]*bg-clip-text[^>]*text-transparent)', html)
        if not gradient_text_patterns:
            # Also check reverse order
            gradient_text_patterns = re.findall(r'(text-transparent[^>]*bg-clip-text[^>]*bg-gradient)', html)
        # If gradient text classes exist, they should be on inline elements
        for gp in gradient_text_patterns:
            # Find the tag that has this class
            for m in re.finditer(re.escape(gp), html):
                # Get the tag start
                tag_start = html.rfind('<', 0, m.start())
                if tag_start >= 0:
                    tag = html[tag_start:m.start()]
                    tag_name = re.match(r'<(\w+)', tag)
                    if tag_name and tag_name.group(1).lower() in ('header', 'section', 'div', 'nav', 'footer', 'main'):
                        report.warn(f"{path}: gradient text classes (bg-clip-text text-transparent) are on a <{tag_name.group(1)}> "
                                    "— they should be on <h1>, <span>, or <p> for text to be visible")

        # Template syntax in plain HTML (not processed by server)
        if '{{' in html and '}}' in html:
            report.warn(f"{path}: contains template syntax ({{{{ }}}}) that won't work in plain HTML. "
                        "Use JavaScript to render dynamic data, not template tags.")

        # Conflicting inline CSS with Tailwind classes
        if '<style>' in html_lower:
            # Check for body background-color that conflicts with Tailwind
            if re.search(r'body\s*\{[^}]*background-color', html):
                report.error(f"{path}: has <style> block with body background-color that overrides Tailwind. "
                             "Remove the entire <style>...</style> block. Use ONLY Tailwind utility classes.")

        # Must have a fetch/XHR call to the backend OR inline product data
        has_fetch = 'fetch(' in html or 'XMLHttpRequest' in html or 'axios' in html
        has_inline_data = bool(re.search(r'(?:products|items)\s*=\s*\[', html))
        if not has_fetch and not has_inline_data:
            report.warn(f"{path}: no fetch() call to backend and no inline product data")

        # Fetch error handling
        if has_fetch and '.catch' not in html and 'catch' not in html:
            report.warn(f"{path}: fetch() without error handling (.catch or try/catch)")

        # Gradient text bugs:
        # 1. bg-clip-text + text-transparent WITHOUT bg-gradient -> invisible text
        # 2. text-transparent + bg-gradient WITHOUT bg-clip-text -> invisible text (gradient is background, not text)
        for m in re.finditer(r'class="([^"]*text-transparent[^"]*)"', html):
            classes = m.group(1)
            has_clip = 'bg-clip-text' in classes
            has_gradient = 'bg-gradient' in classes
            if has_clip and not has_gradient:
                report.error(f"{path}: element has bg-clip-text + text-transparent but NO bg-gradient. "
                             "Text will be invisible! Add bg-gradient-to-r from-purple-400 to-blue-400")
                break
            if has_gradient and not has_clip:
                report.error(f"{path}: element has text-transparent + bg-gradient but NO bg-clip-text. "
                             "Text will be invisible! Add bg-clip-text to make gradient apply to text.")
                break
            if not has_clip and not has_gradient:
                report.error(f"{path}: element has text-transparent but no gradient classes. "
                             "Text will be invisible! Either remove text-transparent or add "
                             "bg-gradient-to-r bg-clip-text for gradient text effect.")
                break

        # Fetch must render products to DOM, not just console.log
        if has_fetch:
            fetch_idx = html.index('fetch(')
            fetch_block = html[fetch_idx:fetch_idx+1000]
            renders_dom = any(kw in fetch_block for kw in
                              ['innerHTML', 'appendChild', 'insertAdjacentHTML', 'createElement', 'textContent'])
            if not renders_dom:
                report.error(f"{path}: fetch() does not render products to the page (only logs to console). "
                             "Use innerHTML or appendChild to create product cards dynamically.")

        # Valid HTML structure
        if '<html' not in html_lower:
            report.warn(f"{path}: missing <html> tag")
        if '<head' not in html_lower:
            report.warn(f"{path}: missing <head> tag")

        # Has buttons (add to cart, buy, etc)
        if '<button' not in html_lower:
            report.warn(f"{path}: no buttons (expected add-to-cart or buy buttons)")

        # Has hover effects
        if 'hover:' not in html:
            report.warn(f"{path}: no hover effects (expected hover: Tailwind classes)")

        # Responsive design
        if 'md:' not in html and 'lg:' not in html:
            report.warn(f"{path}: not responsive (no md: or lg: breakpoints)")

    # --- 5b. Body must use Tailwind dark classes ---
    for path, html in html_files.items():
        body_tag = re.search(r'<body[^>]*class="([^"]*)"', html)
        if body_tag:
            classes = body_tag.group(1)
            if 'bg-gray-9' not in classes and 'bg-slate-9' not in classes and 'bg-zinc-9' not in classes:
                report.warn(f"{path}: <body> missing dark background Tailwind class (e.g. bg-gray-950)")

    # --- 5c. Backend prices must be numeric ---
    for path, content in py_files.items():
        # Look for price values that are strings instead of numbers
        string_prices = re.findall(r'"price"\s*:\s*"[^"]*"', content)
        if string_prices:
            report.warn(f"{path}: product prices are strings ({string_prices[0]}) — should be numbers (e.g. \"price\": 39900)")

    # --- 5d. Backend serves index.html correctly ---
    for path, content in py_files.items():
        if 'render_template' in content:
            # render_template looks in templates/ dir, but our HTML is in static/
            if not os.path.isdir(os.path.join(WORK_DIR, "templates")):
                # Check if the referenced template exists
                templates = re.findall(r"render_template\(['\"]([^'\"]+)['\"]", content)
                for t in templates:
                    tpath = os.path.join(WORK_DIR, "templates", t)
                    if not os.path.exists(tpath):
                        report.error(f"{path}: uses render_template('{t}') but templates/{t} doesn't exist. "
                                     "Either move the HTML to templates/ or use send_from_directory('static', 'index.html')")

    # --- 5e. Product names should be descriptive, not generic ---
    for path, content in py_files.items():
        generic = re.findall(r'"name"\s*:\s*"(Товар \d+|Product \d+|Item \d+)"', content)
        if len(generic) >= 3:
            report.warn(f"{path}: products have generic names ({generic[0]}, {generic[1]}...). "
                        "Use real product names like 'iPhone 15', 'MacBook Pro', etc.")

    # --- 6. Backend must have product data or API ---
    backend_has_products = False
    for path, content in py_files.items():
        if 'product' in content.lower() or 'товар' in content.lower():
            backend_has_products = True
            break
    if not backend_has_products:
        report.error("Backend has no product data or product-related code")

    # --- 7. Backend must serve static files or have API routes ---
    backend_has_routes = False
    for path, content in py_files.items():
        if '@' in content and 'route' in content:
            backend_has_routes = True
            break
    if not backend_has_routes:
        report.error("Backend has no routes defined")

    # --- 8. requirements.txt ---
    if "requirements.txt" in files:
        req = files["requirements.txt"].lower()
        if "flask" not in req:
            report.warn("requirements.txt missing 'flask'")
    else:
        report.warn("No requirements.txt file")

    return report


# ---------------------------------------------------------------------------
# Run OpenAgent programmatically
# ---------------------------------------------------------------------------

def run_openagent_turn(client, messages, max_tokens):
    """Run one conversation turn (model generates + executes tools)."""
    from main import run_conversation_turn
    return run_conversation_turn(client, messages, max_tokens=max_tokens)


def build_feedback_prompt(report, round_num):
    """Build a feedback message from quality report."""
    if report.ok and not report.warnings:
        return None  # All good!

    parts = []
    if report.errors:
        parts.append("CRITICAL ISSUES that MUST be fixed:")
        for e in report.errors:
            parts.append(f"  - {e}")
        parts.append("")

    if report.warnings:
        parts.append("Issues to improve:")
        for w in report.warnings:
            parts.append(f"  - {w}")
        parts.append("")

    parts.append(
        "Fix ALL critical issues listed above. "
        "If you need to fix the HTML file, use write_file to REWRITE it completely — "
        "do NOT try edit_file for HTML, rewrite the whole file. "
        "Make sure the file is COMPLETE and valid."
    )

    return "\n".join(parts)


TASK_PROMPT = """Создай сайт магазина техники с бекендом на Python (Flask).

Нужно создать 3 файла: app.py, templates/index.html, requirements.txt

=== app.py ===
Flask сервер с:
- products = список из 6+ товаров с реальными названиями (iPhone, MacBook, AirPods, iPad, Apple Watch, HomePod), price как ЧИСЛО (не строка): {"id": 1, "name": "iPhone 15", "price": 89900, "image": "https://via.placeholder.com/300"}
- GET /api/products -> jsonify(products)
- GET / -> render_template('index.html')
- app.run(debug=True)

=== templates/index.html ===
ВАЖНО: Tailwind подключается ТОЛЬКО так:
<script src="https://cdn.tailwindcss.com"></script>
НЕ <link>! Это JavaScript файл, не CSS!

НЕ использовать <style> блоки — ТОЛЬКО Tailwind utility классы!

Структура:
- <body class="bg-gray-950 text-gray-100 min-h-screen">
- <header> с gradient: from-purple-600 to-blue-500
- <h1> с gradient text: ОБЯЗАТЕЛЬНО все 4 класса вместе: bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent
- <main> с grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6
- Карточки: bg-white text-gray-800 rounded-2xl shadow-lg p-6 hover:scale-105 transition-transform
- Цена: text-2xl font-bold text-purple-600
- Кнопка "В корзину": bg-purple-600 text-white rounded-lg hover:bg-purple-700
- JavaScript: fetch('/api/products').then().catch(error => ...)
- <meta name="viewport" content="width=device-width, initial-scale=1.0">

=== requirements.txt ===
Flask

Начинай СРАЗУ с write_file для app.py, потом static/index.html, потом requirements.txt. Создай каждый файл ОДИН раз."""


def main():
    setup_workdir()

    # Import OpenAgent internals
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace(
        WORK_DIR, r"C:\Users\User\source\repos\Qwen-Code"
    ))
    # Ensure we import from the right place
    qwen_dir = r"C:\Users\User\source\repos\Qwen-Code"
    if qwen_dir not in sys.path:
        sys.path.insert(0, qwen_dir)

    # Monkey-patch UI functions that need user input
    import ui
    ui.show_dpo_prompt = lambda *a, **kw: False
    ui.confirm_action = lambda *a, **kw: (True, None)

    import executor
    executor.skip_permissions = True

    from client import LMClient
    from config import SYSTEM_PROMPT, MAX_CONTEXT_TOKENS

    client = LMClient()

    # Pick the first available model
    models = client.list_models()
    if not models:
        print("[test] ERROR: No models loaded in LM Studio!")
        sys.exit(1)

    model_key, model_ctx = models[0]
    client.model = model_key
    max_tokens = model_ctx if model_ctx else MAX_CONTEXT_TOKENS
    print(f"[test] Using model: {model_key} (ctx: {model_ctx})")

    # Build system prompt
    system_prompt = SYSTEM_PROMPT.format(
        cwd=WORK_DIR,
        platform="Windows",
        del_cmd="del",
    )
    messages = [{"role": "system", "content": system_prompt}]

    MAX_ROUNDS = 5
    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"\n{'='*60}")
        print(f"[test] === ROUND {round_num}/{MAX_ROUNDS} ===")
        print(f"{'='*60}")

        if round_num == 1:
            messages.append({"role": "user", "content": TASK_PROMPT})
        # else: feedback was already appended

        try:
            response = run_openagent_turn(client, messages, max_tokens)
        except KeyboardInterrupt:
            print("\n[test] Interrupted by user")
            break
        except Exception as e:
            print(f"\n[test] ERROR in round {round_num}: {e}")
            import traceback
            traceback.print_exc()
            break

        # Check generated files
        print(f"\n[test] --- Quality Check (Round {round_num}) ---")
        files = find_files()
        print(f"[test] Files found: {list(files.keys())}")

        report = check_quality(files)
        print(f"\n[test] {report.summary()}")

        if report.ok and not report.warnings:
            print(f"\n[test] ALL CHECKS PASSED in round {round_num}!")
            break
        elif report.ok and report.warnings:
            # Only warnings — give one more round to polish
            if round_num >= MAX_ROUNDS:
                print(f"\n[test] PASSED with {len(report.warnings)} warnings after {round_num} rounds")
                break
            feedback = build_feedback_prompt(report, round_num)
            if feedback:
                print(f"\n[test] Sending feedback for polishing...")
                messages.append({"role": "user", "content": feedback})
            else:
                print(f"\n[test] PASSED!")
                break
        else:
            # Has errors — must fix
            feedback = build_feedback_prompt(report, round_num)
            if feedback and round_num < MAX_ROUNDS:
                print(f"\n[test] Sending feedback with {len(report.errors)} errors to fix...")
                messages.append({"role": "user", "content": feedback})
            else:
                print(f"\n[test] FAILED after {round_num} rounds with {len(report.errors)} errors")
                break

    # Final summary
    print(f"\n{'='*60}")
    print("[test] FINAL STATE")
    print(f"{'='*60}")
    files = find_files()
    for fname, content in sorted(files.items()):
        lines = len(content.splitlines())
        size = len(content)
        print(f"  {fname}: {lines} lines, {size} chars")

    report = check_quality(files)
    print(f"\n{report.summary()}")

    if report.ok:
        print("\n[test] SUCCESS!")
    else:
        print(f"\n[test] FAILED with {len(report.errors)} errors, {len(report.warnings)} warnings")

    return report.ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
