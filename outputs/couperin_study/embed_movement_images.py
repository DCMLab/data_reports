#!/usr/bin/env python3
"""
Post-processes differences_roo_top10.html: inserts (or re-inserts) a second column
right after the index with bass movement PNGs embedded as base64.
Idempotent: strips any previously inserted column first, then re-inserts cleanly.
"""
import base64
import re
from pathlib import Path

from bs4 import BeautifulSoup

HERE = Path(__file__).parent
HTML_PATH = HERE / "differences_roo_top10.html"
IMG_DIR = HERE / "bass_movement"

TALL_ROWS = {"to_leap", "from_leap", "to_and_from_leap"}
ROW_HEIGHT = 40  # px — height for the 16 non-leap rows
LEAP_ROW_HEIGHT = 67  # px — height for the 3 leap rows

html = HTML_PATH.read_text(encoding="utf-8")
soup = BeautifulSoup(html, "html.parser")
style_el = soup.find("style")

# --- Strip any previously inserted image column(s) back to clean 9-col state ---
# CSS class attributes (col0, col1, …) are hardcoded on elements and don't change
# when new columns are inserted, so CSS selectors keep matching the right elements.
# However, earlier script versions wrongly shifted the CSS — detect and undo that.
extra_cols = len(soup.find("tbody").find("tr").find_all(["th", "td"])) - 9

if extra_cols > 0:
    # Detect how many times the CSS was wrongly shifted: the smallest colN that carries
    # a border-left rule should be col0 in the correct state; if it's col1, col2, …
    # the CSS was shifted that many times.
    border_cols = re.findall(r"\.col(\d+)\s*\{[^}]*border-left", style_el.string)
    n_wrong_shifts = int(min(border_cols)) if border_cols else 0
    if n_wrong_shifts > 0:
        style_el.string = re.sub(
            r"col(\d+)",
            lambda m: f"col{int(m.group(1)) - n_wrong_shifts}",
            style_el.string,
        )
    end = 1 + extra_cols
    for tr in soup.find("thead").find_all("tr"):
        for th in tr.find_all("th")[1:end]:
            th.decompose()
    for tr in soup.find("tbody").find_all("tr"):
        cells = tr.find_all(["th", "td"])
        for cell in cells[1:end]:
            cell.decompose()

# --- Insert blank <th> after index <th> in every thead row ---
# No CSS changes needed: inserting a new element doesn't affect existing class-based selectors.
for tr in soup.find("thead").find_all("tr"):
    tr.find("th").insert_after(soup.new_tag("th"))


def _td_style(tall):
    h = LEAP_ROW_HEIGHT if tall else ROW_HEIGHT
    return f"text-align:center; vertical-align:middle; padding:2px; min-width:80px; height:{h}px; overflow:hidden;"


def _img_style(tall):
    h = LEAP_ROW_HEIGHT if tall else ROW_HEIGHT
    return f"display:block; margin:auto; max-height:{h}px;"


# --- Insert image <td> after the row-heading <th> in every tbody row ---
for tr in soup.find("tbody").find_all("tr"):
    th = tr.find("th")
    index_val = th.get_text(strip=True)
    tall = index_val in TALL_ROWS

    td = soup.new_tag("td", style=_td_style(tall))
    img_path = IMG_DIR / f"{index_val}.png"
    if img_path.exists():
        b64 = base64.b64encode(img_path.read_bytes()).decode()
        td.append(
            soup.new_tag(
                "img", src=f"data:image/png;base64,{b64}", style=_img_style(tall)
            )
        )
    th.insert_after(td)

output = str(style_el) + "\n" + str(soup.find("table")) + "\n"
HTML_PATH.write_text(output, encoding="utf-8")
print(f"Done → {HTML_PATH}")
