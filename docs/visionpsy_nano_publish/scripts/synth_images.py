"""Generate the synthesized benchmark images (chart, receipt, diagram, text)."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


OUT = Path(os.path.expanduser("~/movie/visionpsy_bench/synth"))
OUT.mkdir(parents=True, exist_ok=True)


def _font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def make_chart() -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    categories = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    revenue = [120, 148, 132, 175, 210, 232]
    cost = [95, 100, 108, 118, 129, 138]
    x = list(range(len(categories)))
    ax.bar([xi - 0.2 for xi in x], revenue, width=0.4, label="Revenue", color="#3b82f6")
    ax.bar([xi + 0.2 for xi in x], cost, width=0.4, label="Cost", color="#ef4444")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("USD (thousands)")
    ax.set_title("Q1-Q2 2026 Revenue vs Cost")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    out = OUT / "chart_revenue.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def make_receipt() -> Path:
    W, H = 480, 720
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    big = _font(22)
    med = _font(16)
    sm = _font(14)

    d.text((W // 2 - 90, 30), "BLUE BOTTLE CAFE", fill="black", font=big)
    d.text((W // 2 - 100, 62), "1234 Market St, San Francisco", fill="black", font=sm)
    d.text((W // 2 - 60, 82), "Tel: (415) 555-0123", fill="black", font=sm)
    d.line([(30, 115), (W - 30, 115)], fill="black", width=1)
    d.text((30, 125), "Receipt #  A-084213", fill="black", font=med)
    d.text((30, 148), "Date: 2026-07-31 14:32", fill="black", font=med)
    d.text((30, 171), "Cashier: Maya", fill="black", font=med)
    d.line([(30, 200), (W - 30, 200)], fill="black", width=1)

    items = [
        ("Latte (12 oz)", 1, 5.50),
        ("Cappuccino",   1, 5.00),
        ("Croissant",    2, 4.25),
        ("Blueberry Muffin", 1, 3.75),
        ("Sparkling Water", 2, 2.50),
    ]
    y = 215
    for name, qty, price in items:
        d.text((30, y), f"{qty} x {name}", fill="black", font=med)
        d.text((W - 90, y), f"${qty * price:6.2f}", fill="black", font=med)
        y += 26

    d.line([(30, y + 6), (W - 30, y + 6)], fill="black", width=1)
    y += 20
    subtotal = sum(q * p for _, q, p in items)
    tax = round(subtotal * 0.0875, 2)
    total = subtotal + tax
    d.text((30, y), "Subtotal", fill="black", font=med); d.text((W - 90, y), f"${subtotal:6.2f}", fill="black", font=med)
    y += 24
    d.text((30, y), "Tax (8.75%)", fill="black", font=med); d.text((W - 90, y), f"${tax:6.2f}", fill="black", font=med)
    y += 24
    d.text((30, y), "TOTAL", fill="black", font=big); d.text((W - 100, y), f"${total:6.2f}", fill="black", font=big)
    y += 40
    d.text((30, y), "Payment: Visa ****4471", fill="black", font=sm); y += 20
    d.text((30, y), "Approved  Auth #: 209411", fill="black", font=sm); y += 40
    d.text((W // 2 - 55, y), "Thank you!", fill="black", font=big)
    out = OUT / "receipt.png"
    img.save(out)
    return out


def make_diagram() -> Path:
    W, H = 720, 480
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    med = _font(16)
    big = _font(20)

    boxes = [
        (60, 40, 260, 100, "User uploads image"),
        (60, 190, 260, 250, "Preprocessor tiles"),
        (60, 340, 260, 400, "SigLIP2 vision"),
        (460, 40, 660, 100, "Prompt tokens"),
        (460, 190, 660, 250, "SmolLM2 decoder"),
        (460, 340, 660, 400, "Generated text"),
    ]
    for (x1, y1, x2, y2, label) in boxes:
        d.rectangle((x1, y1, x2, y2), outline="black", width=2, fill="#eef2ff")
        d.text((x1 + 20, y1 + 22), label, fill="black", font=med)

    # arrows
    d.line((160, 100, 160, 190), fill="black", width=2)
    d.polygon([(155, 185), (165, 185), (160, 195)], fill="black")
    d.line((160, 250, 160, 340), fill="black", width=2)
    d.polygon([(155, 335), (165, 335), (160, 345)], fill="black")
    d.line((260, 370, 460, 220), fill="black", width=2)
    d.polygon([(455, 215), (462, 225), (450, 225)], fill="black")
    d.line((560, 100, 560, 190), fill="black", width=2)
    d.polygon([(555, 185), (565, 185), (560, 195)], fill="black")
    d.line((560, 250, 560, 340), fill="black", width=2)
    d.polygon([(555, 335), (565, 335), (560, 345)], fill="black")

    d.text((W // 2 - 100, 8), "VisionPsy-Nano flow", fill="black", font=big)
    out = OUT / "diagram_flow.png"
    img.save(out)
    return out


if __name__ == "__main__":
    print("chart:", make_chart())
    print("receipt:", make_receipt())
    print("diagram:", make_diagram())
