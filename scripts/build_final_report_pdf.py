"""Build a formatted PDF report from FINAL_REPORT.md and exported figures."""

from __future__ import annotations

import html
import re
import shutil
import tempfile
from pathlib import Path

from matplotlib.mathtext import math_to_image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "FINAL_REPORT.md"
OUTPUT = ROOT / "output" / "pdf" / "FINAL_REPORT.pdf"
SUBMISSION_OUTPUT = ROOT / "FINAL_REPORT.pdf"


def _normalise_math(expression: str) -> str:
    """Normalise Markdown math commands supported by Matplotlib mathtext."""
    expression = expression.replace(r"\lVert", r"\|").replace(r"\rVert", r"\|")
    return re.sub(r"\\operatorname\{([^}]+)\}", r"\\mathrm{\1}", expression)


def _inline_markup(
    text: str,
    math_directory: Path | None = None,
    math_counter: list[int] | None = None,
) -> str:
    """Convert a small safe subset of Markdown inline markup to ReportLab XML."""
    text = html.escape(text, quote=False)
    links: list[tuple[str, str]] = []

    def stash_link(match: re.Match[str]) -> str:
        links.append((match.group(1), match.group(2)))
        return f"@@LINK{len(links) - 1}@@"

    text = re.sub(r"\[([^]]+)\]\(([^)]+)\)", stash_link, text)
    def inline_math(match: re.Match[str]) -> str:
        expression = _normalise_math(match.group(1))
        if math_directory is None or math_counter is None:
            return f"<font name='Courier'>{html.escape(expression)}</font>"
        index = math_counter[0]
        math_counter[0] += 1
        path = math_directory / f"inline-equation-{index:03d}.png"
        math_to_image(f"${expression}$", str(path), dpi=220, format="png", color="#173f5f")
        # Keep inline equations inside the narrowest report-table cell.
        width = min(72, max(18, 3.1 * len(expression)))
        height = 8 if len(expression) < 24 else 9
        return f'<img src="{path.as_posix()}" width="{width:.1f}" height="{height}" valign="middle"/>'

    text = re.sub(r"\$([^$]+)\$", inline_math, text)
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    for index, (label, url) in enumerate(links):
        text = text.replace(
            f"@@LINK{index}@@",
            f"<link href='{html.escape(url, quote=True)}' color='#1f5f8b'><u>{html.escape(label)}</u></link>",
        )
    return text


def _paragraph(
    text: str,
    style: ParagraphStyle,
    math_directory: Path | None = None,
    math_counter: list[int] | None = None,
) -> Paragraph:
    """Create a paragraph with sanitized inline Markdown formatting."""
    return Paragraph(_inline_markup(text, math_directory, math_counter), style)


def _image(path: Path, alt: str) -> Image:
    """Create a report image with a bounded width and a caption."""
    image = Image(str(path))
    image._restrictSize(6.55 * inch, 4.25 * inch)
    image.hAlign = "CENTER"
    return image


def _math_image(expression: str, directory: Path, index: int) -> Image:
    """Render one LaTeX math expression as a sharp report image."""
    path = directory / f"equation-{index:03d}.png"
    expression = _normalise_math(expression)
    math_to_image(f"${expression}$", str(path), dpi=220, format="png", color="#173f5f")
    image = Image(str(path))
    image._restrictSize(4.6 * inch, 0.30 * inch)
    image.hAlign = "LEFT"
    return image


def _table(
    rows: list[list[str]],
    styles: dict[str, ParagraphStyle],
    math_directory: Path,
    math_counter: list[int],
) -> Table:
    """Create a styled table from Markdown table rows."""
    wrapped = [
        [_paragraph(cell.strip(), styles["table"], math_directory, math_counter) for cell in row]
        for row in rows
    ]
    column_count = len(rows[0]) if rows else 0
    if column_count == 4:
        widths = [0.85 * inch, 1.65 * inch, 2.65 * inch, 1.4 * inch]
    elif column_count == 3:
        widths = [1.35 * inch, 1.7 * inch, 1.7 * inch]
    elif column_count:
        widths = [6.55 * inch / column_count] * column_count
    else:
        widths = None
    table = Table(wrapped, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173f5f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b7c9d6")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef4f7")]),
            ]
        )
    )
    return table


def _build_story(math_directory: Path) -> list[object]:
    """Parse the Markdown report into a ReportLab story."""
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("ReportTitle", parent=base["Title"], fontName="Helvetica-Bold", fontSize=19, leading=23, textColor=colors.HexColor("#173f5f"), alignment=TA_CENTER, spaceAfter=18),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=colors.HexColor("#173f5f"), spaceBefore=14, spaceAfter=7),
        "h3": ParagraphStyle("H3", parent=base["Heading3"], fontName="Helvetica-Bold", fontSize=11.5, leading=15, textColor=colors.HexColor("#28627f"), spaceBefore=10, spaceAfter=5),
        "body": ParagraphStyle("Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.2, leading=13, alignment=TA_LEFT, spaceAfter=6),
        "bullet": ParagraphStyle("Bullet", parent=base["BodyText"], fontName="Helvetica", fontSize=9.2, leading=13, leftIndent=14, firstLineIndent=-8, spaceAfter=3),
        "quote": ParagraphStyle("Quote", parent=base["BodyText"], fontName="Helvetica-Oblique", fontSize=9.2, leading=13, leftIndent=18, borderPadding=7, borderColor=colors.HexColor("#9bb8c8"), borderWidth=0.5, borderLeft=True, spaceAfter=8),
        "table": ParagraphStyle("Table", parent=base["BodyText"], fontName="Helvetica", fontSize=7.4, leading=9.2),
        "caption": ParagraphStyle("Caption", parent=base["BodyText"], fontName="Helvetica-Oblique", fontSize=8, leading=10, alignment=TA_CENTER, textColor=colors.HexColor("#4f6570"), spaceAfter=10),
    "code": ParagraphStyle("Code", parent=base["Code"], fontName="Courier", fontSize=5.8, leading=7.2, leftIndent=6, rightIndent=6, backColor=colors.HexColor("#f1f4f5"), borderPadding=6, spaceAfter=8),
    }
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    story: list[object] = []
    index = 0
    equation_index = 0
    inline_math_counter = [1000]
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if line.startswith("!["):
            match = re.match(r"!\[([^]]*)\]\(([^)]+)\)", line)
            if match:
                figure = ROOT / match.group(2)
                if figure.exists():
                    story.extend([Spacer(1, 6), _image(figure, match.group(1)), _paragraph(match.group(1), styles["caption"])])
            index += 1
            continue
        if line.startswith("```"):
            code: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].startswith("```"):
                code.append(lines[index])
                index += 1
            story.append(Preformatted("\n".join(code), styles["code"]))
            index += 1
            continue
        if line.strip() == "$$" or (line.strip().startswith("$$") and line.strip().endswith("$$") and len(line.strip()) > 4):
            if line.strip() == "$$":
                index += 1
                expression: list[str] = []
                while index < len(lines) and lines[index].strip() != "$$":
                    expression.append(lines[index].strip())
                    index += 1
                index += 1
                formula = " ".join(expression)
            else:
                formula = line.strip()[2:-2].strip()
                index += 1
            story.extend([Spacer(1, 3), _math_image(formula, math_directory, equation_index), Spacer(1, 5)])
            equation_index += 1
            continue
        if line.startswith("# "):
            story.append(_paragraph(line[2:], styles["title"], math_directory, inline_math_counter))
            index += 1
            continue
        if line.startswith("## "):
            story.append(_paragraph(line[3:], styles["h2"], math_directory, inline_math_counter))
            index += 1
            continue
        if line.startswith("### "):
            story.append(_paragraph(line[4:], styles["h3"], math_directory, inline_math_counter))
            index += 1
            continue
        if line.startswith("> "):
            story.append(_paragraph(line[2:], styles["quote"], math_directory, inline_math_counter))
            index += 1
            continue
        if line.lstrip().startswith("|") and index + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{3,}", lines[index + 1]):
            rows: list[list[str]] = []
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                if re.match(r"^\s*\|?\s*:?-{3,}", lines[index]):
                    index += 1
                    continue
                rows.append([cell for cell in lines[index].strip().strip("|").split("|")])
                index += 1
            story.extend([_table(rows, styles, math_directory, inline_math_counter), Spacer(1, 8)])
            continue
        if re.match(r"^\d+\. ", line) or line.startswith("- "):
            story.append(_paragraph(line, styles["bullet"], math_directory, inline_math_counter))
            index += 1
            continue
        paragraph = [line]
        index += 1
        while index < len(lines) and lines[index].strip() and not re.match(r"^(#|>|!\[|```|\| |\d+\. |- )", lines[index]):
            paragraph.append(lines[index])
            index += 1
        story.append(_paragraph(" ".join(paragraph), styles["body"], math_directory, inline_math_counter))
    return story


def _footer(canvas, document) -> None:
    """Draw the report footer and page number."""
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#b7c9d6"))
    canvas.line(0.7 * inch, 0.55 * inch, 7.8 * inch, 0.55 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#4f6570"))
    canvas.drawString(0.7 * inch, 0.35 * inch, "Retrieval-Augmented Robotic Manipulation")
    canvas.drawRightString(7.8 * inch, 0.35 * inch, f"Page {document.page}")
    canvas.restoreState()


def main() -> None:
    """Build the final PDF report at the stable output path."""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=0.7 * inch, leftMargin=0.7 * inch, topMargin=0.65 * inch, bottomMargin=0.75 * inch, title="Retrieval-Augmented Robotic Manipulation")
    with tempfile.TemporaryDirectory(prefix="report-math-", dir=OUTPUT.parent) as directory:
        document.build(_build_story(Path(directory)), onFirstPage=_footer, onLaterPages=_footer)
    shutil.copyfile(OUTPUT, SUBMISSION_OUTPUT)
    print(f"PDF written to: {OUTPUT}")
    print(f"Submission copy written to: {SUBMISSION_OUTPUT}")


if __name__ == "__main__":
    main()
