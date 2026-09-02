"""Build a formatted PDF report from FINAL_REPORT.md and exported figures."""

from __future__ import annotations

import html
import re
import shutil
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "FINAL_REPORT.md"
OUTPUT = ROOT / "output" / "pdf" / "FINAL_REPORT.pdf"
SUBMISSION_OUTPUT = ROOT / "FINAL_REPORT.pdf"


def _inline_markup(text: str) -> str:
    """Convert a small safe subset of Markdown inline markup to ReportLab XML."""
    text = html.escape(text, quote=False)
    links: list[tuple[str, str]] = []

    def stash_link(match: re.Match[str]) -> str:
        links.append((match.group(1), match.group(2)))
        return f"@@LINK{len(links) - 1}@@"

    text = re.sub(r"\[([^]]+)\]\(([^)]+)\)", stash_link, text)
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    for index, (label, url) in enumerate(links):
        text = text.replace(
            f"@@LINK{index}@@",
            f"<link href='{html.escape(url, quote=True)}' color='#1f5f8b'><u>{html.escape(label)}</u></link>",
        )
    return text


def _paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    """Create a paragraph with sanitized inline Markdown formatting."""
    return Paragraph(_inline_markup(text), style)


def _image(path: Path, alt: str) -> Image:
    """Create a report image with a bounded width and a caption."""
    image = Image(str(path))
    image._restrictSize(6.55 * inch, 4.25 * inch)
    image.hAlign = "CENTER"
    return image


def _table(rows: list[list[str]], styles: dict[str, ParagraphStyle]) -> Table:
    """Create a styled table from Markdown table rows."""
    wrapped = [[_paragraph(cell.strip(), styles["table"]) for cell in row] for row in rows]
    table = Table(wrapped, repeatRows=1, hAlign="LEFT")
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


def _build_story() -> list[object]:
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
        "code": ParagraphStyle("Code", parent=base["Code"], fontName="Courier", fontSize=7.2, leading=9, leftIndent=10, backColor=colors.HexColor("#f1f4f5"), borderPadding=6, spaceAfter=8),
    }
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    story: list[object] = []
    index = 0
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
            story.append(_paragraph("<br/>".join(html.escape(x) for x in code), styles["code"]))
            index += 1
            continue
        if line.startswith("# "):
            story.append(_paragraph(line[2:], styles["title"]))
            index += 1
            continue
        if line.startswith("## "):
            story.append(_paragraph(line[3:], styles["h2"]))
            index += 1
            continue
        if line.startswith("### "):
            story.append(_paragraph(line[4:], styles["h3"]))
            index += 1
            continue
        if line.startswith("> "):
            story.append(_paragraph(line[2:], styles["quote"]))
            index += 1
            continue
        if line.startswith("| ") and index + 1 < len(lines) and "|---" in lines[index + 1]:
            rows: list[list[str]] = []
            while index < len(lines) and lines[index].startswith("|"):
                if index != 0 and set(lines[index].replace("|", "").replace("-", "").replace(":", "").strip()) == set():
                    index += 1
                    continue
                rows.append([cell for cell in lines[index].strip().strip("|").split("|")])
                index += 1
            story.extend([_table(rows, styles), Spacer(1, 8)])
            continue
        if re.match(r"^\d+\. ", line) or line.startswith("- "):
            story.append(_paragraph(line, styles["bullet"]))
            index += 1
            continue
        paragraph = [line]
        index += 1
        while index < len(lines) and lines[index].strip() and not re.match(r"^(#|>|!\[|```|\| |\d+\. |- )", lines[index]):
            paragraph.append(lines[index])
            index += 1
        story.append(_paragraph(" ".join(paragraph), styles["body"]))
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
    document.build(_build_story(), onFirstPage=_footer, onLaterPages=_footer)
    shutil.copyfile(OUTPUT, SUBMISSION_OUTPUT)
    print(f"PDF written to: {OUTPUT}")
    print(f"Submission copy written to: {SUBMISSION_OUTPUT}")


if __name__ == "__main__":
    main()
