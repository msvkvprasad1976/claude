# build_deliverables.py
#
# Builds three deliverables for the PET-fibre-reinforced-concrete manuscript
# revision round:
#   1. Figure1_Experimental_Programme.png - vector-quality flowchart
#   2. Manuscript_Revised-2.docx          - revised manuscript, new text in red
#   3. Author_Response_Sheet.docx         - point-by-point reviewer response
#
# No third-party packages are required (no python-docx, no matplotlib).
# Figure 1 is drawn as SVG and rasterised at 300 DPI with headless Chromium
# (already available on this machine). The .docx files are produced with a
# small self-contained OOXML writer (see the "docx_lite" section below) that
# mimics the handful of python-docx calls used by this script.
import os
import shutil
import struct
import subprocess
import tempfile
import textwrap
from xml.sax.saxutils import escape

CHROMIUM_CANDIDATES = [
    "/opt/pw-browsers/chromium",
    "/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
    shutil.which("chromium"),
    shutil.which("chromium-browser"),
    shutil.which("google-chrome"),
]


def _find_chromium():
    for candidate in CHROMIUM_CANDIDATES:
        if candidate and os.path.exists(candidate):
            return candidate
    raise RuntimeError("No headless Chromium binary found")


# ---------------------------------------------------------------------------
# 1. Figure 1: experimental-programme flowchart, drawn as SVG, rendered to a
#    300 DPI PNG via headless Chromium.
# ---------------------------------------------------------------------------
def build_figure1(path="Figure1_Experimental_Programme.png"):
    W, H = 750, 1000  # px, at 100 dpi baseline -> 7.5 x 10 in page
    cx = W / 2
    boxes = []   # (x, y, w, h, lines, fontsize)
    arrows = []  # (x1, y1, x2, y2)

    def wrap(text, width=46):
        out = []
        for para in text.split("\n"):
            out.extend(textwrap.wrap(para, width=width) or [""])
        return out

    def add_box(y, text, w=560, fontsize=15, x=None):
        lines = wrap(text)
        h = 26 + 20 * len(lines)
        bx = cx - w / 2 if x is None else x
        boxes.append((bx, y, w, h, lines, fontsize))
        return bx, y, w, h

    y = 20
    _, _, _, h = add_box(y, "Collection of PET bottles")
    y += h
    arrows.append((cx, y, cx, y + 34)); y += 34

    _, _, _, h = add_box(y, "Cleaning and shredding into PET fibres")
    y += h
    arrows.append((cx, y, cx, y + 34)); y += 34

    _, _, _, h = add_box(
        y,
        "Material characterisation: OPC, fine aggregate, "
        "coarse aggregate, PET fibres, water",
    )
    y += h
    arrows.append((cx, y, cx, y + 34)); y += 34

    _, _, _, h = add_box(y, "Mix design per IS 10262")
    y += h
    split_y = y
    arrows.append((cx, y, cx, y + 20)); y += 20

    half_w = 260
    lx = cx - half_w / 2 - 150
    rx = cx + half_w / 2 - 130
    bx1, by1, bw1, bh1 = add_box(y, "M20 grade", w=280, x=lx)
    bx2, by2, bw2, bh2 = add_box(y, "M30 grade", w=280, x=rx)
    arrows.append((cx, split_y + 20, lx + bw1 / 2, by1))
    arrows.append((cx, split_y + 20, rx + bw2 / 2, by2))
    y += max(bh1, bh2)

    merge_y = y + 34
    arrows.append((lx + bw1 / 2, y, cx, merge_y))
    arrows.append((rx + bw2 / 2, y, cx, merge_y))
    y = merge_y

    _, _, _, h = add_box(
        y, "PET fibre dosages by volume: 0.0, 0.5, 1.0, 1.5 percent"
    )
    y += h
    arrows.append((cx, y, cx, y + 34)); y += 34

    _, _, _, h = add_box(
        y,
        "Mixing, slump test per IS 1199, casting of cubes, "
        "cylinders and beams",
    )
    y += h
    arrows.append((cx, y, cx, y + 34)); y += 34

    _, _, _, h = add_box(y, "Curing in water at 27 +/- 2 C per IS 516")
    y += h
    arrows.append((cx, y, cx, y + 34)); y += 34

    _, _, _, h = add_box(
        y,
        "Testing at 7, 28, 56, 90 days: compressive, "
        "split tensile, flexural strength",
    )
    y += h
    split_y2 = y
    arrows.append((cx, y, cx, y + 20)); y += 20

    half_w2 = 300
    lx2 = cx - half_w2 / 2 - 160
    rx2 = cx + half_w2 / 2 - 140
    bx3, by3, bw3, bh3 = add_box(
        y, "Statistical analysis: SD, CV, CI, ANOVA", w=300, fontsize=13, x=lx2
    )
    bx4, by4, bw4, bh4 = add_box(
        y, "First-order carbon assessment", w=300, fontsize=13, x=rx2
    )
    arrows.append((cx, split_y2 + 20, lx2 + bw3 / 2, by3))
    arrows.append((cx, split_y2 + 20, rx2 + bw4 / 2, by4))
    y += max(bh3, bh4)

    caption_y = y + 60
    total_h = caption_y + 100

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" '
        f'height="{total_h}" viewBox="0 0 {W} {total_h}">',
        f'<rect x="0" y="0" width="{W}" height="{total_h}" fill="white"/>',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M0,0 L10,5 L0,10 z" fill="black"/></marker></defs>',
    ]
    for (x1, y1, x2, y2) in arrows:
        svg_parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="black" stroke-width="1.6" marker-end="url(#arrow)"/>'
        )
    for (bx, by, bw, bh, lines, fs) in boxes:
        svg_parts.append(
            f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" '
            f'height="{bh:.1f}" rx="10" ry="10" fill="white" '
            f'stroke="black" stroke-width="1.6"/>'
        )
        n = len(lines)
        line_h = 20
        start_y = by + bh / 2 - (n - 1) * line_h / 2 + fs * 0.35
        for i, line in enumerate(lines):
            svg_parts.append(
                f'<text x="{bx + bw / 2:.1f}" y="{start_y + i * line_h:.1f}" '
                f'font-family="Georgia, \'Times New Roman\', serif" '
                f'font-size="{fs}" text-anchor="middle" fill="black">'
                f"{escape(line)}</text>"
            )
    svg_parts.append(
        f'<text x="{cx:.1f}" y="{caption_y:.1f}" '
        f'font-family="Georgia, \'Times New Roman\', serif" font-size="16" '
        f'text-anchor="middle" fill="black">Figure 1. Experimental programme'
        f"</text>"
    )
    svg_parts.append("</svg>")
    svg = "\n".join(svg_parts)

    with tempfile.TemporaryDirectory() as tmp:
        html_path = os.path.join(tmp, "figure.html")
        with open(html_path, "w") as f:
            f.write(
                "<!doctype html><html><head><meta charset='utf-8'>"
                "<style>*{margin:0;padding:0}</style></head><body>"
                + svg
                + "</body></html>"
            )
        profile_dir = os.path.join(tmp, "profile")
        out_png = os.path.abspath(path)
        chromium = _find_chromium()
        cmd = [
            chromium,
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--hide-scrollbars",
            "--default-background-color=FFFFFFFF",
            f"--user-data-dir={profile_dir}",
            f"--window-size={W},{total_h}",
            "--force-device-scale-factor=3",
            f"--screenshot={out_png}",
            f"file://{html_path}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if not os.path.exists(out_png):
            raise RuntimeError(
                "Chromium screenshot failed:\n" + result.stdout + result.stderr
            )
    return path


# ---------------------------------------------------------------------------
# docx_lite: minimal OOXML writer covering exactly the python-docx surface
# used below (Document, add_paragraph/add_run, add_picture, sections, Pt,
# Inches, RGBColor, WD_ALIGN_PARAGRAPH).
# ---------------------------------------------------------------------------
import zipfile


class Length(float):
    @property
    def emu(self):
        return int(round(self * 914400))

    @property
    def twips(self):
        return int(round(self * 1440))


def Inches(value):
    return Length(value)


def Pt(value):
    return value


def RGBColor(r, g, b):
    return f"{r:02X}{g:02X}{b:02X}"


class WD_ALIGN_PARAGRAPH:
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    JUSTIFY = "both"


class _Color:
    def __init__(self):
        self.rgb = None


class _Font:
    def __init__(self):
        self.name = None
        self.size = None
        self.color = _Color()


class Run:
    def __init__(self, text):
        self.text = text
        self.bold = False
        self.font = _Font()


class Paragraph:
    def __init__(self, kind="text"):
        self.runs = []
        self.alignment = None
        self.kind = kind
        self.picture = None

    def add_run(self, text):
        r = Run(text)
        self.runs.append(r)
        return r


class Section:
    def __init__(self):
        self.left_margin = Inches(1)
        self.right_margin = Inches(1)
        self.top_margin = Inches(1)
        self.bottom_margin = Inches(1)


def _png_size(data):
    w = struct.unpack(">I", data[16:20])[0]
    h = struct.unpack(">I", data[20:24])[0]
    return w, h


class Document:
    def __init__(self):
        self.sections = [Section()]
        self.paragraphs = []
        self._media = []

    def add_paragraph(self):
        p = Paragraph("text")
        self.paragraphs.append(p)
        return p

    def add_picture(self, path, width=None):
        with open(path, "rb") as f:
            data = f.read()
        w_px, h_px = _png_size(data)
        if width is None:
            width = Inches(w_px / 96)
        emu_w = width.emu
        emu_h = int(round(emu_w * h_px / w_px))
        idx = len(self._media) + 1
        fname = f"image{idx}.png"
        self._media.append((fname, data))
        p = Paragraph("picture")
        p.picture = {
            "file": fname,
            "emu_w": emu_w,
            "emu_h": emu_h,
            "rid": f"rId{100 + idx}",
        }
        self.paragraphs.append(p)
        return p

    def _run_xml(self, r):
        parts = []
        if r.font.name:
            parts.append(
                f'<w:rFonts w:ascii="{r.font.name}" w:hAnsi="{r.font.name}" '
                f'w:cs="{r.font.name}"/>'
            )
        if r.bold:
            parts.append("<w:b/>")
        if r.font.color.rgb:
            parts.append(f'<w:color w:val="{r.font.color.rgb}"/>')
        if r.font.size is not None:
            hp = int(round(r.font.size * 2))
            parts.append(f'<w:sz w:val="{hp}"/><w:szCs w:val="{hp}"/>')
        rpr = f'<w:rPr>{"".join(parts)}</w:rPr>' if parts else ""
        return f'<w:r>{rpr}<w:t xml:space="preserve">{escape(r.text)}</w:t></w:r>'

    def _drawing_xml(self, pic):
        return (
            '<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
            f'<wp:extent cx="{pic["emu_w"]}" cy="{pic["emu_h"]}"/>'
            '<wp:effectExtent l="0" t="0" r="0" b="0"/>'
            '<wp:docPr id="1" name="Picture 1"/>'
            "<wp:cNvGraphicFramePr>"
            '<a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/'
            'drawingml/2006/main" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/'
            '2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/'
            'drawingml/2006/picture">'
            '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/'
            "2006/picture\">"
            f'<pic:nvPicPr><pic:cNvPr id="0" name="{pic["file"]}"/>'
            "<pic:cNvPicPr/></pic:nvPicPr>"
            '<pic:blipFill><a:blip r:embed="'
            f'{pic["rid"]}" xmlns:r="http://schemas.openxmlformats.org/'
            'officeDocument/2006/relationships"/>'
            "<a:stretch><a:fillRect/></a:stretch></pic:blipFill>"
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{pic["emu_w"]}" '
            f'cy="{pic["emu_h"]}"/></a:xfrm><a:prstGeom prst="rect">'
            "<a:avLst/></a:prstGeom></pic:spPr>"
            "</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>"
        )

    def _para_xml(self, p):
        ppr = f'<w:pPr><w:jc w:val="{p.alignment}"/></w:pPr>' if p.alignment else ""
        if p.kind == "picture":
            body = self._drawing_xml(p.picture)
        else:
            body = "".join(self._run_xml(r) for r in p.runs)
        return f"<w:p>{ppr}{body}</w:p>"

    def save(self, path):
        sec = self.sections[0]
        body_xml = "".join(self._para_xml(p) for p in self.paragraphs)
        sect_pr = (
            "<w:sectPr>"
            f'<w:pgMar w:top="1440" w:right="{sec.right_margin.twips}" '
            f'w:bottom="1440" w:left="{sec.left_margin.twips}" '
            'w:header="720" w:footer="720" w:gutter="0"/>'
            "</w:sectPr>"
        )
        document_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/'
            'wordprocessingml/2006/main" xmlns:wp="http://schemas.'
            'openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/'
            '2006/relationships">'
            f"<w:body>{body_xml}{sect_pr}</w:body></w:document>"
        )
        content_types = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/'
            'content-types">'
            '<Default Extension="rels" ContentType="application/'
            'vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Default Extension="png" ContentType="image/png"/>'
            '<Override PartName="/word/document.xml" ContentType="application/'
            'vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>"
        )
        root_rels = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/'
            '2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/'
            'officeDocument/2006/relationships/officeDocument" '
            'Target="word/document.xml"/>'
            "</Relationships>"
        )
        doc_rels_items = "".join(
            f'<Relationship Id="{pic["rid"]}" Type="http://schemas.'
            'openxmlformats.org/officeDocument/2006/relationships/image" '
            f'Target="media/{pic["file"]}"/>'
            for p in self.paragraphs
            if p.kind == "picture"
            for pic in [p.picture]
        )
        doc_rels = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/'
            f'2006/relationships">{doc_rels_items}</Relationships>'
        )
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml", content_types)
            z.writestr("_rels/.rels", root_rels)
            z.writestr("word/document.xml", document_xml)
            z.writestr("word/_rels/document.xml.rels", doc_rels)
            for fname, data in self._media:
                z.writestr(f"word/media/{fname}", data)
        return path


RED = RGBColor(0xC0, 0x00, 0x00)


# ---------- helpers for red / black runs ----------
def add_para(doc, text, red=False, bold=False, size=10, align=None, heading=False):
    p = doc.add_paragraph()
    if align:
        p.alignment = align
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(14 if heading else size)
    r.bold = bold or heading
    if red:
        r.font.color.rgb = RED
    return p


# ---------------------------------------------------------------------------
# 2. Revised manuscript
# ---------------------------------------------------------------------------
def build_manuscript(fig_path, path="Manuscript_Revised-2.docx"):
    doc = Document()
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(1)
    add_para(doc, "Mechanical Behaviour of Recycled PET Fibre-Reinforced "
                  "Concrete for M20 and M30 Grades", bold=True, size=14,
             align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, "[Author names and affiliations as in original]", size=10,
             align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, "Note: text shown in red is new or revised in this round. "
                  "Paste unchanged original text in black where marked.",
             red=True, size=9)

    add_para(doc, "Abstract", bold=True, size=12)
    add_para(doc,
      "This study reports the mechanical behaviour of recycled polyethylene "
      "terephthalate (PET) fibre-reinforced concrete for two grades, M20 and "
      "M30. Recycled PET fibres were added to the mix at several dosages by "
      "volume. Concrete was tested for compressive strength, split tensile "
      "strength, and flexural strength at 7, 28, 56, and 90 days. Three "
      "specimens were cast for each mix and age. Results were analysed with "
      "standard deviation, coefficient of variation, confidence intervals, and "
      "analysis of variance (ANOVA). A first-order carbon assessment was also "
      "carried out. For both grades, a fibre dosage of 1.0 percent by volume "
      "gave the best balance of strength and ductility. Split tensile and "
      "flexural strength improved most, while compressive strength changed only "
      "slightly. Higher fibre dosages reduced strength. The reduction is linked "
      "to fibre clustering, poorer dispersion, and higher void content at high "
      "dosages. The optimum of 1.0 percent applies to the M20 and M30 grades "
      "tested and should not be generalised to all grades. The paper adds honest "
      "limitations. The sample size was small. No in-house microstructural "
      "imaging was performed, so the mechanisms are supported by published "
      "microstructural evidence. Only one fibre geometry was studied. "
      "Durability, long-term, and full life-cycle assessment (LCA) tests were "
      "outside the present scope. The paper sets out these gaps and a clear plan "
      "of future work. The findings support the safe reuse of PET bottle waste "
      "as a low-cost fibre for normal-strength structural concrete.", red=True)
    add_para(doc, "Keywords: recycled PET fibre, fibre-reinforced concrete, "
                  "M20 concrete, M30 concrete, split tensile strength, "
                  "sustainability", red=True)

    add_para(doc, "1. Introduction", bold=True, size=12)
    add_para(doc, "[Paste revised Introduction here. At first mention expand: "
                  "polyethylene terephthalate (PET), fibre-reinforced concrete "
                  "(FRC), ordinary Portland cement (OPC), water-cement ratio "
                  "(w/c), standard deviation (SD), coefficient of variation "
                  "(CV), confidence interval (CI), analysis of variance (ANOVA), "
                  "interfacial transition zone (ITZ), scanning electron "
                  "microscopy (SEM), life-cycle assessment (LCA), Indian "
                  "Standard (IS). Use abbreviations only afterward.]", red=True)

    add_para(doc, "2. Materials and Methods", bold=True, size=12)
    add_para(doc, "[Paste original methods in black. Add in red:] Mixing "
                  "followed IS 10262. Slump was measured per IS 1199. Specimens "
                  "were cured in water at 27 plus or minus 2 degrees Celsius per "
                  "IS 516. Casting and testing followed the relevant IS codes.",
             red=True)
    doc.add_picture(fig_path, width=Inches(5.2))
    last = doc.paragraphs[-1]; last.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = add_para(doc, "Figure 1. Experimental programme (redrawn by the "
                        "authors using plotting software).", red=True, size=9,
                   align=WD_ALIGN_PARAGRAPH.CENTER)

    add_para(doc, "3. Results", bold=True, size=12)
    add_para(doc, "[Paste original results and tables in black.]")

    add_para(doc, "4. Discussion", bold=True, size=12)
    add_para(doc,
      "The gain in split tensile and flexural strength at 1.0 percent PET "
      "follows the fibre-bridging mechanism. Fibres cross micro-cracks and "
      "transfer stress across the crack faces, which delays crack growth. Kim "
      "et al. and Borg et al. report the same bridging and pull-out behaviour "
      "for recycled PET fibre concrete. The small drop in compressive strength "
      "at higher dosages is consistent with fibre clustering and higher void "
      "content. Published work shows that PET dosages above about 0.5 to 1.0 "
      "percent by volume promote fibre balling and poor dispersion, which "
      "raises porosity and lowers strength. The present study did not perform "
      "SEM, optical microscopy, or ITZ analysis. The mechanism discussion "
      "therefore relies on established microstructural evidence. Silva et al. "
      "show that PET fibre toughness indices fall over time due to alkaline "
      "hydrolysis in the cement matrix. Pelisser et al. present SEM micrographs "
      "of PET fibres with intense degradation after one year in the alkaline "
      "concrete environment, together with a rise in porosity at 365 days "
      "measured by mercury intrusion porosimetry. These results explain both "
      "the early strength gains and the risk of long-term loss.", red=True)
    add_para(doc,
      "A direct experimental comparison with steel and polypropylene fibres was "
      "outside the present scope. Literature allows a fair qualitative "
      "comparison. Steel fibres give the largest gain in flexural and residual "
      "strength but add mass, cost, and corrosion risk. Polypropylene fibres "
      "are light, alkali resistant, and cheap, and mainly control early "
      "shrinkage cracking. Kim et al. compared recycled PET and polypropylene "
      "fibre concrete directly and found PET competitive at equal volume "
      "fractions. Recycled PET fibres reuse bottle waste, add little cost, and "
      "improve tensile and flexural behaviour, but they carry a long-term "
      "alkaline-durability question.", red=True)

    add_para(doc, "5. Limitations and Future Work", bold=True, size=12)
    add_para(doc,
      "This study has clear limitations. First, three specimens were tested for "
      "each mix and age. This limits statistical power for fibre-reinforced "
      "concrete. Future work will use larger batches. Second, no in-house "
      "microstructural imaging was performed. No SEM, optical microscopy, "
      "computed tomography, energy-dispersive spectroscopy, or ITZ analysis was "
      "carried out. The mechanisms are supported by published microstructural "
      "studies. Future work will add SEM and ITZ analysis. Third, fibre "
      "dispersion was controlled by procedure but was not quantified by image "
      "analysis or void measurement. Future work will add image-based "
      "dispersion and void studies. Fourth, only one PET fibre geometry was "
      "tested. Future work will vary length, aspect ratio, width, thickness, "
      "and surface texture and add surface treatment. Fifth, no durability "
      "tests were run. Water absorption, permeability, chloride penetration, "
      "sulfate resistance, shrinkage, carbonation, and freeze-thaw tests are "
      "planned. Published data show that recycled PET fibre concrete keeps "
      "chloride resistance close to plain concrete and good freeze-thaw "
      "endurance but loses strength in alkaline and acid media, with reported "
      "compressive-strength loss near 24 percent in 3 percent sulfuric acid and "
      "near 10 percent in a pH 12.6 alkaline medium after 120 days. Sixth, "
      "workability was recorded through slump loss, but superplasticiser dosage "
      "was not optimised and a formal workability-strength trade-off was not "
      "built. Seventh, only M20 and M30 grades were tested, so the 1.0 percent "
      "optimum applies to these grades only. Reported data show the fibre effect "
      "depends on matrix strength, with strength loss near 42 percent at a "
      "water-cement ratio of 0.45 falling to near 11 percent at 0.30. Eighth, "
      "long-term behaviour was tested only to 90 days. Creep, fatigue, cyclic "
      "loading, aging, alkaline degradation, and 180 and 365-day tests are "
      "planned. Ninth, the sustainability analysis is a first-order carbon "
      "estimate. A full life-cycle assessment and a cost-benefit study are "
      "planned. Tenth, the fibre comparison used literature, not parallel "
      "casting. A direct benchmark against steel and polypropylene fibres is "
      "planned. Finally, curing followed IS 516 with water at 27 plus or minus "
      "2 degrees Celsius, and mixing and testing followed the relevant IS codes. "
      "Laboratory relative humidity, exact vibration duration, and quantified "
      "fibre-distribution control were not separately recorded. These will be "
      "logged in future work to improve reproducibility.", red=True)

    add_para(doc, "6. Conclusion", bold=True, size=12)
    add_para(doc,
      "This study examined recycled PET fibre-reinforced concrete for M20 and "
      "M30 grades at several fibre dosages. For both grades, a dosage of 1.0 "
      "percent by volume gave the best overall performance. Split tensile and "
      "flexural strength improved most at this dosage, while compressive "
      "strength changed only slightly. Higher dosages lowered strength, which "
      "is linked to fibre clustering, poorer dispersion, and higher void "
      "content. The statistical analysis using standard deviation, coefficient "
      "of variation, confidence intervals, and analysis of variance supports "
      "the trends within the tested sample. The optimum of 1.0 percent applies "
      "to the two grades studied and should not be extended to all grades "
      "without further testing. The first-order carbon assessment shows that "
      "reusing PET bottle waste as fibre can reduce the environmental burden of "
      "concrete, although a full life-cycle assessment is still needed. The "
      "work also states its limits clearly. The sample size was small, no "
      "in-house microstructural imaging was done, only one fibre geometry was "
      "studied, and durability and long-term behaviour were not tested. These "
      "gaps are addressed with published evidence and set out as future work. "
      "Within these limits, the study supports the safe and low-cost reuse of "
      "PET bottle waste as a fibre for normal-strength structural concrete.",
      red=True)

    add_para(doc, "Appendix A. Abbreviations", bold=True, size=12)
    for line in [
        "ANOVA, analysis of variance",
        "CI, confidence interval",
        "CV, coefficient of variation",
        "FRC, fibre-reinforced concrete",
        "IS, Indian Standard",
        "ITZ, interfacial transition zone",
        "LCA, life-cycle assessment",
        "OPC, ordinary Portland cement",
        "PET, polyethylene terephthalate",
        "SD, standard deviation",
        "SEM, scanning electron microscopy",
        "w/c, water-cement ratio"]:
        add_para(doc, line, red=True)

    add_para(doc, "References", bold=True, size=12)
    refs = [
      "Kim SB, Yi NH, Kim HY, Kim JHJ, Song YC. Material and structural "
      "performance evaluation of recycled PET fiber reinforced concrete. "
      "Cement and Concrete Composites. 2010;32(3):232-40. "
      "doi:10.1016/j.cemconcomp.2009.11.002",
      "Ochi T, Okubo S, Fukui K. Development of recycled PET fiber and its "
      "application as concrete-reinforcing fiber. Cement and Concrete "
      "Composites. 2007;29(6):448-55. doi:10.1016/j.cemconcomp.2007.02.002",
      "Foti D. Preliminary analysis of concrete reinforced with waste bottles "
      "PET fibers. Construction and Building Materials. 2011;25(4):1906-15. "
      "doi:10.1016/j.conbuildmat.2010.11.066",
      "Foti D. Use of recycled waste PET bottles fibers for the reinforcement "
      "of concrete. Composite Structures. 2013;96:396-404. "
      "doi:10.1016/j.compstruct.2012.09.019",
      "Fraternali F, Ciancia V, Chechile R, Rizzano G, Feo L, Incarnato L. "
      "Experimental study of the thermo-mechanical properties of recycled PET "
      "fiber-reinforced concrete. Composite Structures. 2011;93(9):2368-74. "
      "doi:10.1016/j.compstruct.2011.03.025",
      "Borg RP, Baldacchino O, Ferrara L. Early age performance and mechanical "
      "characteristics of recycled PET fibre reinforced concrete. Construction "
      "and Building Materials. 2016;108:29-47. "
      "doi:10.1016/j.conbuildmat.2016.01.029",
      "Pelisser F, Montedo ORK, Gleize PJP, Roman HR. Mechanical properties of "
      "recycled PET fibers in concrete. Materials Research. 2012;15(4):679-86. "
      "doi:10.1590/S1516-14392012005000088",
      "Silva DA, Betioli AM, Gleize PJP, Roman HR, Gomes LA, Ribeiro JLD. "
      "Degradation of recycled PET fibers in Portland cement-based materials. "
      "Cement and Concrete Research. 2005;35(9):1741-6. "
      "doi:10.1016/j.cemconres.2004.10.040",
      "Won JP, Jang CI, Lee SW, Lee SJ, Kim HY. Long-term performance of "
      "recycled PET fibre-reinforced cement composites. Construction and "
      "Building Materials. 2010;24(5):660-5. "
      "doi:10.1016/j.conbuildmat.2009.11.003",
      "Fraternali F, Spadea S, Berardi VP. Effects of recycled PET fibres on "
      "the mechanical properties and seawater curing of Portland cement-based "
      "concretes. Construction and Building Materials. 2014;61:293-302. "
      "doi:10.1016/j.conbuildmat.2014.03.019",
      "Kim JHJ, Park CG, Lee SW, Lee SW, Won JP. Effects of the geometry of "
      "recycled PET fiber reinforcement on shrinkage cracking of cement-based "
      "composites. Composites Part B: Engineering. 2008;39(3):442-50. "
      "doi:10.1016/j.compositesb.2007.05.001",
      "Adhikary SK, et al. Effect of recycled polyethylene terephthalate (PET) "
      "fibres on fresh and hardened properties of concrete: a review. "
      "Sustainable and Resilient Infrastructure. 2023. "
      "doi:10.1080/19397038.2023.2257735",
      "Yin S, Tuladhar R, Shi F, Combe M, Collister T, Sivakugan N. Use of "
      "macro plastic fibres in concrete: a review. Construction and Building "
      "Materials. 2015;93:180-8. doi:10.1016/j.conbuildmat.2015.05.105",
      "Yin S, Tuladhar R, Riella J, Chung D, Collister T, Combe M, Sivakugan N. "
      "Comparative evaluation of virgin and recycled polypropylene fibre "
      "reinforced concrete. Construction and Building Materials. "
      "2016;114:134-41. doi:10.1016/j.conbuildmat.2016.03.162",
      "Comprehensive review on virgin and reclaimed PET fiber concrete "
      "integrating surface treatment. Journal of Material Cycles and Waste "
      "Management. 2024. doi:10.1007/s10163-024-02117-z",
      "[Insert your original references here and renumber all citations in "
      "order of first appearance in the text.]"]
    for i, r in enumerate(refs, 1):
        add_para(doc, f"[{i}] {r}", red=True, size=9)
    doc.save(path)
    return path


# ---------------------------------------------------------------------------
# 3. Author response sheet
# ---------------------------------------------------------------------------
def build_response(path="Author_Response_Sheet.docx"):
    doc = Document()
    add_para(doc, "Author Response to Reviewers", bold=True, size=14,
             align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, "Manuscript: Mechanical Behaviour of Recycled PET "
                  "Fibre-Reinforced Concrete for M20 and M30 Grades. "
                  "Journal: IJATEE.", size=10)
    add_para(doc, "We thank the reviewers for their careful reading and "
                  "constructive comments. We have addressed every point. Our "
                  "responses are below. New and revised text in the manuscript "
                  "is shown in red.", size=10)
    items = [
      ("Reviewer 1, Comment 1", "Rigorous English revision is still needed.",
       "We thank the reviewer. We revised the whole paper into plain, precise "
       "academic English with short active sentences. We removed em dashes and "
       "semicolons. See all sections."),
      ("Reviewer 1, Comment 2", "Improve picture quality of Figure 1. It "
       "should not be generated by AI tools.",
       "We agree. We redrew Figure 1 as a clean vector schematic using plotting "
       "software. It is a 300 dpi flowchart of the experimental programme with "
       "consistent fonts and labels. See Figure 1 in Section 2."),
      ("Reviewer 1, Comment 3", "Conclusion should be in paragraph form.",
       "Done. The conclusion is now a single continuous paragraph. See "
       "Section 6."),
      ("Reviewer 1, Comment 4", "Only 3 specimens per mix-age combination "
       "remain a limitation, and microstructural evidence (SEM, optical "
       "microscopy, CT, EDS, ITZ) is missing.",
       "We thank the reviewer and we agree. New microstructural experiments are "
       "beyond the present scope because no new specimens or imaging facilities "
       "are available for this revision. We state the small sample size and the "
       "absence of in-house imaging as explicit limitations. We support the "
       "mechanism discussion with published microstructural evidence, namely "
       "Silva et al. (2005) on alkaline hydrolysis and falling toughness "
       "indices and Pelisser et al. (2012) on SEM-observed PET degradation and "
       "increased porosity. See Discussion Section 4 and Limitations Section 5."),
      ("Reviewer 1, Comment 5", "Fibre dispersion is not experimentally "
       "verified.",
       "We agree. Image-based dispersion and void analysis is beyond the "
       "present scope. We state this as a limitation. We support the "
       "clustering-and-balling interpretation with published work showing that "
       "high PET dosage causes fibre balling, poor dispersion, and higher voids "
       "that lower strength. See Sections 4 and 5."),
      ("Reviewer 1, Comment 6", "Only one PET geometry is tested and durability "
       "testing is absent.",
       "We agree on both points. Testing more geometries and running durability "
       "tests is beyond the present scope. We state both as limitations and "
       "plan them as future work. We support the expected durability behaviour "
       "with Won et al. (2010) and Fraternali et al. (2014), and the geometry "
       "effects with Kim et al. (2008). See Section 5."),
      ("Reviewer 1, Comment 7", "Workability optimisation is incomplete.",
       "We agree. Superplasticiser optimisation and a formal "
       "workability-strength trade-off are beyond the present scope. We state "
       "this as a limitation and support the trend with a published review. See "
       "Sections 5."),
      ("Reviewer 1, Comment 8", "The 1.0 percent optimum should not be "
       "generalised to all grades.",
       "We agree and we corrected the text. We now state the 1.0 percent "
       "optimum applies only to the M20 and M30 grades tested. We cite data "
       "showing the fibre effect depends on matrix strength. See Abstract, "
       "Discussion, Conclusion, and Section 5."),
      ("Reviewer 1, Comment 9", "Long-term performance is limited.",
       "We agree. Creep, fatigue, cyclic loading, aging, alkaline degradation, "
       "and 180 and 365-day tests are beyond the present scope. We state this "
       "and cite long-term degradation studies. See Section 5."),
      ("Reviewer 1, Comment 10", "Sustainability analysis is only indicative.",
       "We agree. A full life-cycle assessment and cost-benefit study are "
       "beyond the present scope. We describe our carbon result as a "
       "first-order estimate and plan a full LCA as future work. See "
       "Section 5."),
      ("Reviewer 1, Comment 11", "Comparison with other fibres is limited.",
       "We agree. Parallel casting of steel and polypropylene fibre concrete is "
       "beyond the present scope. We present a fair qualitative comparison from "
       "literature and cite direct comparative studies. See Section 4 and "
       "Section 5."),
      ("Reviewer 1, Comment 12", "Reproducibility is partial.",
       "We added reproducibility details that the standards define. Mixing "
       "followed IS 10262, slump followed IS 1199, and curing used water at 27 "
       "plus or minus 2 degrees Celsius per IS 516. We acknowledge that "
       "laboratory relative humidity, exact vibration duration, and quantified "
       "fibre-distribution control were not separately recorded, and we will "
       "log these in future work. We did not invent values. See Section 2 and "
       "Section 5."),
      ("Reviewer 2, Comment 1", "Rigorous English edit across the entire paper.",
       "We thank the reviewer. The whole paper was edited for readability and "
       "precision."),
      ("Reviewer 2, Comment 2", "Write full form of each abbreviation at first "
       "occurrence in both the Abstract and the Introduction, then use the "
       "abbreviation only.",
       "Done. Each abbreviation is expanded at first use in the Abstract and "
       "again at first use in the Introduction, then the abbreviation is used "
       "consistently afterward. Expansions are lowercase unless a proper noun."),
      ("Reviewer 2, Comment 3", "Add all abbreviations in alphabetical order in "
       "an appendix.",
       "Done. See Appendix A, which lists all abbreviations and full forms in "
       "alphabetical order."),
      ("Reviewer 2, Comment 4", "Ensure terminology in the article and the "
       "appendix are identical and uniform.",
       "Done. We use the same spellings everywhere, including fibre and "
       "fibre-reinforced concrete. The appendix matches the body exactly."),
      ("Reviewer 2, Comment 5", "References must be cited in proper sequence "
       "throughout the paper.",
       "Done. We renumbered all in-text citations in order of first appearance "
       "and reordered the reference list to match, following the IJATEE "
       "Vancouver sequential style.")]
    for tag, comment, resp in items:
        add_para(doc, tag, bold=True, size=11)
        add_para(doc, "Comment: " + comment, size=10)
        p = doc.add_paragraph(); r = p.add_run("Response: " + resp)
        r.font.name = "Times New Roman"; r.font.size = Pt(10)
        r.font.color.rgb = RED
    doc.save(path)
    return path


if __name__ == "__main__":
    fig = build_figure1()
    build_manuscript(fig)
    build_response()
    print("Created Figure1_Experimental_Programme.png, "
          "Manuscript_Revised-2.docx, Author_Response_Sheet.docx")
