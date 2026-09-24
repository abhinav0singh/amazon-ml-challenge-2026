"""
make_briefing_pdf.py: build docs/TEAM_BRIEFING.pdf, the one-document onboarding
for the whole team -- problem, data, strategy, workflow and the 3-day plan.

Not part of the scoring pipeline. Needs only reportlab:

    python -m pip install reportlab
    python docs/make_briefing_pdf.py

Every number in here was measured from the dataset TSVs on 25 Sep 2026 and is
mirrored in AGENTS.md section 1. If a real run ever contradicts one, fix both
files rather than letting them drift apart.

Note on characters: reportlab's built-in fonts use WinAnsi encoding, which has
no glyphs for arrows or the <= / >= / ~= symbols -- they render as black boxes.
Use ASCII equivalents ("->", "<=", "approx.") in all strings below.
"""
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

OUT = "docs/TEAM_BRIEFING.pdf"

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#5a5f66")
ACCENT = colors.HexColor("#16507a")
RULE = colors.HexColor("#c8cdd3")
BAND = colors.HexColor("#eef2f6")
WARN = colors.HexColor("#8a3a12")

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("title", parent=ss["Title"], fontName="Helvetica-Bold",
                            fontSize=26, leading=31, textColor=INK, spaceAfter=4),
    "sub": ParagraphStyle("sub", parent=ss["Normal"], fontName="Helvetica",
                          fontSize=11.5, leading=16, textColor=MUTED, alignment=TA_CENTER),
    "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold",
                         fontSize=16, leading=20, textColor=ACCENT,
                         spaceBefore=16, spaceAfter=7),
    "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold",
                         fontSize=11.5, leading=15, textColor=INK,
                         spaceBefore=11, spaceAfter=4),
    "body": ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica",
                           fontSize=9.7, leading=14.2, textColor=INK, spaceAfter=6),
    "bullet": ParagraphStyle("bullet", parent=ss["Normal"], fontName="Helvetica",
                             fontSize=9.7, leading=14.2, textColor=INK,
                             leftIndent=11, bulletIndent=2, spaceAfter=3.5),
    "cell": ParagraphStyle("cell", parent=ss["Normal"], fontName="Helvetica",
                           fontSize=8.7, leading=12.2, textColor=INK),
    "cellb": ParagraphStyle("cellb", parent=ss["Normal"], fontName="Helvetica-Bold",
                            fontSize=8.7, leading=12.2, textColor=INK),
    "callout": ParagraphStyle("callout", parent=ss["Normal"], fontName="Helvetica",
                              fontSize=9.7, leading=14.2, textColor=WARN),
    "foot": ParagraphStyle("foot", parent=ss["Normal"], fontName="Helvetica",
                           fontSize=8, leading=11, textColor=MUTED),
}


def P(t, s="body"):
    return Paragraph(t, S[s])


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def table(rows, widths, head=True):
    """Grid table; first row is a header band when `head` is set."""
    data = [[Paragraph(c, S["cellb"] if (head and r == 0) else S["cell"])
             for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if head else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, RULE),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 5),
             ("RIGHTPADDING", (0, 0), (-1, -1), 5),
             ("TOPPADDING", (0, 0), (-1, -1), 4),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    if head:
        style.append(("BACKGROUND", (0, 0), (-1, 0), BAND))
    t.setStyle(TableStyle(style))
    return t


def callout(title, body):
    """Boxed warning block, kept on one page."""
    inner = [Paragraph(f"<b>{title}</b>", S["callout"]), Paragraph(body, S["callout"])]
    t = Table([[inner]], colWidths=[165 * mm])
    t.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.8, WARN),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fdf4ee")),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return t


def decorate(canvas, doc):
    """Footer: page number and a standing reminder of what the doc is."""
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(22 * mm, 15 * mm, 188 * mm, 15 * mm)
    canvas.setFont("Helvetica", 7.6)
    canvas.setFillColor(MUTED)
    canvas.drawString(22 * mm, 10.5 * mm,
                      "Amazon ML Challenge 2026 - Business Entity Resolution - team briefing")
    canvas.drawRightString(188 * mm, 10.5 * mm, str(doc.page))
    canvas.restoreState()


def story():
    s = []

    # ---------------- cover ----------------
    s += [Spacer(1, 42 * mm),
          P("Business Entity Resolution", "title"),
          P("Amazon ML Challenge 2026 &middot; team briefing", "sub"),
          Spacer(1, 7 * mm),
          P("What we are building, what the data says, how we intend to win,<br/>"
            "and who does what across the three days.", "sub"),
          Spacer(1, 14 * mm)]
    s += [table([
        ["Window", "25 Sep 2026 00:00 IST -> 27 Sep 2026 23:59 IST"],
        ["Uploads", "5 per day, 15 total, then the submit button disables"],
        ["Ranking", "Private leaderboard decides the final result"],
        ["Metric", "F0.5 per Source-1 entity, macro-averaged over all entities"],
        ["Team", "P1 Abhinav (lead) &middot; P2 blocking &middot; P3 features &middot; P4 decision and docs"],
        ["Contract", "AGENTS.md is binding. This document explains it; it does not replace it."],
    ], [30 * mm, 135 * mm], head=False)]
    s += [Spacer(1, 10 * mm),
          P("Every number in this document was measured from the dataset files on "
            "25 Sep 2026. Nothing here is estimated or remembered. Where a figure is "
            "not yet measured, it says so.", "foot")]
    s.append(PageBreak())

    # ---------------- 1. problem ----------------
    s += [P("1. The problem", "h1"),
          P("Three independent sources describe the same set of real businesses. They share "
            "no identifiers. Names and addresses are noisy: abbreviations, legal-suffix "
            "differences, transliterations, typos, missing components, landmark-based "
            "directions. <b>Source 1 is already deduplicated</b> and acts as the reference."),
          P("For <b>every</b> Source 1 record in the test set, we output which Source 2 and "
            "Source 3 records describe the same business. The answer may be zero, one, or many."),
          P("The deliverable is one tab-separated file, <b>1,732,544 rows plus a header</b>, "
            "one row per test Source 1 entity, with an empty field where the answer is "
            "\"nothing matches this\"."),
          P("This is a record-linkage problem, and its defining difficulty is arithmetic: "
            "1.7 million by 10 million is 17 trillion possible pairs. We cannot score them. "
            "So the pipeline must cheaply narrow the field, then judge what survives "
            "precisely. Everything else follows from that constraint.")]

    s += [P("Noise we must absorb", "h2")]
    s += [table([
        ["Field", "Variation we have been told to expect"],
        ["Name", "Corp / Corporation, Pvt / Private, Ltd / Limited; DBA and trade names; "
                 "&amp; versus \"and\"; word-order transposition; typos"],
        ["Address", "Rd / Road, St / Street; transliteration variants; missing PIN or state; "
                    "landmark references (\"Near SBI ATM\"); municipal numbering; reordering"],
        ["Country", "An <b>open set</b> of labels. Train has US and India. Test adds France. "
                    "Never hard-code the set."],
    ], [26 * mm, 139 * mm])]

    # ---------------- 2. data ----------------
    s += [P("2. The data", "h1"),
          P("Counted directly from the TSVs, not sampled or estimated.")]
    s += [table([
        ["", "Train", "Test"],
        ["Source 1 entities", "2,206,822", "<b>1,732,544</b>"],
        ["Source 2 records", "5,034,616", "4,887,273"],
        ["Source 3 records", "5,285,603", "5,082,316"],
        ["Countries (S1)", "US 1,323,633<br/>India 883,188",
         "US 663,106<br/>India 809,986<br/><b>France 259,452 (15.0%)</b>"],
        ["Total size on disk", "approx. 1.3 GB", "approx. 1.2 GB"],
    ], [42 * mm, 58 * mm, 65 * mm])]

    s += [P("The five findings that shape our strategy", "h2")]
    s += bullets([
        "<b>Singletons are 5.58% of entities</b> (123,247 of 2,206,822). Predicting an empty "
        "list for everything scores approximately 0.056. This matters enormously and is "
        "covered in the callout below.",
        "<b>Mean 3.67 true matches</b> per non-singleton entity; 7,638,365 true pairs in "
        "training. Most entities have several matches, and finding them is the work.",
        "<b>No Source 2 or 3 record belongs to two Source 1 entities.</b> Exactly zero across "
        "all 7.6 million pairs. One-to-one is a hard constraint we get for free, and it is a "
        "pure precision gain, which this metric rewards double.",
        "<b>France is 15.0% of the test set</b> and appears nowhere in training. Real, bounded, "
        "and the reason nothing in the pipeline may branch on a country name.",
        "<b>Test Source 2 and 3 records carry country labels, France included</b>, so blocking "
        "can group by country on every split without a fallback that would compare against all "
        "10 million records.",
    ])

    s.append(PageBreak())

    # ---------------- 3. metric ----------------
    s += [P("3. The metric, and what it actually implies", "h1"),
          P("F0.5 is computed <b>per Source 1 entity</b>, then averaged over <b>all</b> "
            "entities including singletons. An entity with one candidate counts exactly as "
            "much as an entity with forty. Pair-level accuracy, AUC and logloss are proxies "
            "only; none of them is the objective."),
          P("F0.5 = (1.25 &times; P &times; R) / (0.25 &times; P + R)")]

    s += [P("Worked consequences, for an entity with 4 true matches", "h2")]
    s += [table([
        ["What we predict", "Precision", "Recall", "F0.5"],
        ["3 correct, 0 wrong", "1.00", "0.75", "<b>0.938</b>"],
        ["4 correct, 1 wrong", "0.80", "1.00", "<b>0.833</b>"],
        ["Nothing at all", "-", "0.00", "<b>0.000</b>"],
    ], [66 * mm, 33 * mm, 33 * mm, 33 * mm])]

    s += [Spacer(1, 3 * mm),
          P("Read those three rows together. Dropping an uncertain candidate beats adding a "
            "wrong one - that is the precision weighting. But refusing to answer at all is "
            "catastrophic, and that is the part a precision-first reflex gets wrong.")]

    s += [Spacer(1, 3 * mm), callout(
        "The correction that changes our strategy",
        "Our own AGENTS.md originally said singletons were \"a large share of the total "
        "score\". They are 5.58%. Predicting empty everywhere scores 0.056, not something "
        "near 0.5. <b>94.4% of the available score requires actually finding matches.</b> "
        "The right posture is high precision that still commits on most entities - not "
        "maximum caution. A team that tunes toward silence caps itself near 0.056. "
        "AGENTS.md section 2 has been corrected; work from the corrected version.")]

    # ---------------- 4. approach ----------------
    s += [P("4. How the pipeline works", "h1"),
          P("Five stages, all in <font face=\"Courier\">src/</font>. Each one narrows or "
            "sharpens what the next one sees.")]
    s += [table([
        ["Stage", "File", "What it does", "Why it matters"],
        ["1. Normalise", "normalize.py",
         "Lowercase, strip accents so Societe matches Soci&eacute;t&eacute;, expand "
         "abbreviations, split the name into full and legal-suffix-free \"core\" forms.",
         "Country-agnostic by construction. This is what makes an unseen France survivable."],
        ["2. Block", "blocking.py",
         "Three TF-IDF views - char n-grams on the core name, char n-grams on name plus "
         "address, word n-grams on the address - take the nearest few per entity, union them. "
         "Grouped by country.",
         "<b>Sets the ceiling on everything.</b> A true match dropped here can never be "
         "recovered downstream."],
        ["3. Pair features", "pair_features.py",
         "28 features per surviving pair: fuzzy ratios, Jaro-Winkler, TF-IDF cosines, postal "
         "and number agreement, plus rank and competition context.",
         "The context features let the model reason about alternatives, not just one pair "
         "in isolation."],
        ["4. Match", "run_pipeline.py",
         "LightGBM, 5 folds grouped by Source 1 entity, predicts P(this pair is a match).",
         "MIT licensed. Fast enough to retrain repeatedly, which matters more than raw "
         "ceiling over three days."],
        ["5. Decide", "decide.py",
         "Threshold the probabilities, then enforce one-to-one so each Source 2 or 3 record "
         "is claimed by at most one entity.",
         "<b>Where the score is actually won.</b> The only stage that directly sets "
         "precision, and currently the least invested in."],
    ], [22 * mm, 26 * mm, 62 * mm, 55 * mm])]

    s.append(PageBreak())

    # ---------------- 5. strategy ----------------
    s += [P("5. How we intend to win", "h1")]

    s += [P("Priority order, highest leverage first", "h2")]
    s += [table([
        ["#", "Lever", "Why it ranks here"],
        ["1", "Make blocking run at full scale",
         "It is the ceiling on every other number, and today the implementation cannot "
         "finish. Until this lands we have no submission at all, so its expected value "
         "dominates everything else on the board."],
        ["2", "The decision layer: threshold plus a real one-to-one assignment",
         "Precision is doubly weighted and this is the only stage that sets it directly. "
         "The one-to-one constraint is exactly true in training, and we currently exploit "
         "it only greedily. Cheap to improve, measurable in one run."],
        ["3", "Feature quality",
         "Real but incremental. Worth doing once stages 1 and 2 are solid, and only with "
         "cross-fitted evidence behind each change."],
        ["4", "France robustness",
         "15% of the score. The pipeline is already country-agnostic, so this is insurance "
         "against a regression rather than a source of upside."],
    ], [8 * mm, 52 * mm, 105 * mm])]

    s += [P("The disciplines that protect the score", "h2")]
    s += bullets([
        "<b>Report only cross-fitted numbers.</b> tune() picks its threshold on the same rows "
        "it scores, so it flatters itself. cross_fitted_score() is the honest one, and it is "
        "the only number that belongs in STATUS.md.",
        "<b>Accept a change only if</b> it beats seed noise by more than 2x <b>and</b> improves "
        "at least 4 of 5 folds. Anything smaller is logged as inconclusive and dropped. Most "
        "ideas will not clear this bar, and that is the point.",
        "<b>Score every entity, including the ones blocking found nothing for.</b> An entity "
        "with no candidates is a real prediction of \"empty\" and still scores. Filtering them "
        "out inflates the number and fools us.",
        "<b>Trust CV over the public leaderboard.</b> The public board scores a subset; the "
        "private board decides the result. When they disagree, suspect a submission bug first, "
        "then a metric mismatch, then leakage, then shift. Noise is the last explanation, "
        "never the first.",
        "<b>Check LOCO on anything that touches features.</b> A change that lifts CV but drops "
        "leave-one-country-out will hurt us on 15% of the test set.",
    ])

    s += [P("How we spend 15 uploads", "h2"),
          P("Uploads answer questions cross-validation cannot. They do not settle threshold "
            "nudges, hyperparameter tweaks or curiosity - a public-board difference smaller "
            "than its own noise is not evidence of anything.")]
    s += [table([
        ["Worth an upload", "Never worth an upload"],
        ["The format check (the empty submission)<br/>"
         "The first working end-to-end pipeline<br/>"
         "A structurally different blocking or feature family<br/>"
         "An ensemble<br/>"
         "The two reserved final picks",
         "A threshold nudge<br/>"
         "A hyperparameter tweak<br/>"
         "A blend-weight change<br/>"
         "\"Let's just see what happens\"<br/>"
         "Anything we could have answered with CV"],
    ], [82 * mm, 83 * mm])]

    s.append(PageBreak())

    # ---------------- 6. workflow ----------------
    s += [P("6. Team workflow", "h1")]
    s += [table([
        ["Who", "Owns", "Current task"],
        ["<b>P1 Abhinav</b><br/>Team lead",
         "metric.py, cv_folds.py, run_pipeline.py, STATUS.md, all uploads, final zip",
         "Issue #5. Runs the pipeline, owns every upload, generates and commits the locked "
         "folds file."],
        ["<b>P2</b><br/>Data analyst", "normalize.py, blocking.py",
         "<b>Issue #1 - critical path.</b> Replace the dense top-k with a sparse one so "
         "blocking can finish."],
        ["<b>P3</b><br/>ML engineer", "pair_features.py, model params",
         "<b>Issue #2.</b> Vectorise feature construction and remove the pair-length reindex."],
        ["<b>P4</b><br/>Decision and docs", "decide.py, approach_document.md, STATUS.md header",
         "Issues #3 and #4. Decision layer, and start the methodology document on day 1."],
    ], [28 * mm, 58 * mm, 79 * mm])]

    s += [P("Rules of the road", "h2")]
    s += bullets([
        "Branch per person. Small commits. Merge to main fast - <b>self-merge plus a chat "
        "ping beats waiting for review</b>. With 68 hours left, a pull request sitting "
        "unreviewed overnight is the expensive failure, not an unreviewed merge.",
        "Before any push: <font face=\"Courier\">metric.py</font> passes, and your change ran "
        "end to end at least once.",
        "<b>Never run <font face=\"Courier\">git add .</font></b> - name the paths. A blanket "
        "add has already pulled an entire virtual environment into a commit once.",
        "Always use <font face=\"Courier\">.venv\\Scripts\\python</font>, never bare python. "
        "Build the environment with Python 3.12; 3.14 cannot install our pinned versions.",
        "After every experiment: one row in STATUS.md section 5, one line in section 8. "
        "Section 8's headings match the organisers' template, so the write-up assembles itself "
        "instead of being written from scratch at midnight on day 3.",
        "Every upload gets a git tag and a ledger row <b>before</b> it goes up. Version history "
        "is required for shortlisting.",
        "Develop against <font face=\"Courier\">--sample 2000</font> for a three-minute feedback "
        "loop. Its scores are not CV and must never be logged as results.",
    ])

    s += [Spacer(1, 2 * mm), callout(
        "The rule that ends our run if broken",
        "No external data lookup of any kind. No geocoding APIs, no business registries, no "
        "commercial entity-resolution services, no scraped or downloaded reference data. "
        "Hand-written abbreviation dictionaries are fine - they rewrite tokens already in the "
        "record. A list of real place names would not be: that is a gazetteer, which is "
        "external data. Code packages are reviewed by humans, and the penalty is "
        "disqualification for the whole team regardless of who wrote the line.")]

    s.append(PageBreak())

    # ---------------- 7. three days ----------------
    s += [P("7. The three days", "h1"),
          P("Each day has one job. A day that achieves its job and nothing else is a good day; "
            "a day that produces five interesting experiments and no working pipeline is not.")]

    s += [P("Day 1 (25 Sep) - make it run", "h2"),
          P("<b>Job: turn code that has never executed into a pipeline that finishes.</b> "
            "Nothing else on this list matters until blocking and features scale.")]
    s += bullets([
        "P2 and P3 work issues #1 and #2 in parallel. These are the critical path.",
        "P1 uploads D1-1, the empty submission. It proves the upload path end to end and its "
        "public score reveals the singleton share of the public subset - information we cannot "
        "obtain any other way.",
        "P1 sends the organisers' query form the one question that matters: does the private "
        "leaderboard score our last submission, our best, or one we select? Neither official "
        "document states it, and it decides the entire day 3 endgame.",
        "P4 starts the methodology document and the decision-layer work against sampled output.",
        "<b>Day 1 done when:</b> a full-scale run has been started, and D1-1 is on the board.",
    ])

    s += [P("Day 2 (26 Sep) - make it good", "h2"),
          P("<b>Job: produce the first honest CV number, then improve it deliberately.</b>")]
    s += bullets([
        "First full-scale run completes. P1 records blocking recall, cross-fitted CV and LOCO "
        "in STATUS.md, and commits <font face=\"Courier\">work/folds.csv</font> so every "
        "teammate shares the same folds from that point on.",
        "First real model submission. Compare its public score against CV and reconcile any "
        "gap before trusting either.",
        "P4 takes the decision layer seriously: proper one-to-one assignment, threshold "
        "behaviour by candidate count. This is the highest-value modelling work available.",
        "Feature and blocking iterations, each one accepted or rejected on the 2x-noise and "
        "4-of-5-folds rule.",
        "<b>Day 2 done when:</b> we have a cross-fitted CV number we believe, and at least one "
        "real model submission scored on the public board.",
    ])

    s += [P("Day 3 (27 Sep) - make it safe", "h2"),
          P("<b>Job: convert whatever we have into the best defensible submission, and stop "
            "in time.</b> Day 3 is where good scores get lost to rushing.")]
    s += bullets([
        "Morning: last structural experiment. Anything not working by midday is abandoned, not "
        "rescued.",
        "Afternoon: freeze the pipeline. Final full run. Validate with "
        "<font face=\"Courier\">--check-ids</font>.",
        "<b>By 22:00 IST:</b> both reserved uploads in - the best-CV version, and one hedge "
        "that differs on our biggest open risk, France robustness. Do not leave these to the "
        "last hour; the portal is busiest then.",
        "<b>By 23:00 IST:</b> final zip ready - both TSVs, a runnable code folder with README "
        "and pinned requirements, and the filled methodology document.",
        "<b>Day 3 done when:</b> the zip would survive a reviewer opening it cold, without us "
        "in the room to explain anything.",
    ])

    s += [P("Inside any day", "h2"),
          P("Pull main. Take everyone's work. Do one thing. Run it. Log the number - whether it "
            "helped or not, because an inconclusive result recorded is worth more than a "
            "promising one forgotten. Merge. Ping the chat. Repeat.")]
    s += [Spacer(1, 4 * mm),
          P("Post in chat when you start something, when you finish, and immediately when you "
            "are blocked. That last one is what actually saves hours. Nobody is watching GitHub "
            "notifications at three in the morning.", "foot")]

    return s


def main():
    doc = SimpleDocTemplate(
        OUT, pagesize=A4,
        leftMargin=22 * mm, rightMargin=22 * mm, topMargin=20 * mm, bottomMargin=22 * mm,
        title="Business Entity Resolution - Amazon ML Challenge 2026 team briefing",
        author="Team briefing")
    doc.build(story(), onFirstPage=decorate, onLaterPages=decorate)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
