// Builds TEAM_HANDBOOK.docx — the working document for the team.
// Run: node make_handbook.js
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  PageBreak, LevelFormat, Footer, PageNumber,
} = require("docx");

const W = 9026;                       // A4 content width in DXA
const INK = "1A1A1A", ACCENT = "16507A", MUTED = "5A5F66", WARN = "8A3A12";
const BAND = "EEF2F6", WARNBG = "FDF4EE";

const P = (text, o = {}) => new Paragraph({
  spacing: { after: o.after ?? 120, line: 280 },
  alignment: o.align,
  children: [new TextRun({ text, bold: o.bold, italics: o.italics,
    color: o.color ?? INK, size: o.size ?? 20, font: o.font })],
});

const H1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 140 },
  children: [new TextRun({ text, bold: true, color: ACCENT, size: 30 })],
});
const H2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2, spacing: { before: 220, after: 100 },
  children: [new TextRun({ text, bold: true, color: INK, size: 23 })],
});

const BULLET = (text, o = {}) => new Paragraph({
  numbering: { reference: "bullets", level: 0 },
  spacing: { after: 70, line: 280 },
  children: [new TextRun({ text, size: 20, color: o.color ?? INK, bold: o.bold })],
});

// Rich paragraph: array of [text, {bold?, code?}] pairs
const RP = (runs, o = {}) => new Paragraph({
  spacing: { after: o.after ?? 120, line: 280 },
  numbering: o.bullet ? { reference: "bullets", level: 0 } : undefined,
  children: runs.map(([t, f = {}]) => new TextRun({
    text: t, bold: f.b, italics: f.i, size: 20,
    color: f.color ?? INK, font: f.code ? "Consolas" : undefined,
  })),
});

const CODE = (text) => new Paragraph({
  spacing: { before: 60, after: 120, line: 260 },
  shading: { type: ShadingType.CLEAR, fill: "F4F6F8" },
  border: { left: { style: BorderStyle.SINGLE, size: 12, color: ACCENT, space: 6 } },
  indent: { left: 120 },
  children: [new TextRun({ text, font: "Consolas", size: 17, color: INK })],
});

function cell(text, { bold, fill, width, code } = {}) {
  const lines = String(text).split("|||");
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: fill ? { type: ShadingType.CLEAR, fill } : undefined,
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    children: lines.map((l) => new Paragraph({
      spacing: { after: 0, line: 260 },
      children: [new TextRun({ text: l, bold, size: 18, color: INK,
        font: code ? "Consolas" : undefined })],
    })),
  });
}

function table(rows, widths, { code = false } = {}) {
  return new Table({
    columnWidths: widths,
    width: { size: W, type: WidthType.DXA },
    rows: rows.map((r, i) => new TableRow({
      tableHeader: i === 0,
      children: r.map((c, j) => cell(c, {
        bold: i === 0, fill: i === 0 ? BAND : undefined, width: widths[j],
        code: code && i > 0 && j === 0,
      })),
    })),
  });
}

function callout(title, body) {
  return new Table({
    columnWidths: [W], width: { size: W, type: WidthType.DXA },
    rows: [new TableRow({
      children: [new TableCell({
        width: { size: W, type: WidthType.DXA },
        shading: { type: ShadingType.CLEAR, fill: WARNBG },
        margins: { top: 130, bottom: 130, left: 150, right: 150 },
        children: [
          new Paragraph({ spacing: { after: 70 },
            children: [new TextRun({ text: title, bold: true, color: WARN, size: 20 })] }),
          ...body.map((b) => new Paragraph({ spacing: { after: 60, line: 280 },
            children: [new TextRun({ text: b, color: WARN, size: 19 })] })),
        ],
      })],
    })],
  });
}

const SP = (h = 140) => new Paragraph({ spacing: { after: h }, children: [] });

const doc = new Document({
  creator: "Amazon ML Challenge 2026 team",
  title: "Business Entity Resolution — Team Handbook",
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 340, hanging: 200 } } } }],
    }],
  },
  styles: { default: { document: { run: { font: "Calibri", size: 20 } } } },
  sections: [{
    properties: { page: { margin: { top: 1200, bottom: 1200, left: 1440, right: 1440 } } },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.RIGHT,
        children: [new TextRun({ text: "Business Entity Resolution — team handbook      ",
          size: 15, color: MUTED }),
          new TextRun({ children: [PageNumber.CURRENT], size: 15, color: MUTED })],
      })] }),
    },
    children: build(),
  }],
});

function build() {
  const c = [];

  // ---------- cover ----------
  c.push(SP(1700));
  c.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 90 },
    children: [new TextRun({ text: "Business Entity Resolution", bold: true, size: 52, color: INK })] }));
  c.push(P("Amazon ML Challenge 2026 — team handbook", { align: AlignmentType.CENTER, size: 24, color: MUTED }));
  c.push(SP(300));
  c.push(P("What we are solving, what the data actually contains, how the pipeline works,",
    { align: AlignmentType.CENTER, size: 21, color: MUTED, after: 20 }));
  c.push(P("what is assigned to whom, and exactly what to run.",
    { align: AlignmentType.CENTER, size: 21, color: MUTED }));
  c.push(SP(420));
  c.push(table([
    ["Written", "25 September 2026, ~05:30 IST (Day 1, overnight)"],
    ["Deadline", "27 September 2026, 23:59 IST"],
    ["Uploads", "5 per day, 15 total. Final ranking from the PRIVATE leaderboard"],
    ["Status", "Phase 0. No validation score measured yet. Zero uploads used"],
    ["Contract", "AGENTS.md is binding. This handbook explains it; it does not replace it"],
  ], [1900, 7126]));
  c.push(SP(260));
  c.push(P("Every number in this handbook was measured from the dataset or from a run that actually happened. Estimates are labelled as estimates. If you find a number here that a run contradicts, the run wins — fix the document.",
    { size: 18, color: MUTED, align: AlignmentType.CENTER }));
  c.push(new Paragraph({ children: [new PageBreak()] }));

  // ---------- 1 problem ----------
  c.push(H1("1. The problem"));
  c.push(P("Three independent sources describe the same real businesses. They share no identifiers. Names and addresses are noisy. Source 1 is already deduplicated and acts as the reference."));
  c.push(RP([["For ", {}], ["every", { b: true }], [" Source 1 record in the test set we must output which Source 2 and Source 3 records describe the same business. The answer may be zero, one, or many.", {}]]));
  c.push(RP([["The deliverable is one tab-separated file with ", {}], ["1,732,544", { b: true }], [" data rows plus a header, one row per test Source 1 entity, with an empty field where the answer is \u201cnothing matches this\u201d.", {}]]));
  c.push(P("The defining difficulty is arithmetic. 1.7 million by 10 million is 17 trillion possible pairs. We cannot score them. So the pipeline must narrow the field cheaply, then judge what survives precisely. Every design decision follows from that."));

  c.push(H2("Noise we must absorb"));
  c.push(table([
    ["Field", "Variation the organisers told us to expect"],
    ["Name", "Corp / Corporation, Pvt / Private, Ltd / Limited; DBA and trade names; \u201c&\u201d versus \u201cand\u201d; word-order transposition; typos"],
    ["Address", "Rd / Road, St / Street; transliteration variants; missing PIN or state; landmark references (\u201cNear SBI ATM\u201d); municipal numbering; component reordering"],
    ["Country", "An OPEN SET of labels. Train has US and India. Test adds France. Never hard-code the set."],
  ], [1500, 7526]));

  // ---------- 2 data ----------
  c.push(H1("2. The dataset"));
  c.push(P("Counted directly from the TSVs, not sampled or estimated."));
  c.push(table([
    ["", "Train", "Test"],
    ["Source 1 entities", "2,206,822", "1,732,544"],
    ["Source 2 records", "5,034,616", "4,887,273"],
    ["Source 3 records", "5,285,603", "5,082,316"],
    ["Countries (Source 1)", "US 1,323,633|||India 883,188", "US 663,106|||India 809,986|||France 259,452 (15.0%)"],
  ], [2400, 3213, 3413]));

  c.push(H2("The five findings that shape everything"));
  c.push(BULLET("Singletons are 5.58% of entities (123,247 of 2,206,822). Predicting an empty list for everything scores about 0.056."));
  c.push(BULLET("Mean 3.67 true matches per non-singleton entity; 7,638,365 true pairs in training. Most entities have several matches and finding them is the work."));
  c.push(BULLET("No Source 2 or 3 record belongs to two Source 1 entities. Exactly zero across all 7.6 million pairs. One-to-one is a hard constraint we get for free, and it is a pure precision gain."));
  c.push(BULLET("France is 15.0% of the test set and appears nowhere in training. This is why nothing in the pipeline may branch on a country name."));
  c.push(BULLET("Test Source 2 and 3 records carry country labels including France, so blocking can group by country on every split."));

  c.push(new Paragraph({ children: [new PageBreak()] }));

  // ---------- 3 metric ----------
  c.push(H1("3. The metric, and what it actually implies"));
  c.push(P("F0.5 is computed per Source 1 entity, then averaged over all entities including singletons. An entity with one candidate counts exactly as much as one with forty. Pair-level accuracy, AUC and logloss are proxies only \u2014 none of them is the objective."));
  c.push(CODE("F0.5 = (1.25 \u00d7 P \u00d7 R) / (0.25 \u00d7 P + R)"));
  c.push(H2("Worked consequences, for an entity with 4 true matches"));
  c.push(table([
    ["What we predict", "Precision", "Recall", "F0.5"],
    ["3 correct, 0 wrong", "1.00", "0.75", "0.938"],
    ["4 correct, 1 wrong", "0.80", "1.00", "0.833"],
    ["Nothing at all", "\u2014", "0.00", "0.000"],
  ], [3626, 1800, 1800, 1800]));
  c.push(SP(120));
  c.push(P("Read those three rows together. Dropping an uncertain candidate beats adding a wrong one \u2014 that is the precision weighting. But refusing to answer at all is catastrophic, and that is the part a precision-first reflex gets wrong."));
  c.push(callout("The correction that changed our strategy", [
    "Our own AGENTS.md originally said singletons were \u201ca large share of the total score\u201d. They are 5.58%. Predicting empty everywhere scores 0.056, not something near 0.5.",
    "94.4% of the available score requires actually finding matches. The right posture is high precision that still commits on most entities \u2014 not maximum caution. A team that tunes toward silence caps itself near 0.056.",
    "AGENTS.md section 2 has been corrected. Work from the corrected version.",
  ]));

  // ---------- 4 approach ----------
  c.push(H1("4. Our approach"));
  c.push(P("Five stages. Each narrows or sharpens what the next one sees."));
  c.push(table([
    ["Stage", "File", "What it does", "Why it matters"],
    ["1. Normalise", "normalize.py", "Lowercase, strip accents so Societe matches Soci\u00e9t\u00e9, expand abbreviations, split the name into full and legal-suffix-free \u201ccore\u201d forms.", "Country-agnostic by construction. This is what makes an unseen France survivable."],
    ["2. Block", "blocking.py", "Index every record under several complementary keys, then run three-view TF-IDF top-k inside each small block and union the results.", "Sets the ceiling on everything. A true match dropped here can never be recovered."],
    ["3. Features", "pair_features.py", "28 features per surviving pair: fuzzy ratios, Jaro-Winkler, TF-IDF cosines, postal and number agreement, plus rank and competition context.", "Context features let the model reason about alternatives, not one pair in isolation."],
    ["4. Match", "run_pipeline.py", "LightGBM, 5 folds grouped by Source 1 entity, predicting P(this pair is a match).", "MIT licensed. Fast enough to retrain repeatedly, which matters more than raw ceiling."],
    ["5. Decide", "decide.py", "Threshold the probabilities, then enforce one-to-one so each Source 2/3 record is claimed by at most one entity.", "Where the score is actually won. The only stage that directly sets precision."],
  ], [1250, 1550, 3213, 3013]));

  c.push(H2("What changed overnight on 25 September"));
  c.push(P("The pipeline had never been executed \u2014 no machine had the dependencies installed. Once it ran, blocking turned out not to scale: cost exponent 1.60 and rising, and entity cover falling from 0.995 to 0.955 as the corpus grew. Both trends were fatal at 2.2M entities."));
  c.push(RP([["Blocking was redesigned around a ", {}], ["rare-token inverted index", { b: true }], [": index each record under its 3 rarest name tokens and 2 rarest address tokens, plus a name prefix and a postal key. A match survives if it shares any one key, so a single distinctive word is enough. An earlier version using only exact whole-name keys scored recall 0.879 \u2014 exact keys break on any typo or reordering, which is exactly the noise this dataset is made of.", {}]]));
  c.push(table([
    ["Sample", "Recall", "Entity cover", "Cands / entity", "Blocking time"],
    ["2,000", "0.9788", "0.9435", "19.2", "9 s"],
    ["10,000", "0.9821", "0.9530", "39.9", "42 s"],
    ["40,000", "0.9833", "0.9546", "60.6", "148 s"],
  ], [1626, 1700, 1900, 1900, 1900]));
  c.push(SP(120));
  c.push(RP([["Cost exponent is now ", {}], ["0.91 \u2014 sub-linear", { b: true }], [", and entity cover is stable instead of falling. Estimated full-scale train blocking: about 1.6 hours instead of about 7.7. That estimate is an extrapolation from three points and has not been confirmed by a full-scale run.", {}]]));

  c.push(new Paragraph({ children: [new PageBreak()] }));

  // ---------- 5 issues ----------
  c.push(H1("5. Issues, and what the labels mean"));
  c.push(table([
    ["Label", "Meaning"],
    ["A \u00b7 safe", "Self-contained. Cannot break the pipeline or corrupt a score. Take these if you are working alone with nobody to check you."],
    ["B \u00b7 critical", "On the critical path, but each has a hard self-check you can run yourself before merging. Do the check \u2014 it is the whole reason these are not C."],
    ["C \u00b7 critical + judgement", "Critical and easy to produce a number that looks fine and is wrong. Read the warnings on the issue twice. If in doubt, stop and ask in chat rather than guessing."],
  ], [2000, 7026]));

  c.push(H2("Open issues"));
  c.push(table([
    ["#", "Label", "Title", "Owner"],
    ["4", "A", "Final submission package and methodology document", "P4"],
    ["7", "A", "Repo hygiene: placeholders, requirements, Python 3.12", "any"],
    ["8", "A", "Tests for normalize.py, with a France focus", "any"],
    ["2", "B", "Vectorise build_pair_features, remove pair-length reindex", "P3"],
    ["9", "B", "Verify multi-key blocking at full scale, tune its knobs", "P2"],
    ["3", "C", "Decision layer: one-to-one assignment and threshold", "P4"],
    ["10", "C", "France / unseen-country robustness (LOCO + adversarial)", "P4 or P2"],
  ], [700, 900, 5426, 2000]));
  c.push(SP(120));
  c.push(P("Closed: #1 (blocking \u2014 superseded, the work is on branch p1-blocking-scale) and #5 (AGENTS.md correction \u2014 merged)."));
  c.push(RP([["Issue ", {}], ["#2 is the single blocker on the critical path", { b: true }], [". Until pair features are vectorised, no full-scale run can finish, so no validation number exists and nothing downstream can be measured. If you are choosing what to work on and #2 is unclaimed, take #2.", {}]]));

  // ---------- 6 phases ----------
  c.push(H1("6. Phases"));
  c.push(table([
    ["Phase", "Window (IST)", "The one job", "Gate to exit"],
    ["0 \u2014 Make it run", "now \u2192 25 Sep ~18:00", "Environment, validator, baseline uploaded, blocking and features made to scale", "A full-scale run has started; D1-1 is on the board"],
    ["1 \u2014 First honest number", "25 Sep 18:00 \u2192 26 Sep 10:00", "Full run completes. Blocking recall, cross-fitted CV and LOCO measured. folds.csv committed. First model upload", "A CV number we believe, and one real submission scored"],
    ["2 \u2014 Push", "26 Sep 10:00 \u2192 27 Sep 08:00", "Decision layer, feature work, blocking knobs, model diversity. Every change on the noise rule", "Gains have flattened, or ~75% of time gone"],
    ["3 \u2014 Consolidate", "27 Sep 08:00 \u2192 16:00", "Ensemble and blend on OOF only. Seed bagging. Features frozen. NO NEW IDEAS", "Best pipeline chosen on cross-fitted CV"],
    ["4 \u2014 Land it", "27 Sep 16:00 \u2192 23:00", "Clean-clone reproduce, full audit, best-CV uploaded LAST before 22:00, zip and document by 23:00", "The zip would survive a reviewer opening it cold"],
  ], [1500, 1900, 3213, 2413]));
  c.push(SP(140));
  c.push(RP([["We are ", {}], ["late in Phase 0", { b: true }], [" and that is the honest position. The phase was budgeted at about three hours and we are past that, because the pipeline could not produce a full-scale output. Phases 1 to 4 do not compress \u2014 so the answer is to finish Phase 0, not to skip ahead.", {}]]));

  c.push(H2("Inside any working session"));
  c.push(P("Pull main. Take everyone's work. Do one thing. Run it. Log the number \u2014 whether it helped or not, because an inconclusive result recorded is worth more than a promising one forgotten. Merge. Post in chat. Repeat."));
  c.push(P("Post when you start something, when you finish, and immediately when you are blocked. That last one is what actually saves hours."));

  c.push(new Paragraph({ children: [new PageBreak()] }));

  // ---------- 7 commands ----------
  c.push(H1("7. Commands"));
  c.push(RP([["Always use ", {}], [".venv\\Scripts\\python", { code: true }], [", never bare ", {}], ["python", { code: true }], [". The interpreter first on PATH may be 3.14, which cannot install our pinned versions.", {}]]));

  c.push(H2("First-time setup"));
  c.push(CODE("py -3.12 -m venv .venv"));
  c.push(CODE(".venv\\Scripts\\python -m pip install -r requirements.txt"));
  c.push(CODE(".venv\\Scripts\\python src\\metric.py"));
  c.push(P("The last command must print: metric OK: example = 0.714. If it prints anything else, stop and tell P1 \u2014 every number the team reports depends on that file."));

  c.push(H2("The data"));
  c.push(RP([["Unzip the organisers' archive OUTSIDE OneDrive. On P1's machine it lives at ", {}], ["C:\\amlc\\student_resource\\dataset", { code: true }], [", with the organisers' ", {}], ["utils\\", { code: true }], [" beside it. OneDrive will lock a 2 GB folder mid-run.", {}]]));

  c.push(H2("Run the pipeline"));
  c.push(P("Smoke test \u2014 about 90 seconds. Use this for your whole development loop:"));
  c.push(CODE(".venv\\Scripts\\python src\\run_pipeline.py --data C:\\amlc\\student_resource\\dataset --out output_smoke --sample 2000"));
  c.push(P("Blocking cost and recall sweep, for anyone touching blocking:"));
  c.push(CODE(".venv\\Scripts\\python src\\sweep_blocking.py --data C:\\amlc\\student_resource\\dataset --n 40000"));
  c.push(P("The real run, with the unseen-country check. This does not finish yet \u2014 issue #2:"));
  c.push(CODE(".venv\\Scripts\\python src\\run_pipeline.py --data C:\\amlc\\student_resource\\dataset --out output --work work --loco"));

  c.push(H2("Validate before any upload"));
  c.push(CODE(".venv\\Scripts\\python C:\\amlc\\student_resource\\utils\\validate_submission.py --matching output\\matching_results.tsv --candidate output\\candidate_pairs.tsv --test-dir C:\\amlc\\student_resource\\dataset\\test --check-ids"));
  c.push(P("Must print PASS. Only P1 uploads, from one machine \u2014 simultaneous logins can terminate the whole attempt."));

  c.push(H2("Git"));
  c.push(CODE("git checkout main && git pull\ngit checkout -b <your-branch>"));
  c.push(CODE("git add src docs STATUS.md\ngit commit -m \"...\"\ngit push -u origin <your-branch>"));
  c.push(callout("Never run git add .", [
    "Name the paths you mean. A blanket add has already pulled an entire virtual environment into a commit once, and with work/ holding parquet caches it gets far worse.",
    "work/ and output/ are gitignored. The single deliberate exception is work/folds.csv, which P1 commits once with git add -f after the first full run. Everyone else pulls it and never regenerates it.",
  ]));

  c.push(new Paragraph({ children: [new PageBreak()] }));

  // ---------- 8 what next ----------
  c.push(H1("8. If every issue is closed, what next"));
  c.push(P("In priority order. Do not start further down this list while something above it is unfinished."));
  c.push(table([
    ["#", "Work", "Why it ranks here"],
    ["1", "Measure the full-scale blocking recall ceiling and entity cover, and write them into STATUS.md \u00a74", "This is the cap on every score we can ever achieve, and it is still unmeasured. Nothing else deserves attention while that is true."],
    ["2", "Error analysis on out-of-fold predictions", "Once a real CV number exists, sort entities by loss contribution and look at the worst 30\u201350. That tells you which feature to build next, instead of guessing. See the model-doctor skill."],
    ["3", "Model diversity for a later ensemble", "CatBoost or XGBoost on the SAME folds, saving OOF and test predictions. A model scoring slightly worse but with OOF correlation below ~0.97 is valuable. See oof-engineer."],
    ["4", "Feature batches driven by error analysis", "Themed batches of 5\u201320 with a written hypothesis each, accepted only on the noise rule. See feature-hunter."],
    ["5", "Blocking recall recovery", "Now that blocking cost is linear, recall can be bought linearly: raise rare tokens from 3 to 4, or raise MAX_CANDS. Measure cover before and after."],
    ["6", "Threshold refinement", "Does the best threshold differ by candidate count? An entity with 40 candidates has more chances to err than one with 2. Must adapt to a measured property, never to a country name."],
  ], [700, 3126, 5200]));

  c.push(H2("The acceptance rule, for all of it"));
  c.push(RP([["Accept a change only if it gains more than ", {}], ["2\u00d7 the seed-to-seed noise", { b: true }], [" AND improves at least ", {}], ["4 of the 5 folds", { b: true }], [". Anything smaller is logged as inconclusive and dropped. Most ideas will not clear this bar \u2014 that is the point of having it. And the number you report is always the cross-fitted one: tune() picks its threshold on the same rows it scores, so it flatters itself.", {}]]));

  c.push(H1("9. The rules that end our run"));
  c.push(callout("No external data lookup, of any kind", [
    "No geocoding APIs. No business registries. No commercial entity-resolution services. No scraped or downloaded reference data. Not even \u201cjust to normalise addresses\u201d.",
    "Hand-written abbreviation dictionaries are fine \u2014 rd to road is a statement about English applied blindly to every row, and it adds no information that was not already in the string. Adding rue, avenue, boulevard or chemin is equally fine.",
    "A list of real French communes, postcodes or company names is NOT fine. That is a gazetteer, which is external data. If you cannot tell which side of the line something is on, do not add it \u2014 ask.",
    "Code packages are reviewed by humans. The penalty is disqualification of the entire team, regardless of who wrote the line.",
  ]));
  c.push(SP(160));
  c.push(BULLET("Final model must be MIT or Apache-2.0 licensed and at most 8B parameters. Record the licence and parameter count of anything you add BEFORE you use it."));
  c.push(BULLET("Only P1 uploads, from one machine. Simultaneous logins can terminate the attempt entirely \u2014 do not open the portal to \u201cjust check something\u201d."));
  c.push(BULLET("Every Source 1 test entity gets exactly one row. No duplicate IDs in a list, no duplicate rows, only S2/S3 IDs that exist in the test set. A malformed file is rejected and burns a submission."));
  c.push(BULLET("Never branch on country == \"US\" or \"India\". Country may be a grouping key. It may never be a condition. That is a France bug and it costs us 15% of the test set."));
  c.push(BULLET("Do not change the metric, the folds or the seed. If you believe one is wrong, say so in chat and stop."));

  return c;
}

Packer.toBuffer(doc).then((b) => {
  fs.writeFileSync(process.argv[2] || "TEAM_HANDBOOK.docx", b);
  console.log("wrote", process.argv[2] || "TEAM_HANDBOOK.docx", b.length, "bytes");
});
