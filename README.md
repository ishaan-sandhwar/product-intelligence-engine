# Product Intelligence Engine

Takes a part number and a 35-character distributor description, and returns a
complete, validated, delivery-ready product record — with the evidence for every
value it produces.

Built for the UniHack product-intelligence challenge: *generate, enrich and
validate product intelligence from limited product information.*

---

## The problem, exactly as the data states it

The supplied catalogue is 1,000 rows × 6 columns:

```
Mfg_Part_Num   Part_Desc                                   E1_Brand         Unilog_Brand           DIB_Brand           Part_Manuf
DCB518ASTS06G  Diablo 1/2"x18" - Sanding Belt 6pc          -- Unbranded --  -- No Unilog Brand --  Diablo              Freud Inc (2435)
564856         564856 40W Led ST19 27k 2pk                 -- Unbranded --  -- No Unilog Brand --  Philips             Phillips Lighting (5831)
543140016      1nx6-16' Biscayne Sq Edge - Trex Lineage    TREX             -- No Unilog Brand --  -- No DIB Brand --  U S Lumber (3073)
```

Not one specification column. Three of the six columns are usually placeholder
markers. The delivery template on the other side wants **252 columns**: a
three-level taxonomy, six description variants, twenty feature slots, sixty
attribute triplets, asset references and commerce fields.

That gap is the whole problem, and every design decision below follows from it.

| Challenge outcome | How it is met |
|---|---|
| Structured intelligence from limited inputs | Trade-shorthand decoding (`P150` → grit 150, `27k` → 2700 K, `1/2"x18"` → width × length, `5"x.045"x7/8"` → diameter/thickness/arbor) before any model call, then LLM extraction and knowledge-graph inference for what the text cannot yield |
| Improved data quality and consistency | 26-class canonical schema with imperial canonical units, unit conversion, alias resolution that *refuses* ambiguous headers, deterministic rule engine |
| Validated, traceable outputs | Every value carries source, evidence snippet, method, confidence and rejected alternatives; adversarial LLM judge; full audit JSON |
| Scale across large catalogues | Two-phase thread-pooled batch, response cache, deterministic-first extraction so most fields never cost a token, SQLite store, headless CLI |

---

## Architecture

```
6 input columns  ·  optional PDF / image / URL
        │
        ▼
  1 classify      keyword vote over the taxonomy, LLM only when unsure     ─┐
  2 extract       every source independently:                               │  per record,
                  trade shorthand · unit miner · regex · LLM · VLM          │  thread-pooled
  3 resolve       candidates → one golden value, losers kept as             │
                  alternatives with the reason they lost                   ─┘
        │
        ▼  catalogue-wide context built here: RAG index + product graph
        │
  4 rules         deterministic cross-field and range checks               ─┐
  5 enrich        gaps from catalogue peers first, then the model           │  per record,
  6 judge         adversarial audit of the assembled record                 │  thread-pooled
  7 route         confidence + open issues → human review queue             │
  8 score         quality measured on the same yardstick as the input      ─┘
        │
        ▼
golden record  →  delivery CSV (252 cols)  |  flat CSV  |  audit JSON  |  review queue
```

Stage order is not arbitrary. Classification comes first because the category
decides which attributes exist at all. Rules run before the model judge because
they are free and more reliable. The RAG index and product graph are built
*between* the two phases, so enrichment can borrow from peers that have already
been extracted — otherwise the first products processed would have no peers.

### Design decisions worth knowing

- **Deterministic before probabilistic.** A distributor description is not
  prose, it is compressed trade shorthand. `3M 775L Stikit Film P150 - Cubitron
  II 50 Disc/Box` yields form, grit, attachment system, pack quantity and
  selling unit with no API call at all. The model only ever sees what the
  decoders could not resolve.
- **The supplier column is not the manufacturer.** `Part_Manuf` names the account
  the distributor buys from. For a 3M abrasive that is "Jam Industrial Supply
  LLC"; for a Diablo belt it is "Freud Inc" — the same column, a reseller in one
  row and the maker in the next. `src/normalize/identity.py` publishes it as
  `supplier`, then resolves brand from the brand columns, the description's own
  brand vocabulary, and finally maker-style account names. That fills 552 brands
  and 885 manufacturers across the file with no API call.
- **Inference stays inside its scope.** Peer consensus is powerful and blind: a
  brandless Diablo belt sitting in a 3M-dominated abrasives group will inherit
  3M's `775L` series unless something stops it. Brand-scoped attributes (series,
  model, UPC) are refused from a category-only peer group.
- **Ambiguity is refused, not guessed.** When a messy header fuzzy-matches two
  different canonical keys within the margin, it maps to neither. Silent
  mis-mapping is the most expensive error in a catalogue.
- **Peer inference is flagged, not published.** A value inferred from catalogue
  siblings ("88% of peers ship 50 per box") is written at reduced confidence,
  tagged `catalog_sibling`, and routed to a human — it never enters the delivery
  file as if it were sourced.
- **Blank beats wrong.** Anything below the publish floor is held back. A blank
  cell is recoverable; a confident wrong value is not.
- **The human loop is closed.** Every approve / edit / reject is written to
  `review_decisions`, and the Quality page turns those into an agreement rate and
  a per-method approval rate — which is how you discover that a confidence
  constant in `config.py` is set too high.

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows.  source .venv/bin/activate elsewhere
pip install -r requirements.txt

copy .env.example .env             # then paste in one API key (any provider)
streamlit run Overview.py
```

Load `data/raw/Unihack_ Sample Dataset - Input.csv` on the **Ingest & Run** page.

No API key? It still runs: the deterministic path (shorthand decoding, unit
conversion, rules, catalogue peers) does the work and content generation is
skipped. Gemini's free tier via Google AI Studio is enough for the full pipeline.

### Headless, for scale

```bash
python scripts/run_pipeline.py "data/raw/Unihack_ Sample Dataset - Input.csv" --limit 150
python scripts/run_pipeline.py catalog.csv --no-llm                    # deterministic only
python scripts/run_pipeline.py catalog.csv --concurrency 8 --audit out.json
python scripts/run_pipeline.py catalog.csv --delivery submission.csv   # the 252-column file
```

### Tests

```bash
pytest tests -q        # 35 tests: identity resolution, dimension shorthand, request pacing
```

The suite covers the decoding rules that are expensive to get wrong — that a
distributor account never reaches the manufacturer line, that a round abrasive is
read as diameter and not width, and that `1x6-16'` is a one-inch board six inches
wide and sixteen feet long.

### The app

| Page | What it is for |
|---|---|
| **Overview** | Catalogue KPIs, quality dimensions, grade mix, run history |
| **Ingest & Run** | Load the catalogue or build one product from a part number; attach datasheets, images, URLs |
| **Catalog** | Per-product inspection: values, evidence snippets, rejected alternatives, issues, stage log |
| **Review Queue** | Approve / correct / reject flagged values, and fill required fields nothing could source |
| **Quality** | Score decomposition, field coverage, issue mix, reviewer calibration |
| **Export** | Delivery template CSV, flat CSV, audit JSON, open-queue CSV |

---

## The delivery template

`src/export/delivery.py` renders the golden records into the supplied
`resources/Unihack_ Expected Output - Delivery Format.csv` contract. The header
row of that file *is* the column order — it is read at export time, so a revised
template drops in without a code change.

| Template block | Where it comes from |
|---|---|
| `Dept` / `Class` / `Fine` / `Classpath` | the taxonomy attached to the classified category in `schemas/attributes.json` |
| `PART_NUMBER`, `Mfg_Part_Num`, brand and manufacturer columns | the input row, echoed verbatim (placeholders included) plus the cleaned canonical values |
| `MOBILE_DESC` … `MARKETING_DESCRIPTION` | six generated variants, each with its own register and length taken from the template's worked example |
| `ITEM_FEATURES_1..20` | generated feature phrases, one specification each |
| `ATTRIBUTE_LABEL/VALUE/UOM 1..60` | verified attributes packed in priority order — required first, then schema weight, then confidence |
| `LENGTH`/`WIDTH`/`HEIGHT`/`WEIGHT` + `_UOM` | canonical dimension attributes with their units |
| Image and document columns | attached sources, matched to document type by filename |

Canonical units are **imperial** (`in`, `ft`, `lb`) because that is what this
trade publishes and the template carries its own UOM column — converting a 24-inch
dishwasher to 609.6 mm would be correct and useless.

---

## Providers

The client picks the first configured provider in `PROVIDER_PRIORITY`
(`config.py`) and skips the rest, so one key is enough:

| Provider | Env var | Text | Vision |
|---|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | claude-sonnet-5 | ✓ |
| Gemini | `GEMINI_API_KEY` | gemini-3.5-flash-lite | gemini-3.6-flash |
| OpenAI | `OPENAI_API_KEY` | gpt-4o-mini | ✓ |
| Groq | `GROQ_API_KEY` | llama-3.3-70b | — |

Responses are cached by `(provider, model, payload)` hash, so re-running the same
catalogue costs nothing.

Every provider is paced by a client-side token bucket (`LLM_RPM_LIMITS`), because
a free-tier 429 still consumes a request slot — discovering the ceiling by
hitting it poisons the following minute too. Gemini's free tier allows 15
requests per minute per model; the default of 13 leaves room for retries. When a
429 does arrive, the server's own `retryDelay` sets the backoff rather than an
exponential guess that would land early and burn another slot.

---

## Layout

```
config.py                 paths, model ids, thresholds, scoring weights, content targets
schemas/attributes.json   the canonical dictionary: 26 classes, taxonomy, attributes, content fields
schemas/attributes_industrial.json   the same contract aimed at industrial equipment - swap to re-target
resources/                the delivery template (its header row is the export contract)
Overview.py               Streamlit entry (Overview page)
pages/                    Ingest · Catalog · Review Queue · Quality · Export
scripts/                  headless batch runner, messy-sample generator
src/
  models.py               ProductRecord, AttributeValue, Provenance, QualityScore
  schema.py               attribute dictionary, alias resolution, taxonomy, category guessing
  pipeline.py             stage orchestration, batch runner, human-edit handlers
  store.py                SQLite: records, runs, review decisions
  ingest/                 table loader, document intelligence (PDF/web/image), extractors
  normalize/              unit families and conversion, canonical value shaping
  llm/                    provider abstraction (4 SDKs), response cache, prompts
  validate/               rule engine, conflict resolution, adversarial judge
  enrich/                 RAG retriever, product knowledge graph, enrichment agent
  score/                  the four-dimension quality score
  export/                 delivery-template renderer
  ui/                     shared Streamlit helpers
```

### Extending the schema

`schemas/attributes.json` drives extraction prompts, validation, scoring and
exports. Adding a product class is a JSON edit, not a code change: give it a
label, a Dept and Class, keywords for the classifier, and its attributes with
`datatype`, `unit`, `unit_family`, `aliases`, `required`.

Keywords must describe **products, not brands** — "milw" as a keyword put every
Milwaukee cut-off disc into Cordless Power Tool at full confidence, because a
brand keyword matches everything that brand sells.

---

## Quality score

```
overall = 0.35·completeness + 0.25·accuracy + 0.20·consistency + 0.20·richness
```

- **completeness** — weighted fill rate over the attributes that apply to the class
- **accuracy** — mean confidence, penalised for missing evidence and open errors
- **consistency** — survives cross-field and unit checks (absence is *not* counted
  here; that is completeness's job, and double-counting it made records score
  lower after processing than before)
- **richness** — presence and adequacy of the delivery content set against
  `CONTENT_TARGETS`

Both `quality_before` and `quality_after` are stored, so uplift is measured on
the same yardstick rather than asserted.

---

## Known limits

- Content generation needs an API key; without one, richness scores zero and the
  six description columns export blank rather than guessed.
- A full 1000-row LLM pass is ~4000 calls, which is about five hours at a free
  tier's 15 requests per minute. The deterministic pass covers all 1000 rows in
  7 seconds; the model is pointed at a slice.
- The keyword classifier settles ~71% of the supplied catalogue on its own; the
  rest fall to the LLM classifier, and to `General Product` when no key is set.
- Web extraction obeys nothing but a timeout — no robots.txt handling, no rate
  limiter. Fine for a demo, not for a crawl.
- The product graph infers from co-occurrence within the loaded catalogue only;
  it is rebuilt per run rather than persisted.
- `SKU - MY_PART_NUMBER`, `List Price` and `Prop 65` are left blank: they are
  distributor-side data that no amount of enrichment can honestly invent.
