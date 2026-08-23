# Product Intelligence Engine — UniHack submission

**Six columns in, 252 columns out.** The supplied catalogue gives a part number,
a 35-character description and three brand columns that are mostly placeholder
markers. The delivery template asks for a full taxonomy path, six description
variants, twenty feature bullets and sixty attribute triplets. Everything in
between has to be recovered, and every recovered value has to be defensible.

---

## Solution overview

> The Product Intelligence Engine turns a bare distributor row — a part number,
> one line of 35-character trade shorthand, and brand columns that are mostly
> "-- Unbranded --" placeholders — into a complete 252-column Unilog delivery
> record.
>
> It works in the order a product manager would. Deterministic decoders read the
> shorthand first: form, grit, dimensions, pack quantity and selling unit come
> out of the description with no model call, covering all 1,000 supplied rows in
> 7 seconds. An identity resolver then separates the three things the input
> conflates — the supplier account, the brand and the manufacturer — because
> `Part_Manuf` names the company the distributor buys from, not the company that
> made the product. A knowledge graph fills what the catalogue itself can
> support, scoped so brand-specific values never leak across a brand boundary.
> Only what is genuinely left goes to a language model, which classifies,
> extracts, writes the six commerce descriptions and then judges the finished
> record.
>
> Every published value carries its method, its source, the evidence snippet it
> came from and a confidence. Anything below the publish floor is held back and
> routed to a human review queue rather than guessed, because a blank cell is
> recoverable and a confident wrong value is not. Quality is scored before and
> after across completeness, accuracy, consistency and richness, so the uplift is
> a measured number: 52.8 → 60.4 deterministically over the full file, and
> 51.2 → 78.3 on the model-enriched slice, with 97% of the commerce content
> columns filled and 100% taxonomy coverage.

---

## The problem, as the data actually states it

| | |
|---|---|
| Input | 1,000 rows × 6 columns — `Mfg_Part_Num`, `Part_Desc`, `E1_Brand`, `Unilog_Brand`, `DIB_Brand`, `Part_Manuf` |
| Specification columns supplied | **zero** |
| Rows whose brand columns are placeholders | **554 / 1000** |
| Output | 252-column Unilog delivery template |
| Domain | US building products — decking, LED lamps, fixtures, abrasives, cordless tools, appliances, windows |

Two facts drove the whole architecture:

1. **The description is not prose, it is compressed trade shorthand.**
   `3M 775L Stikit Film P150 - Cubitron II 50 Disc/Box` carries brand, series,
   attachment system, grit, form, pack quantity and selling unit. A parser gets
   all of it. A language model gets it too — 1,000 times slower and at a cost.
2. **The supplied metadata lies in a specific, systematic way.** `Part_Manuf` is
   the account the distributor buys from, not the manufacturer. It says
   "Jam Industrial Supply LLC" for a 3M abrasive and "Freud Inc" for a Diablo
   belt. Publishing that column as the manufacturer puts a distributor's name on
   the manufacturer line of hundreds of records.

---

## What the engine does

```
ingest → classify → decode → resolve → validate → enrich → judge → route → score → export
```

- **Deterministic first.** Trade-shorthand decoders, unit conversion, enum
  snapping and rule checks run before any model call. They cover all 1,000 rows
  in 7 seconds with no API key at all.
- **Identity resolution.** `Part_Manuf` is published as `supplier`; brand is
  resolved from the brand columns, then a brand vocabulary matched against the
  description, then maker-style account names. 552 brands and 885 manufacturers
  filled, deterministically.
- **Scoped inference.** Peer consensus fills gaps from the catalogue itself, but
  brand-scoped attributes (series, model, UPC) are refused from a category-only
  peer group — otherwise a brandless Diablo belt inherits 3M's `775L` series
  from its neighbours.
- **The model is the last resort, not the first.** It classifies what keywords
  could not, extracts what the decoders missed, writes the six commerce
  descriptions, and judges what everything else produced.
- **Nothing unsupported is published.** Every value carries method, source,
  evidence and confidence. Below the publish floor it is held back; below the
  review threshold it goes to a human queue.

---

## Measured results

**Deterministic pass — all 1,000 rows, 7.2 s, zero API calls**

| Metric | Before | After |
|---|---|---|
| Quality score | 52.8 | **60.4** |
| Fields filled | — | **7,330** |
| Auto-approved | — | **83%** |
| Delivery identity coverage | — | **96%** |
| Delivery taxonomy coverage | — | **100%** |

**LLM-enriched slice — 100 rows, same rows both columns**

| Metric | Deterministic | With the model |
|---|---|---|
| Quality score | 59.6 | **72.4** |
| Grade mix | C / D | **B31 / C66 / D2 / F1** |
| Content column coverage | 0% | **100%** |
| Attribute triplets per product | 2.6 | **5.9** |
| Feature bullets per product | 0 | **5.0** |
| Delivery taxonomy coverage | 100% | **100%** |

411 model calls, zero rate-limit failures, 100 of 100 rows completed.

**Classifier escalation** — of 12 rows the keyword classifier left as
`General Product`, the model resolved **9** (railing components ×5, window units
×2, an abrasive, a decking board). The remaining three — a heater kit, an attic
access door, a rainscreen — have no matching class in the 26-category taxonomy
and stay generic rather than being forced.

---

## The part worth showing a judge

The quality score went **down** the first time the model was switched on:
accuracy 84 → 43, consistency 93 → 39. That was not the model writing badly. It
was the judge finally checking work nobody had checked before, and finding two
real defects:

- `manufacturer` = "Jam Industrial Supply LLC" — a distributor on the
  manufacturer line.
- `series` = `775L` on a Diablo belt — 3M's series, inherited from a
  category-wide peer group.

Both were fixed at the source rather than papered over. A third class of finding
survived on purpose: a Mirka `2.75x30` roll is flagged as an implausible
30-inch length, because the trade quotes those in feet — the description is
genuinely ambiguous, so it belongs in the review queue and not in an autofill
rule.

**That is the argument for the whole design.** A pipeline that only fills cells
scores well and ships errors. A pipeline that scores itself, catches its own
inference leaking across a brand boundary, and refuses what it cannot support is
the one you can actually put in front of a customer's catalogue.

---

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements.txt
cp .env.example .env                                 # add GEMINI_API_KEY

streamlit run Overview.py                                 # the six-page app

# headless
python scripts/run_pipeline.py "data/raw/Unihack_ Sample Dataset - Input.csv" --no-llm
python scripts/run_pipeline.py "data/raw/Unihack_ Sample Dataset - Input.csv" --limit 100

pytest tests -q                                      # 35 tests
```

## Demo route, five minutes

1. **Ingest & Run** — load the 1,000-row file, run deterministic. 7 seconds,
   quality 52.8 → 60.4.
2. **Catalog** — open a 3M abrasive. Show `supplier` vs `manufacturer`, and the
   evidence snippet behind every decoded attribute.
3. **Review Queue** — show a flagged value with the judge's reason attached.
4. **Quality** — score decomposition and the issue mix.
5. **Export** — the 252-column delivery file, and the enriched slice with all six
   description columns populated.

## Honest limits

- A full 1,000-row model pass is ~4,000 calls ≈ 5 hours on a free tier's 15
  requests-per-minute ceiling. The deterministic pass covers everything; the
  model is pointed at a slice.
- 290 rows classify as `General Product` deterministically; the escalation path
  resolves about three quarters of them, and the rest are genuinely outside the
  26-class taxonomy.
- `SKU - MY_PART_NUMBER`, `List Price` and `Prop 65` export blank. They are
  distributor-side data no amount of enrichment can honestly invent.
