# Demo video script — 3 minutes

Record at 1600×1000 or larger, browser zoomed so the numbers are readable.
Have the app already running (`streamlit run Overview.py`) with the 1,000-row
catalogue loaded, and the Overview page open before you press record.

---

## 0:00 – 0:25 · The problem, in one screen

> "A distributor sends this: a part number, one line of trade shorthand, and
> three brand columns that mostly say 'Unbranded'. Six columns. The Unilog
> delivery template wants 252 — a full taxonomy path, six description variants,
> twenty feature bullets, sixty attribute triplets. Everything in between has to
> be recovered, and every value has to be defensible."

**On screen:** the input CSV open, or the Ingest page showing the six columns.

---

## 0:25 – 0:55 · It runs, and it measures itself

Run the deterministic pass on all 1,000 rows.

> "No API key, no network — this is the parser reading trade shorthand the way a
> product manager does. A thousand rows in seven seconds. Quality goes from 52.8
> to 60.4, and that is a measured before-and-after, not a claim: completeness,
> accuracy, consistency and richness, each scored, each decomposable."

**On screen:** Overview — 1,000 products, quality 60, 83% auto-approved, the
dimension bars and grade mix.

---

## 0:55 – 1:35 · Every value can be interrogated

Open **Catalog**, pick the 3M abrasive `3MABR-7100075678`.

> "Grit P150, form, attachment system, pack quantity — none of that was in a
> spec column. It was decoded from thirty-five characters of shorthand, and each
> value carries its method, its source, the evidence snippet it came from, and a
> confidence."

Point at `supplier` vs `manufacturer`.

> "This is the trap in the supplied data. `Part_Manuf` says 'Jam Industrial
> Supply LLC' — that is the account the distributor buys from, not the company
> that made the abrasive. The same column says 'Freud Inc' on the next row,
> where it *is* the maker. We publish it as supplier, and recover 3M as the
> brand and manufacturer from the description. 552 brands, 885 manufacturers,
> no API call."

---

## 1:35 – 2:10 · The model, and the engine arguing with itself

Switch to the LLM-enriched slice on **Export**.

> "Now the model. It classifies what keywords could not, writes the six commerce
> descriptions, and then judges its own output. Content coverage goes to 97%,
> attribute triplets from 4.3 to 6.2 per product, quality to 78."

Then the honest bit — this is the moment that wins the room:

> "When we first switched the model on, the score went *down*. Accuracy 84 to
> 43. Not because the writing was bad — because the judge had started checking
> work nobody had checked before, and it found two real defects: a distributor's
> name on the manufacturer line, and 3M's product series sitting on a Diablo
> sanding belt, inherited from a category-wide peer group. Both fixed at the
> source. Peer inference is now refused outside a brand-scoped group."

---

## 2:10 – 2:40 · What it refuses to do

Open **Review Queue**.

> "Anything the engine cannot defend lands here — low confidence, an open
> validation error, a conflict between sources. A Mirka roll quoted '2.75x30' is
> flagged, because the trade quotes those in feet and the description does not
> say. We do not guess it. Blank is recoverable; a confident wrong value is not.
> Every approve, correct and reject is stored, and it feeds reviewer
> calibration."

---

## 2:40 – 3:00 · The deliverable

Back to **Export**, Delivery template tab.

> "And this is the contract: 1,000 rows by 252 columns, taxonomy 100%, identity
> 96%, column order read from the template file itself so a revised template
> needs no code change. Distributor-side columns — list price, Prop 65 — stay
> blank, because no amount of enrichment can honestly invent them."

Download the CSV on camera. End.

---

## Things to have open in tabs beforehand

- The app, on Overview
- The input CSV
- The exported delivery CSV, so the download is instant

## Do not

- Do not run the LLM slice live on camera — free tier is 15 requests per minute
  and it will stall mid-sentence. Run it beforehand; the prompt cache makes any
  repeat instant.
- Do not read the numbers off this script. Read them off the screen.
