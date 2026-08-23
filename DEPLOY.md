# Deploying the prototype

The submission needs a public GitHub repository and a live prototype link.
Everything below is already prepared in the repo — these are the three steps
that need your accounts.

---

## 1 · Push to GitHub

The repository is initialised and committed locally on branch `main`. Create an
**empty public** repo on GitHub (no README, no .gitignore — the repo already has
both), then:

```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

That URL is the **GitHub Repository** link for the submission form and for
slide 14.

**Already handled:** `.env` is git-ignored, so your API key never leaves the
machine. The virtualenv, caches and run logs are ignored too. The supplied
catalogue, the delivery template and a processed database snapshot *are*
committed, so the deployed app opens with 1,000 products already in it.

---

## 2 · Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and sign in with the same GitHub account.
2. **Create app** → pick the repository, branch `main`, main file `Overview.py`.
3. Open **Advanced settings** before deploying and set the Python version to
   **3.12** or **3.13**.
4. Under **Secrets**, paste:

   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```

   `config.py` copies that into the environment on startup, so nothing else
   needs changing. Deploying *without* a key still works — the app runs the
   deterministic pipeline and the content columns stay blank.
5. Deploy. First build takes a few minutes.

The resulting `https://<something>.streamlit.app` URL is the **Working
Prototype** link.

### If the build fails

- **Memory limit (1 GB on the free tier).** The 1,000-row catalogue fits, but if
  a build gets killed, drop `pymupdf` from `requirements.txt` — it is only
  needed for datasheet ingestion, which the demo does not use.
- **Slow first load.** The app reads the committed SQLite snapshot; this is
  normal on a cold start.

---

## 3 · Record the demo video

Follow `DEMO_SCRIPT.md` — it is timed to three minutes and marks the two moments
that matter: the supplier-vs-manufacturer trap in the supplied data, and the
quality score *falling* when the judge was switched on.

Record locally rather than against the deployed app: it is faster, and the free
tier's rate limit cannot stall you mid-sentence.

---

## Before you submit — checklist

- [ ] GitHub repo is **public** and the push succeeded
- [ ] Streamlit app loads and shows 1,000 products on the Overview page
- [ ] `GEMINI_API_KEY` set in Streamlit secrets (optional, but the content
      columns need it)
- [ ] Slide 2 of the deck: team name and team leader name filled in
- [ ] Slide 14 of the deck: the three URLs pasted in
- [ ] Demo video uploaded and the link is set to public / anyone-with-link
- [ ] Delivery CSV attached if the form asks for the output file:
      `outputs/exports/submission_llm100_delivery.csv`
