# Style-Based Coordinated-Account Detection — Explainer

`cluster_campaign1_style.py` finds new coordinated/bot accounts by matching their
**writing style** against the known campaign-1 accounts — *not* their topic. This
document walks through what it does, why it replaces the old clustering, and what
every value means.

---

## 1. What is the purpose of the script

It detects **coordinated/bot accounts in a new campaign by using known seed accounts
from a previous campaign** — even when the topic has completely changed (campaign 1
talked about K-pop, campaign 2 talks about AI).

It does this by matching accounts on **how they write** (writing style / typing
fingerprint), not **what they write about** (topic). The output is every candidate
account ranked and flagged by how bot-like its writing is, written to
`clustering/campaign_1_style/{window}/style_ranking.csv`.

The core question it answers for every account:

> *"Does this account write like the known campaign-1 bots, and unlike a normal
> user — regardless of what it's posting about?"*

---

## 2. Why the previous account clustering is not as effective

The old `account_clustering.py` is not ideal for **cross-campaign**
detection (K-pop seeds → AI candidates):

1. **Topic Matching** 
The main features are word TF-IDF (~35% weight) + semantic sentence embedding (`all-MiniLM-L6-v2`, ~20–30%). That means **~55–60%
   of the score encodes subject matter.** K-pop and AI vocabulary barely overlap, so
   a genuinely coordinated account gets pushed *away* from the seeds simply because
   it's talking about a different topic. The semantic block, in particular, is built
   to capture meaning — exactly the thing that does **not** transfer across campaigns.

2. **The score scale is rebuilt every run.** It min-max-rescales all similarities
   into a fixed 0.4–0.95 band per run. Consequences:
   - The most-similar pair *always* maps to ~0.95, so the run **always produces
     "top" matches even when nothing is genuinely similar.**
   - A fixed downstream threshold (e.g. `> 0.55`) means a different thing every day,
     because the scale depends on whoever was scraped that day.

3. **There is no negative control.** With nothing representing "a normal user," the
   threshold can't be calibrated and you can't tell whether a given score is actually
   high or just average for that day.

Two secondary issues:
- The **KMeans/Hierarchical/DBSCAN clustering is decorative** — it's computed but the
  decision is really *nearest-seed similarity*, so the "clustering" name is misleading.
- The decision used the **average similarity to all seeds**, which *dilutes* the
  signal: a bot that strongly matches one persona template but not the others gets
  washed out by the averaging.

---

## 3. What the campaign-1 bots actually look like (why style works)

Profiling all 31 usable seed accounts (~23.6k words) shows a strong, topic-independent
**generation fingerprint** — which is exactly why matching on style survives the topic
switch. Each figure is the share of seed accounts showing the trait:

| Signal | Seed accounts |
|--------|---------------|
| Use any non-ASCII character | **100%** |
| Use the em-dash `—` | 61% |
| Use curly apostrophe `’` (vs straight `'`) | 87% |
| Use curly double-quotes `“ ”` (vs straight `"`) | 65% |
| Use ellipsis `…` | 42% |
| Use non-breaking hyphen `‑` (U+2011) | 29% |
| Use slang (e.g. `lol`, `tbh`, `lmao`) | 71% |
| Mean sentence length (words) | 19.3 |
| Mean word length (chars) | 4.73 |

Compared against 600 random normal accounts from the same subreddits, the **em-dash is
the standout tell**: 61% of seeds use it vs **4%** of normal users, ~26× more often by
rate — while *never* falling back to ASCII `--`. These accounts read as **LLM-generated
text styled to look casual** (slang) but betrayed by smart-quote typography no normal
keyboard produces. None of that depends on K-pop, which is why it carries over to the
AI campaign.

> Emoji and Korean Hangul are **excluded** from the fingerprint — emoji are common among
> normal users (weak tell), and Korean is *topic* (K-pop), not authorship.

---

## 4. What are the changes

| Old (`account_clustering.py`) | New (`cluster_campaign1_style.py`) |
|-----|-----|
| Topic features (word TF-IDF + semantic embedding) | **Topic-invariant style features** (char n-grams, function words, structural) |
| Average similarity to *all* seeds | **Top-5** nearest-seed similarity |
| No baseline | **Background negative set** → separation score |
| Per-run min-max rescale (0.4–0.95) | **z-score vs a calibrated null** (no rescale) |
| No way to measure quality | **Leave-one-seed-out validation** (AUC + recall) |
| Single-day | **N-day windowing** for more text per account |
| Decorative clustering | Direct nearest-seed scoring |
| Non-ASCII blended / ignored | Non-ASCII a **separate reported channel** (no topic leak) |

---

## 5. What are the steps in the new methodology

1. **Collate** — for an N-day window, group every account's posts across *all*
   subreddits (`load_data()` groups by `redditorName`) and aggregate into one text
   blob per account. Seeds = all campaign_1 accounts.
2. **Featurise** — convert each account's text into one topic-blind **style vector**
   from 3 weighted blocks (`build_style_matrix`).
3. **Score** — for each candidate:
   `style_score = mean(top-5 cosine to seeds) − mean(cosine to background)`.
4. **Calibrate** — convert to a **z-score** against a null distribution built by
   scoring the background accounts the same way (`background_null`).
5. **Flag** — flag if `z_score > 2.0`.
6. **Validate** — leave-one-seed-out: score each seed as if it were unknown and
   measure AUC + recall, proving the features separate known bots from normal users.
7. **Output** — a CSV per window, ranked by `style_score`.

---

## 6. What are the 3 weighted blocks and what they look out for

Each account's text becomes three topic-blind feature groups, combined into one
**style vector**:

| Block | Weight | What it looks out for |
|-------|--------|------------------------|
| **Character n-grams (3–5)** | 0.50 | The author's sub-word fingerprint — spacing, morphology, recurring letter/punctuation sequences. The strongest, most topic-robust authorship signal (the workhorse of authorship attribution), hence the highest weight. |
| **Function words** | 0.30 | Frequency of grammatical glue words (*the, of, i, but, you…*) — content-free habits that **cannot** encode topic. |
| **Structural** | 0.20 | Writing mechanics — ASCII punctuation profile, sentence length, type-token ratio, ALL-CAPS ratio, digit ratio, whitespace rhythm. A useful but noisier signal, so the lowest weight. |

**How they combine** (`build_style_matrix`): each block is independently
**L2-normalised** (so no block dominates by raw magnitude), multiplied by its weight,
the three are concatenated, and the result is L2-normalised again. The weights encode
**how reliable each signal is**; the validation AUC (~0.94) confirms the mix.
Weights are tunable with `--w-char --w-func --w-struct`.

---

## 7. What are the different scores and values, and what they mean

Using a real flagged row as the running example —
**`OkBuffalo1925`**: `style_score=0.4232, z=4.33, seed_topk_sim=0.7572,
background_sim=0.3340, non_ascii=3.38, seed_avg=5.14`.

| Value | Meaning | Example |
|-------|---------|---------|
| `seed_topk_sim` | Mean cosine to the **5 most similar seeds** → "how close to known bots." | 0.7572 |
| `background_sim` | Mean cosine to the background sample → "how close to a *normal* user." Everyone scores ~0.3 here (shared-English floor). | 0.3340 |
| `style_score` | `seed_topk_sim − background_sim` → the **separation**: writes like bots **and** unlike normal users. The ranking is sorted by this. | 0.4232 |
| `z_score` | `(style_score − null_mean) / null_sd` → **how many standard deviations above a normal user.** Calibrated and comparable across runs. | 4.33 |
| `flagged` | 1 if `z_score > 2.0`. | 1 |
| `most_similar_seed` | The single closest seed — *which* bot it most resembles. | `whiskydeep1` |
| `non_ascii_normalized` | Weighted non-ASCII (smart-punctuation) density per 250 chars. **Reported only — not scored or flagged.** | 3.38 |
| `seed_avg_non_ascii` | Mean `non_ascii_normalized` across all seeds — a constant reference baseline (same on every row). | 5.14 |
| `n_posts` | Posts by the account in the window. | 22 |

### The background null and the z-score (why they exist)

A raw `style_score` of 0.42 is meaningless on its own — even normal users get a
positive score (~0.15), because the top-5 of 32 seeds always finds a few coincidental
lookalikes. So the script builds a **null distribution**: it runs the *same* scoring
formula on the background accounts (`background_null`), producing the `style_score` a
**normal user** typically gets:

```
null_mean ≈ 0.153    null_sd ≈ 0.062     (from a sample run)
z_score   = (style_score − null_mean) / null_sd
flag if   z_score > 2.0   ⇔   style_score > null_mean + 2.0·null_sd
```

The null is the **control group**. Its purpose:
- **Reference for "normal"** — turns 0.42 into "4.3σ above a normal user → genuinely unusual."
- **Principled threshold** — `z > 2` means "more bot-like than ~98% of normal users," not a hand-picked constant.
- **Comparability across runs** — standardising against each window's null makes
  "flagged" mean the same thing every day (the exact failure of the old per-run rescale).

### Why each value is used (justification)

| Choice | Why |
|--------|-----|
| Style features, not topic | Topic vocabulary doesn't transfer K-pop → AI; idiolect does. |
| Cosine similarity | Vectors are L2-normalised → compares the *shape* of a style profile, independent of length. |
| **Top-5** seeds (not mean-of-all, not max) | Mean-of-all dilutes; max (top-1) is noisy; top-5 = "resembles a *cluster* of known bots." |
| Subtract `background_sim` | Removes the ~0.3 shared-English floor → isolates *distinctively* bot-like, not *generically English*. |
| z-score vs null | Makes the score interpretable, calibrated, and portable across windows. |
| `z > 2.0` | Sits in the genuine tail; balances recall vs review load (1.5 scoops normal users; 3.0 misses half). |

---

## 8. What other params / features are present

- **`--window N`** — days collated per run. More text per account → stabler
  fingerprints and sparse tells appear; trades time resolution for reliability.
- **`--min-posts`** — drop text-poor accounts that can't be fingerprinted reliably.
- **`--k`** — number of nearest seeds to average (default 5).
- **`--bg-size`** — background sample size (default 400; 1000 gives a steadier null).
- **`--z`** — flag threshold in standard deviations (operating point: 2.0).
- **`--w-char / --w-func / --w-struct`** — tunable block weights.
- **`--validate`** — run the leave-one-seed-out AUC/recall.
- **`--style-embed MODEL`** — optional purpose-built topic-invariant *style* embedding
  (e.g. a sentence-transformers authorship model); off by default, degrades gracefully.
- **`--after / --before / --date`** — date range (chunked into `--window` blocks).
- **All-seeds handling** (`_force_all_seeds`) — uses *every* campaign_1 account as a
  seed, overriding the engine's hardcoded skip list.
- **Weighted non-breaking hyphen** — `‑` (U+2011) is weighted 2.5 in `NonAsciiScorer`
  as a strong LLM/copy-paste tell.
- **Background centroid optimisation** — the per-candidate `background_sim` is computed
  against the background's centroid, so a large `--bg-size` stays fast.

---

## 9. How to run

```powershell
# 7-day windows over late Feb, validated, flag at z >= 2.0, large background
python cluster_campaign1_style.py --after 23-02-2026 --before 01-03-2026 \
    --window 7 --min-posts 2 --validate --z 2.0 --bg-size 1000
```

Output: one `clustering/campaign_1_style/{window}/style_ranking.csv` per window,
ranked by `style_score`, plus the `[VALIDATION]` AUC/recall printed to the console.

**Flags are a high-recall filter, not a verdict** — they feed the downstream LLM
writing-style comparison and interaction-tracing, which make the final call.

---

## 10. Sample results (7-day window, z=2.0, bg=1000)

| Metric | Value | Reading |
|--------|-------|---------|
| **AUC** | **0.943** | Strong — a real bot outscores a normal user 94% of the time. The method works. |
| **Recall @ z>2** | 65.6% | ~21/32 seeds caught; misses are mostly **thin seeds** that can't be fingerprinted → real recall on substantive seeds ≈ **84%**. |
| **Flagged** | 1,850 (4.4%) | The suspicious tail; `z>2.5` tightens to ~856. |
| **Null mean** | 0.153 = median candidate | Calibration is honest — the typical account scores like a normal user. |
| **Distribution** | mass at ~0.14, thin tail to 0.43 | Clean separation: large normal population + a distinct suspicious tail. |

---

## 11. Caveats

- **Background contamination** — the background is a random sample of candidates
  assumed mostly-normal (bots rare). If real bots land in it, the null inflates and the
  threshold becomes conservative (lower recall). A curated clean-account background
  would remove this.
- **Thin seeds** — seeds with very little text (e.g. a 1-post account) can't be
  fingerprinted, attract no matches, and drag the recall metric down without adding
  detection power. Rescrape or drop them.
- **Non-normal tail** — `z>2` ≈ "top 2.3%" only if the null is Gaussian; the candidate
  pool has a heavier tail (~4.4% flagged), reflecting real enrichment.
- **Non-ASCII is topic-aware** — counting all non-ASCII includes Korean (topic) and
  accents, which is why it's a *reported* column, not a flagging signal.

---

## 12. Lifecycle: this is a bootstrap — revert to the old clustering once new seeds exist

The style-based script is **not meant to run forever**. It exists to solve a
**cold-start problem**: when a *new* campaign appears, you have no seeds for it — only
the *previous* campaign's seeds, on a different topic. Topic-based clustering can't
bridge that gap, so we match on topic-invariant **style** to catch the first
new-campaign accounts.

**Once the first ~2 new-campaign accounts are caught and added to daily scraping, the
situation changes — and the old clustering (`account_clustering.py`) becomes the
better tool again.** Why:

1. **You now have in-campaign seeds.** The accounts you just confirmed belong to the
   *new* campaign — same topic, same narrative, same time window — and daily scraping
   makes their data richer every day. You no longer have to reach across a topic gap.

2. **Topic flips from a liability to an asset.** The *only* reason the style script
   drops topic features is that K-pop seeds couldn't match AI candidates on subject
   matter. But a new-campaign seed vs a new-campaign candidate now **share the topic**
   (the same coordinated narrative) *and* the style. So the old pipeline's content
   features — word TF-IDF + the semantic embedding — **add discriminating signal
   instead of subtracting it.** Within one campaign, "they all push the same message"
   is a strong, legitimate clustering signal.

3. **The old pipeline is the production system.** It's automated end-to-end
   (`daily_pipeline.py`), and the style script's flags feed straight into it:
   PRAW karma enrichment → Gemini writing-style comparison → interaction tracing →
   `coordinated_accounts.csv`. The style script is a standalone discovery tool, not
   wired into the daily run.

**So the workflow is a hand-off:**

```
New campaign, no seeds, topic changed
        │
        ▼
[ style-based detection ]  ← cross-campaign bootstrap (this script)
        │  catches first new-campaign accounts
        ▼
Confirm 1–2 new seeds → add to daily scraping
        │  now you have in-campaign, same-topic seeds
        ▼
[ old account_clustering.py ]  ← steady-state monitoring
   topic + semantic features now help, fully automated daily pipeline
```

In short: **use style-matching to break into a new campaign, then switch back to the
topic-aware clustering to monitor it at scale** — because once your seeds and your
targets are in the same campaign, the topic signal you deliberately threw away is
exactly what makes the old method strong.
