"""Cross-topic, style-based bot detection using campaign_1 hits as seeds.

WHY THIS EXISTS
---------------
account_clustering.py (and cluster_campaign1.py) score candidates by similarity
to the seed set using content-laden features (word TF-IDF + a topic-semantic
MiniLM embedding) and then MIN-MAX RESCALE every run into a 0.4-0.95 band. That
has three problems when the seeds (K-pop, campaign 1) and the candidates
(AI/anti-AI, campaign 2) talk about DIFFERENT topics:

  1. Word TF-IDF and the semantic embedding encode TOPIC, which does not transfer
     across campaigns -> true matches get pushed apart.
  2. The per-run 0.4-0.95 rescale makes the score relative to whoever was scraped
     that day, so the downstream `> 0.5` threshold means a different thing every
     day and ALWAYS yields "top" matches even when nothing is truly similar.
  3. There is no negative control, so the threshold cannot be calibrated.

This prototype reframes the task as authorship verification on TOPIC-INVARIANT
features and calibrates against a background of normal users:

  * Style identity (3 topic-blind blocks):
      - character n-grams (3-5)         -> idiolect / spacing / morphology
      - function-word relative freqs    -> content-free author fingerprint
      - structural stylometry           -> ASCII punctuation profile, sentence
                                           rhythm, type-token ratio, caps/digit
  * Non-ASCII typography is reported as a side-column only (non_ascii_normalized):
    NonAsciiScorer tallies ALL non-ASCII chars (smart-punctuation weighted high,
    everything else at base weight; emoji excluded). It is NOT z-scored or flagged
    — flagging is by style score alone.
  * Score (no per-run rescale): for each candidate,
        style_score = mean(top-k cosine to seeds) - mean(cosine to background)
    i.e. "writes like a known bot AND unlike a normal user". This is a
    likelihood-ratio-style separation, stable across days.
  * Calibration: the same score is computed for every background account, giving
    a null distribution. Candidates are flagged by z-score / percentile against
    that distribution rather than a hand-picked constant.
  * Validation (--validate): leave-one-seed-out. Each seed is scored as if it
    were a candidate (top-k to the OTHER seeds minus background). We report AUC
    (seed scores vs background scores) and recall@flag-threshold, so you can
    measure whether the features actually separate campaign-1 authorship from
    normal users BEFORE trusting the ranking on campaign 2.

Optionally (--style-embed MODEL) a purpose-built topic-invariant *style*
embedding (e.g. a sentence-transformers authorship model) can be added as a
fourth feature block. It is OFF by default and degrades gracefully if the model
or sentence-transformers is unavailable.

USAGE
-----
    python cluster_campaign1_style.py
    python cluster_campaign1_style.py --after 27-02-2026 --before 15-03-2026 --min-posts 2
    python cluster_campaign1_style.py --date 2026-03-10 --validate
    python cluster_campaign1_style.py --validate            # validate over the window

Outputs one CSV per day at clustering/campaign_1_style/{date}/style_ranking.csv.
"""
import argparse
import csv
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, normalize

from account_clustering import AccountClusteringEngine, NonAsciiScorer, parse_date

BASE_DIR = Path(__file__).parent
NEW_DATA_DIR = BASE_DIR / "daily_data"
HITS_DIR = BASE_DIR / "hits" / "campaign_1"   # campaign_1 accounts as seeds

DEFAULT_AFTER = "2026-02-27"
DEFAULT_BEFORE = "2026-03-15"

# Topic-invariant feature block weights (style identity only). Non-ASCII typography
# is scored SEPARATELY (the non-ASCII channel), not blended into the style identity.
DEFAULT_W_CHAR = 0.50    # character n-grams: strongest topic-robust idiolect signal
DEFAULT_W_FUNC = 0.30    # function-word frequencies: content-free
DEFAULT_W_STRUCT = 0.20  # ASCII punctuation / structure / rhythm

# ~150 common English function words (articles, pronouns, prepositions, conjunctions,
# auxiliaries, determiners). These carry author style, not topic.
FUNCTION_WORDS = [
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing",
    "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more",
    "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same",
    "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with",
    "would", "you", "your", "yours", "yourself", "yourselves", "aint", "cant",
    "couldnt", "didnt", "doesnt", "dont", "isnt", "wasnt", "wont", "wouldnt",
    "im", "ive", "ill", "youre", "youve", "hes", "shes", "thats", "theres",
]
_FW_INDEX = {w: i for i, w in enumerate(FUNCTION_WORDS)}

# Structural punctuation marks profiled individually (relative frequency per char).
_STRUCT_PUNCT = list(".,!?;:'\"-()[]")

# Smart-punctuation tells (em-dash, curly quotes, ellipsis, non-breaking hyphen, ...).
# NOTE: retained for reference only and NOT used in the 3-block style identity — these
# are handled by the SEPARATE non-ASCII channel (NonAsciiScorer) below.
_TYPO_CHARS = [
    ("—", "em_dash"),            # —
    ("–", "en_dash"),            # –
    ("’", "curly_apostrophe"),   # ’
    ("“", "curly_dquote_open"),  # “
    ("”", "curly_dquote_close"), # ”
    ("‘", "left_single_curly"),  # ‘
    ("…", "ellipsis"),           # …
    ("‑", "non_breaking_hyphen"),# ‑
    ("→", "arrow"),              # →
    ("≠", "not_equal"),          # ≠
    (" ", "nbsp"),               # no-break space
]
# Slang / informal markers (whole-word). Casual register is part of the fingerprint.
_INFORMAL = {"lol", "lmao", "tbh", "ngl", "imo", "idk", "omg", "fr", "rn", "tho",
             "btw", "iirc", "ofc", "lowkey", "ikr", "smh", "fwiw"}

_WORD_RE = re.compile(r"[a-z']+")


# ==============================================================================
# TOPIC-INVARIANT FEATURE EXTRACTION
# ==============================================================================

def function_word_matrix(texts):
    """Relative frequency of each function word per document. Shape (n, |FW|)."""
    mat = np.zeros((len(texts), len(FUNCTION_WORDS)), dtype=np.float64)
    for r, t in enumerate(texts):
        words = _WORD_RE.findall(t.lower())
        n = len(words)
        if n == 0:
            continue
        for w in words:
            j = _FW_INDEX.get(w)
            if j is not None:
                mat[r, j] += 1.0
        mat[r] /= n
    return mat


def structural_matrix(texts):
    """Compact topic-blind structural stylometry per document."""
    rows = []
    for t in texts:
        n_chars = max(len(t), 1)
        words = t.split()
        n_words = max(len(words), 1)
        sentences = [s for s in re.split(r"[.!?]+", t) if s.strip()]
        n_sent = max(len(sentences), 1)

        feats = [t.count(p) / n_chars for p in _STRUCT_PUNCT]   # punctuation profile
        feats.append(np.mean([len(w) for w in words]) if words else 0.0)  # avg word len
        feats.append(n_words / n_sent)                          # avg sentence length
        feats.append(len(set(w.lower() for w in words)) / n_words)  # type-token ratio
        feats.append(sum(1 for w in words if w.isupper() and len(w) > 1) / n_words)  # ALLCAPS
        feats.append(sum(c.isdigit() for c in t) / n_chars)     # digit ratio
        feats.append(t.count(" ") / n_chars)                    # whitespace rhythm
        feats.append(t.count("\n") / n_chars)                   # newline habit
        rows.append(feats)
    return np.asarray(rows, dtype=np.float64)


def build_style_matrix(texts, weights, style_embed_model=None):
    """Combine topic-invariant blocks into one L2-normalised style embedding.

    Each block is L2-normalised per row then scaled by its weight, so the weights
    control relative influence regardless of raw magnitude. There is NO per-run
    min-max rescale of similarities downstream -- raw cosine is used.
    """
    # 1. Character n-grams (3-5) -- the workhorse of topic-robust authorship ID.
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        min_df=2, max_features=4000, sublinear_tf=True,
    )
    char_tfidf = char_vec.fit_transform(texts)
    n_comp = min(150, char_tfidf.shape[1] - 1, char_tfidf.shape[0] - 1)
    if n_comp >= 2:
        char_reduced = TruncatedSVD(n_components=n_comp, random_state=42).fit_transform(char_tfidf)
    else:
        char_reduced = char_tfidf.toarray()
    char_norm = normalize(char_reduced, norm="l2")

    # 2. Function-word frequencies (z-scored so each word is comparable).
    fw = function_word_matrix(texts)
    fw_norm = normalize(StandardScaler().fit_transform(fw), norm="l2")

    # 3. Structural stylometry (z-scored).
    st = structural_matrix(texts)
    st_norm = normalize(StandardScaler().fit_transform(st), norm="l2")

    blocks = [
        char_norm * weights["char"],
        fw_norm * weights["func"],
        st_norm * weights["struct"],
    ]

    # 4. Optional purpose-built style embedding (topic-invariant author rep).
    if style_embed_model is not None:
        emb = _try_style_embedding(texts, style_embed_model)
        if emb is not None:
            blocks.append(normalize(emb, norm="l2") * weights.get("embed", 0.5))

    combined = np.hstack(blocks)
    return normalize(combined, norm="l2")


def _try_style_embedding(texts, model_name):
    """Encode texts with a sentence-transformers style/authorship model, or None."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(f"   [style-embed] sentence-transformers not installed; skipping {model_name}")
        return None
    try:
        model = SentenceTransformer(model_name)
        return model.encode([t[:5000] for t in texts], convert_to_numpy=True,
                            show_progress_bar=False)
    except Exception as e:  # noqa: BLE001 - any load/encode failure degrades gracefully
        print(f"   [style-embed] could not use {model_name}: {e}")
        return None


# ==============================================================================
# SCORING  (separation against a background negative set, no per-run rescale)
# ==============================================================================

def topk_mean(sim_block, k):
    """Mean of the top-k values in each row. sim_block shape (rows, n_refs)."""
    k = max(1, min(k, sim_block.shape[1]))
    part = np.sort(sim_block, axis=1)[:, -k:]
    return part.mean(axis=1)


def score_accounts(style, seed_idx, target_idx, bg_idx, target_names, bg_names, k):
    """style_score = mean(top-k cosine to seeds) - mean(cosine to background).

    Self-matches (a target that also appears in the background sample) are masked
    out of the background mean so an account never boosts its own null.
    """
    seed_sims = cosine_similarity(style[target_idx], style[seed_idx])   # (T, S)
    bg_sims = cosine_similarity(style[target_idx], style[bg_idx])       # (T, B)

    bg_name_arr = np.asarray(bg_names)
    bg_mean = np.empty(len(target_idx))
    for r, name in enumerate(target_names):
        mask = bg_name_arr != name
        row = bg_sims[r][mask] if mask.any() else bg_sims[r]
        bg_mean[r] = row.mean() if row.size else 0.0

    seed_top = topk_mean(seed_sims, k)
    most_similar_seed_pos = seed_sims.argmax(axis=1)
    return seed_top, bg_mean, seed_top - bg_mean, most_similar_seed_pos


def background_null(style, seed_idx, bg_idx, k):
    """Score every background account the same way to build a null distribution."""
    bg_seed = cosine_similarity(style[bg_idx], style[seed_idx])
    bg_bg = cosine_similarity(style[bg_idx])
    np.fill_diagonal(bg_bg, np.nan)                 # exclude self
    seed_top = topk_mean(bg_seed, k)
    bg_mean = np.nanmean(bg_bg, axis=1)
    return seed_top - bg_mean


def leave_one_seed_out(style, seed_idx, bg_idx, k):
    """Score each seed against the OTHER seeds (held out) minus background."""
    ss = cosine_similarity(style[seed_idx])
    np.fill_diagonal(ss, np.nan)                    # never match a seed to itself
    sb = cosine_similarity(style[seed_idx], style[bg_idx])
    scores = np.empty(len(seed_idx))
    for r in range(len(seed_idx)):
        others = ss[r][~np.isnan(ss[r])]
        if others.size == 0:
            scores[r] = np.nan
            continue
        kk = max(1, min(k, others.size))
        seed_top = np.sort(others)[-kk:].mean()
        scores[r] = seed_top - sb[r].mean()
    return scores


def auc_separation(positive, negative):
    """P(positive > negative) over all pairs == ROC AUC. Ignores NaNs."""
    pos = positive[~np.isnan(positive)]
    neg = negative[~np.isnan(negative)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    pairwise = pos[:, None] - neg[None, :]            # >0 where seed beats background
    wins = (pairwise > 0).sum()
    ties = (pairwise == 0).sum()
    return (wins + 0.5 * ties) / (pos.size * neg.size)


# ==============================================================================
# WINDOWED DRIVER  (collate all unique accounts across an N-day window)
# ==============================================================================

def _dates_in_range(start_date, end_date):
    """List of YYYY-MM-DD strings from start_date to end_date inclusive."""
    cur = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    out = []
    while cur <= end:
        out.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return out


def _force_all_seeds(engine):
    """Use EVERY hits/campaign_1 account as a seed.

    The engine's load_data() drops a hardcoded skip list and may have pulled some
    campaign_1 accounts in as ordinary candidates. Reload all seed files directly,
    label them 'hit', and override any candidate copy with the seed-file text.
    Returns the final seed count.
    """
    agg = engine.text_aggregator
    short = []
    for jf in sorted(HITS_DIR.glob("*_posts.json")):
        u = jf.stem.replace("_posts", "")
        try:
            with open(jf, "r", encoding="utf-8") as f:
                posts = json.load(f)
        except Exception:
            continue
        text = agg.preprocess_text(agg.aggregate_account_text(posts))
        if not text:
            continue  # genuinely empty -> nothing to fingerprint
        if len(text.split()) < 10:
            short.append(u)  # thin fingerprint, but include it anyway
        engine.all_accounts[u] = text
        engine.account_labels[u] = "hit"
        engine.hit_accounts[u] = text
    if short:
        print(f"  Note: {len(short)} seed(s) have <10 words (thin fingerprint, still used): "
              f"{', '.join(short)}")
    return sum(1 for v in engine.account_labels.values() if v == "hit")


def run_window(start_date, end_date, label, args, weights):
    output_dir = BASE_DIR / "clustering" / "campaign_1_style" / label
    output_dir.mkdir(parents=True, exist_ok=True)

    have = [d for d in _dates_in_range(start_date, end_date) if (NEW_DATA_DIR / d).is_dir()]
    if not have:
        print(f"  No data folders in window {label} -- skipping.")
        return
    print(f"  Window {label}: collating {len(have)} day(s): {', '.join(have)}")

    # Reuse the existing engine purely for data loading (semantic model OFF).
    # start_date..end_date makes load_data() collate every account's posts across
    # ALL days in the window AND all subreddits, grouped by redditorName.
    engine = AccountClusteringEngine(
        new_data_dir=str(NEW_DATA_DIR),
        hits_dir=str(HITS_DIR),
        output_dir=str(output_dir),
        use_semantic_embeddings=False,
        start_date=start_date,
        end_date=end_date,
        redditor_joined_after=None,
    )
    if args.min_posts is not None:
        engine.min_posts = args.min_posts
    if args.max_posts is not None:
        engine.max_posts = args.max_posts
    engine.load_data()
    n_seeds = _force_all_seeds(engine)   # use ALL campaign_1 accounts as seeds
    print(f"  Using {n_seeds} seeds (all campaign_1 accounts with sufficient text)")

    usernames = list(engine.all_accounts.keys())
    if len(usernames) < 3:
        print(f"  Too few accounts ({len(usernames)}) for {label} -- skipping.")
        return

    labels = engine.account_labels
    seed_names = [u for u in usernames if labels.get(u) == "hit"]
    cand_names = [u for u in usernames if labels.get(u) == "new_account"]
    if not seed_names or not cand_names:
        print(f"  Need both seeds and candidates (seeds={len(seed_names)}, "
              f"cands={len(cand_names)}) -- skipping {label}.")
        return

    texts = [engine.all_accounts[u] for u in usernames]
    pos = {u: i for i, u in enumerate(usernames)}
    seed_idx = np.array([pos[u] for u in seed_names])
    cand_idx = np.array([pos[u] for u in cand_names])

    # Background = a stable random sample of candidates (proxy for "normal" users;
    # coordinated bots are assumed rare in the general pool).
    rng = np.random.RandomState(42)
    if len(cand_names) > args.bg_size:
        bg_local = rng.choice(len(cand_names), size=args.bg_size, replace=False)
        bg_names = [cand_names[i] for i in bg_local]
    else:
        bg_names = list(cand_names)
    bg_idx = np.array([pos[u] for u in bg_names])

    k = min(args.k, len(seed_names))
    print(f"\n  {label}: seeds={len(seed_names)}  candidates={len(cand_names)}  "
          f"background={len(bg_names)}  top-k={k}")

    # --- Features + scoring -----------------------------------------------------
    style = build_style_matrix(texts, weights, style_embed_model=args.style_embed)

    seed_top, bg_mean, score, ms_pos = score_accounts(
        style, seed_idx, cand_idx, bg_idx, cand_names, bg_names, k)

    null = background_null(style, seed_idx, bg_idx, k)
    null = null[~np.isnan(null)]
    null_mu, null_sd = float(np.mean(null)), float(np.std(null) or 1e-9)
    flag_thresh = null_mu + args.z * null_sd
    z_scores = (score - null_mu) / null_sd

    # Non-ASCII normalized score — reported as a side-column only (NOT z-scored or
    # flagged). NonAsciiScorer tallies all non-ASCII chars (smart-punctuation
    # weighted high, everything else at base weight 1.0; emoji excluded).
    scorer = NonAsciiScorer()
    seed_na = float(np.mean([scorer.score_text(engine.all_accounts[u]).get("normalized_score", 0.0)
                             for u in seed_names]))
    cand_na = np.array([scorer.score_text(t).get("normalized_score", 0.0)
                        for t in (engine.all_accounts[u] for u in cand_names)])

    order = np.argsort(-score)
    n_flag = int((score > flag_thresh).sum())
    print(f"  background null: mean={null_mu:.4f} sd={null_sd:.4f}  "
          f"flag threshold (z>{args.z}) = {flag_thresh:.4f}  -> {n_flag} flagged"
          f"   (seed avg non-ASCII = {seed_na:.2f})")

    # --- Optional validation ----------------------------------------------------
    if args.validate:
        loo = leave_one_seed_out(style, seed_idx, bg_idx, k)
        auc = auc_separation(loo, background_null(style, seed_idx, bg_idx, k))
        recall = float(np.mean(loo[~np.isnan(loo)] > flag_thresh)) if loo.size else float("nan")
        print(f"  [VALIDATION] leave-one-seed-out AUC vs background = {auc:.3f}  "
              f"(0.5=useless, 1.0=perfect)")
        print(f"  [VALIDATION] held-out seed recall @ flag threshold = {recall:.2%}  "
              f"(how many real campaign-1 authors we'd catch)")

    # --- Write ranking ----------------------------------------------------------
    out_csv = output_dir / "style_ranking.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "account", "postURL", "style_score", "z_score", "flagged",
            "seed_topk_sim", "background_sim", "most_similar_seed",
            "non_ascii_normalized", "seed_avg_non_ascii", "n_posts",
        ])
        for i in order:
            u = cand_names[i]
            w.writerow([
                u, f"https://reddit.com/user/{u}",
                f"{score[i]:.4f}", f"{z_scores[i]:.3f}",
                int(score[i] > flag_thresh),
                f"{seed_top[i]:.4f}", f"{bg_mean[i]:.4f}",
                seed_names[ms_pos[i]],
                f"{cand_na[i]:.2f}", f"{seed_na:.2f}",
                engine.account_post_counts.get(u, 0),
            ])
    print(f"  saved {len(cand_names)} ranked candidates -> {out_csv}")

    # Console preview of the top hits ("*" = style-flagged; na = non-ASCII score).
    top = order[:args.top_n]
    print(f"\n  Top {min(args.top_n, len(top))} by style separation ({label}):")
    print(f"  {'account':<24} {'score':>7} {'z':>6} {'seedK':>7} {'bg':>7} {'na':>7}  most_similar_seed")
    for i in top:
        flag = "*" if score[i] > flag_thresh else " "
        print(f"  {flag}{cand_names[i]:<23} {score[i]:>7.4f} {z_scores[i]:>6.2f} "
              f"{seed_top[i]:>7.4f} {bg_mean[i]:>7.4f} {cand_na[i]:>7.2f}  {seed_names[ms_pos[i]]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--after", default=DEFAULT_AFTER, help="Start date DD-MM-YYYY or YYYY-MM-DD")
    ap.add_argument("--before", default=DEFAULT_BEFORE, help="End date DD-MM-YYYY or YYYY-MM-DD")
    ap.add_argument("--date", default=None, help="Single date (overrides --after/--before)")
    ap.add_argument("--window", type=int, default=1,
                    help="Collate every N consecutive days into one run (default 1 = per-day). "
                         "e.g. --window 3 gives more text per account for stabler stylometry.")
    ap.add_argument("--min-posts", type=int, default=2, help="Min posts per candidate (default 2)")
    ap.add_argument("--max-posts", type=int, default=None, help="Max posts per candidate")
    ap.add_argument("--k", type=int, default=5, help="top-k nearest seeds to average (default 5)")
    ap.add_argument("--bg-size", type=int, default=400, help="Background sample size (default 400)")
    ap.add_argument("--z", type=float, default=3.0, help="Flag threshold in std-devs above background null (default 3)")
    ap.add_argument("--top-n", type=int, default=30, help="How many to preview per day (default 30)")
    ap.add_argument("--w-char", type=float, default=DEFAULT_W_CHAR, help="Char n-gram weight")
    ap.add_argument("--w-func", type=float, default=DEFAULT_W_FUNC, help="Function-word weight")
    ap.add_argument("--w-struct", type=float, default=DEFAULT_W_STRUCT, help="Structural weight")
    ap.add_argument("--style-embed", default=None, metavar="MODEL",
                    help="Optional sentence-transformers style/authorship model "
                         "(e.g. AnnaWegmann/Style-Embedding). Off by default.")
    ap.add_argument("--validate", action="store_true",
                    help="Run leave-one-seed-out validation (AUC + recall) per day")
    args = ap.parse_args()

    weights = {"char": args.w_char, "func": args.w_func, "struct": args.w_struct, "embed": 0.5}

    if not HITS_DIR.is_dir():
        print(f"Seed dir not found: {HITS_DIR}")
        return
    n_seeds = len(list(HITS_DIR.glob("*_posts.json")))

    if args.date:
        dates = [parse_date(args.date)]
    else:
        dates = _dates_in_range(parse_date(args.after), parse_date(args.before))

    # Chunk the date list into non-overlapping windows of --window days.
    win = max(1, args.window)
    windows = []
    for i in range(0, len(dates), win):
        block = dates[i:i + win]
        start, end = block[0], block[-1]
        label = start if start == end else f"{start}_to_{end}"
        windows.append((start, end, label))

    print("#" * 80)
    print(f"# CAMPAIGN_1 STYLE-BASED (TOPIC-INVARIANT) DETECTION  {dates[0]} -> {dates[-1]}")
    print(f"# Seeds: up to {n_seeds} accounts from {HITS_DIR} (all used)")
    print(f"# Window: {win} day(s) per run  ->  {len(windows)} window(s)")
    print(f"# Feature weights: char={weights['char']} func={weights['func']} "
          f"struct={weights['struct']}"
          + (f"  embed={weights['embed']} ({args.style_embed})" if args.style_embed else ""))
    print(f"# Score = mean(top-{args.k} cosine to seeds) - mean(cosine to background)  (no per-run rescale)")
    print("#" * 80)

    for i, (start, end, label) in enumerate(windows, 1):
        print("\n" + "#" * 80)
        print(f"# [{i}/{len(windows)}] {label}")
        print("#" * 80)
        run_window(start, end, label, args, weights)


if __name__ == "__main__":
    main()
