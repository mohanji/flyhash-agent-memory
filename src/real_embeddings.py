"""
real_embeddings.py — Real text embeddings via sentence-transformers.

RUN THIS ON YOUR OWN MACHINE, not in a network-locked sandbox: it needs to
download the model (~90MB) from huggingface.co the first time.

This is the loader referenced in benchmark.py / recall_vs_bits.py — it has
the exact same signature as load_embeddings() in benchmark.py, so you can
swap it in with a one-line change (see bottom of this file).
"""

from __future__ import annotations
import numpy as np


# A small, varied corpus so you can see topic clustering in the results.
# Replace this with your own texts (support tickets, chat transcripts,
# doc chunks, whatever your agent's memory should actually hold).
#
# IMPORTANT: keep every sentence here UNIQUE. If n_docs (passed to
# load_embeddings) exceeds len(SAMPLE_CORPUS), sentences get repeated to
# fill the count — and repeated sentences produce IDENTICAL embeddings,
# which creates hundreds of exact-distance-zero "neighbors" for the same
# query. With that many ties, which specific ones land in a "top-10" comes
# down to arbitrary tie-breaking in sorting, not hash quality — so recall
# numbers become noisy and uninformative. For a meaningful benchmark, call
# load_embeddings(n_docs=len(SAMPLE_CORPUS)) so nothing repeats, or add
# enough real unique sentences that duplication isn't needed.
SAMPLE_CORPUS = [
    # tech/support
    "How do I reset my password on the login page?",
    "The app keeps crashing when I open the settings menu.",
    "Payment failed with error code 402, please advise.",
    "Can I export my data to CSV format?",
    "The dashboard is loading very slowly today.",
    "Why can't I log into my account anymore?",
    "The mobile app won't sync with the desktop version.",
    "I'm getting a 500 error when I try to save changes.",
    "How do I enable two-factor authentication?",
    "My subscription renewed but I wanted to cancel it.",
    "The search feature isn't returning any results.",
    "Can I integrate this with our Slack workspace?",
    "Where can I download the API documentation?",
    "The file upload keeps timing out on large files.",
    "How do I change my billing address?",
    "I accidentally deleted a project, can it be restored?",
    "The notifications aren't showing up on my phone.",
    "Is there a dark mode option in the settings?",
    "How do I add a new team member to the workspace?",
    "The chart on my dashboard shows the wrong date range.",
    # cooking
    "What's a good substitute for butter in this recipe?",
    "How long should I bake chicken at 400 degrees?",
    "The soup tastes bland, what spices should I add?",
    "Can I freeze leftover pasta sauce for later?",
    "What's the best way to caramelize onions?",
    "How do I keep my bread from getting too dense?",
    "What temperature should I sear steak at?",
    "Can I substitute Greek yogurt for sour cream?",
    "How do I know when the rice is fully cooked?",
    "What's a quick marinade for grilled vegetables?",
    "How long can I keep cooked rice in the fridge?",
    "What's the difference between baking soda and baking powder?",
    "How do I stop my cake from sinking in the middle?",
    "What herbs pair well with roasted lamb?",
    "Can I use olive oil instead of vegetable oil for frying?",
    "How do I make a roux without lumps?",
    "What's the best way to store fresh basil?",
    "How do I get crispy skin on roasted chicken?",
    "Can I make this recipe vegan by swapping the eggs?",
    "How do I properly reheat leftover fried rice?",
    # travel
    "What documents do I need for international travel?",
    "Is it safe to visit that city right now?",
    "How early should I arrive at the airport?",
    "What's the best time of year to visit Japan?",
    "Do I need a visa to enter that country?",
    "What's the baggage allowance for economy class?",
    "Should I exchange currency before or after arriving?",
    "What vaccines are recommended before this trip?",
    "How do I get a refund for a cancelled flight?",
    "Is travel insurance worth it for a short trip?",
    "What's the best way to get from the airport to downtown?",
    "Can I bring a carry-on and a personal item both?",
    "What's the local emergency number in that country?",
    "Are credit cards widely accepted there or should I bring cash?",
    "How many days do I need to see the main sights?",
    "What's the etiquette around tipping in that country?",
    "Is it better to book tours in advance or on arrival?",
    "What should I pack for a trip with unpredictable weather?",
    "How do I find affordable accommodation in the city center?",
    "What's the safest way to get around at night there?",
    # fitness
    "How many days a week should I lift weights?",
    "What's a good warm-up before running?",
    "Is it normal to feel sore two days after a workout?",
    "How much protein do I need to build muscle?",
    "What stretches help with lower back pain?",
    "Should I do cardio before or after strength training?",
    "How long should I rest between sets for hypertrophy?",
    "What's a beginner-friendly running plan for a 5k?",
    "How do I fix my form on squats to avoid knee pain?",
    "Is it okay to work out on an empty stomach?",
    "What's the best way to improve my flexibility?",
    "How many rest days do I actually need per week?",
    "What exercises target the core without straining the back?",
    "How long does it take to see visible muscle gain?",
    "Should I switch up my workout routine every few weeks?",
    "What's a good post-workout meal for recovery?",
    "How do I know if I'm overtraining?",
    "What's the right way to progress dumbbell weight over time?",
    "Are bodyweight exercises enough to build strength?",
    "How do I prevent shin splints when starting to run?",
    # finance
    "Should I pay off debt or invest first?",
    "What's the difference between a Roth and traditional IRA?",
    "How much should I keep in an emergency fund?",
    "Is it a good time to refinance my mortgage?",
    "What are the tax implications of selling stock?",
    "How do I start building credit from scratch?",
    "What's a reasonable percentage of income to save?",
    "Should I choose a fixed or variable rate loan?",
    "How do index funds differ from actively managed funds?",
    "What's the best way to consolidate credit card debt?",
    "How much life insurance coverage do I actually need?",
    "Is it smart to invest in individual stocks as a beginner?",
    "What's the difference between a 401k and a pension?",
    "How do capital gains taxes work on rental property?",
    "Should I lease or buy a car if I want to save money?",
    "What's a good strategy for paying off student loans faster?",
    "How do I estimate how much I need to retire comfortably?",
    "What's the safest place to keep short-term savings?",
    "How does compound interest actually work over time?",
    "Is a high-yield savings account worth switching to?",
]



def load_embeddings(n_docs: int = 500, embed_dim: int = None, seed: int = 0,
                     model_name: str = "all-MiniLM-L6-v2", **_ignored):
    """
    Same interface as benchmark.load_embeddings(), so it's a drop-in swap.
    `embed_dim` is ignored (the model fixes it — 384 for MiniLM-L6-v2) but
    kept in the signature so existing call sites don't break.
    """
    from sentence_transformers import SentenceTransformer

    rng = np.random.default_rng(seed)
    base = SAMPLE_CORPUS
    if n_docs > len(base):
        print(f"WARNING: n_docs={n_docs} > {len(base)} unique sentences in "
              f"SAMPLE_CORPUS — sentences will repeat, which creates exact-"
              f"duplicate embeddings and can make recall numbers noisy/"
              f"uninformative (see comment above SAMPLE_CORPUS). Consider "
              f"calling with n_docs<={len(base)}, or adding more unique text.")
    texts = [base[i % len(base)] for i in range(n_docs)]

    print(f"Loading sentence-transformers model '{model_name}' "
          f"(downloads on first run, then cached)...")
    model = SentenceTransformer(model_name)

    print(f"Embedding {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    return texts, embeddings


if __name__ == "__main__":
    texts, embeddings = load_embeddings(n_docs=25)
    print(f"\nEmbedded {len(texts)} texts -> shape {embeddings.shape}")
    print("Run benchmark.py / recall_vs_bits.py with this loader by changing:")
    print("    from benchmark import load_embeddings")
    print("to:")
    print("    from real_embeddings import load_embeddings")
