"""
agent_memory_dataset.py — supervisor feedback item #5: build a realistically
sized agent-memory retrieval benchmark with semantically confusable entries
and ACTUAL query-to-memory relevance labels, rather than relying on
nearest-neighbor-by-embedding as a proxy for relevance (the approach used
everywhere else in this project, inherited from the original FlyHash
paper's own evaluation methodology).

Design note (first version had a real flaw, worth recording): an earlier
version of this generator gave many "personas" independently random
values for the SAME fact category (e.g. many different favorite colors)
with no way for a query to indicate WHICH persona it was asking about.
That made the labeled task unsolvable by any text/embedding method in
principle, not just hard — nothing in a bare question like "What is my
favorite color?" can tell a retriever which of 150 same-template
statements is "mine." Measured mAP was near zero for both FlyHash and
SimHash, which looked like a bug but was actually the benchmark asking
an ill-posed question. Fixed by anchoring every fact and every question
to a persona NAME that appears in both, so the query always contains a
real, unique-enough token (the name) that correctly distinguishes the
true answer from same-template distractors about other people — the
same way a real multi-user agent-memory system would actually scope
retrieval, if not by an explicit session ID then at least by a named
reference in the text itself.

Construction, by design:
  - ~150 distinct first names, each one a "persona."
  - ~10 fact slots per persona (favorite_color, allergy, job_title, ...),
    each slot given exactly one ground-truth value, expressed via several
    paraphrase templates that all mention the persona's name — these are
    the true positives for that persona+slot.
  - The full corpus mixes all personas' statements together, so for any
    given query ("What is Jordan's favorite color?"), the corpus is full
    of near-identical, same-template statements about OTHER people's
    favorite colors — genuinely confusable clutter, since the only
    thing that correctly disambiguates the right answer is the name.
  - QUERIES are phrased as questions, worded differently from the
    statement templates, and always include the target persona's name.
  - RELEVANCE LABELS are that persona+slot's ground-truth paraphrase
    sentences — defined by construction, not derived from any embedding
    distance.

Honest limitation, stated explicitly: this is templated, synthetic text,
not naturally-occurring language — real sentence diversity is richer and
messier than any template set, and real agent deployments would likely
scope retrieval by session/user ID rather than relying on a name
mentioned in free text. This dataset is a labeled stand-in for a real,
large-scale, human-written agent-memory corpus; a stronger version using
real sentence-transformer embeddings on a machine with internet access is
a natural follow-up, the same pattern used earlier in this project for
real MiniLM and full MNIST.
"""

from __future__ import annotations
import random


FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie", "Avery",
    "Quinn", "Peyton", "Sam", "Drew", "Reese", "Skyler", "Dakota", "Emerson",
    "Rowan", "Finley", "Harper", "Kendall", "Blake", "Hayden", "Parker", "Sage",
    "Elliot", "Cameron", "Devon", "Shawn", "Marley", "Remy", "Micah", "Lane",
    "Noa", "River", "Phoenix", "Sawyer", "Wren", "Ellis", "Arden", "Brook",
    "Tatum", "Shiloh", "Indigo", "Oakley", "Story", "Winter", "Vesper", "Lux",
    "Zion", "Kai", "Nico", "Jules", "Max", "Theo", "Leo", "Ezra",
    "Mila", "Nora", "Luna", "Ivy", "Hazel", "Ruby", "Clara", "Violet",
    "Nina", "Iris", "June", "Greta", "Freya", "Stella", "Daisy", "Esme",
    "Owen", "Felix", "Milo", "Arlo", "Jasper", "Oscar", "Hugo", "Silas",
    "Eli", "Asher", "Ezekiel", "Gideon", "Soren", "August", "Rhys", "Dante",
    "Amara", "Zara", "Layla", "Priya", "Nadia", "Yasmin", "Leila", "Amira",
    "Kiran", "Rohan", "Arjun", "Dev", "Ishaan", "Vikram", "Sanjay", "Neel",
    "Mei", "Yuki", "Haruto", "Sora", "Aiko", "Kenji", "Hana", "Ren",
    "Diego", "Mateo", "Santiago", "Luca", "Valentina", "Camila", "Sofia", "Emilia",
    "Anya", "Katya", "Dmitri", "Ivan", "Olga", "Mikhail", "Nika", "Pavel",
    "Fatima", "Omar", "Hassan", "Layth", "Sara", "Yara", "Zayd", "Amir",
    "Kofi", "Amara2", "Zuri", "Jabari", "Nia", "Kwame", "Asha", "Themba",
    "Elin", "Freja", "Oskar", "Sigrid", "Liv", "Magnus", "Astrid", "Bjorn",
    "Mara", "Tomasz", "Zofia", "Piotr", "Kasia", "Wiktor", "Alicja", "Marek",
]


SLOT_SPECS = [
    ("favorite_color", ["blue", "red", "green", "purple", "yellow", "black", "orange", "teal"],
     ["{name}'s favorite color is {value}.", "{name} really likes the color {value}.",
      "If you ask {name}, {value} is their favorite color.",
      "{value} has always been {name}'s favorite color."],
     ["What is {name}'s favorite color?", "Which color does {name} like best?",
      "Can you remind me what color {name} prefers?"]),
    ("favorite_food", ["sushi", "pizza", "tacos", "ramen", "pasta", "curry", "burgers", "dumplings"],
     ["{name}'s favorite food is {value}.", "{name} loves eating {value}.",
      "Honestly, {value} is {name}'s favorite thing to eat.",
      "{name} could eat {value} every day."],
     ["What's {name}'s favorite food?", "What does {name} love eating most?",
      "Which dish does {name} prefer above others?"]),
    ("allergy", ["peanuts", "shellfish", "gluten", "dairy", "pollen", "cats", "penicillin", "soy"],
     ["{name} is allergic to {value}.", "{name} has an allergy to {value}.",
      "{name} needs to avoid {value} because of an allergy.",
      "{value} causes an allergic reaction for {name}."],
     ["What is {name} allergic to?", "What should {name} avoid because of an allergy?",
      "Does {name} have any allergies?"]),
    ("hometown", ["Austin", "Seattle", "Denver", "Boston", "Portland", "Chicago", "Phoenix", "Nashville"],
     ["{name} grew up in {value}.", "{name}'s hometown is {value}.",
      "{name} is originally from {value}.",
      "{value} is where {name} was born and raised."],
     ["Where is {name} from?", "What's {name}'s hometown?", "Where did {name} grow up?"]),
    ("pet_name", ["Milo", "Luna", "Biscuit", "Rex", "Olive", "Charlie", "Nova", "Peanut"],
     ["{name}'s pet is named {value}.", "{name} has a pet called {value}.",
      "{value} is the name of {name}'s dog.", "{name}'s cat is called {value}."],
     ["What's {name}'s pet's name?", "What is {name}'s pet called?",
      "Do you remember {name}'s pet's name?"]),
    ("job_title", ["a software engineer", "a nurse", "a teacher", "an accountant",
                   "a graphic designer", "an electrician", "a chef", "a data analyst"],
     ["{name} works as {value}.", "{name}'s job is {value}.",
      "Professionally, {name} is {value}.",
      "{name} has been {value} for a few years now."],
     ["What's {name}'s job?", "What does {name} do for work?", "What is {name}'s profession?"]),
    ("favorite_language", ["Python", "Rust", "JavaScript", "Go", "C++", "Java", "TypeScript", "Swift"],
     ["{name}'s favorite programming language is {value}.", "{name} prefers coding in {value}.",
      "{value} is the language {name} enjoys using most.",
      "{name} mostly writes code in {value}."],
     ["What's {name}'s favorite programming language?",
      "Which language does {name} prefer to code in?",
      "What language does {name} like using most?"]),
    ("hobby", ["rock climbing", "painting", "chess", "gardening", "photography", "running",
               "woodworking", "birdwatching"],
     ["{name}'s hobby is {value}.", "In their free time {name} enjoys {value}.",
      "{name} spends a lot of weekends doing {value}.",
      "{value} is something {name} does for fun."],
     ["What's {name}'s hobby?", "What does {name} like doing in their free time?",
      "What does {name} do for fun?"]),
    ("commute_mode", ["biking", "the subway", "driving", "walking", "carpooling", "the bus",
                       "a scooter", "the train"],
     ["{name} gets to work by {value}.", "{name}'s commute is usually {value}.",
      "{name} usually commutes via {value}.",
      "Most days, {name} gets around by {value}."],
     ["How does {name} get to work?", "What's {name}'s usual commute?",
      "How does {name} normally travel to work?"]),
    ("coffee_order", ["a flat white", "an iced latte", "black coffee", "a cappuccino",
                      "a cold brew", "an americano", "a mocha", "decaf"],
     ["{name}'s usual coffee order is {value}.", "{name} always orders {value}.",
      "When {name} gets coffee, {name} gets {value}.",
      "{value} is what {name} usually drinks."],
     ["What's {name}'s usual coffee order?", "What does {name} normally order at a coffee shop?",
      "What coffee drink does {name} usually get?"]),
]


def build_dataset(n_personas: int = 150, seed: int = 0, n_query_slots_per_persona: int = 1):
    """
    Builds a labeled agent-memory retrieval dataset.

    n_personas: number of distinct named personas, each independently
        assigned a random ground-truth value per slot. This is the main
        lever for corpus SIZE (personas x slots x paraphrases).
    n_query_slots_per_persona: how many of the 10 slots to generate a
        query for, per persona (keeps the query set a manageable size
        while still exercising every slot type across many personas).

    Returns:
        memories: list[str]              -- all memory sentences
        queries: list[str]               -- all question sentences
        relevant_sets: list[set[int]]    -- for queries[i], the set of
                                             indices into `memories` that
                                             are the TRUE ground-truth
                                             answer (by construction)
    """
    rng = random.Random(seed)
    if n_personas > len(FIRST_NAMES):
        raise ValueError(f"Only {len(FIRST_NAMES)} names available, requested {n_personas}")
    names = rng.sample(FIRST_NAMES, n_personas)

    memories: list[str] = []
    queries: list[str] = []
    relevant_sets: list[set] = []

    # persona_slot_indices[(name, slot_name)] = list of memory indices
    persona_slot_indices: dict = {}

    def render(tmpl: str, name: str, value: str) -> str:
        s = tmpl.format(name=name, value=value)
        return s[0].upper() + s[1:] if s else s

    for name in names:
        for slot_name, values, templates, questions in SLOT_SPECS:
            truth_value = rng.choice(values)
            idxs = []
            for tmpl in templates:
                memories.append(render(tmpl, name, truth_value))
                idxs.append(len(memories) - 1)
            persona_slot_indices[(name, slot_name)] = idxs

    for name in names:
        chosen_slots = rng.sample(SLOT_SPECS, n_query_slots_per_persona)
        for slot_name, values, templates, questions in chosen_slots:
            q = rng.choice(questions).format(name=name)
            queries.append(q)
            relevant_sets.append(set(persona_slot_indices[(name, slot_name)]))

    return memories, queries, relevant_sets


if __name__ == "__main__":
    memories, queries, relevant_sets = build_dataset()
    print(f"Memories: {len(memories)}")
    print(f"Queries: {len(queries)}")
    print(f"Avg relevant memories per query: "
          f"{sum(len(s) for s in relevant_sets) / len(relevant_sets):.2f}")
    print("\nExample memory sentences:")
    for m in memories[:6]:
        print(f"  - {m}")
    print("\nExample query + its labeled relevant memories:")
    print(f"  Q: {queries[0]}")
    for idx in sorted(relevant_sets[0]):
        print(f"    relevant -> {memories[idx]}")
