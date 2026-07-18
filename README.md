# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

Replace this paragraph with your own summary of what your version does.

---

## How The System Works

Real-world recommenders (Spotify, YouTube, etc.) generally work by representing both items and users as vectors in some feature space, then ranking items by how close they are to a user's taste profile — often blended with collaborative signals (what similar users liked) and business logic (freshness, diversity, promoted content) layered on top of the raw similarity score. They're also continuously updated from implicit feedback like skips, replays, and dwell time, not just explicit ratings.

My version is a simplified, transparent slice of that idea: each song is represented as a numeric feature vector (e.g. energy, valence, tempo), and the user's taste profile is a single "seed" vector in that same space. The `Recommender` scores every candidate song purely by Euclidean distance to that seed — closer songs get a higher match score — then ranks and truncates to the top N. It intentionally leaves out collaborative filtering, popularity/freshness signals, and feedback loops, prioritizing a scoring rule that's easy to inspect and reason about over one that maximizes engagement or accounts for other listeners' behavior.

### Algorithm Recipe

1. Represent the user's taste profile as a single numeric seed vector over the same features as the songs (e.g. energy, valence, tempo).
2. For each candidate song, build its feature vector and compute the Euclidean distance between it and the seed vector.
3. Convert distance to a bounded match score: `match_score = 1 / (1 + distance) * 100`, so closer songs score near 100 and distant songs approach 0.
4. Partition the candidates to find the top N highest match scores, then sort just that subset descending by score.
5. Return the top N songs (name, artists, match score) as the final recommendation list.

### Potential Biases

- This system might over-prioritize whichever features have the largest numeric range (e.g. tempo in BPM could dominate over a 0–1 valence value) simply because raw Euclidean distance treats all feature dimensions as equally scaled.
- It might over-prioritize genre or whichever feature is most consistently populated in the data, ignoring great songs that match the user's mood but differ on other axes.
- Because it only compares against a single seed vector, it can't capture that a user's taste is often multi-modal (e.g. likes both chill acoustic and high-energy dance), so it may systematically under-recommend one side of a user's actual taste.
- It has no popularity, freshness, or diversity signal, so it could keep surfacing near-duplicate songs instead of varied ones.

Some prompts to answer:

- What features does each `Song` use in your system
  - For example: genre, mood, energy, tempo
- What information does your `UserProfile` store
- How does your `Recommender` compute a score for each song
- How do you choose which songs to recommend

You can include a simple diagram or bullet list if helpful.

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Sample Recommendation Output

Paste a sample of your recommender's output here as a text block so a reader can see what it produces:

```
# e.g.:
# User profile: genre=indie, mood=chill, energy=low
# Recommendations:
#   1. ...
#   2. ...
#   3. ...
```

**Screenshot or video** *(optional)*: <!-- Insert a screenshot or demo video link here -->

### Prior Iteration (Different Implementation)

The output below is from an earlier version of this project, using a genre/mood/energy point-allocation scoring approach rather than the Euclidean-distance vector matching used in the current codebase.

```text
================================================================================
🎵 ALGORITHMIC MUSIC RECOMMENDER — TERMINAL OUTPUT VERIFICATION
================================================================================
Target User Profile: 
  - Preferred Genre: pop
  - Target Mood: happy
  - Desired Energy Level: 0.80
--------------------------------------------------------------------------------

Top 5 Curated Recommendations:

1. "Cruel Summer" — Taylor Swift
   🏆 Final Score: 9.6 / 10.0
   📝 Matching Analysis:
      - [✓] Genre match found: 'pop' (+4.0 pts)
      - [✓] Mood match found: 'happy' (+3.0 pts)
      - [✓] Energy alignment: Song energy 0.78 matches target (+2.6 pts)

2. "Don't Start Now" — Dua Lipa
   🏆 Final Score: 9.3 / 10.0
   📝 Matching Analysis:
      - [✓] Genre match found: 'pop' (+4.0 pts)
      - [✓] Mood match found: 'happy' (+3.0 pts)
      - [✓] Energy alignment: Song energy 0.83 matches target (+2.3 pts)

3. "Blinding Lights" — The Weeknd
   🏆 Final Score: 9.1 / 10.0
   📝 Matching Analysis:
      - [✓] Genre match found: 'pop' (+4.0 pts)
      - [✓] Mood match found: 'happy' (+3.0 pts)
      - [✓] Energy alignment: Song energy 0.73 matches target (+2.1 pts)

4. "Walking On Sunshine" — Katrina and the Waves
   🏆 Final Score: 8.5 / 10.0
   📝 Matching Analysis:
      - [✗] Genre mismatch: 'rock' (+0.0 pts)
      - [✓] Mood match found: 'happy' (+3.0 pts)
      - [✓] Energy alignment: Song energy 0.81 matches target (+5.5 pts)

5. "Dynamite" — BTS
   🏆 Final Score: 8.2 / 10.0
   📝 Matching Analysis:
      - [✓] Genre match found: 'pop' (+4.0 pts)
      - [✗] Mood mismatch: 'energetic' (+0.0 pts)
      - [✓] Energy alignment: Song energy 0.76 matches target (+4.2 pts)

```

---

## Experiments You Tried

Use this section to document the experiments you ran. For example:

- What happened when you changed the weight on genre from 2.0 to 0.5
- What happened when you added tempo or valence to the score
- How did your system behave for different types of users

---

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this



