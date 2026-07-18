# 🎵 VectorTune

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

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

- **What features does each `Song` use in your system?**
  There's no `Song` class in the current implementation — songs live as rows in a pandas DataFrame (`df`) loaded from `tracks_features.csv` in `src/data_loader.py`. Each row carries `name`, `artists`, `album_id`, plus 11 numeric audio features used for scoring: `danceability`, `energy`, `key`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo_norm`, and `loudness_norm` (the last two are min-max normalized versions of `tempo` and `loudness`). There's no genre or explicit mood tag — `valence` (musical positivity) and `energy` are the closest stand-ins for "mood."

- **What information does your `UserProfile` store?**
  There's no `UserProfile` class either. A user's taste is represented as a single "seed vector" — a NumPy array with one value per feature, in the same order as `feature_cols`. In the Streamlit app (`main.py`), that seed vector comes from looking up an existing song by name/artist and using its own feature values as the target. In the CLI stress-test mode, seed vectors are defined directly as feature dictionaries in `STRESS_TEST_PROFILES`.

- **How does your `Recommender` compute a score for each song?**
  `apply_scoring_rule` in `src/recommender.py` computes the Euclidean distance between the seed vector and every song's feature vector (vectorized across the whole matrix with NumPy broadcasting), then converts distance to a bounded 0-100 `match_score` via `1 / (1 + distance) * 100` — closer songs score higher, and all 11 features are weighted equally.

- **How do you choose which songs to recommend?**
  `apply_ranking_rule` takes those match scores and picks the top N. It first pulls a wider candidate pool via `argpartition` (fast, avoids sorting the full 1.2M-row catalog), sorts just that pool by score, then applies an anti-crowding diversity filter (`diversify=True` by default) that skips any candidate whose `artists` or `album_id` already occupies a slot in the results — so near-duplicate remixes/covers/same-album tracks can't crowd out the top N.


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

Seed: searched for the song **"Blinding Lights"** (matched to the Pentatonix cover — the only version present in this dataset build) in the Streamlit app. Its feature vector became the seed, and `apply_scoring_rule` + `apply_ranking_rule` (with the anti-crowding diversity filter on) ran against the full 1.2M-track catalog.

```text
Matched: Blinding Lights ['Pentatonix']

                  name                artists  match_score
              Сердечко ['Natalia Podolskaya']    90.800003
                   Fly        ['Nine Lashes']    90.699997
              Yourself      ['Anybody There']    90.300003
       Congratulations ['The Juliana Theory']    90.199997
When Lightning Strikes         ['Sean Earle']    90.099998
```

**Screenshot or video** *(optional)*: 
![VectorTune app walkthrough](data/vector-tune-app-walkthrough.gif)

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

### Stress Test: Diverse & Adversarial Profiles

Ran via `python -m src.main --stress-test` (see the `STRESS_TEST_PROFILES` dict and `run_stress_test()` in [src/main.py](src/main.py)), against the full 1.2M-track `tracks_features.csv` catalog. Profiles are expressed as normalized feature vectors matching `feature_cols` from `data_loader.py`. Three are "distinct taste" profiles; three are adversarial/edge-case profiles designed to probe whether the Euclidean-distance scoring breaks down (conflicting feature values, all-extreme values, and an all-neutral flatline).

```text
================================================================================
Profile: High-Energy Pop
  {'danceability': 0.8, 'energy': 0.9, 'key': 5, 'mode': 1, 'speechiness': 0.06, 'acousticness': 0.05, 'instrumentalness': 0.0, 'liveness': 0.15, 'valence': 0.85, 'tempo_norm': 0.6, 'loudness_norm': 0.85}
--------------------------------------------------------------------------------
                                name                        artists  match_score
                       We Gotta Talk             ['Jennifer Lopez']    91.800003
                      It Goes Around          ['ILLEGAL SUBSTANCE']    91.099998
                             Chomeur ['Omar Pene', 'Super Diamono']    91.000000
            Arremangala, Arrempujala    ['Internacionales Conejos']    90.400002
 Oh Happy Gay (Extended Party Remix)             ['Manuel Sanchez']    90.199997
The Difference Between Love And Hell           ['Sahara Hotnights']    89.800003

================================================================================
Profile: Chill Lofi
  {'danceability': 0.5, 'energy': 0.25, 'key': 0, 'mode': 1, 'speechiness': 0.04, 'acousticness': 0.85, 'instrumentalness': 0.4, 'liveness': 0.1, 'valence': 0.4, 'tempo_norm': 0.3, 'loudness_norm': 0.3}
--------------------------------------------------------------------------------
                                                   name                                                             artists  match_score
                                                   Wolf                                                 ['Heikki Sarmanto']    76.699997
Den forlorade sonen (The Prodigal Son) Suite: VI. Polka ['Hugo Alfvén', 'RTÉ National Symphony Orchestra', 'Niklas Willen']    76.300003
                  Chicago Moves: II. The Spaghetti Bowl                                  ['David Sampson', 'Gaudete Brass']    75.800003
                                          Lil's Darlin'                                                   ['George Benson']    75.599998
                                             Lil Darlin                                                   ['George Benson']    75.599998
                  To the Roof - from Vanilla Sky (2001)                             ['Nancy Wilson', 'Original Soundtrack']    75.400002

================================================================================
Profile: Deep Intense Rock
  {'danceability': 0.35, 'energy': 0.95, 'key': 2, 'mode': 0, 'speechiness': 0.08, 'acousticness': 0.02, 'instrumentalness': 0.05, 'liveness': 0.35, 'valence': 0.3, 'tempo_norm': 0.75, 'loudness_norm': 0.95}
--------------------------------------------------------------------------------
                        name                 artists  match_score
            Sound of Madness           ['Shinedown']    90.099998
                  Mesikämmen             ['Aeolian']    87.599998
            Genom snö och is             ['Snöfrid']    86.900002
                     Sadness          ['Magg Dylan']    86.000000
Through the Ages of Atrocity           ['Artillery']    85.900002
        Fair Weather Friends ['A Breach of Silence']    85.900002

================================================================================
Profile: Conflicting: High Energy + Sad
  {'danceability': 0.3, 'energy': 0.9, 'key': 1, 'mode': 0, 'speechiness': 0.05, 'acousticness': 0.1, 'instrumentalness': 0.1, 'liveness': 0.2, 'valence': 0.05, 'tempo_norm': 0.7, 'loudness_norm': 0.8}
--------------------------------------------------------------------------------
                         name             artists  match_score
                New Beginning  ['Primary Weapon']    87.599998
                      Longing      ['Karsh Kale']    86.199997
                 Found My Way ['Navion', 'Oryon']    85.400002
                Bring It Back  ['Leeroy Stagger']    83.900002
I DON'T WANNA DIE IN NEW YORK           ['SPICE']    83.699997
                         Prey ['The Agony Scene']    83.500000

================================================================================
Profile: All Extremes (min/max mix)
  {'danceability': 1.0, 'energy': 0.0, 'key': 11, 'mode': 1, 'speechiness': 0.95, 'acousticness': 1.0, 'instrumentalness': 1.0, 'liveness': 1.0, 'valence': 0.0, 'tempo_norm': 0.0, 'loudness_norm': 1.0}
--------------------------------------------------------------------------------
                                                                                                                                                                                                    name                                               artists  match_score
                                                                                                                                                                      Chet Atkins Voice Mail Liner Notes                                        ['Paul Craft']    51.400002
Quiet Bay - Northern Flicker, Robin, Red-Winged Blackbird, Great Blue Heron, Common Grackle, Tree Swallows, Common Yellowthroat, Osprey, Olive-Sided Flycatcher, Ruffled Grouse, Common Loon (In Flight)                            ["Dan Gibson's Solitudes"]    48.799999
                                                                                                                                                                                         Phone Message 1                          ['Vinnie & The Stardusters']    48.099998
                                                                                                                                                                                      Non Military Intro                                      ['Ian Simmonds']    46.900002
                                                                                                                                                                              The Need to Upset Congress                                       ['Howard Zinn']    46.299999
                                                                                                                                                                                 Two2: No. 27, 49'32.665 ['John Cage', 'Rob Haskins', 'Laurel Karlik Sheehan']    46.099998

================================================================================
Profile: Flatline (all neutral 0.5)
  {'danceability': 0.5, 'energy': 0.5, 'key': 6, 'mode': 1, 'speechiness': 0.5, 'acousticness': 0.5, 'instrumentalness': 0.5, 'liveness': 0.5, 'valence': 0.5, 'tempo_norm': 0.5, 'loudness_norm': 0.5}
--------------------------------------------------------------------------------
                name                                           artists  match_score
             Isolate                                  ['Brock Wilson']    69.000000
              Rewind                           ['Pearls Before Swine']    68.800003
José María - En Vivo                                  ['Óscar Chávez']    67.000000
                Upzy ['Jeff Morris', 'Karl Berger', 'Joe Hertenstein']    66.300003
 The Torture Chamber                                      ['Dj Swamp']    65.599998
            Colombia                                       ['IIndman']    65.300003
```

**Observations:** the "conflicting" profile (high energy + low valence/sad) didn't error or degenerate — it just returned high-energy, low-valence tracks, since the scorer has no notion that energy and sadness are "supposed" to conflict; it only measures distance. The "all extremes" profile produced noticeably lower match scores (~46-51 vs. 65-92 elsewhere) and surfaced spoken-word/ambient oddities (voicemail liner notes, nature-sound recordings) rather than music — a sign that an unrealistic combination of extreme feature values pushes the seed vector into a sparse region of the space where only unusual tracks are "close." The flatline (all-0.5) profile didn't collapse either, landing on mid-scored tracks, though its scores (~65-69) were lower than the well-formed taste profiles, suggesting a truly "average" seed is actually harder to match well than a coherent one.

### Update — v1.0.1: Anti-Crowding Diversity Filter

`apply_ranking_rule` in [src/recommender.py](src/recommender.py) now takes a `diversify=True` flag (default on). Instead of just taking the raw top N by score, it pulls a wider candidate pool (`top_n * 20`) and walks it in score order, skipping any track whose `artists` or `album_id` has already filled a slot — so near-duplicate remixes/live cuts/same-album tracks can no longer crowd out the rest of the top 5. This required adding `album_id` to the columns loaded in [src/data_loader.py](src/data_loader.py). Re-running the same stress-test profiles after this change:

```text
================================================================================
Profile: High-Energy Pop
  {'danceability': 0.8, 'energy': 0.9, 'key': 5, 'mode': 1, 'speechiness': 0.06, 'acousticness': 0.05, 'instrumentalness': 0.0, 'liveness': 0.15, 'valence': 0.85, 'tempo_norm': 0.6, 'loudness_norm': 0.85}
--------------------------------------------------------------------------------
                               name                        artists  match_score
                      We Gotta Talk             ['Jennifer Lopez']    91.800003
                     It Goes Around          ['ILLEGAL SUBSTANCE']    91.099998
                            Chomeur ['Omar Pene', 'Super Diamono']    91.000000
           Arremangala, Arrempujala    ['Internacionales Conejos']    90.400002
Oh Happy Gay (Extended Party Remix)             ['Manuel Sanchez']    90.199997

================================================================================
Profile: Chill Lofi
  {'danceability': 0.5, 'energy': 0.25, 'key': 0, 'mode': 1, 'speechiness': 0.04, 'acousticness': 0.85, 'instrumentalness': 0.4, 'liveness': 0.1, 'valence': 0.4, 'tempo_norm': 0.3, 'loudness_norm': 0.3}
--------------------------------------------------------------------------------
                                                   name                                                             artists  match_score
                                                   Wolf                                                 ['Heikki Sarmanto']    76.699997
Den forlorade sonen (The Prodigal Son) Suite: VI. Polka ['Hugo Alfvén', 'RTÉ National Symphony Orchestra', 'Niklas Willen']    76.300003
                  Chicago Moves: II. The Spaghetti Bowl                                  ['David Sampson', 'Gaudete Brass']    75.800003
                                          Lil's Darlin'                                                   ['George Benson']    75.599998
                  To the Roof - from Vanilla Sky (2001)                             ['Nancy Wilson', 'Original Soundtrack']    75.400002

================================================================================
Profile: Deep Intense Rock
  {'danceability': 0.35, 'energy': 0.95, 'key': 2, 'mode': 0, 'speechiness': 0.08, 'acousticness': 0.02, 'instrumentalness': 0.05, 'liveness': 0.35, 'valence': 0.3, 'tempo_norm': 0.75, 'loudness_norm': 0.95}
--------------------------------------------------------------------------------
                        name        artists  match_score
            Sound of Madness  ['Shinedown']    90.099998
                  Mesikämmen    ['Aeolian']    87.599998
            Genom snö och is    ['Snöfrid']    86.900002
                     Sadness ['Magg Dylan']    86.000000
Through the Ages of Atrocity  ['Artillery']    85.900002

================================================================================
Profile: Conflicting: High Energy + Sad
  {'danceability': 0.3, 'energy': 0.9, 'key': 1, 'mode': 0, 'speechiness': 0.05, 'acousticness': 0.1, 'instrumentalness': 0.1, 'liveness': 0.2, 'valence': 0.05, 'tempo_norm': 0.7, 'loudness_norm': 0.8}
--------------------------------------------------------------------------------
                         name             artists  match_score
                New Beginning  ['Primary Weapon']    87.599998
                      Longing      ['Karsh Kale']    86.199997
                 Found My Way ['Navion', 'Oryon']    85.400002
                Bring It Back  ['Leeroy Stagger']    83.900002
I DON'T WANNA DIE IN NEW YORK           ['SPICE']    83.699997

================================================================================
Profile: All Extremes (min/max mix)
  {'danceability': 1.0, 'energy': 0.0, 'key': 11, 'mode': 1, 'speechiness': 0.95, 'acousticness': 1.0, 'instrumentalness': 1.0, 'liveness': 1.0, 'valence': 0.0, 'tempo_norm': 0.0, 'loudness_norm': 1.0}
--------------------------------------------------------------------------------
                                                                                                                                                                                                    name                      artists  match_score
                                                                                                                                                                      Chet Atkins Voice Mail Liner Notes               ['Paul Craft']    51.400002
Quiet Bay - Northern Flicker, Robin, Red-Winged Blackbird, Great Blue Heron, Common Grackle, Tree Swallows, Common Yellowthroat, Osprey, Olive-Sided Flycatcher, Ruffled Grouse, Common Loon (In Flight)   ["Dan Gibson's Solitudes"]    48.799999
                                                                                                                                                                                         Phone Message 1 ['Vinnie & The Stardusters']    48.099998
                                                                                                                                                                                      Non Military Intro             ['Ian Simmonds']    46.900002
                                                                                                                                                                              The Need to Upset Congress              ['Howard Zinn']    46.299999

================================================================================
Profile: Flatline (all neutral 0.5)
  {'danceability': 0.5, 'energy': 0.5, 'key': 6, 'mode': 1, 'speechiness': 0.5, 'acousticness': 0.5, 'instrumentalness': 0.5, 'liveness': 0.5, 'valence': 0.5, 'tempo_norm': 0.5, 'loudness_norm': 0.5}
--------------------------------------------------------------------------------
                name                                           artists  match_score
             Isolate                                  ['Brock Wilson']    69.000000
              Rewind                           ['Pearls Before Swine']    68.800003
José María - En Vivo                                  ['Óscar Chávez']    67.000000
                Upzy ['Jeff Morris', 'Karl Berger', 'Joe Hertenstein']    66.300003
 The Torture Chamber                                      ['Dj Swamp']    65.599998
```

**What changed:** each profile now returns exactly 5 results (the old runs sometimes returned a 6th row from the `top_n + 1` slicing) and no artist or album repeats within a single profile's list. For these six profiles the top 5 tracks themselves happened to already be from distinct artists/albums, so the visible song list looks the same — the filter's effect here is mainly removing the extra row and guaranteeing no crowding, rather than swapping out songs. On denser regions of the catalog (e.g. a seed song with many remixes/covers), the same filter would be expected to swap out later slots that used to be filled by near-duplicates.

### Update — v1.0.2: Visual Summary Table + "Reasons" Column

Terminal output is now rendered as a formatted table via the `tabulate` library instead of a raw pandas `to_string()` dump, and each row now includes a `reasons` column explaining *why* it matched — the two features where the track sits closest to the seed profile (`explain_match()` in `src/recommender.py`, wired through `apply_ranking_rule`'s new optional `seed_vector`/`feature_matrix`/`feature_cols` args). This also shows up in the Streamlit UI as a caption under each track. Re-running the stress test:

```text
================================================================================
Profile: High-Energy Pop
  {'danceability': 0.8, 'energy': 0.9, 'key': 5, 'mode': 1, 'speechiness': 0.06, 'acousticness': 0.05, 'instrumentalness': 0.0, 'liveness': 0.15, 'valence': 0.85, 'tempo_norm': 0.6, 'loudness_norm': 0.85}
--------------------------------------------------------------------------------
╒═════════════════════════════════════╤════════════════════════════════╤═══════════════╤════════════════════════════════════════════════╕
│ name                                │ artists                        │   match_score │ reasons                                        │
╞═════════════════════════════════════╪════════════════════════════════╪═══════════════╪════════════════════════════════════════════════╡
│ We Gotta Talk                       │ ['Jennifer Lopez']             │          91.8 │ Closely matches on key, mode, instrumentalness │
├─────────────────────────────────────┼────────────────────────────────┼───────────────┼────────────────────────────────────────────────┤
│ It Goes Around                      │ ['ILLEGAL SUBSTANCE']          │          91.1 │ Closely matches on key, mode, instrumentalness │
├─────────────────────────────────────┼────────────────────────────────┼───────────────┼────────────────────────────────────────────────┤
│ Chomeur                             │ ['Omar Pene', 'Super Diamono'] │          91   │ Closely matches on key, mode, danceability     │
├─────────────────────────────────────┼────────────────────────────────┼───────────────┼────────────────────────────────────────────────┤
│ Arremangala, Arrempujala            │ ['Internacionales Conejos']    │          90.4 │ Closely matches on key, mode, instrumentalness │
├─────────────────────────────────────┼────────────────────────────────┼───────────────┼────────────────────────────────────────────────┤
│ Oh Happy Gay (Extended Party Remix) │ ['Manuel Sanchez']             │          90.2 │ Closely matches on key, mode, acousticness     │
╘═════════════════════════════════════╧════════════════════════════════╧═══════════════╧════════════════════════════════════════════════╛

================================================================================
Profile: Chill Lofi
  {'danceability': 0.5, 'energy': 0.25, 'key': 0, 'mode': 1, 'speechiness': 0.04, 'acousticness': 0.85, 'instrumentalness': 0.4, 'liveness': 0.1, 'valence': 0.4, 'tempo_norm': 0.3, 'loudness_norm': 0.3}
--------------------------------------------------------------------------------
╒═════════════════════════════════════════════════════════╤═════════════════════════════════════════════════════════════════════╤═══════════════╤═══════════════════════════════════════════╕
│ name                                                    │ artists                                                             │   match_score │ reasons                                   │
╞═════════════════════════════════════════════════════════╪═════════════════════════════════════════════════════════════════════╪═══════════════╪═══════════════════════════════════════════╡
│ Wolf                                                    │ ['Heikki Sarmanto']                                                 │          76.7 │ Closely matches on key, mode, liveness    │
├─────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┼───────────────┼───────────────────────────────────────────┤
│ Den forlorade sonen (The Prodigal Son) Suite: VI. Polka │ ['Hugo Alfvén', 'RTÉ National Symphony Orchestra', 'Niklas Willen'] │          76.3 │ Closely matches on key, mode, speechiness │
├─────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┼───────────────┼───────────────────────────────────────────┤
│ Chicago Moves: II. The Spaghetti Bowl                   │ ['David Sampson', 'Gaudete Brass']                                  │          75.8 │ Closely matches on key, mode, liveness    │
├─────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┼───────────────┼───────────────────────────────────────────┤
│ Lil's Darlin'                                           │ ['George Benson']                                                   │          75.6 │ Closely matches on key, mode, speechiness │
├─────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┼───────────────┼───────────────────────────────────────────┤
│ To the Roof - from Vanilla Sky (2001)                   │ ['Nancy Wilson', 'Original Soundtrack']                             │          75.4 │ Closely matches on key, mode, speechiness │
╘═════════════════════════════════════════════════════════╧═════════════════════════════════════════════════════════════════════╧═══════════════╧═══════════════════════════════════════════╛

================================================================================
Profile: Deep Intense Rock
  {'danceability': 0.35, 'energy': 0.95, 'key': 2, 'mode': 0, 'speechiness': 0.08, 'acousticness': 0.02, 'instrumentalness': 0.05, 'liveness': 0.35, 'valence': 0.3, 'tempo_norm': 0.75, 'loudness_norm': 0.95}
--------------------------------------------------------------------------------
╒══════════════════════════════╤════════════════╤═══════════════╤════════════════════════════════════════════════╕
│ name                         │ artists        │   match_score │ reasons                                        │
╞══════════════════════════════╪════════════════╪═══════════════╪════════════════════════════════════════════════╡
│ Sound of Madness             │ ['Shinedown']  │          90.1 │ Closely matches on key, mode, tempo_norm       │
├──────────────────────────────┼────────────────┼───────────────┼────────────────────────────────────────────────┤
│ Mesikämmen                   │ ['Aeolian']    │          87.6 │ Closely matches on key, mode, liveness         │
├──────────────────────────────┼────────────────┼───────────────┼────────────────────────────────────────────────┤
│ Genom snö och is             │ ['Snöfrid']    │          86.9 │ Closely matches on key, mode, instrumentalness │
├──────────────────────────────┼────────────────┼───────────────┼────────────────────────────────────────────────┤
│ Sadness                      │ ['Magg Dylan'] │          86   │ Closely matches on key, mode, speechiness      │
├──────────────────────────────┼────────────────┼───────────────┼────────────────────────────────────────────────┤
│ Through the Ages of Atrocity │ ['Artillery']  │          85.9 │ Closely matches on key, mode, liveness         │
╘══════════════════════════════╧════════════════╧═══════════════╧════════════════════════════════════════════════╛

================================================================================
Profile: Conflicting: High Energy + Sad
  {'danceability': 0.3, 'energy': 0.9, 'key': 1, 'mode': 0, 'speechiness': 0.05, 'acousticness': 0.1, 'instrumentalness': 0.1, 'liveness': 0.2, 'valence': 0.05, 'tempo_norm': 0.7, 'loudness_norm': 0.8}
--------------------------------------------------------------------------------
╒═══════════════════════════════╤═════════════════════╤═══════════════╤════════════════════════════════════════════╕
│ name                          │ artists             │   match_score │ reasons                                    │
╞═══════════════════════════════╪═════════════════════╪═══════════════╪════════════════════════════════════════════╡
│ New Beginning                 │ ['Primary Weapon']  │          87.6 │ Closely matches on key, mode, tempo_norm   │
├───────────────────────────────┼─────────────────────┼───────────────┼────────────────────────────────────────────┤
│ Longing                       │ ['Karsh Kale']      │          86.2 │ Closely matches on key, mode, acousticness │
├───────────────────────────────┼─────────────────────┼───────────────┼────────────────────────────────────────────┤
│ Found My Way                  │ ['Navion', 'Oryon'] │          85.4 │ Closely matches on key, mode, energy       │
├───────────────────────────────┼─────────────────────┼───────────────┼────────────────────────────────────────────┤
│ Bring It Back                 │ ['Leeroy Stagger']  │          83.9 │ Closely matches on key, mode, speechiness  │
├───────────────────────────────┼─────────────────────┼───────────────┼────────────────────────────────────────────┤
│ I DON'T WANNA DIE IN NEW YORK │ ['SPICE']           │          83.7 │ Closely matches on key, mode, tempo_norm   │
╘═══════════════════════════════╧═════════════════════╧═══════════════╧════════════════════════════════════════════╛

================================================================================
Profile: All Extremes (min/max mix)
  {'danceability': 1.0, 'energy': 0.0, 'key': 11, 'mode': 1, 'speechiness': 0.95, 'acousticness': 1.0, 'instrumentalness': 1.0, 'liveness': 1.0, 'valence': 0.0, 'tempo_norm': 0.0, 'loudness_norm': 1.0}
--------------------------------------------------------------------------------
╒══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╤══════════════════════════════╤═══════════════╤════════════════════════════════════════════╕
│ name                                                                                                                                                                                                     │ artists                      │   match_score │ reasons                                    │
╞══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╪══════════════════════════════╪═══════════════╪════════════════════════════════════════════╡
│ Chet Atkins Voice Mail Liner Notes                                                                                                                                                                       │ ['Paul Craft']               │          51.4 │ Closely matches on key, mode, acousticness │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────┼───────────────┼────────────────────────────────────────────┤
│ Quiet Bay - Northern Flicker, Robin, Red-Winged Blackbird, Great Blue Heron, Common Grackle, Tree Swallows, Common Yellowthroat, Osprey, Olive-Sided Flycatcher, Ruffled Grouse, Common Loon (In Flight) │ ["Dan Gibson's Solitudes"]   │          48.8 │ Closely matches on key, mode, valence      │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────┼───────────────┼────────────────────────────────────────────┤
│ Phone Message 1                                                                                                                                                                                          │ ['Vinnie & The Stardusters'] │          48.1 │ Closely matches on key, mode, acousticness │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────┼───────────────┼────────────────────────────────────────────┤
│ Non Military Intro                                                                                                                                                                                       │ ['Ian Simmonds']             │          46.9 │ Closely matches on key, mode, acousticness │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────┼───────────────┼────────────────────────────────────────────┤
│ The Need to Upset Congress                                                                                                                                                                               │ ['Howard Zinn']              │          46.3 │ Closely matches on key, mode, acousticness │
╘══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╛

================================================================================
Profile: Flatline (all neutral 0.5)
  {'danceability': 0.5, 'energy': 0.5, 'key': 6, 'mode': 1, 'speechiness': 0.5, 'acousticness': 0.5, 'instrumentalness': 0.5, 'liveness': 0.5, 'valence': 0.5, 'tempo_norm': 0.5, 'loudness_norm': 0.5}
--------------------------------------------------------------------------------
╒══════════════════════╤═══════════════════════════════════════════════════╤═══════════════╤═════════════════════════════════════════════╕
│ name                 │ artists                                           │   match_score │ reasons                                     │
╞══════════════════════╪═══════════════════════════════════════════════════╪═══════════════╪═════════════════════════════════════════════╡
│ Isolate              │ ['Brock Wilson']                                  │          69   │ Closely matches on key, mode, energy        │
├──────────────────────┼───────────────────────────────────────────────────┼───────────────┼─────────────────────────────────────────────┤
│ Rewind               │ ['Pearls Before Swine']                           │          68.8 │ Closely matches on key, mode, loudness_norm │
├──────────────────────┼───────────────────────────────────────────────────┼───────────────┼─────────────────────────────────────────────┤
│ José María - En Vivo │ ['Óscar Chávez']                                  │          67   │ Closely matches on key, mode, energy        │
├──────────────────────┼───────────────────────────────────────────────────┼───────────────┼─────────────────────────────────────────────┤
│ Upzy                 │ ['Jeff Morris', 'Karl Berger', 'Joe Hertenstein'] │          66.3 │ Closely matches on key, mode, liveness      │
├──────────────────────┼───────────────────────────────────────────────────┼───────────────┼─────────────────────────────────────────────┤
│ The Torture Chamber  │ ['Dj Swamp']                                      │          65.6 │ Closely matches on key, mode, loudness_norm │
╘══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╛
```

**Unexpected discovery:** every single result across all six profiles lists `key` and `mode` as its top two reasons, with only the third slot varying. That's not a display bug — it's a real property of `apply_scoring_rule`'s scoring: because `key` (an unnormalized 0-11 integer) and `mode` (a 0/1 binary) aren't scaled down like the other 0-1 continuous features, they dominate the raw Euclidean distance far more than intended, so the actual winners of the ranking are, almost always, tracks that happen to share the seed's exact key and mode. This confirms — with direct evidence — the "largest numeric range dominates" bias already called out in the Potential Biases section above, and it means the current 0-100 `match_score` is quietly weighted much more heavily toward key/mode agreement than a listener would expect from "energy," "danceability," etc. (Note: `explain_match()`'s own diff comparisons *are* normalized by each feature's catalog range — this isn't an artifact of how the reasons are computed, it reflects what the un-normalized scoring rule actually optimized for.)

### Update — v1.0.3: Circular + Normalized Key Encoding

Two fixes to `key` in `src/data_loader.py`:

1. **Circular encoding.** Musical key is circular (key 11 is right next to key 0, not maximally far from it), so raw integer key values were replaced with `key_x = 0.5·cos(2π·key/12)` and `key_y = 0.5·sin(2π·key/12)` — a point on a circle, so adjacent keys are actually close in feature space.
2. **Scale normalization.** The unit circle naturally spans `[-1, 1]` (range 2), still twice the span of the other 0-1 features (range 1). Scaling by `0.5` shrinks `key_x`/`key_y` to `[-0.5, 0.5]` (range 1), so key now contributes to the Euclidean distance on equal footing with every other feature instead of being over- or under-weighted.

`src/main.py`'s `build_seed_vector()` applies the identical `0.5` scaling when expanding a profile's human-readable `key` value, so hand-crafted stress-test profiles stay in the same coordinate space as the catalog.

Re-running the stress test, two profiles' results actually changed as a result (not just cosmetic): "All Extremes" swapped in "Hijinx and a Child" and "Understood" for two of its lower slots, and "Flatline" swapped in "Sweater Day / Shelter" and "Solace" — concrete proof the rescaling changed real rankings, not just internal bookkeeping.

**Refined discovery:** `mode` still shows up in nearly every result's reasons, but now that it's properly scaled, that's no longer a normalization artifact — it's because `mode` is *binary* (major/minor), so roughly half the catalog ties the seed exactly (diff = 0) on that one feature alone, while the continuous 0-1 features (energy, valence, etc.) almost never hit an exact match. A binary feature will structurally out-compete continuous features for "closest match" even at equal scale, simply because it's much easier to tie exactly. That's a distinct bias from the one v1.0.3 fixed, and normalizing scale alone doesn't resolve it — it would need either dropping binary features from the distance calculation or explicitly down-weighting them.

---

## Experiments

---

## Limitations and Risks

- **No genre or textual understanding at all.** The scorer only sees 11 numeric audio features (danceability, energy, key, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, loudness) — it has no idea what a song is *about*, what language it's in, or what its lyrics say. Two songs can land at nearly identical coordinates in that feature space despite being totally different genres or moods to a human listener (see the "Blinding Lights" sample above, whose top matches are a mix of pop, rock, and worship tracks).
- **Favors "average" taste over niche or contradictory taste.** Because all features are weighted equally in a plain Euclidean distance, users whose preferences sit near the dense center of the catalog's feature distribution get tight, high-confidence matches, while users with unusual or internally conflicting preferences (e.g. very high energy + very low valence) get lower match scores and sometimes surface odd non-music tracks (spoken-word intros, nature-sound recordings) — see the stress-test results above.
- **No feedback loop.** The system doesn't learn from skips, replays, likes, or repeated searches — every recommendation is a fresh, stateless nearest-neighbor lookup with no memory of past interactions.
- **No popularity, freshness, or business-logic layer.** Unlike real-world recommenders, there's no signal for how popular/recent a track is, so it can just as easily surface an obscure decades-old track as a chart hit, for better or worse.
- **Anti-crowding filter is same-artist/same-album only.** The diversity filter added in v1.0.1 prevents duplicate artists/albums in a single result set, but does nothing to prevent five results that are musically near-identical if they happen to come from different artists — real diversity (of sound, not just of artist name) isn't addressed.
- **Single-seed representation.** A user's taste is reduced to one seed vector (or one lookup song), so multi-modal listeners (e.g. someone who likes both chill acoustic and high-energy dance) will only ever get recommendations clustered around whichever single seed they searched for.
- **Scale/performance risk, not just accuracy risk.** Every search reprocesses the full ~1.2M-row feature matrix in memory; this works today but has no caching/indexing strategy (e.g. approximate nearest neighbors) that would be needed to scale further or run under tighter memory/latency constraints.

---

## Reflection


[**Model Card**](model_card.md)

Building this made concrete something I'd only understood abstractly before: a recommender doesn't need to know what a song "means" to a person in order to predict what they'll like — it just needs the right numbers and a distance function. Turning "taste" into a handful of floats (danceability, energy, valence, and so on) and reducing "similarity" to a Euclidean distance felt almost too simple to work, but running it against a 1.2M-track catalog kept producing recommendations that were genuinely defensible. That was the core lesson: prediction, at its most basic, is just measuring how close two points are in a space you've chosen — and the whole system lives or dies on whether that space actually captures what matters.

The stress-testing step is where the bias became visible rather than theoretical. Because every feature is weighted equally and distance is measured the same way for everyone, the system quietly favors listeners whose taste sits near the dense middle of the catalog's feature distribution — they get tight, confident matches — while listeners with unusual or self-contradictory preferences get weaker matches and occasionally nonsensical results (spoken-word clips, ambient nature recordings) that a person would never call music recommendations. Nothing in the code intends to treat anyone differently; the unfairness comes entirely from the shape of the data and an averaging formula that has no notion of "this doesn't apply to you." That's the part I'll carry forward: bias in a system like this doesn't require a bad rule, just a rule that's blind to the fact that data isn't evenly distributed across the people it's serving.



