# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

**VectorTune 1.0.1**

---

## 2. Intended Use  

VectorTune generates a top-5 list of similar tracks given either a seed song (searched by name/artist in the Streamlit UI) or a hand-crafted taste profile expressed as audio-feature values (used in the CLI stress-test mode). It's a "more like this" recommender, not a personalized-for-you engine — it has no concept of an individual user account, listening history, or long-term preferences; each request is a single, stateless lookup.

It assumes the person issuing a query can either name a song they already like or articulate their taste directly in terms of audio features (energy, valence, danceability, etc.) — it makes no attempt to infer taste from behavior, demographics, or free-text descriptions like "sad" or "upbeat."

This is a classroom exploration project built to demonstrate and critique a simple content-based recommendation approach — it is not intended for production use or real end users. The catalog, scoring rule, and lack of feedback/personalization mean it should be treated as a teaching artifact for understanding how vector-similarity recommenders work (and where they fall short), not as a deployable product.

---

## 3. How the Model Works  

Imagine every song is a dot floating in space, placed there based on things like how energetic it sounds, how happy or sad it feels, how danceable it is, how loud it is, and a handful of other audio qualities — eleven in total. Songs that sound similar end up close together in that space, and songs that sound very different end up far apart.

To get recommendations, you pick a song you like (by searching for its name), and the app treats that song's spot in the space as your "target." Then it measures the straight-line distance from your target to every other song in the catalog — over a million of them — and turns that distance into a friendlier "match score" out of 100: the closer a song is, the higher its score. It doesn't weigh any one quality more than another; energy, mood-ish qualities like valence, and everything else all count equally toward the distance.

Finally, it doesn't just hand you the five closest songs blindly — it also checks that it isn't giving you five near-copies of the same song (like the same artist's other tracks, or different versions from the same album) crowding out the list. If it spots a repeat artist or album already in your results, it skips that song and keeps looking until it has five genuinely distinct picks.

The starter logic only picked the closest songs by distance, with no protection against near-duplicates. The main change I made was adding that "don't crowd the list" check, so the top 5 feel like five different discoveries rather than five versions of the same thing.

---

## 4. Data  

The catalog comes from `data/tracks_features.csv`, a ~1.2M-row Spotify audio-features export (`initialize_engine()` in `src/data_loader.py`). After dropping duplicate `(name, artists)` pairs, **1,141,556 unique tracks** remain in the working catalog.

There are no genre or mood labels in this dataset at all — no "pop," "lofi," "rock" tags, and no explicit mood field. The closest stand-ins are the numeric audio features: `valence` (musical positivity, a rough proxy for "happy" vs. "sad") and `energy` (intensity/activity, a rough proxy for "chill" vs. "intense"). Because of this, the system can't reason about genre or mood the way a human would — it only sees where a song lands in an 11-dimensional numeric space (danceability, energy, key, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, loudness).

I didn't add or remove any rows beyond the built-in deduplication and the normalization already done in `data_loader.py` (min-max scaling `tempo` and `loudness` into `tempo_norm`/`loudness_norm` so they sit on the same 0-1 scale as the other features, and downcasting to `float32`/`int8` to fit in memory). A separate, much smaller `data/songs.csv` (10 hand-crafted example songs with genre/mood/tempo_bpm columns) exists in the repo but isn't actually used by the current recommender — it's leftover from an earlier design and doesn't match today's feature set.

Given the dataset spans a huge, unfiltered swath of Spotify's catalog, whole categories of "musical taste" are still effectively missing from what the model can express: there's no representation of lyrical content/language, no cultural or regional context, no information about a track's popularity or era beyond what tempo/loudness/production style implicitly encode, and no way to express taste in terms a person would actually use, like "sounds like the 90s" or "feels nostalgic."

---

## 5. Strengths  

The system works best for listeners with a coherent, "mainstream-shaped" taste — someone whose preferences sit near the dense middle of the catalog's feature distribution (moderate-to-high energy, clear valence, typical tempo/loudness) consistently gets tight, high-confidence matches, often in the high-80s to low-90s out of 100. The High-Energy Pop, Chill Lofi, and Deep Intense Rock stress-test profiles all landed sensible, genre-coherent results this way.

The scoring also correctly captures the *shape* of a taste profile, not just one dominant feature: the "Conflicting: High Energy + Sad" profile proved this well, since it shared the same energy target as the Pop profile but flipped valence to near-zero, and the results shifted noticeably toward moodier, lower-valence tracks instead of just repeating the Pop results. That matched my intuition — a single flipped feature meaningfully changing the output shows the model is genuinely weighing the whole vector, not just the loudest signal.

It also matched intuition on the "more like this" use case specifically: searching for an existing song (e.g. "Blinding Lights") and getting back tracks with a plausible similar feel, rather than random catalog noise, confirms the core distance-based approach is sound for that narrow job — even without any notion of genre, lyrics, or explicit mood.

---

## 6. Limitations and Bias 

**Weakness discovered during stress testing:** because the scoring rule is unweighted Euclidean distance across all 11 audio features at once, the recommender implicitly favors users whose taste sits near the "average" of the catalog (mid-range energy, valence, danceability, etc.), since that's where tracks are most densely clustered and distances are naturally small. When I tested a "conflicting" profile (energy: 0.9, valence: 0.05 — high energy but sad) it still returned coherent high-energy tracks, but the "all extremes" profile (max/min values across every feature at once) scored 30-45 points lower than the mainstream profiles and surfaced non-music oddities like voicemail recordings and nature-sound tracks instead of songs, because so few real tracks exist near that corner of the feature space. This means users with a genuinely unusual combination of tastes get both lower-confidence matches and lower-quality (sometimes irrelevant) recommendations, while listeners with common, "average" taste profiles get a filter-bubble-like advantage of tighter, more reliable matches — the system isn't unfair by design, but it structurally rewards mainstream taste over niche or self-contradictory taste combinations.

---

## 7. Evaluation  

**Profiles tested:** six feature-vector profiles run against the full 1.2M-track catalog via `python -m src.main --stress-test`: three "normal" taste profiles (High-Energy Pop, Chill Lofi, Deep Intense Rock) and three adversarial ones (Conflicting: High Energy + Sad, All Extremes, Flatline/all-neutral). For each I looked at the top-5 match scores and whether the actual songs "made sense" for the stated taste, or whether the scorer had clearly lost the plot.

**Pairwise comparisons:**

- **High-Energy Pop vs. Chill Lofi:** these sit at opposite ends of energy, danceability, and acousticness. Pop returned upbeat, party-leaning tracks around a 90/100 match score; Lofi returned quiet, acoustic/instrumental tracks around a 76/100 match score. This is exactly what should happen — the two profiles are far apart in feature space, so their nearest neighbors barely overlap.
- **Deep Intense Rock vs. Chill Lofi:** same story, even more extreme — high-energy, low-acousticness "rock" pulled in metal-leaning bands, while Lofi pulled in soft jazz/orchestral pieces. Makes sense: acousticness and energy are doing most of the separating work here.
- **High-Energy Pop vs. Deep Intense Rock:** both profiles have similarly high energy, but Pop has high valence (happy) and major mode, while Rock has low valence (dark) and minor mode. Even though "energy" matched, the results diverged into upbeat dance tracks vs. moodier/heavier tracks — a good sign that valence and mode are contributing real signal beyond raw energy.
- **Conflicting (High Energy + Sad) vs. High-Energy Pop:** same energy target as Pop, but valence flipped to near-zero (sad). The recommendations shifted away from party tracks toward moodier, lower-valence songs, even though "energy" alone would have suggested similar results to Pop. This showed the scorer isn't just reacting to one feature — it's genuinely averaging distance across all of them, so a single flipped feature can meaningfully change the output.
- **All Extremes vs. Flatline:** both are unrealistic edge cases, but they failed differently. All Extremes (max/min values on every feature simultaneously) scored 30-45 points lower than any normal profile and returned non-music oddities (voicemail recordings, nature-sound tracks) because almost nothing in the catalog lives in that corner of the space. Flatline (every feature at exactly 0.5) didn't break down the same way — it returned real songs — but its match scores (65-69) were still noticeably weaker than a coherent, well-formed profile like Deep Intense Rock (86-90), suggesting "average everything" isn't actually a strong signal either.

**What surprised me:** I expected the adversarial profiles to either crash, get filtered out, or just return random noise. Instead the system degraded gracefully but silently — it always returns *something* with a plausible-looking match percentage, even when that percentage is quietly telling you the match is weak or the input didn't make sense. A non-technical user has no way to tell "92% because this is a great match" apart from "51% because your preferences don't correspond to any real music" — both just look like a number and a track list.

**Plain-language example:** imagine a listener who says they want "Happy Pop" but a song like a high-energy, upbeat "workout anthem" (e.g. a hypothetical "Gym Hero"-style track) keeps showing up in their results even though they never asked for a gym playlist. That happens because the recommender doesn't know the *label* "pop" or "happy" at all — it only sees numbers like energy, valence, and danceability. A high-energy workout song and a happy pop song can land at nearly the same coordinates in that numeric space, so to the scoring rule they look like near-identical matches, even though a person would never describe them the same way.

---

## 8. Future Work  

Ideas for how you would improve the model next.  

Prompts:  

- Additional features or preferences  
- Better ways to explain recommendations  
- Improving diversity among the top results  
- Handling more complex user tastes  

---

## 9. Personal Reflection   

### Engineering Process Reflection

#### 1. What was your biggest learning moment during this project?

My biggest learning moment came from wrestling with infrastructure and scale constraints rather than the code logic itself. Encountering a fatal push rejection due to a **345 MB dataset file** taught me a vital lesson about Git tracking behavior. I learned that updating a `.gitignore` *after* a heavy file has been staged does not prevent Git from attempting to upload it. Learning how to forcefully reset local branch pointers (`git reset origin/main`) and purge cached files from the tracking index was an intense but foundational lesson in data configuration management.

Furthermore, optimizing memory footprints by downcasting `float64` primitives to `float32` and `int64` to `int8` showed me that clean data architecture directly impacts an application's ability to stay alive under strict resource caps like Streamlit's 1 GB RAM limit.

#### 2. How did using AI tools help you, and when did you need to double-check them?

AI tools acted as a force-multiplier for prototyping, acting like an expert code reviewer sitting over my shoulder. They were incredibly helpful for generating fast NumPy-broadcasted matrix operations and optimizing file I/O operations (such as streaming chunks into a dataframe using `usecols`).

However, I had to actively double-check the AI's assumptions regarding directory environments and scope architecture. For instance, the AI generated absolute import modules (`from src.data_loader import...`) that completely broke python's path routing because it failed to account for the execution context of the immediate local folder. I had to manually step in and correct these paths to local relative imports. I also had to catch formatting slip-ups, like unquoted string variables in target URLs that triggered immediate `SyntaxErrors`.

#### 3. What surprised you about how simple algorithms can still "feel" like recommendations?

I was surprised by how much computational "magic" can be extracted from simple, primitive linear algebra. At its core, the vector-matching engine relies entirely on basic multi-dimensional distances—literally subtracting coordinates in space and using the Pythagorean theorem to calculate distance.

Despite the total absence of complex deep neural networks or modern machine learning frameworks, mapping these spatial coordinates to human emotional vectors (like "danceability," "energy," and "valence") creates an output list that genuinely matches a user's acoustic expectations. It proved that if you structure data thoughtfully, a pure mathematical proximity formula can mimic human taste with remarkable accuracy.

#### 4. What would you try next if you extended this project?

If I were to extend this project, I would implement two key features:

1. **Relational Database Sharding:** Instead of loading raw `.csv` data into memory, I would parse the full 200 GB Spotify primary metadata file into a cloud-hosted SQLite/PostgreSQL instance. This would allow me to run true SQL relational joins across artist tables to filter tracks dynamically by **Genre** before any vectorized math begins.
2. **Diversity Filters (Anti-Crowding):** I would inject a deduplication layer into the ranking rule. Often, identical mathematical distances cause the top 5 results to be filled with live versions, acoustic cuts, or different remixes of the exact same song. I would add a conditional check that drops any track if its `artist` or parent `album_id` already occupies a slot in the current recommendation array, ensuring a diverse, satisfying list of discoveries for the listener.
