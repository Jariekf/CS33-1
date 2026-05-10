# McKinsey-Style Reference Skeletons

This file is loaded directly into the BAT prompt at runtime as a few-shot
*structural* anchor. It is intentionally NOT pushed into the vector store -
embedding outline-only text would dilute the retrieval space and the
narrative content already covers the same headings.

After ingesting your 10 McKinsey reports, replace the placeholders below
with the actual H1-H3 outline of each report (titles only - no body text).
The goal is to teach the LLM the *shape* of a McKinsey deliverable
(Pyramid Principle, SCQA opening, MECE structure).

---

## Reference Skeleton 1 - <McKinsey Report Title 1>

1. Executive summary
   1.1 Bottom-line answer (one paragraph)
   1.2 Three key takeaways
2. Where we are today
   2.1 Market context
   2.2 Why this matters now
3. What we found
   3.1 Theme A
   3.2 Theme B
   3.3 Theme C
4. Strategic implications
5. Where to start (90-day priorities)

---

## Reference Skeleton 2 - <McKinsey Report Title 2>

(Replace with a real outline.)

---

## Generic McKinsey Pattern (always applies)

- Open with the answer (Pyramid Principle: top-down, not chronological).
- Frame the *Situation*, *Complication*, *Question*, *Answer* (SCQA) early.
- Group findings into 3-5 mutually exclusive, collectively exhaustive (MECE) themes.
- Each theme has: claim -> evidence -> "so what" implication.
- Close with concrete, sequenced actions (no vague "consider exploring").
