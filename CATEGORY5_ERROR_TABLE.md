# Category 5 Error Table

Scope: current `sample0` A/B against `baseline_wt`, restricted to `category 5` questions where current retrieval missed the gold evidence but baseline `hit@5` succeeded.

Source files:
- [category5_regressed.json](/C:/Users/Meteor/Downloads/專題/MAMGA/category5_regressed.json)
- [category5_regressed_detail.json](/C:/Users/Meteor/Downloads/專題/MAMGA/category5_regressed_detail.json)

## Counts
- Total `category 5` questions: `47`
- Current misses: `33`
- Regressed vs baseline: `10`

Primary buckets for the `10` regressed cases:
- `speaker` 對調: `4`
- `relation` 對調: `3`
- `pet / family ownership` 對調: `3`
- `same-session wrong turn`: `4`
- `duplicated event across sessions`: `2`

Note:
- Several questions belong to more than one bucket.
- The most useful split is not mutually exclusive taxonomy; it is "what intervention is needed next".

## Error Table
| Question | Gold Evidence | Primary Bucket | Secondary Bucket | What Went Wrong | Next Intervention |
| --- | --- | --- | --- | --- | --- |
| `What was grandma's gift to Melanie?` | `D4:3` | `relation` 對調 | `speaker` 對調 | Gold says `grandma -> Caroline`, but question swaps recipient to `Melanie`; retriever floats generic `gift/family` lines instead of contradiction evidence. | `contradiction filter` |
| `What was grandpa's gift to Caroline?` | `D4:3` | `relation` 對調 | `same-session wrong turn` | Gold says `grandma`, not `grandpa`; current lands on `D4:4`, an adjacent turn in the same session, rather than the contradictory source turn. | `contradiction filter` |
| `What did Caroline and her family do while camping?` | `D4:8` | `pet / family ownership` 對調 | `duplicated event across sessions` | Gold is `Melanie` talking about her own family camping; question rewrites ownership to `Caroline`. Retrieval drifts to other family/camping sessions. | `contradiction filter` |
| `What did Caroline and her family see during their camping trip last year?` | `D10:14` | `pet / family ownership` 對調 | `duplicated event across sessions` | Gold is `Melanie`'s family seeing the meteor shower; current prefers other camping sessions with similar surface words. | `contradiction filter` + `question decomposition` |
| `Where did Oscar hide his bone once?` | `D13:6` | `pet / family ownership` 對調 | `same-session wrong turn` | `Oscar` is Caroline's guinea pig; gold evidence is actually about `Oliver` hiding a bone in a slipper. Current gets the right session but not the answer turn. | `contradiction filter` |
| `Who is Caroline a fan of in terms of modern music?` | `D15:28` | `speaker` 對調 | `same-session wrong turn` | Gold is `Melanie` stating *her* modern-music preference; current stays in the music cluster but misses the exact answer turn. | `contradiction filter` + answer-turn rerank |
| `What precautionary sign did Caroline see at the café?` | `D16:16` | `speaker` 對調 | `same-session wrong turn` | Gold is `Melanie` sending the image/sign; current stays in session `16` but keeps selecting Caroline follow-up turns instead of the image-answer turn. | answer-turn rerank |
| `What setback did Caroline face recently?` | `D17:8` | `speaker` 對調 | none | Gold is `Melanie` describing her setback; current jumps to unrelated Caroline turns from other sessions. | `contradiction filter` |
| `What does Caroline do to keep herself busy during her pottery break?` | `D17:10` | `speaker` 對調 | none | Gold is `Melanie` saying she reads and paints during *her* pottery break; question rewrites subject to Caroline. Current retrieves pottery/reading neighbors but not the contradiction turn. | `contradiction filter` |
| `What was the poetry reading that Melanie attended about?` | `D17:18` | `relation` 對調 | `same-session wrong turn` | Gold evidence is a question turn showing Melanie did **not** attend; the actual description is in Caroline's next turn (`D17:19`). This needs resolving the false premise before answer extraction. | `question decomposition` |

## By Bucket
### `speaker` 對調
- Typical pattern: question asks about `Caroline`, but gold evidence is `Melanie` answering about herself or replying to Caroline.
- Affected cases:
  - `Who is Caroline a fan of in terms of modern music?`
  - `What precautionary sign did Caroline see at the café?`
  - `What setback did Caroline face recently?`
  - `What does Caroline do to keep herself busy during her pottery break?`
- Recommendation:
  - Add a `contradiction filter` that checks whether the proposition is actually asserted by the same subject as the question.
  - Keep answer-turn rerank, but only as a secondary ranking aid.

### `relation` 對調
- Typical pattern: kinship or event participation is rewritten with the wrong relation.
- Affected cases:
  - `What was grandma's gift to Melanie?`
  - `What was grandpa's gift to Caroline?`
  - `What was the poetry reading that Melanie attended about?`
- Recommendation:
  - For kinship and participation claims, use `question decomposition` first:
    - extract relation slot: `grandma/grandpa`, `attended/did not attend`
    - verify that slot against evidence before ranking by lexical overlap
  - Then apply `contradiction filter` if the slot mismatches.

### `pet / family ownership` 對調
- Typical pattern: ownership is reassigned to the wrong person or pet.
- Affected cases:
  - `What did Caroline and her family do while camping?`
  - `What did Caroline and her family see during their camping trip last year?`
  - `Where did Oscar hide his bone once?`
- Recommendation:
  - Build a lightweight ownership verifier:
    - pet owner: `Oscar -> Caroline`, `Oliver -> Melanie`
    - family trip ownership: which speaker says `my family`
  - This is best treated as `contradiction filter`, not generic semantic rerank.

### `same-session wrong turn`
- Typical pattern: the system reaches the correct dialogue/session but selects a neighboring turn rather than the true answer/contradiction turn.
- Affected cases:
  - `What was grandpa's gift to Caroline?`
  - `Where did Oscar hide his bone once?`
  - `Who is Caroline a fan of in terms of modern music?`
  - `What precautionary sign did Caroline see at the café?`
  - `What was the poetry reading that Melanie attended about?`
- Recommendation:
  - Keep `same-session answer-turn promotion`, but extend it with explicit local turn windows:
    - if a candidate is in the target session, also score `dia_id +/- 1`
    - promote image-caption turns when the question asks `what/where/who`

### `duplicated event across sessions`
- Typical pattern: multiple sessions contain similar camping/family/trip language, so lexical overlap overwhelms the exact event.
- Affected cases:
  - `What did Caroline and her family do while camping?`
  - `What did Caroline and her family see during their camping trip last year?`
- Recommendation:
  - Use `question decomposition` to isolate event anchors such as `last year`, `camping trip`, `meteor shower`.
  - Then route retrieval by event tuple rather than raw keyword overlap.

## What This Implies
- The next highest-value change is not another general heuristic on rerank weights.
- The dominant gap is proposition verification:
  - `Who/whose relation is this?`
  - `Whose family/pet/activity is this?`
  - `Did this person actually attend/do/own the thing in the question?`
- So the implementation order should be:
  1. Add a narrow `contradiction filter` for `category 5`-style subject/relation/ownership checks.
  2. Add lightweight `question decomposition` for kinship, ownership, and participation slots.
  3. Only then revisit answer-turn rerank if the contradiction-filtered candidate set is still noisy.
