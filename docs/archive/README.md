# Archive — point-in-time documents

Everything in this folder describes the project **as it was on the date in the
document**, not as it is now. The findings were acted on when they were written;
what came of each one is recorded in [`CHANGELOG.md`](../../CHANGELOG.md) under
the release that followed.

They live here rather than at the repository root because a file named
`ADVERSARIAL-REVIEW` sitting next to `README.md` reads as a current description
of the code, and stops being true the moment the next release ships.

| Document | Baseline | What it is |
|---|---|---|
| `ADVERSARIAL-REVIEW-v0.1.13.md` | v0.1.13 (2026-06-29) | Adversarial read of the strategies and the upstream adapter. |
| `ADVERSARIAL-REVIEW-v0.1.30.md` | v0.1.30 (2026-08-21) | The same exercise repeated 17 releases later. |
| `ADVERSARIAL-REVIEW-v0.1.30-followup.md` | v0.1.30 working tree | Re-check of the above against the in-progress fixes. |
| `PROPOSAL-fusion-efficiency.md` | v0.1.36 | Cost/latency proposal for the fusion pipeline. |

Untracked local audits may also sit here; they are ignored by git and never ship.
