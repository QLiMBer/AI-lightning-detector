Open Questions (updated)

- Processing latency: no strict constraint for MVP; we’ll prioritize correctness first.
- JSON schema: v1 approved to “make it work”; we can extend later with rationales or thumbnails if useful.
- Non‑lightning classes: start with lightning only; we may add common confounders later as `non_lightning_events` if observed.
- Error handling: propose to skip unreadable/corrupt videos but log them in a top‑level `reports/index.txt` and continue.
