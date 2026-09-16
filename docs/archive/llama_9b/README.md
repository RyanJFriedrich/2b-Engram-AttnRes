# Legacy Archive: Llama-9B (Spec v1.x & v2.x)

> [!WARNING]
> **RETIRED & ARCHIVED**: The Llama-9B model architecture and Meta Llama-3.1 donor foundation are completely retired due to Meta Llama licensing restrictions.
>
> The project has pivoted to an Apache 2.0 clean-room foundation: **OLMo-2B (Spec v3.0)** using AllenAI's `allenai/Olmo-3.1-32B-Think` as the donor.
>
> For current, authoritative specifications and operational runbooks, refer to:
> - [`docs/olmo-2b-spec.md`](../../olmo-2b-spec.md) — The authoritative v3.0 architecture specification.
> - [`docs/olmo-2b-runbook.md`](../../olmo-2b-runbook.md) — Operational instructions for local RTX 4090 and 96 GB cloud box.
> - [`docs/engram-addressing-spec.md`](../../engram-addressing-spec.md) — Engram sidecar addressing and hashing specification.

---

### Archived Documents in this Directory

1. `llama-9b-refit-spec.md` — The legacy v2.0 refit/cold-start 9.4B model specification ($d_{\text{model}} = 4096$, 33 layers, 128k Llama-3.1 vocab).
2. `llama-9b-refit-first-steps.md` — The initial bring-up run order and owner deliverables for the 9.4B model.
3. `llama-9b-agent-addendum.md` — Engineering rationale and deliberate exclusions for the 9.4B architecture.
4. `llama-9b-box-runbook.md` — Historical bring-up runbook for the 96 GB Blackwell instance.
5. `llama-9b-refit-data-pipeline.md` — Historical data generation instructions targeting Llama-3.1 scoring.
