# Clinical Trial Eligibility Rule Verification

This repository contains the code, schemas, prompts, and result summaries used for my MSc thesis on extracting and verifying structured clinical trial eligibility criteria.

The project compares two extraction branches:
- Branch A: BERT-based span extraction with deterministic rule completion.
- Branch B: LLM-based clinical field extraction.

It also includes a three-layer verification and rescue framework:
1. Deterministic verification
2. Branch-specific risk/support verification
3. Rescue, repair, or review decision

The CHIA dataset is publicly available and was used as the source dataset. Raw data and API credentials are not included in this repository.