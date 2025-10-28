# Asset Appraisal Agent (Standalone Workflow)

This folder contains the collateral appraisal workflow that is intended to live in its own
repository (`asset-appraisal-agent`).  It produces CSV exports that the credit appraisal agent
can consume through the new asset bridge API.

## Features

- Multi-stage appraisal pipeline (KYC, document validation, valuation modelling, field inspection,
  credit committee) with synthetic decision logic.
- Generates realistic collateral outcomes: `validated`, `under_verification`, `re_evaluate`,
  `pending_asset_review`, `monitor`, and `denied_fraud`.
- Produces a CSV export with per-loan collateral value, legitimacy/confidence scores, workflow
  trace, and inclusion flag for the credit pipeline.
- CLI helper (`python -m asset_agent.workflow`) to generate synthetic loans or process a provided
  dataset.

## Usage

```bash
cd asset_appraisal_agent
python -m asset_agent.workflow --loans 100 --ratio 0.75 --out exports/sample_asset_appraisals.csv
```

The resulting CSV can be uploaded through the credit platform UI ("Collateral Asset Bridge" panel)
or posted via the `/v1/asset-bridge/upload` API in the credit appraisal repository.

## Export schema

Key columns produced by the workflow:

| Column                   | Description |
| ------------------------ | ----------- |
| `application_id`         | Loan identifier used to join with credit appraisal datasets. |
| `asset_type`             | Collateral category (property, vehicle, equipment, etc.). |
| `collateral_value`       | Latest valuation figure after all stages. |
| `collateral_status`      | Final asset status (`validated`, `under_verification`, `re_evaluate`, etc.). |
| `verification_stage`     | Stage that produced the latest decision. |
| `confidence`             | 0-1 score representing valuation confidence. |
| `legitimacy_score`       | 0-1 score representing document legitimacy. |
| `include_in_credit`      | Whether the loan should continue to the credit appraisal pipeline. |
| `notes`                  | Short free-text note for human reviewers. |
| `workflow_trace`         | JSON trace of all appraisal stages and decisions. |

To integrate with the credit appraisal agent, ensure that the exported CSV includes an
`application_id` column matching the loan dataset fed into the credit pipeline.
