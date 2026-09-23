# Blockers and Recovery

## Track A full Maven CRM benchmark — `BLOCKED_EXTERNAL_DATA`

**Exact blocker:** Maven's official public page exposes five 499-row/metadata preview CSV assets. The complete CRM Sales Opportunities package required for primary Track-A evidence is not present locally and requires the site's interactive “Free Download” workflow. The preview is retained only for schema and pipeline smoke tests and cannot support the primary leaderboard or locked claims.

**Why external:** the complete official file is controlled by an interactive upstream acquisition flow; substituting a third-party mirror would violate source governance.

**Resume action:** use a browser to complete the official download at `https://mavenanalytics.io/data-playground/crm-sales-opportunities`; place the unmodified files `sales_pipeline.csv`, `accounts.csv`, `products.csv`, `sales_teams.csv`, and `data_dictionary.csv` in `data/incoming/maven/`; then run `make acquire && make develop`. Acquisition code must reject the 499-row preview for primary evidence.

## Track D Criteo v2.1 — resolved

The historical endpoint remains unavailable, but the dataset was recovered from Criteo's
verified official Hugging Face organization at immutable revision
`2424920019e49d52d72c13ac1143ec5d53af276b`. The required SHA-256 matched exactly and E28
completed under causal protocol D2.

## Git initialization — sandbox filesystem blocker

`git init -b main` was attempted but `.git/info/` is mounted read-only in this session.
After the session, run exactly: `cd /mnt/c/Users/bjw-0/Downloads/AccountPulse && git init -b main`

The `.git` path is an empty read-only mount here; initialization must be performed by the
user outside this sandbox. Nothing was published or pushed.

## Protected Track-A stages

E30/E31 cannot execute until the complete official Track-A source passes G0-G4. This is a scientific protection, not an implementation failure. Do not freeze or open a pseudo-LOCKED result from the preview.
