# Data Card

## Track A — Maven CRM

Official provider preview assets were acquired and hashed, but the pipeline file has only 499 rows. It is schema-smoke evidence only (`FICTITIOUS_PUBLIC_PREVIEW_ONLY`). The complete official download is required for primary CRM science.

## Track B — Olist

Official Olist Marketing Funnel and Brazilian E-Commerce Kaggle packages (CC-BY-NC-SA-4.0), 8,000 leads. Predictors are limited to origin, landing-page ID, and first-contact month. Closed-deal fields and seller ID construct outcomes only. The primary external value label is seller marketplace item value in 90 days after first contact. R2 uses FIT/VALIDATION/TEST with 90-day purge gaps.

## Track C — UCI Online Retail II

Official UCI dataset 502 (CC BY 4.0). The workbook has 1,067,371 raw lines; 824,364 lines with a customer identifier support customer-level modeling, producing 44,941 baskets and 70,782 snapshots. It is a wholesale-heavy retail lifecycle benchmark, not CRM. R2 uses cutoff cohorts through March 2011 for FIT, June for VALIDATION, September for TEST, with intervening snapshots purged for label maturity.

## Track D — Criteo

Official Criteo Uplift v2.1 was acquired from Criteo's verified Hugging Face organization at
revision `2424920019e49d52d72c13ac1143ec5d53af276b`. The 311,422,618-byte file contains
13,979,592 rows and matches SHA-256 `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`.
License: CC BY-NC-SA 4.0. It is a non-uniformly subsampled randomized advertising benchmark;
it supports a causal-method bridge, not transportability or enterprise CRM claims.
