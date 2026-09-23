# Data Sources

Raw data is ignored by Git. Hashes and shapes are authoritative in `SOURCE_MANIFEST.yaml`.

## maven_crm_sales_opportunities

- Provider: Maven Analytics
- Official identity: https://mavenanalytics.io/data-playground/crm-sales-opportunities
- Version: complete official package acquired 2026-09-23
- License: Maven Analytics usage terms; redistribution not assumed
- Acquisition: user-authenticated official Maven download
- Claim boundary: `FICTITIOUS_PUBLIC_CRM_NOT_PRODUCTION`
- Limitations: Fictitious public CRM benchmark; not enterprise production evidence. The prior 499-row preview remains schema/audit evidence only.

## olist_marketing_and_ecommerce

- Provider: Olist via Kaggle
- Official identity: olistbr/marketing-funnel-olist and olistbr/brazilian-ecommerce
- Version: Kaggle current 2026-09-22
- License: CC-BY-NC-SA-4.0
- Acquisition: authenticated Kaggle API
- Claim boundary: `PUBLIC_REAL_SPARSE_FUNNEL`
- Limitations: First-contact predictors only; seller linkage is outcome-only; CC-BY-NC-SA-4.0.

## uci_online_retail_ii

- Provider: UCI Machine Learning Repository
- Official identity: dataset 502 Online Retail II
- Version: official archive current 2026-09-22
- License: CC BY 4.0
- Acquisition: direct UCI archive download
- Claim boundary: `PUBLIC_REAL_WHOLESALE_HEAVY_RETAIL`
- Limitations: Retail lifecycle benchmark, not CRM or enterprise production evidence; CC BY 4.0.

## criteo_uplift_v2_1

- Provider: Criteo AI Lab via verified Criteo Hugging Face organization
- Official identity: https://huggingface.co/datasets/criteo/criteo-uplift
- Version: v2.1; Hugging Face revision 2424920019e49d52d72c13ac1143ec5d53af276b
- License: CC BY-NC-SA 4.0
- Acquisition: direct immutable-revision download from verified provider organization
- Claim boundary: `RANDOMIZED_CAUSAL_BENCHMARK_NOT_TRANSPORTABILITY_EVIDENCE`
- Limitations: Non-uniformly subsampled advertising experiment; absolute effects do not generalize beyond the released benchmark population.
