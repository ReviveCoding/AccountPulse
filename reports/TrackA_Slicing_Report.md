# Track A Slicing Report

base_protocol: `AP-V1-TRACKA-20260923-A1`
diagnostic_protocol: `AP-V1-TRACKA-20260923-A1-DIAG1`
result_label: `POST_LOCK_DIAGNOSTIC_ONLY`
post_lock: `true`
decision_eligible: `false`


Local slice ranking and global policy behavior are separate rows and must not be mixed. Zero history is a **sparse-history proxy, not a true cold account**.

## Largest local AP-EV minus B1 weaknesses

| slice | slice_value | support_n | wins | value_capture_10 | difference_ci_low | difference_ci_high |
|---|---|---|---|---|---|---|
| sector | entertainment | 24 | 12 | -0.3288 | -0.5858 | 0.4502 |
| apev_score_decile | D8 | 40 | 17 | -0.2837 | -0.3991 | -0.0585 |
| b1_score_decile | D4 | 40 | 16 | -0.2096 | -0.3735 | 0.0663 |
| employees_quartile | Q3 | 94 | 51 | -0.1977 | -0.3802 | 0.0876 |
| retrospective_high_value_tail | HIGH_VALUE_TAIL | 19 | 19 | -0.1673 | -0.3105 | 0.0027 |
| company_age_quartile | Q3 | 98 | 49 | -0.1379 | -0.3306 | 0.1532 |
| b1_score_decile | D6 | 40 | 17 | -0.1349 | -0.2678 | 0.0911 |
| region | Kenya | 7 | 7 | -0.1275 | -0.1275 | -0.1275 |
| office_location | Kenya | 7 | 7 | -0.1275 | -0.1275 | -0.1275 |
| engagement_period | 2017-10-09/2017-10-15 | 52 | 22 | -0.1251 | -0.3284 | 0.1304 |
| revenue_quartile | Q4 | 90 | 44 | -0.1190 | -0.3212 | 0.1860 |
| sales_price_quartile | Q4 | 55 | 26 | -0.1045 | -0.3414 | 0.1769 |
| engagement_period | 2017-10-23/2017-10-29 | 114 | 53 | -0.0984 | -0.4110 | 0.1974 |
| sector | retail | 68 | 34 | -0.0857 | -0.2690 | 0.0576 |
| b1_score_decile | D10 | 41 | 22 | -0.0831 | -0.4067 | 0.1863 |
| company_age_quartile | Q1 | 98 | 47 | -0.0810 | -0.2563 | 0.0717 |
| product | MG Advanced | 61 | 28 | -0.0713 | -0.1763 | 0.0804 |
| apev_score_decile | D7 | 40 | 21 | -0.0655 | -0.1991 | 0.0549 |
| region | United States | 313 | 161 | -0.0607 | -0.1849 | 0.0521 |
| office_location | United States | 313 | 161 | -0.0607 | -0.1849 | 0.0521 |

## Interpretation boundary

Small slices with insufficient outcome classes explicitly carry `NOT_EVALUABLE`; intersectional results are exploratory and limited to the six predefined pairs.
