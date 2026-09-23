# Track A Error Analysis

base_protocol: `AP-V1-TRACKA-20260923-A1`
diagnostic_protocol: `AP-V1-TRACKA-20260923-A1-DIAG1`
result_label: `POST_LOCK_DIAGNOSTIC_ONLY`
post_lock: `true`
decision_eligible: `false`


Cases are frozen diagnostic examples and were not used to modify any model.

## Top missed whales

| case_model | opportunity_id | account | product | actual_value | B1_rank | AP-EV_rank |
|---|---|---|---|---|---|---|
| B1 | WXOL5HTS | Ontomedia | GTX Plus Pro | 6818.0000 | 50 | 31 |
| B1 | JWWVG8KG | Inity | GTX Pro | 5729.0000 | 58 | 10 |
| B1 | 7PEL3AIZ | Isdom | GTX Pro | 5685.0000 | 110 | 54 |
| B1 | TLAWJR5V | Ontomedia | GTX Plus Pro | 5646.0000 | 49 | 33 |
| B1 | XO4ZYI3G | Xx-holding | GTX Plus Pro | 5569.0000 | 44 | 36 |
| B1 | S3GVO59B | Stanredtax | GTX Pro | 5542.0000 | 119 | 29 |
| B1 | C933ZQBB | Cancity | GTX Pro | 5458.0000 | 89 | 108 |
| B1 | B894QWQM | Kan-code | GTX Pro | 5360.0000 | 66 | 77 |
| B1 | 4TD018BE | Fasehatice | GTX Pro | 5344.0000 | 81 | 134 |
| B1 | A5IABHP1 | Bioplex | GTX Pro | 5275.0000 | 104 | 164 |
| B1 | HFER9GET | Kinnamplus | GTX Pro | 5239.0000 | 88 | 135 |
| B1 | Q6MGPT50 | Inity | GTX Pro | 5192.0000 | 54 | 11 |
| B1 | OFEUX5PI | Hatfan | GTX Pro | 5190.0000 | 84 | 118 |
| B1 | 0SQM849O | Xx-zobam | GTX Pro | 5144.0000 | 51 | 19 |
| B1 | BUFF2EP5 | Kinnamplus | GTX Pro | 5113.0000 | 80 | 136 |
| B1 | M2VKMZPH | Stanredtax | GTX Pro | 5046.0000 | 112 | 27 |
| B1 | U9QMLC5Z | Cheers | GTX Pro | 4991.0000 | 52 | 22 |
| B1 | 4P0SDD91 | Iselectrics | GTX Pro | 4970.0000 | 72 | 140 |
| B1 | U220GK9W | Streethex | GTX Pro | 4936.0000 | 77 | 112 |
| B1 | 14P6935C | Dontechi | GTX Pro | 4885.0000 | 59 | 59 |

## False priorities

| case_model | opportunity_id | account | product | actual_outcome | B1_rank | AP-EV_rank |
|---|---|---|---|---|---|---|
| B1 | 4MXSHU7X | Kan-code | GTK 500 | LOST | 2 | 313 |
| B1 | PP4ZY6TZ | Treequote | GTK 500 | LOST | 3 | 319 |
| B1 | FOFS0O2G | Xx-zobam | GTX Plus Pro | LOST | 4 | 17 |
| B1 | 8R0HTKUK | Xx-zobam | GTX Plus Pro | LOST | 5 | 23 |
| B1 | ECB42PGL | Zotware | GTX Plus Pro | LOST | 10 | 41 |
| B1 | 56D231XJ | Kan-code | GTX Plus Pro | LOST | 13 | 43 |
| B1 | C3Y94J5R | Globex Corporation | GTX Plus Pro | LOST | 16 | 47 |
| B1 | 6Z6SRB1Y | Hottechi | GTX Plus Pro | LOST | 18 | 57 |
| B1 | X3DSI0TQ | Globex Corporation | GTX Plus Pro | LOST | 20 | 9 |
| B1 | 99HAV7SY | Hottechi | GTX Plus Pro | LOST | 22 | 56 |
| B1 | IDO6PU1B | Hatfan | GTX Plus Pro | LOST | 24 | 146 |
| B1 | LH03MJX5 | Yearin | GTX Plus Pro | LOST | 25 | 12 |
| B1 | CZC1J6FQ | Cancity | GTX Plus Pro | LOST | 28 | 70 |
| B1 | G9H0MWBL | Ganjaflex | GTX Plus Pro | LOST | 29 | 39 |
| B1 | 129P48QP | Ganjaflex | GTX Plus Pro | LOST | 31 | 34 |
| B1 | DHUG18WU | Sonron | GTX Plus Pro | LOST | 39 | 6 |
| AP-EV | QG7RGODO | Plexzap | GTX Pro | LOST | 83 | 2 |
| AP-EV | DHUG18WU | Sonron | GTX Plus Pro | LOST | 39 | 6 |
| AP-EV | KOK62RF4 | Sonron | GTX Pro | LOST | 95 | 7 |
| AP-EV | X3DSI0TQ | Globex Corporation | GTX Plus Pro | LOST | 20 | 9 |

Near-cutoff cases and all requested diagnostic fields are in the canonical Parquet artifact.
