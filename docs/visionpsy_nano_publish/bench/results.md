# VisionPsy-Nano-460M MLX benchmark

- Config: dtype=bf16, max_new_tokens=64
- Variants: standard,flash
- Images: 7, Prompts: 5

## Summary: avg decode tok/s and peak GPU per variant

| Variant | Avg decode tok/s | Median decode tok/s | Avg peak GPU (GB) | Runs |
|---|---|---|---|---|
| standard | 99.1 | 90.3 | 2.64 | 35 |
| flash | 152.2 | 156.9 | 2.64 | 35 |

## Image: portrait

| Variant | Prompt | Prompt tok | Tiles | Prefill s | Decode tok | Decode tok/s | Peak GPU GB | Output |
|---|---|---|---|---|---|---|---|---|
| standard | describe_en | 861 | 13 | 0.953 | 45 | 155.55 | 2.437 | A smiling man in a white lab coat, labeled with the name "OICOMELVANG" and a medical cross emblem, gestures with his right hand while seated at a desk in a well-lit office. |
| standard | text_en | 861 | 13 | 0.112 | 7 | 53.06 | 2.437 | OICOMELVANG |
| standard | count_en | 863 | 13 | 0.112 | 10 | 70.85 | 2.437 | There are 3 people in the image. |
| standard | subject_en | 860 | 13 | 0.131 | 13 | 86.98 | 2.437 | The main subject is a man wearing a white lab coat. |
| standard | describe_zh | 874 | 13 | 0.113 | 22 | 122.12 | 2.438 | 他說：“OICOMELVANG，25158”。 |
| flash | describe_en | 471 | 7 | 0.27 | 30 | 198.49 | 2.359 | A man in a white lab coat with the name "De WARON" on it stands behind a desk, gesturing with his right hand. |
| flash | text_en | 471 | 7 | 0.063 | 5 | 67.35 | 2.359 | De Wooning |
| flash | count_en | 473 | 7 | 0.062 | 2 | 30.58 | 2.359 | 1 |
| flash | subject_en | 470 | 7 | 0.062 | 26 | 186.45 | 2.359 | The main subject is a man wearing a white lab coat with the name 'De Woon' and a logo on it. |
| flash | describe_zh | 484 | 7 | 0.062 | 64 | 248.98 | 2.36 | 这是一�lic控股的,話是柯尼特,黃色胸部有壊身跡,身ailed上有藥物處單 |

## Image: anime_grid

| Variant | Prompt | Prompt tok | Tiles | Prefill s | Decode tok | Decode tok/s | Peak GPU GB | Output |
|---|---|---|---|---|---|---|---|---|
| standard | describe_en | 1121 | 17 | 0.304 | 56 | 158.74 | 2.906 | This image displays a digital fashion design software interface, showcasing various portrait and profile views of a woman in different outfits, including a cropped tank top, high-waisted jeans, and a white dress, along with motion-blurring and face-close-up options. |
| standard | text_en | 1121 | 17 | 0.147 | 19 | 91.5 | 2.906 | Cama loitapinal! Koma Side Profile Kana 35:4 |
| standard | count_en | 1123 | 17 | 0.147 | 3 | 19.67 | 2.906 | 10 |
| standard | subject_en | 1120 | 17 | 0.204 | 3 | 19.74 | 2.906 | A woman |
| standard | describe_zh | 1134 | 17 | 0.147 | 22 | 100.7 | 2.907 | This is a digital fashion design software interface showcasing various portrait and profile views of a woman in different outfits. |
| flash | describe_en | 1121 | 17 | 0.156 | 48 | 156.47 | 2.906 | This image displays a collection of digital portrait templates featuring various angles of a woman in different outfits, including a full-body shot, a back view, and a close-up, alongside a side profile and a face close-up. |
| flash | text_en | 1121 | 17 | 0.146 | 14 | 73.67 | 2.906 | front - Full Body Kana 3/4 Portrait Side Profile |
| flash | count_en | 1123 | 17 | 0.146 | 11 | 61.44 | 2.906 | There are 12 people in the image. |
| flash | subject_en | 1120 | 17 | 0.147 | 3 | 19.54 | 2.906 | A woman |
| flash | describe_zh | 1134 | 17 | 0.148 | 44 | 149.52 | 2.907 | A collection of fashion photos showcasing a model in various outfits, from a full-body front view to a close-up of her back, with a focus on the Canna Loiteral and Kama brands. |

## Image: neon_text

| Variant | Prompt | Prompt tok | Tiles | Prefill s | Decode tok | Decode tok/s | Peak GPU GB | Output |
|---|---|---|---|---|---|---|---|---|
| standard | describe_en | 1121 | 17 | 0.147 | 39 | 141.99 | 2.906 | A vibrant red neon sign reading "HELLO KADE" glows brightly on a rainy night down a wet city street, reflecting the lights of the surrounding buildings and streetlamps. |
| standard | text_en | 1121 | 17 | 0.146 | 6 | 36.6 | 2.906 | HELLO KADE |
| standard | count_en | 1123 | 17 | 0.147 | 11 | 60.84 | 2.906 | There are 2 people visible in the image. |
| standard | subject_en | 1120 | 17 | 0.147 | 6 | 36.73 | 2.906 | HELLO KADE |
| standard | describe_zh | 1134 | 17 | 0.147 | 7 | 41.71 | 2.907 | A rainy night in kade |
| flash | describe_en | 341 | 5 | 0.052 | 41 | 243.83 | 2.482 | A vibrant red neon sign reading "HELLO KADE" glows brightly on a rainy night in a city street, reflecting the lights and creating a striking contrast against the dark, wet pavement. |
| flash | text_en | 341 | 5 | 0.047 | 6 | 96.9 | 2.482 | HELLO KADE |
| flash | count_en | 343 | 5 | 0.048 | 11 | 142.06 | 2.482 | I cannot count objects or people in the image. |
| flash | subject_en | 340 | 5 | 0.046 | 15 | 170.06 | 2.482 | The main subject is a neon sign that reads 'Hello Kade'. |
| flash | describe_zh | 354 | 5 | 0.047 | 29 | 221.25 | 2.483 | A rainy night in a small town, with a glowing red neon sign that reads "HELLO KADE" hanging above the street. |

## Image: screenshot

| Variant | Prompt | Prompt tok | Tiles | Prefill s | Decode tok | Decode tok/s | Peak GPU GB | Output |
|---|---|---|---|---|---|---|---|---|
| standard | describe_en | 1121 | 17 | 0.146 | 64 | 178.41 | 2.906 | This infographic compares four AI creation packages: PLUS, ULTRA, SEEDANCE 2.0, and UNLIMITED & FREE GENS, highlighting cost savings ($49, $428, $864) and features like parallel generations, Supercomputer access, and unlimited models. |
| standard | text_en | 1121 | 17 | 0.146 | 64 | 177.8 | 2.906 | PLUS 49% OFF For everyday AI creation ULTRA 56% OFF Best Value Select Offer Save $288 compared to monthly $49 $25 per month, billed annually  yellow Select Offer $429 $57 per month, billed annually |
| standard | count_en | 1123 | 17 | 0.146 | 57 | 169.36 | 2.906 | Based on the image provided, here is a count of the objects/people present:  **Objects/People:** *   **PLUS:** 1 *   **ULTRA:** 2 *   **SEEDANCE 2.0:** 3 |
| standard | subject_en | 1120 | 17 | 0.146 | 3 | 19.77 | 2.906 | AI creation |
| standard | describe_zh | 1134 | 17 | 0.148 | 31 | 124.27 | 2.907 | This is an infographic showcasing two different AI creation packages, PLUS and Ultra, highlighting their cost savings, features, and benefits for various AI projects. |
| flash | describe_en | 1121 | 17 | 0.147 | 49 | 156.88 | 2.906 | This infographic compares the cost savings of two AI creation platforms, PLUS and ULTRA, highlighting the benefits of the ULTRA plan for ambitious projects, such as parallel generations, access to supercomputers, and unlimited models and features. |
| flash | text_en | 1121 | 17 | 0.147 | 64 | 176.67 | 2.906 | PLUS 49% OFF For everyday AI creation  ULTRA 56% OFF For ambitious AI projects  Select Offer Save $288 compared to monthly  Parallel generations: up to 6 Videos, 8 Images Access to Supercomputer Access to all Seedance models |
| flash | count_en | 1123 | 17 | 0.147 | 3 | 19.53 | 2.906 | 12 |
| flash | subject_en | 1120 | 17 | 0.146 | 3 | 19.58 | 2.906 | AI creation |
| flash | describe_zh | 1134 | 17 | 0.147 | 28 | 117.5 | 2.907 | This infographic compares the cost and benefits of two AI creation platforms, PLUS and ULTRA, highlighting their respective advantages and savings. |

## Image: chart

| Variant | Prompt | Prompt tok | Tiles | Prefill s | Decode tok | Decode tok/s | Peak GPU GB | Output |
|---|---|---|---|---|---|---|---|---|
| standard | describe_en | 861 | 13 | 0.113 | 37 | 160.66 | 2.437 | This bar chart compares revenue and cost for Q1-Q2 2026, showing that revenue exceeds cost by 100,000 in June. |
| standard | text_en | 861 | 13 | 0.113 | 14 | 90.34 | 2.437 | Q1-Q2 2026 Revenue vs Cost |
| standard | count_en | 863 | 13 | 0.113 | 2 | 17.28 | 2.437 | Six |
| standard | subject_en | 860 | 13 | 0.112 | 5 | 40.01 | 2.437 | Revenue vs Cost |
| standard | describe_zh | 874 | 13 | 0.112 | 24 | 130.43 | 2.438 | This bar chart compares the revenue and cost of a company during Q1-Q2 2026. |
| flash | describe_en | 471 | 7 | 0.063 | 29 | 193.57 | 2.359 | The bar chart compares the revenue and cost of two departments, Revenue and Cost, for the six months ended June 2026. |
| flash | text_en | 471 | 7 | 0.062 | 4 | 55.95 | 2.359 | Revenue Cost |
| flash | count_en | 473 | 7 | 0.062 | 2 | 30.64 | 2.359 | 6 |
| flash | subject_en | 470 | 7 | 0.062 | 3 | 43.71 | 2.359 | Revenue |
| flash | describe_zh | 484 | 7 | 0.062 | 32 | 202.19 | 2.36 | This bar chart compares the revenue and cost of two departments, Revenue and Cost, for the Q1-Q2 2026 period. |

## Image: receipt

| Variant | Prompt | Prompt tok | Tiles | Prefill s | Decode tok | Decode tok/s | Peak GPU GB | Output |
|---|---|---|---|---|---|---|---|---|
| standard | describe_en | 861 | 13 | 0.112 | 64 | 204.22 | 2.437 | This image displays a receipt from Blue Bottle Cafe at 1234 Market St, San Francisco, showing the transaction details for a purchase of coffee, including items like Latte, Cappuccino, Croissant, Blueberry Muffin, and Sparkling Water, with a total cost of |
| standard | text_en | 861 | 13 | 0.113 | 64 | 202.68 | 2.437 | BLUE BOTTLE CAFE 1234 Market St, San Francisco Tel: (415) 555-0123 Receipt # A-084213 Date: 2026-07-31 1 |
| standard | count_en | 863 | 13 | 0.112 | 2 | 17.36 | 2.437 | 7 |
| standard | subject_en | 860 | 13 | 0.112 | 11 | 75.81 | 2.437 | The main subject is Blue Bottle Cafe. |
| standard | describe_zh | 874 | 13 | 0.112 | 64 | 202.53 | 2.438 | This is a receipt from Blue Bottle Cafe at 1234 Market St, San Francisco, showing the payment for 2026-07-31 14:32, including items like Latte, Cappuccino, Croissant, Blueberry Muffin |
| flash | describe_en | 211 | 3 | 0.04 | 36 | 268.12 | 2.723 | This image displays a receipt from Blue Bottle Cafe in San Francisco, detailing the purchase of various beverages and taxes, including a total of $30.18. |
| flash | text_en | 211 | 3 | 0.031 | 64 | 295.0 | 2.723 | Blue Bottle Cafe   1234 Market St, San Francisco    Tel: (415) 556-0123    Receipt # A-084213    Date: 2026-07-31 14:32 |
| flash | count_en | 213 | 3 | 0.03 | 64 | 295.19 | 2.724 | <think> I want to count the total number of items listed in the receipt above. There's a list of items with prices and quantities. I need to make sure I don't miss any or mix up names/people. Let's recount carefully.  1.  X Latte (12 oz) |
| flash | subject_en | 210 | 3 | 0.031 | 14 | 203.66 | 2.723 | The main subject is a receipt from Blue Bottle Cafe. |
| flash | describe_zh | 224 | 3 | 0.031 | 64 | 292.75 | 2.725 | This image shows a receipt from Blue Bottle Cafe, located at 1234 Market St, San Francisco. The receipt includes items such as one latte, one cappuccino, two croissants, one blueberry muffin, and two sparkling water. The total cost of the items is |

## Image: diagram

| Variant | Prompt | Prompt tok | Tiles | Prefill s | Decode tok | Decode tok/s | Peak GPU GB | Output |
|---|---|---|---|---|---|---|---|---|
| standard | describe_en | 861 | 13 | 0.112 | 43 | 175.34 | 2.437 | This flowchart illustrates the process of visionpsi-nano, starting with user uploads an image, passing through a preprocessor tiles, then to aSmoIL2 decoder, and finally generating text. |
| standard | text_en | 861 | 13 | 0.112 | 8 | 59.1 | 2.437 | VisionPsy-Nano flow |
| standard | count_en | 863 | 13 | 0.113 | 2 | 17.27 | 2.437 | 5 |
| standard | subject_en | 860 | 13 | 0.113 | 8 | 59.24 | 2.437 | VisionPsy-Nano flow |
| standard | describe_zh | 874 | 13 | 0.113 | 32 | 149.86 | 2.438 | The image flows from the user uploads an image to the preprocessor tiles, then to the SmoIL2 decoder, and finally to generated text. |
| flash | describe_en | 211 | 3 | 0.031 | 29 | 257.37 | 2.723 | This flowchart illustrates the process of generating a vision-based Nano flow from user uploads of images and prompts, culminating in generated text. |
| flash | text_en | 211 | 3 | 0.031 | 8 | 157.12 | 2.723 | VisionPsy-Nano flow |
| flash | count_en | 213 | 3 | 0.031 | 2 | 60.2 | 2.724 | 1 |
| flash | subject_en | 210 | 3 | 0.031 | 8 | 156.81 | 2.723 | VisionPsy-Nano flow |
| flash | describe_zh | 224 | 3 | 0.03 | 29 | 257.05 | 2.725 | This flowchart illustrates the process of generating a vision-based Nano flow, starting from user uploads of images and culminating in generated text. |
