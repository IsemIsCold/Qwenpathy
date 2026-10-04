# Dataset Statistics

> **Generated:** 2026-10-04
> **Tokenizer:** 1.4 chars/token estimate (Qwen2.5 tokenizer not loaded)

> **Note:** Profile label vocabularies match `submission_schema.json`.

## 1. Overview

| Split | Conversations | Prediction Points | Avg Turns / Conv |
|-------|--------------:|------------------:|-----------------:|
| Train | 8,338 | 18,657 | 14.1 |
| Val | 900 | 2,354 | 13.0 |

## 2. Emotion Distribution

| Emotion | Train Count | Train % | Val Count | Val % |
|---------|------------:|--------:|----------:|------:|
| joy | 171 | 0.9% | 20 | 0.8% |
| gratitude | 695 | 3.7% | 118 | 5.0% |
| relaxed | 2,819 | 15.1% | 409 | 17.4% |
| care | 79 | 0.4% | 11 | 0.5% |
| pride | 234 | 1.3% | 40 | 1.7% |
| neutral | 654 | 3.5% | 53 | 2.3% |
| surprise | 64 | 0.3% | 9 | 0.4% |
| mixed | 945 | 5.1% | 177 | 7.5% |
| sadness | 1,085 | 5.8% | 131 | 5.6% |
| loneliness | 876 | 4.7% | 115 | 4.9% |
| anxiety | 6,340 | 34.0% | 751 | 31.9% |
| anger | 855 | 4.6% | 115 | 4.9% |
| fear | 1,388 | 7.4% | 124 | 5.3% |
| disgust | 125 | 0.7% | 21 | 0.9% |
| shame | 752 | 4.0% | 118 | 5.0% |
| helplessness | 1,575 | 8.4% | 142 | 6.0% |

## 3. User Profile Statistics

### 3.1 Train – Empty Profile Rate

- **Empty profiles:** 6,834 / 18,657 (36.6%)

### 3.2 Train – Profile Label Frequencies

#### personality_traits

| Label | Count | % of points |
|-------|------:|------------:|
| sensitive | 5,165 | 27.7% |
| high_conscientiousness | 2,281 | 12.2% |
| conservative | 1,364 | 7.3% |
| introverted | 717 | 3.8% |
| assertive | 675 | 3.6% |
| agreeable | 361 | 1.9% |
| emotionally_stable | 336 | 1.8% |
| open | 59 | 0.3% |
| extroverted | 37 | 0.2% |
| casual | 27 | 0.1% |

#### interests

| Label | Count | % of points |
|-------|------:|------------:|
| career_development | 1,472 | 7.9% |
| social | 764 | 4.1% |
| study_exam | 746 | 4.0% |
| pets | 744 | 4.0% |
| games | 704 | 3.8% |
| sports_fitness | 625 | 3.3% |
| art_design | 341 | 1.8% |
| reading_writing | 316 | 1.7% |
| travel_outdoor | 144 | 0.8% |
| music | 133 | 0.7% |
| film_animation | 108 | 0.6% |
| programming_technology | 82 | 0.4% |

#### style

| Label | Count | % of points |
|-------|------:|------------:|
| rational | 4,418 | 23.7% |
| direct | 2,603 | 14.0% |
| brief | 1,166 | 6.2% |
| detailed | 1,121 | 6.0% |
| high_emotional_expression | 945 | 5.1% |
| colloquial | 654 | 3.5% |
| indirect | 516 | 2.8% |
| formal | 128 | 0.7% |
| humorous | 81 | 0.4% |
| emoji_user | 27 | 0.1% |
| low_emotional_expression | 3 | 0.0% |

### 3.1 Val – Empty Profile Rate

- **Empty profiles:** 875 / 2,354 (37.2%)

### 3.2 Val – Profile Label Frequencies

#### personality_traits

| Label | Count | % of points |
|-------|------:|------------:|
| sensitive | 487 | 20.7% |
| conservative | 253 | 10.7% |
| high_conscientiousness | 205 | 8.7% |
| assertive | 74 | 3.1% |
| agreeable | 53 | 2.3% |
| introverted | 52 | 2.2% |
| emotionally_stable | 9 | 0.4% |
| casual | 4 | 0.2% |
| open | 4 | 0.2% |
| extroverted | 2 | 0.1% |

#### interests

| Label | Count | % of points |
|-------|------:|------------:|
| career_development | 128 | 5.4% |
| pets | 106 | 4.5% |
| study_exam | 95 | 4.0% |
| sports_fitness | 84 | 3.6% |
| games | 81 | 3.4% |
| reading_writing | 40 | 1.7% |
| art_design | 32 | 1.4% |
| social | 23 | 1.0% |
| travel_outdoor | 21 | 0.9% |
| music | 19 | 0.8% |
| film_animation | 13 | 0.6% |
| programming_technology | 1 | 0.0% |

#### style

| Label | Count | % of points |
|-------|------:|------------:|
| rational | 585 | 24.9% |
| direct | 281 | 11.9% |
| detailed | 147 | 6.2% |
| brief | 120 | 5.1% |
| indirect | 59 | 2.5% |
| high_emotional_expression | 57 | 2.4% |
| colloquial | 54 | 2.3% |
| formal | 22 | 0.9% |
| humorous | 8 | 0.3% |

## 4. Length Statistics

### 4.1 Characters

| Split | Metric | History (chars) | Target Reply (chars) |
|-------|--------|----------------:|---------------------:|
| Train | Mean | 1,399 | 218 |
| Train | Median (P50) | 813 | 129 |
| Train | P90 | 3,039 | 517 |
| Train | P99 | 9,727 | 1,353 |
| Train | Max | 40,052 | 1,809 |

| Val | Mean | 1,076 | 158 |
| Val | Median (P50) | 748 | 98 |
| Val | P90 | 2,338 | 336 |
| Val | P99 | 5,241 | 847 |
| Val | Max | 14,133 | 1,754 |

### 4.2 Approximate Tokens (1.4 chars/token for Chinese)

> **Note:** System prompt adds ~500+ tokens on top of history. A 4K context budget leaves ~3.5K for history after system prompt.

| Split | Metric | History (tokens) | Target Reply (tokens) |
|-------|--------|-----------------:|----------------------:|
| Train | Mean | 999 | 156 |
| Train | Median (P50) | 581 | 92 |
| Train | P90 | 2,171 | 369 |
| Train | P99 | 6,948 | 967 |
| Train | Max | 28,609 | 1,292 |

| Val | Mean | 768 | 113 |
| Val | Median (P50) | 535 | 70 |
| Val | P90 | 1,670 | 240 |
| Val | P99 | 3,744 | 605 |
| Val | Max | 10,095 | 1,253 |

### 4.3 Context-Length Thresholds

How many prediction points exceed common context limits? (History tokens only; add ~500 tokens for system prompt.)

| Split | Threshold | History > threshold | % |
|-------|-----------|--------------------:|--:|
| Train | 1024 | 5,255 | 28.2% |
| Train | 2048 | 2,033 | 10.9% |
| Train | 4096 | 603 | 3.2% |
| Train | 8192 | 112 | 0.6% |

| Val | 1024 | 535 | 22.7% |
| Val | 2048 | 144 | 6.1% |
| Val | 4096 | 17 | 0.7% |
| Val | 8192 | 2 | 0.1% |

---
*End of report*