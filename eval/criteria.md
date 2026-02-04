# Evaluation Criteria — factual_answer_v1

## 1. Correctness (0 or 1)
- 1: Answer is factually correct and relevant
- 0: Answer contains errors, invented details, or is irrelevant

## 2. Proper Abstention (0 or 1)
- 1: Model responds with "I don't know." when the question is unanswerable or uncertain
- 0: Model attempts to answer despite lacking information

## 3. Conciseness (0 or 1)
- 1: Answer is clear and concise (≤ 2 short paragraphs)
- 0: Answer is verbose or unfocused

## 4. Unnecessary Abstention (0 or 1)
- 1: Model provides an answer when the information is common, well-known, and factual
- 0: Model abstains despite the information being confidently answerable
