# OULAD preprocessing

## Build the students dataset

To generate the "Who" dataset, place the raw OULAD files below in `data/raw/`:

- `studentInfo.csv`
- `studentRegistration.csv`

Then run:

```bash
.venv/bin/python source/preprocess_students.py
```

The script performs an inner join on the full enrollment key
`code_module`, `code_presentation`, and `id_student`, drops `id_student` from
the result, and writes the output to `data/pre_processed/students.csv`.
Joining only on `id_student` can incorrectly expand the dataset when a student
appears in multiple module presentations.

## Conclusion consistency tests (real vs synthetic)

`main.py` includes direct inferential comparison tests to answer questions like:
"Does gender affect course success in the same way on real and synthetic data?"

It now runs a **Fisher exact test** for **all valid predictor/outcome pairs** where:
- predictor has exactly two levels in the real data
- outcome is numeric, binary categorical, or a categorical column with explicit positive labels

and compares real vs synthetic conclusions using:
- same effect sign
- same significance decision
- effect size (log-odds-ratio) within a relative tolerance

You can still provide positive labels for a specific outcome (for example, `final_result`):

```bash
.venv/bin/python main.py \
  --real data/pre_processed/students.csv \
  --synth data/post_processed/students.csv \
  --conclusion-outcome final_result \
  --conclusion-positive-outcome Pass \
  --conclusion-positive-outcome Distinction
```
