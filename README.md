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

## Conclusion consistency test (real vs synthetic)

`main.py` now includes a direct inferential comparison test to answer questions like:
"Does gender affect course success in the same way on real and synthetic data?"

By default, it runs a **Fisher exact test** for:
- predictor: `gender`
- outcome: `final_result`, collapsed to a positive class of `Pass` + `Distinction`

and compares real vs synthetic conclusions using:
- same effect sign
- same significance decision
- effect size (log-odds-ratio) within a relative tolerance

Example:

```bash
.venv/bin/python main.py \
  --real data/pre_processed/students.csv \
  --synth data/post_processed/students.csv \
  --conclusion-predictor gender \
  --conclusion-outcome final_result \
  --conclusion-positive-outcome Pass \
  --conclusion-positive-outcome Distinction
```
