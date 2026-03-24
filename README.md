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
