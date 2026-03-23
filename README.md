# OULAD preprocessing

## Build the students dataset

To generate the "Who" dataset, place the raw OULAD files below in `data/raw/`:

- `studentInfo.csv`
- `studentRegistration.csv`

Then run:

```bash
.venv/bin/python source/preprocess_students.py
```

The script performs an inner join on `id_student`, drops `id_student` from the
result, and writes the output to `data/pre_processed/students.csv`.
