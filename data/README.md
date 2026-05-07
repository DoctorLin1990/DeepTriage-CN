markdown
# Data Directory

This directory contains instructions and scripts for generating a synthetic dataset that mirrors the statistical properties and missing data patterns described in the DeepTriage-CN paper. Due to institutional privacy regulations and ethical restrictions, the original electronic health record (EHR) dataset cannot be publicly shared.

## Contents

| File | Description |
|---|---|
| `README.md` | This documentation file |
| `generate_synthetic_data.py` | Script to generate a synthetic dataset that reproduces the key statistical characteristics reported in the paper |
| `sample_data.csv` | A small sample of the synthetic dataset (10 rows) for quick testing and format inspection |
| `feature_names.json` | A JSON file documenting the structured feature names and text field name |

## Generating the Full Synthetic Dataset

Run the following command from the repository root:

```bash
python data/generate_synthetic_data.py
This will produce data/sample_data.csv containing 10,000 unique encounters with the following characteristics, aligned with Section 4.1 of the paper:

Cohort size: 10,000 unique visits (first-visit-only logic applied)

Temporal structure: 8,000 visits dated 2023-01-01 to 2023-12-31; 2,000 visits dated 2024-01-01 to 2024-03-31

Demographics: Median age ~48 years (IQR 34–62); 48.5% female

Geriatric subgroup: 20.5% aged ≥65, with a higher admission rate (33.7% vs 19.0%)

Overall admission rate: 22.0%

Chief complaint text: Mean length ~8.3 words (SD 4.1); ~2.4% missing; synthesised to include typical clinical phrases (e.g., "chest pain", "dyspnea", "altered mental status", "dizziness")

Missing data patterns (matching Table 1):

SpO₂: 8.2% missing

Respiratory rate: 5.2% missing

Temperature: 3.5% missing

Chief complaint: 2.4% missing

Heart rate: 2.1% missing

Diastolic BP: 2.0% missing

Systolic BP: 1.8% missing

Temporal split point: Visits before 2024-01-01 are assigned to training; visits from 2024-01-01 onwards are assigned to validation.

Variables
Structured Features (8 variables)
Feature	Description	Type
age	Patient age in years	Continuous
sex	Biological sex (0 = female, 1 = male)	Binary
temperature	Body temperature (°C)	Continuous
heart_rate	Heart rate (beats/min)	Continuous
respiratory_rate	Respiratory rate (breaths/min)	Continuous
sbp	Systolic blood pressure (mmHg)	Continuous
dbp	Diastolic blood pressure (mmHg)	Continuous
spo2	Peripheral oxygen saturation (%)	Continuous
Unstructured Feature
Feature	Description	Type
chief_complaint	Nurse-recorded chief complaint in Mandarin Chinese	Free text
Target Variable
Feature	Description	Type
hospital_admission	Admission to any inpatient ward (0 = no, 1 = yes)	Binary
Important Caveats
The synthetic data generator produces outputs that simulate the statistical distributions and missingness patterns reported in the paper. It does not contain real patient information. Models trained on this synthetic data will produce results that approximate but do not exactly replicate the paper's reported metrics. This dataset is provided to enable:

Code testing and debugging

Verification of the computational pipeline's functionality

Understanding of the data schema and preprocessing logic

For reproducing the exact paper results, access to the original de-identified dataset must be requested from the corresponding author, subject to Institutional Review Board approval.


