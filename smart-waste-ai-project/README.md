# AI-Powered Smart Waste Segregation and Recycling Management System

**Student:** Shravani Kadam | **Roll Number:** 25

An academic software project demonstrating ten prescribed AI practicals themed
around smart waste management, food-waste prediction, and recycling.

---

## 1. Project Overview

This is a **single unified application** built with Streamlit. All ten AI
practicals and the core waste-management demonstration are integrated into one
interface with a single navigation menu. All datasets are synthetic and clearly
labelled. No physical hardware or image recognition is included.

## 2. What Each Practical Does

| # | Practical | File |
|---|-----------|------|
| 1 | PEAS, Problem Formulation, Food-Waste Prediction | `notebooks/practical_01_food_waste_prediction.ipynb` |
| 2 | Prolog Facts, Rules and Queries | `prolog/practical_02_family_and_waste_rules.pl` |
| 3 | BFS and DFS for a Rescue Robot | `src/practical_03_bfs_dfs.py` |
| 4 | A* Informed Search | `src/practical_04_astar.py` |
| 5 | Minimax Territory Control Game | `src/practical_05_minimax.py` |
| 6 | Fuzzy Logic Room Temperature Controller | `src/practical_06_temperature_fuzzy.py` |
| 7 | Mamdani and Sugeno Washing-Machine Controller | `src/practical_07_washing_machine_fuzzy.py` |
| 8 | Hybrid Neuro-Fuzzy System (ANFIS) | `src/practical_08_anfis.py` |
| 9 | Supervised and Unsupervised Neural Learning | `src/practical_09_neural_learning.py` |
| 10 | Micro and Small Language Models for Parking Assistant | `src/practical_10_language_models.py` |

## 3. Folder Structure

```
smart-waste-ai-project/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py                      # Streamlit interface
├── data/                       # Sample CSV datasets
├── notebooks/
│   └── practical_01_food_waste_prediction.ipynb
├── src/
│   ├── waste_management/
│   │   ├── food_waste_prediction.py
│   │   └── waste_rules.py
│   ├── practical_03_bfs_dfs.py
│   ├── practical_04_astar.py
│   ├── practical_05_minimax.py
│   ├── practical_06_temperature_fuzzy.py
│   ├── practical_07_washing_machine_fuzzy.py
│   ├── practical_08_anfis.py
│   ├── practical_09_neural_learning.py
│   └── practical_10_language_models.py
├── prolog/
│   └── practical_02_family_and_waste_rules.pl
└── tests/
    └── test_core_algorithms.py
```

## 4. Required Software

- Python 3.11 or compatible
- pip
- SWI-Prolog (for Practical 2)
- Jupyter (for the notebook)

## 5. Installation on Windows

```powershell
# Clone or extract the project folder, then:
cd smart-waste-ai-project
```

## 6. Create and Activate a Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 7. Install Dependencies

```powershell
pip install -r requirements.txt
```

## 8. Launch the Central Interface

```powershell
streamlit run app.py
```

## 9. Run Each Practical Individually

```powershell
# From the smart-waste-ai-project folder:
python src\practical_03_bfs_dfs.py
python src\practical_04_astar.py
python src\practical_05_minimax.py
python src\practical_06_temperature_fuzzy.py
python src\practical_07_washing_machine_fuzzy.py
python src\practical_08_anfis.py
python src\practical_09_neural_learning.py
python src\practical_10_language_models.py
```

## 10. Open and Run the Jupyter Notebook

```powershell
jupyter notebook notebooks\practical_01_food_waste_prediction.ipynb
```

## 11. Execute the Prolog File

Install SWI-Prolog from https://www.swi-prolog.org, then:

```powershell
swipl -s prolog\practical_02_family_and_waste_rules.pl
```

At the `?-` prompt, try:

```prolog
?- father(john, mary).
?- sibling(mary, tom).
?- grandparent(john, sue).
?- categorize(plastic_bottle, dry).
?- categorize(newspaper, residual).
?- needs_collection(wet_bin).
```

## 12. Example Inputs and Expected Output

**Food-waste prediction:** 150 people, 3 meals, weekday, 20 kg history -> predicted value in kg (varies by model).

**Waste segregation:** "paper" + soiled -> residual; "plastic bottle" -> dry.

**BFS/DFS:** Both find a path on the default grid; BFS finds the shortest.

**A*:** Returns optimal path and cost on the default grid.

**Minimax:** AI plays X vs greedy O on a 5x5 board; final score displayed.

## 13. Known Limitations and Troubleshooting

- All data is synthetic; results are illustrative only.
- Streamlit may not work in restricted environments; use `python app.py` as a CLI fallback.
- If `streamlit` is not on PATH, use `python -m streamlit run app.py`.
- Prolog requires SWI-Prolog installed separately.
- Python 3.14 may have compatibility issues with some packages; Python 3.11 is recommended.

---

## Requirement-to-File Checklist

| Practical | Implementation File | Status |
|-----------|-------------------|--------|
| 1 | `notebooks/practical_01_food_waste_prediction.ipynb` | Implemented |
| 2 | `prolog/practical_02_family_and_waste_rules.pl` | Implemented |
| 3 | `src/practical_03_bfs_dfs.py` | Implemented |
| 4 | `src/practical_04_astar.py` | Implemented |
| 5 | `src/practical_05_minimax.py` | Implemented |
| 6 | `src/practical_06_temperature_fuzzy.py` | Implemented |
| 7 | `src/practical_07_washing_machine_fuzzy.py` | Implemented |
| 8 | `src/practical_08_anfis.py` | Implemented |
| 9 | `src/practical_09_neural_learning.py` | Implemented |
| 10 | `src/practical_10_language_models.py` | Implemented |
