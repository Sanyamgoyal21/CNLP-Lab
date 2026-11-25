# Experiment 1 – Employee Data Analysis and Visualization

**Aim:** Load a structured dataset with **pandas**, answer analytical questions with grouping, sorting and filtering, and visualise the results with **matplotlib**. This is the data-handling foundation for the later NLP experiments.

---

## Tasks

| No. | Task |
|---|---|
| 1.1 | Average salary of each department (bar chart) |
| 1.2 | Number of employees in each department (bar chart) |
| 1.3 | Percentage of male and female employees (pie chart) |
| 1.4 | Salary distribution (histogram) |
| 1.5 | Relationship between experience and salary (scatter plot) |
| 1.6 | Top 10 employees with the highest salary |
| 1.7 | Highest salary in every department |
| 1.8 | Employees earning more than the overall average salary |
| 1.9 | Average years of experience in each department |
| 1.10 | Age distribution (histogram) |

---

## Dataset

| Property | Value |
|---|---|
| File | `employee_information_100.csv` |
| Rows | 100 employees |
| Columns | `Employee_ID`, `Name`, `Department`, `Age`, `Gender`, `Salary`, `Experience_Years` |
| Departments | Finance, HR, IT, Marketing, Operations, Sales |

Sample rows:

| Employee_ID | Name | Department | Age | Gender | Salary | Experience_Years |
|---|---|---|---|---|---|---|
| E001 | Diana1 | HR | 39 | Male | 58000 | 8 |
| E002 | Diana2 | Operations | 56 | Male | 105000 | 27 |

---

## Implementation Steps

1. **Load the data.** `pd.read_csv()` reads the CSV into a DataFrame, and `df.head()` checks the columns.
2. **Aggregate by group.** `df.groupby("Department")["Salary"].mean()`, `.max()` and `["Experience_Years"].mean()` compute per-department statistics (tasks 1.1, 1.7, 1.9).
3. **Count categories.** `value_counts()` on `Department` and `Gender` (tasks 1.2, 1.3).
4. **Sort and filter.** `sort_values("Salary", ascending=False).head(10)` for the top earners (1.6). A boolean mask `df[df["Salary"] > df["Salary"].mean()]` finds above-average earners (1.8).
5. **Visualise.** Bar charts, a pie chart (`autopct="%1.1f%%"`), histograms (`bins=10`) and a scatter plot, each with a title and axis labels.

No text preprocessing is needed because the data is already structured and clean.

---

## How to Run

1. Open `CNLP_Experiment_1.ipynb` in Google Colab or Jupyter.
2. Put `employee_information_100.csv` next to the notebook (or in Google Drive).
3. Change the path in the second cell if needed. The notebook currently reads
   `/content/drive/MyDrive/CNLP-Lab/Experiment-1/employee_information_100.csv`, so in Colab mount Drive first:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. **Runtime → Run all.**

Requirements: `pandas`, `matplotlib`.

---

## Results

**Average salary and experience per department (1.1, 1.7, 1.9):**

| Department | Employees | Average salary | Highest salary | Avg. experience (years) |
|---|---|---|---|---|
| Finance | 12 | 82,416.67 | 117,000 | 11.17 |
| HR | 21 | 72,095.24 | 113,000 | 19.81 |
| IT | 18 | 73,611.11 | 117,000 | 18.39 |
| Marketing | 20 | 76,100.00 | 119,000 | 21.90 |
| Operations | 16 | 90,875.00 | 117,000 | 16.88 |
| Sales | 13 | 74,769.23 | 114,000 | 23.85 |

- **Gender split (1.3):** 64 male (64%), 36 female (36%).
- **Overall average salary (1.8):** 77,760. The notebook lists every employee above it.
- **Highest salary (1.6):** 119,000 (Marketing: E088 and E086).

---

## Observations

- **Operations** has the highest average salary (90,875), even though **HR** has the most employees (21).
- **Sales** has the highest average experience (23.85 years) but only the 4th highest average salary. Across all employees, experience and salary are essentially uncorrelated (Pearson r = 0.03), which matches the scattered points in the 1.5 plot.
- The dataset is synthetic. Some rows are impossible, for example E003 is 27 years old with 32 years of experience. Real data would need validation checks such as `Experience_Years < Age - 18`.
- The maximum salary is almost the same in every department (113k–119k), so the differences come from the averages, not the top earners.

---

## Files

```
Experiment-1/
├── CNLP_Experiment_1.ipynb          # notebook with all 10 tasks and charts
├── employee_information_100.csv     # dataset
└── Experiment-1 - Google Docs.pdf   # problem statement
```
