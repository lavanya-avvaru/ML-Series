# ML Series — Learning Journey

A structured, hands-on Machine Learning learning series. Every day covers one concept — theory + implementation in Python.

---

## 📁 EDA, Preprocessing & Logistic Regression

> Before jumping into classification, I wanted the data to be actually ready.  
> This was a full end-to-end preprocessing session on the Titanic dataset.

---

### 🗂️ Dataset

| Property | Value |
|----------|-------|
| Rows | 891 |
| Columns | 12 |
| Target | `Survived` (0 or 1) |
| Features | Mix of numerical and categorical |

---

### 🔍 EDA

- Explored data using `shape`, `info()`, `describe()`
- Found missing values — `Age` (177), `Cabin` (687), `Embarked` (2)
- Visualized Age distribution using histogram
- Detected outliers using boxplot
- Analyzed feature correlations using heatmap

---

### 🧹 Missing Value Treatment

| Column | Missing | Strategy |
|--------|---------|----------|
| `Age` | 177 | Filled with **median** — outliers present |
| `Embarked` | 2 | Filled with **mode** — categorical column |
| `Cabin` | 687 (77%) | **Dropped** — too much missing data |

---

### 📦 Outlier Treatment

- Method used — **IQR (Interquartile Range)**
- Column — `Age`

```
Lower limit : 2.5
Upper limit : 54.5
Outliers    : 66 rows
```

> Capped instead of dropping — no data loss

---

### ⚖️ Feature Scaling

- Applied `StandardScaler` on `Age` and `Fare`
- Brings both features to the same range so neither dominates the model

---

### 🔤 Encoding

| Column | Method | Reason |
|--------|--------|--------|
| `Sex` | Label Encoding | Binary column — only 2 values |
| `Embarked` | One Hot Encoding | 3 categories, no order |
| `Name`, `Ticket` | Dropped | No predictive value |

---

### 🎯 Feature Selection

Correlation with `Survived`:

| Feature | Correlation | Decision |
|---------|------------|----------|
| `Sex` | -0.54 | ✅ Keep |
| `Pclass` | -0.34 | ✅ Keep |
| `Fare` | +0.26 | ✅ Keep |
| `Embarked_S` | -0.15 | ✅ Keep |
| `SibSp` | -0.03 | ❌ Drop |
| `Embarked_Q` | +0.003 | ❌ Drop |
| `Parch` | +0.08 | ❌ Drop |

---

### 🤖 Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
```

| Metric | Score |
|--------|-------|
| Train/Test Split | 80 / 20 |
| Accuracy | 79.88% |
| F1 Score | 0.75 |
| Precision | 0.76 |
| Recall | 0.74 |

**Confusion Matrix**

```
[[88  17]
 [19  55]]
```


*Documenting every step of this journey as I go.*
