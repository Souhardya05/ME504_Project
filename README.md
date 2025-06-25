
# 📊 Maximum von Mises Stress Prediction Using Neural Networks

This project develops a deep learning model to predict the **maximum von Mises stress** in a material system given its **volume fraction distribution**. It further explores a dimensionality reduction approach using **Principal Component Analysis (PCA)** to improve efficiency without sacrificing accuracy.

## 👨‍🏫 Guide
**Dr. Manish Agarwal**  
Project Duration: **Apr 2025 – May 2025**

## 📁 Dataset
- `volume fraction/prop_*.dat`: Contains the input material volume fraction distributions.
- `stress_node/stress_*.txt`: Contains the stress tensor components: `tau_xx`, `tau_yy`, and `tau_xy`.

Each sample corresponds to a pair of files with the same index (0–9999). The objective is to predict the **maximum von Mises stress** from the corresponding stress field.

---

## 🧠 Objective

1. Build a regression model using a **Neural Network** to predict max von Mises stress from volume fraction data.
2. Implement **PCA** to reduce the dimensionality of the input data.
3. Evaluate performance using metrics such as **Mean Absolute Percentage Error (MAPE)** and **R² Score**.

---

## 🛠 Technologies Used

- Python 3
- NumPy, Matplotlib, scikit-learn
- TensorFlow / Keras
- PCA for Dimensionality Reduction

---

## 🧾 Methodology

### 🔹 Step 1: Data Loading
- Load volume fraction and stress data from 10,000 samples.
- Compute the **maximum von Mises stress** using:
  
  \[
  \sigma_{vm} = \sqrt{\tau_{xx}^2 + \tau_{yy}^2 - \tau_{xx} \tau_{yy} + 3 \tau_{xy}^2}
  \]

### 🔹 Step 2: Preprocessing
- Normalize stress values using `StandardScaler`.
- Split the dataset into training and testing sets.

### 🔹 Step 3: Model Building
- A **Keras Sequential Model** is defined with fully connected dense layers.
- Uses `mean_squared_error` as the loss function and Adam optimizer.

### 🔹 Step 4: Dimensionality Reduction (PCA)
- Apply PCA to reduce the number of features while preserving variance.
- Train the same model using PCA-reduced data to compare results.

---

## 📈 Evaluation Metrics

- **R² Score** (Custom metric implemented in TensorFlow)
- **Mean Absolute Percentage Error (MAPE)**

---

## 📌 Example Code Snippets

```python
# von Mises stress calculation
vm_stress = np.sqrt(tau_xx**2 + tau_yy**2 - tau_xx*tau_yy + 3*tau_xy**2)
```

```python
# PCA transformation
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
```

---

## 📂 Folder Structure

```
.
├── stress_node/
│   └── stress_0.txt ... stress_9999.txt
├── volume fraction/
│   └── prop_0.dat ... prop_9999.dat
├── Project.ipynb
└── README.md
```

---

## 📬 Future Work

- Use convolutional neural networks (CNNs) for spatial feature extraction.
- Integrate real-time prediction interface with UI.
- Study sensitivity of predictions to PCA dimensions.

---

## 🔗 GitHub Repository

[Click here to view the code](https://github.com/Souhardya05/ME504_Project/tree/main)
