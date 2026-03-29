# 🌱 Crop Recommendation System (ML + Streamlit)

## 🚀 Overview

The **Crop Recommendation System** is a Machine Learning-based project that predicts the most suitable crop to grow based on various environmental and soil parameters.

It leverages multiple ML algorithms, compares their performance, and provides predictions through an interactive **Streamlit web app**.

---

## 🎯 Problem Statement

Farmers often struggle to decide which crop to grow due to varying soil conditions, weather, and nutrients.

This project aims to:

* Analyze agricultural data
* Train multiple ML models
* Recommend the best crop for given conditions

---

## 🧠 Features

* 📊 Data preprocessing & feature engineering
* 🤖 Multiple ML algorithms implemented:

  * Logistic Regression
  * Decision Tree
  * Random Forest
  * Support Vector Machine (SVM)
  * K-Nearest Neighbors (KNN)
  * XgBoost
  * CatBoost
* 📈 Model comparison & accuracy evaluation
* 🌐 Interactive UI using **Streamlit**
* ⚡ Real-time crop prediction

---

## 🏗️ Tech Stack

* **Language:** Python
* **Libraries:**

  * NumPy, Pandas
  * Scikit-learn
  * Matplotlib / Seaborn
* **Frontend/UI:** Streamlit
* **Version Control:** Git & GitHub

---

## 📂 Project Structure

crop-recommendation/
│── data/
│── models/
│── app.py
│── train.py
│── requirements.txt
│── README.md

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/abhisar2435/crop-recommendation.git
cd crop-recommendation
```

### 2️⃣ Install dependencies

```bash
pip install -r requirement.txt
```

### 3️⃣ Run the Streamlit app

```bash
streamlit run app.py
```

---

## 📊 Input Parameters

The model predicts crops based on:

* Nitrogen (N)
* Phosphorus (P)
* Potassium (K)
* Temperature
* Humidity
* pH value
* Rainfall

---

## 📈 Model Evaluation

* Models are trained and evaluated using:

  * Accuracy Score
  * Confusion Matrix
  * Cross-validation
* Best-performing model is used in the final app

---

## 🌐 Streamlit App

The project includes a user-friendly web interface where users can:

* Input soil & environmental data
* Get instant crop recommendations

---

## 🔥 Future Improvements

* 🌍 Integrate real-time weather APIs
* 📱 Deploy as a mobile app (Flutter integration)
* ☁️ Deploy model using FastAPI + cloud
* 📊 Add visualization dashboards

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork the repo and submit a pull request.

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

**Abhisar Tyagi**
B.Tech CSE (AI) | Aspiring AI Engineer 🚀

---
