# ♻️ Smart Waste Classification using AI (YOLOv8-L)

This project focuses on building an **AI-based waste classification system** that identifies waste categories such as **paper, plastic, metal, and biological waste** using **image recognition**.
It then recommends the **correct bin color** — **Green** for biodegradable, **Blue** for recyclable, and **Red** for hazardous waste — to reduce human error and improve recycling efficiency.

### 🧠 Model Overview

* Built using the **YOLOv8-L (Large)** classification model, known for its **speed and real-time accuracy**.
* Trained on a **custom dataset** containing **100 images from each class** (total **1000 images**).
* Training took **over 12 hours** to achieve optimal performance.
* The **best performing weights** were saved in the `'best.pt'` file for deployment.

### ⚙️ Data Preprocessing

* All images were resized to **224×224 pixels** for uniform input.
* Applied **data augmentation** techniques such as **rotation, flipping, brightness adjustment, and zooming**.
* These steps improved the model’s **accuracy and ability to handle image variations**.

### 💻 Web Application

A **user-friendly Streamlit web app** was developed to make the system accessible to everyone.

* Users can **upload waste images**.
* The app displays the **predicted waste type**, **confidence score**, and **suggested bin color**.
* Designed to be **clean, responsive, and easy to use** for a professional look and smooth operation.

### 📊 Evaluation Metrics

| Metric    | Value |
| :-------- | :---: |
| Precision |  0.875 |
| Recall    |  0.778 |
| F1 Score  |  0.824 |

✅ *An F1 score of 0.82 indicates strong model performance with balanced precision and recall.*

### 🌍 Impact

This system contributes to **smart city initiatives** by automating waste sorting, **reducing human error**, and **enhancing recycling efficiency**, making waste management **smarter and more sustainable**.

---
