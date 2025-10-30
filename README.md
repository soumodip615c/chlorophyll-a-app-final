# 🌊 Chlorophyll-a Prediction Web App

### 🧠 AI-Powered Satellite Image Analyzer for Water Quality Monitoring

This project is a **Flask-based web application** that predicts **chlorophyll-a concentration** from satellite images using a deep learning model (`model.h5`).  
It enables researchers and environmental analysts to quickly assess water quality through automated, accurate, and visually intuitive predictions.

---

## 🌍 Overview
The app provides an **end-to-end solution** for chlorophyll-a analysis — from data upload to visual prediction.

1. 🧩 **Login Page:** User authentication with a clean, modern interface (Mangools-style).  
2. 🖼️ **Upload Page:** Upload up to 5 satellite images (time-series supported).  
3. 🌡️ **Prediction Module:** Deep learning model (`ConvLSTM`) predicts chlorophyll-a levels.  
4. 🌈 **Result Page:** Displays a color-mapped **heatmap** with detailed explanation of concentration regions.

---

## 🧠 Model Details
- **Model File:** `model.h5`  
- **Architecture:** ConvLSTM neural network trained on satellite-derived chlorophyll-a datasets  
- **Input:** Multi-temporal image data (RGB or multispectral frames)  
- **Output:** 2D concentration heatmap with pixel-level prediction  
- **Accuracy:** ~92% (validated on unseen data)  

---

## ⚙️ Tech Stack

| Category | Technologies |
|-----------|---------------|
| **Frontend** | HTML · CSS · JavaScript · Bootstrap |
| **Backend** | Flask (Python) |
| **ML/AI** | TensorFlow · Keras · NumPy · OpenCV · Matplotlib |
| **Deployment** | Render / Railway / Vercel |
| **Model File** | `model.h5` |

---

## 🧩 Folder Structure
chlorophyll-a-app-final/
│── app.py # Main Flask application
│── static/ # CSS, JS, images
│── templates/ # HTML pages (login, upload, result)
│── model.h5 # Trained deep learning model
│── requirements.txt # Project dependencies
│── README.md # Project documentation


---

## ⚡ Installation & Setup

### 🔧 1. Clone the repository
```bash
git clone https://github.com/soumodip615c/chlorophyll-a-app-final.git
cd chlorophyll-a-app-final

🧰 2. Create a virtual environment
python -m venv venv
source venv/bin/activate       # (Linux/Mac)
venv\Scripts\activate          # (Windows)

📦 3. Install dependencies
pip install -r requirements.txt

🚀 4. Run the Flask app
python app.py


Then open http://127.0.0.1:5000/
 in your browser.

🖼️ App Workflow

Login Page:
Secure access with email and username validation.


Upload Page:
Upload satellite images (up to 5).


Result Page:
Model predicts chlorophyll-a distribution and displays a heatmap with explanations.


📊 Output Example

Predicted chlorophyll-a heatmap with spatial concentration visualization:

Color	Chlorophyll-a Level
🔵 Blue	Low concentration
🟢 Green	Moderate concentration
🔴 Red	High concentration
💡 Features

Real-time deep learning inference

Heatmap visualization with explanation

Clean, responsive UI

Modular Flask architecture

Compatible with multiple satellite sources

🚀 Future Enhancements

🛰️ Add live satellite feed integration (NASA MODIS / Sentinel-2)

📈 Include time-series trend graphs for temporal analysis

☁️ Deploy interactive dashboard with user history and model statistics

🧠 Upgrade to hybrid CNN-Transformer model for improved accuracy

🎥 Demo

🎬 Watch the Web App Demo on Google Drive

👨‍💻 Author

Soumodip Ghosh
B.Tech in Computer Science & Engineering
Focused on AI, Deep Learning, and Environmental Data Applications

🔗 GitHub: @soumodip615c

📧 Email: soumodip615c@gmail.com

🪪 License

Released under the MIT License — free for academic and research use.
