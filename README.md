# Broken Biscuit Detection using Classical Image Processing

##  Project Overview

This project was developed for the EE4216 Image Processing and Computer Vision assignment.

The objective of this project is to detect whether the biscuits captured using a cameracare broken or intact using classical image processing techniques without using Machine Learning or Deep Learning methods.

The system processes biscuit images captured on a white A4 sheet and performs:
- Biscuit detection
- Contour extraction
- Shape analysis
- Classification of biscuits as broken or intact
- Output image annotation

---

# 📂 Dataset Preparation

The dataset was manually created by capturing images of:
- Circular biscuits
- Square biscuits

Each image contains:
- Intact biscuits
- Broken biscuits

Images were captured:
- Using a smartphone camera
- From a top-down view
- Under good lighting conditions
- On a white A4 sheet background

---

# 🛠 Technologies and Libraries Used

- Python
- OpenCV
- NumPy
- Matplotlib
- Jupyter Notebook
- Visual Studio Code

---

# ⚙️ Image Processing Techniques Used

The following classical image processing techniques were used:

1. Grayscale Conversion
2. Thresholding
3. Morphological Operations
4. Contour Detection
5. Area-based Classification
6. Image Annotation

---

# 📁 Project Folder Structure

```bash
biscuit-detection/
│
├── images/
│
├── notebook/
│   └── ass2_v1.ipynb
│
├── src/
│   └── ass2.py
│
└── README.md
```

---

# ▶️ How to Run the Project

## Step 1 — Install Required Libraries

```bash
pip install opencv-python numpy matplotlib
```

## Step 2 — Open Project in VS Code

Open the project folder in Visual Studio Code.

---

## Step 3 — Run the Program

```bash
python src/main.py
```

---

# 🧠 Detection Logic

The program identifies biscuits using contour detection.

- Large contour area → Intact Biscuit
- Small/Irregular contour area → Broken Biscuit

Bounding boxes and labels are drawn around detected biscuits.

---

# ✅ Conclusion

This project demonstrates how classical image processing techniques can be used for industrial quality inspection tasks without using machine learning algorithms.

The developed system successfully detects and classifies biscuits as broken or intact using OpenCV-based image processing operations.

---

# 👨‍💻 Author

Sada  
Faculty of Engineering and Technology  
CINEC Campus
