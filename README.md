<div align="center">
  👋 <b>Welcome!</b> 
  <br>
  This is the official repository for the <b>FibroViT</b> project.
</div>

<p align="center">
  <a href="YOUR_STREAMLIT_APP_LINK_HERE" target="_blank">
    <img src="https://i.imgur.com/your-app-demo.gif" alt="App Demo GIF" width="85%">
  </a>
</p>

# 🩺 AI-Powered Pulmonary Fibrosis Detection using ViT

<p align="center">
<a href="https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2023.1282200/full" target="_blank">
    <img src="https://img.shields.io/badge/Read_The_Paper-00A1DE?style=for-the-badge&logo=read-the-docs&logoColor=white"
         alt="Read The Paper" />
</a>
<a href="YOUR_STREAMLIT_APP_LINK_HERE" target="_blank">
    <img src="https://img.shields.io/badge/🚀_Live_Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"
         alt="Launch Streamlit App" />
</a>
<a href="YOUR_GITHUB_REPO_LINK_HERE" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-Repo-181717?style=for-the-badge&logo=github&logoColor=white"
         alt="GitHub Repo" />
</a>
</p>

---

This repository contains the complete project for our research paper: **"FibroVit—Vision transformer-based framework for detection and classification of pulmonary fibrosis from chest CT images"**. It includes the model training process, the final pretrained model, and an interactive web application for live inference.

## 📂 Repository Contents

This project is organized to provide full transparency and reproducibility of the research.

| File / Folder | Description |
| :--- | :--- |
| 📄 **`app.py`** | The main Python script for the interactive Streamlit web application. |
| 📓 **`FibroViT Model Training.ipynb`** | A detailed Jupyter Notebook covering the entire process of data preprocessing, model training, and evaluation. |
| 📁 **`FibroViT Pretrained Model/`** | Contains the final, trained Vision Transformer model files, ready for inference. |
| 📄 **`requirements.txt`** | A list of all Python libraries required to run the project. |
| 📄 **`LICENSE`** | The open-source license for this project. |
| 📄 **`README.md`** | You are here! |

---

## 🛠️ Technology Stack

The project is built using modern tools for deep learning and data science.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face">
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter">
</p>

---

## 🚀 How to Run the App Locally

Follow these steps to get the Streamlit application running on your own machine.

### 1. Prerequisites
-   [Git](https://git-scm.com/downloads) and [Git LFS](https://git-lfs.github.com/) must be installed.
-   [Python 3.8+](https://www.python.org/downloads/) is required.

### 2. Setup and Installation

```bash
# Clone this repository to your local machine
git clone [https://github.com/waseemkathia/FibroVit.git](https://github.com/waseemkathia/FibroVit.git)
cd FibroVit

# Set up a Python virtual environment (highly recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all the required Python packages
pip install -r requirements.txt
```

### 3. Launch the App

With the dependencies installed, you can now run the Streamlit app with a single command:

```bash
streamlit run app.py
```
Your web browser should automatically open to the application, ready for use!

---

## 📜 Citation

If you use this code or our model in your work, we kindly ask that you cite the original research paper. It helps validate and support our contribution to the scientific community.

```bibtex
@article{sabir2023fibrovit,
  author={Sabir, Muhammad Waseem and Farhan, Muhammad and Almalki, Nabil Sharaf and Alnfiai, Mrim M. and Sampedro, Gabriel Avelino},
  title={FibroVit—Vision transformer-based framework for detection and classification of pulmonary fibrosis from chest CT images},
  journal={Frontiers in Medicine},
  volume={10},
  year={2023},
  url={[https://www.frontiersin.org/articles/10.3389/fmed.2023.1282200](https://www.frontiersin.org/articles/10.3389/fmed.2023.1282200)},
  doi={10.3389/fmed.2023.1282200},
  issn={2296-858X}
}
```

---

## 👨‍💻 Connect with Me

I am passionate about leveraging AI to solve real-world problems in healthcare. Please feel free to connect, ask questions, or discuss potential collaborations.

<p align="center">
  <a href="https://www.linkedin.com/in/waseemkathia/" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
  </a>
  &nbsp;
  <a href="https://github.com/waseemkathia" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
  </a>
  &nbsp;
  <a href="https://waseemkathia.github.io/" target="_blank">
    <img src="https://img.shields.io/badge/Website-4A90E2?style=for-the-badge&logo=blogger&logoColor=white" alt="Website">
  </a>
</p>
