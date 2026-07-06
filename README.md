<div align="center">
  <img src="https://raw.githubusercontent.com/ojash08/Exofind/main/exofind_dash/public/icons.svg" alt="ExoFind Logo" width="120" />

  <h1>🌌 ExoFind: Automated Exoplanet Detection Pipeline</h1>
  
  <p>
    <b>A high-performance pipeline and interactive dashboard for discovering exoplanet transit signatures from light curves.</b>
  </p>

  <p>
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB" alt="React" />
    <img src="https://img.shields.io/badge/Vite-B73BFE?style=for-the-badge&logo=vite&logoColor=FFD62E" alt="Vite" />
    <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  </p>
</div>

<hr />

## 🚀 Overview

**ExoFind** is an end-to-end astronomical pipeline designed to ingest Target Pixel Files (TPF) and Light Curves, process photometry data, and detect potential exoplanetary transits using an optimized Box-fitting Least Squares (BLS) algorithm. 

This repository contains both the **Core Pipeline Engine** and the **Interactive Web Dashboard** which allows for visual vetting, anomaly detection, and data export.

### ✨ Key Features

- **🌠 Core Photometry & Cleaning:** Automated masking, detrending, and outlier removal to clean stellar light curves.
- **🔍 Advanced BLS Search:** Utilizes a highly optimized BLS search algorithm to identify periodic transit dips.
- **📊 Real-time Dashboard:** A gorgeous, dark-themed UI for visualizing light curves, folded phase plots, and periodograms instantly.
- **🧠 LLM Confidence Vetting:** Integrated AI/LLM project that evaluates candidate features to output a definitive Confidence Score.
- **💾 Easy Export:** Export detected planetary candidates and raw data to CSV or JSON directly from the UI.

<br />

<div align="center">
  <img src="https://raw.githubusercontent.com/ojash08/Exofind/main/exofind_dash/public/exofind_bls_v2_results.png" alt="Dashboard Preview" width="800" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.5);" />
  <p><i>The ExoFind Interactive Candidate Dashboard</i></p>
</div>

<br />

## 🛠️ Architecture

The project is split into two main modules:

1. `exofind/`: The heavy-lifting Python backend. Handles the astronomy pipeline (Astropy, Lightkurve), the BLS optimization algorithms, and the validation stages.
2. `exofind_dash/`: The React + Vite frontend and FastAPI backend server for the web interface.

<br />

## 👨‍💻 Contributors

This project was built with ❤️ for space exploration.

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/ojash08">
        <img src="https://github.com/ojash08.png" width="100px;" alt="Ojash Bhatnagar" style="border-radius:50%"/><br />
        <b>Ojash Bhatnagar</b>
      </a>
      <br />
      <i>Core Pipeline, Backend Architecture & Interactive Dashboard UI</i>
    </td>
    <td align="center">
      <!-- REPLACE THE LINK AND IMAGE WITH YOUR TEAMMATE'S GITHUB -->
      <a href="https://github.com/teammate">
        <img src="https://github.com/github.png" width="100px;" alt="Teammate Name" style="border-radius:50%"/><br />
        <b>[Teammate Name]</b>
      </a>
      <br />
      <i>LLM-based Confidence Score & Vetting Automation</i>
    </td>
  </tr>
</table>

<br />

## ⚙️ Quick Start

### 1. Start the Backend Server
```bash
cd exofind_dash/backend
pip install -r requirements.txt
python server.py
