# DataGPT — Your Dedicated Analytics Assistant

An AI-powered data analytics application built with Python and Streamlit that helps users explore datasets, generate visualizations, detect anomalies, clean data, and get intelligent insights using Google's Gemini AI.

## 🚀 Project Overview

DataGPT is designed to make data analysis easier, faster, and more accessible. Users can upload datasets and perform different analytical operations through an interactive dashboard without writing complex code.

The application combines traditional data analysis techniques with generative AI to provide meaningful insights and support data-driven decision-making.

## ✨ Features

- **User Authentication:** Secure login and signup system.
- **Interactive Dashboard:** View dataset statistics, records, columns, and missing values.
- **Data Explorer:** Search, inspect, and understand uploaded datasets.
- **Data Visualization:** Create interactive charts and graphs using Plotly.
- **AI Assistant:** Ask questions about your data using Google Gemini AI.
- **Anomaly Detection:** Identify unusual patterns using Isolation Forest.
- **Data Cleaning:** Handle missing values and duplicate records.
- **Advanced AI Analysis:** Generate business insights, trend analysis, and data quality reports.
- **Report Center:** Generate downloadable PDF and Excel reports.
- **Dataset History:** Access previously saved datasets and analysis history.
- **Dark & Light Themes:** Switch between themes for a better user experience.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Plotly | Interactive visualizations |
| Google Gemini API | AI-powered data analysis |
| Scikit-learn | Machine learning and anomaly detection |
| SQLite | User authentication and history storage |
| ReportLab | PDF report generation |
| XlsxWriter | Excel report generation |

## 📁 Project Structure

```text
DataGPT/
│
├── app.py                  # Main Streamlit application
├── auth.py                 # Authentication system
├── history.py              # Dataset and analysis history
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .gitignore              # Git ignored files
│
├── .streamlit/
│   └── secrets.toml        # API configuration (keep private)
│
└── data/
    └── sales.csv           # Sample dataset (if included)
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/adityat-lab/DataGPT.git
```

### 2. Open the project folder

```bash
cd DataGPT
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure Gemini API

Create a file named `secrets.toml` inside the `.streamlit` folder.

Add your own Gemini API key:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

Get an API key from [Google AI Studio](https://aistudio.google.com/).

**Important:** Never upload your actual API key or `secrets.toml` to GitHub.

### 7. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

## 📊 How to Use

1. Create an account or log in.
2. Upload a CSV, Excel, or supported dataset.
3. Explore the dataset using the dashboard.
4. Generate interactive visualizations.
5. Ask questions using the AI Assistant.
6. Detect anomalies and clean your data.
7. Generate AI-powered analysis and reports.
8. Download your results in PDF or Excel format.

## 🔐 Security

- API credentials are stored in local Streamlit secrets.
- Passwords are handled using secure password hashing.
- Local database files and environment folders should not be committed to GitHub.
- Never share API keys or private credentials publicly.

## 🔮 Future Enhancements

- Live database connectivity.
- More advanced machine learning models.
- Automated predictive analytics.
- Interactive AI-generated dashboards.
- Cloud deployment and scalable data processing.
- More export formats and customizable reports.

## 👨‍💻 Author

**Aditya Thokala**

B.Tech — Artificial Intelligence and Data Science

GitHub: [@adityat-lab](https://github.com/adityat-lab)

---

⭐ If you find this project useful, consider giving it a star on GitHub.

**DataGPT — Turning Data into Decisions with AI.**