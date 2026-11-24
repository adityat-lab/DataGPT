
import os
import io
import hashlib
import time
import requests
import auth
import history
import json

MODEL_NAME = "gemini-3.5-flash-lite"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)
from reportlab.lib.units import inch

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================



# =====================================================
# 1. PAGE CONFIGURATION AND AUTHENTICATION
# =====================================================

st.set_page_config(
    page_title="DataGPT Analytics",
    page_icon="📊",
    layout="wide"
)

# Create authentication database
auth.create_table()
history.create_history_tables()

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = None

if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = None

if "history_dataset_id" not in st.session_state:
    st.session_state.history_dataset_id = None

# Show authentication page when logged out
if not st.session_state.logged_in or not st.session_state.user:

    st.title("📊 DataGPT")
    st.subheader("Your Dedicated Analytics Assistant")

    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

    with login_tab:

        st.subheader("Welcome Back")

        email = st.text_input(
            "Email Address",
            key="login_email"
        )

        password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.button("Login", use_container_width=True):

            if not email.strip() or not password:
                st.warning("Enter your email and password.")

            else:
                success, user = auth.login(email, password)

                if success:
                    st.session_state.logged_in = True
                    st.session_state.user = user
                    st.rerun()
                else:
                    st.error("Invalid email or password.")

    with signup_tab:

        st.subheader("Create Account")

        username = st.text_input(
            "Username",
            key="signup_username"
        )

        new_email = st.text_input(
            "Email Address",
            placeholder="example@gmail.com",
            key="signup_email"
        )

        new_password = st.text_input(
            "Password",
            type="password",
            placeholder="Example@123",
            key="signup_password"
        )

        st.caption(
            "Password must have at least 8 characters, "
            "one uppercase letter, one lowercase letter, "
            "one number and one special character."
        )

        if st.button("Create Account", use_container_width=True):

            if not username.strip() or not new_email.strip() or not new_password:
                st.warning("Please fill in all fields.")

            else:
                success, message = auth.signup(
                    username,
                    new_email,
                    new_password
                )

                if success:
                    st.success(
                        message + " You can now log in."
                    )
                else:
                    st.error(message)

    # IMPORTANT: Stop the rest of the app until login
    st.stop()


# =====================================================
# 2. CUSTOM CSS
# =====================================================

st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        opacity: 0.7;
        margin-bottom: 25px;
    }

    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128,128,128,0.2);
        padding: 16px;
        border-radius: 12px;
    }

    div.stButton > button {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. GEMINI CONFIGURATION
# =====================================================

MODEL_NAME = "gemini-3.5-flash-lite"


@st.cache_resource
def get_gemini_client():

    api_key = ""

    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass

    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY", "")

    if not api_key:
        raise ValueError(
            "Gemini API key not found. "
            "Configure .streamlit/secrets.toml"
        )

    return genai.Client(api_key=api_key)


def ask_gemini(prompt):
    api_key = ""

    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass

    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY", "")

    if not api_key:
        raise ValueError("Gemini API key not found.")

    url = "https://generativelanguage.googleapis.com/v1beta/interactions"

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }

    payload = {
        "model": MODEL_NAME,
        "input": prompt
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=(10, 60)
    )

    response.raise_for_status()
    result = response.json()

    for step in result.get("steps", []):
        if step.get("type") == "model_output":
            for item in step.get("content", []):
                if item.get("type") == "text":
                    return item.get("text", "")

    raise ValueError("Gemini returned no text response.")


# =====================================================
# 4. FILE UPLOAD AND DATA LOADING
# =====================================================

def load_data(file_obj, filename=None):

    # Always obtain the original filename safely
    if filename is None:
        filename = getattr(file_obj, "name", None)

    if not filename:
        raise ValueError("File name is missing.")

    filename = filename.lower()

    # Reset the file pointer
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)

    try:

        if filename.endswith(".csv"):

            return pd.read_csv(file_obj)

        elif filename.endswith((".xlsx", ".xls")):

            return pd.read_excel(file_obj)

        elif filename.endswith(".json"):

            return pd.read_json(file_obj)

        else:

            raise ValueError(
                "Unsupported format. "
                "Upload CSV, XLSX, XLS or JSON."
            )

    except Exception as error:

        raise ValueError(
            f"Could not read {filename}: {error}"
        ) from error


# =====================================================
# 5. DATASET PROFILE
# =====================================================

def dataset_profile(df):

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(
            df.columns.astype(str)
        ),
        "missing_values": int(
            df.isnull().sum().sum()
        ),
        "duplicate_rows": int(
            df.duplicated().sum()
        ),
        "memory_usage_mb": round(
            df.memory_usage(deep=True).sum()
            / (1024 * 1024),
            2
        )
    }


# =====================================================
# 6. AI DATASET CONTEXT
# =====================================================


def prepare_ai_context(df):

    profile = dataset_profile(df)

    # Numerical statistics
    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if numeric_cols:
        numeric_summary = (
            df[numeric_cols]
            .describe()
            .round(2)
            .to_string()
        )
    else:
        numeric_summary = "No numerical columns found."

    # Categorical statistics
    categorical_cols = df.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    categorical_summary = {}

    for col in categorical_cols[:15]:
        categorical_summary[str(col)] = (
            df[col]
            .astype(str)
            .value_counts()
            .head(10)
            .to_dict()
        )

    # Missing values
    missing_summary = (
        df.isnull()
        .sum()
        .to_dict()
    )

    # Correlations
    if len(numeric_cols) >= 2:
        correlation_summary = (
            df[numeric_cols]
            .corr()
            .round(2)
            .to_string()
        )
    else:
        correlation_summary = "Not enough numerical columns."

    # Sample records
    sample = df.head(12).copy()

    sample = sample.replace(
        [np.inf, -np.inf],
        np.nan
    )

    sample = sample.astype(object).where(
        pd.notna(sample),
        None
    )

    return f"""
You are DataGPT, a professional data analytics assistant.

DATASET PROFILE:
{profile}

NUMERICAL STATISTICS:
{numeric_summary}

CATEGORICAL VALUE DISTRIBUTIONS:
{categorical_summary}

MISSING VALUES:
{missing_summary}

CORRELATION MATRIX:
{correlation_summary}

SAMPLE RECORDS:
{sample.to_string(index=False)}

IMPORTANT RULES:
- Use only the supplied dataset information.
- Never invent statistics or values.
- Clearly distinguish actual statistics from sample observations.
- The sample contains only the first 12 records.
- Explain insights in simple professional language.
- Use headings and bullet points where useful.
- If the information is insufficient, say so honestly.
"""

def generate_advanced_analysis(df):

    context = prepare_ai_context(df)

    prompt = f"""
{context}

You are DataGPT's advanced analytics engine.

Perform a comprehensive analysis of this dataset.

Create the following sections:

1. EXECUTIVE SUMMARY
Explain what this dataset contains.

2. KEY DATA INSIGHTS
Identify at least five meaningful insights.
Use actual statistics where available.

3. TREND ANALYSIS
Identify possible trends and patterns.
Do not claim time-based trends unless date information supports them.

4. CORRELATION ANALYSIS
Explain important relationships between numerical columns.
Do not confuse correlation with causation.

5. DATA QUALITY
Identify missing values, duplicate rows and potential data quality issues.

6. BUSINESS RECOMMENDATIONS
Give five practical recommendations based on the dataset.

7. VISUALIZATION RECOMMENDATIONS
Suggest five useful charts.
Mention the recommended columns for each chart.

8. FINAL CONCLUSION
Give a short conclusion about the dataset.

IMPORTANT:
- Never invent numerical values.
- Clearly explain limitations.
- Use simple professional language.
- Separate observations from recommendations.
- Remember that only a sample of records is supplied,
  but the provided summary statistics describe the full dataset.

Format your response using clear Markdown headings and bullet points.
"""

    return ask_gemini(prompt)

def generate_pdf_report(df, ai_analysis=""):

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    story = []

    profile = dataset_profile(df)

    story.append(
        Paragraph("DataGPT Analytics Report", styles["Title"])
    )

    story.append(Spacer(1, 15))

    story.append(
        Paragraph("Dataset Overview", styles["Heading2"])
    )

    overview = [
        ["Metric", "Value"],
        ["Total Rows", str(profile["rows"])],
        ["Total Columns", str(profile["columns"])],
        ["Missing Values", str(profile["missing_values"])],
        ["Duplicate Rows", str(profile["duplicate_rows"])],
        ["Memory Usage", f"{profile['memory_usage_mb']} MB"]
    ]

    table = Table(overview, colWidths=[220, 220])

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243B55")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))

    story.append(table)

    story.append(Spacer(1, 20))

    story.append(
        Paragraph("Column Information", styles["Heading2"])
    )

    column_data = [
        ["Column", "Data Type", "Missing", "Unique"]
    ]

    for col in df.columns[:30]:

        column_data.append([
            str(col)[:30],
            str(df[col].dtype),
            str(int(df[col].isna().sum())),
            str(int(df[col].nunique()))
        ])

    column_table = Table(
        column_data,
        repeatRows=1
    )

    column_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243B55")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("PADDING", (0, 0), (-1, -1), 6)
    ]))

    story.append(column_table)

    story.append(Spacer(1, 20))

    story.append(
        Paragraph("Dataset Preview", styles["Heading2"])
    )

    preview = df.head(10).copy().fillna("")

    preview_data = [
        [str(col)[:18] for col in preview.columns[:6]]
    ]

    for _, row in preview.iterrows():

        preview_data.append([
            str(value)[:25]
            for value in row.iloc[:6]
        ])

    preview_table = Table(
        preview_data,
        repeatRows=1
    )

    preview_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243B55")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("PADDING", (0, 0), (-1, -1), 5)
    ]))

    story.append(preview_table)

    if ai_analysis:

        story.append(PageBreak())

        story.append(
            Paragraph("AI Analysis and Insights", styles["Heading1"])
        )

        # Keep this as plain text to avoid treating AI output as HTML.
        for line in ai_analysis.splitlines():

            clean_line = (
                line.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )

            if clean_line.strip():

                story.append(
                    Paragraph(
                        clean_line,
                        styles["BodyText"]
                    )
                )

                story.append(Spacer(1, 5))

    doc.build(story)

    buffer.seek(0)

    return buffer.getvalue()


def generate_excel_report(df, ai_analysis=""):

    buffer = io.BytesIO()

    profile = dataset_profile(df)

    column_info = pd.DataFrame({
        "Column": df.columns.astype(str),
        "Data Type": [str(dtype) for dtype in df.dtypes],
        "Missing Values": [
            int(df[col].isna().sum())
            for col in df.columns
        ],
        "Unique Values": [
            int(df[col].nunique())
            for col in df.columns
        ]
    })

    overview = pd.DataFrame({
        "Metric": [
            "Total Rows",
            "Total Columns",
            "Missing Values",
            "Duplicate Rows",
            "Memory Usage MB"
        ],
        "Value": [
            profile["rows"],
            profile["columns"],
            profile["missing_values"],
            profile["duplicate_rows"],
            profile["memory_usage_mb"]
        ]
    })

    with pd.ExcelWriter(
        buffer,
        engine="xlsxwriter"
    ) as writer:

        df.to_excel(
            writer,
            sheet_name="Dataset",
            index=False
        )

        overview.to_excel(
            writer,
            sheet_name="Overview",
            index=False
        )

        column_info.to_excel(
            writer,
            sheet_name="Column Info",
            index=False
        )

        if ai_analysis:

            pd.DataFrame({
                "AI Analysis": ai_analysis.splitlines()
            }).to_excel(
                writer,
                sheet_name="AI Insights",
                index=False
            )

        workbook = writer.book

        header_format = workbook.add_format({
            "bold": True,
            "font_color": "white",
            "bg_color": "#243B55",
            "border": 1
        })

        for worksheet in writer.sheets.values():

            worksheet.freeze_panes(1, 0)
            worksheet.set_default_row(20)

            worksheet.autofilter(
                0, 0,
                max(1, worksheet.dim_rowmax),
                max(0, worksheet.dim_colmax)
            )

            worksheet.set_column(
                0,
                max(0, worksheet.dim_colmax),
                20
            )

            for col_num, value in enumerate(
                worksheet.table[0][0] if False else []
            ):
                pass

    buffer.seek(0)

    return buffer.getvalue()




# =====================================================
# 7. SIDEBAR
# =====================================================

with st.sidebar:

    st.title("📊 DataGPT")

    st.caption(
        "Your Dedicated Analytics Assistant"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Data Explorer",
            "Visualizations",
            "AI Assistant",
            "Anomaly Detection",
            "Data Cleaning",
            "Advanced AI Analysis",
            "Report Center",
            "Dataset History"
            
        ]
    )

    st.divider()

    st.subheader("Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["csv", "xlsx", "xls", "json"]
    )

    st.divider()

    st.caption("Powered by Gemini AI")


# =====================================================
# 8. MAIN HEADER
# =====================================================

st.markdown(
    '<p class="main-title">DataGPT Analytics</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">'
    'Transform your data into meaningful insights with AI.'
    '</p>',
    unsafe_allow_html=True
)

# =====================================================
# USER ACCOUNT AND LOGOUT
# =====================================================

if (
    st.session_state.get("logged_in", False)
    and st.session_state.get("user") is not None
):

    st.sidebar.success(
        f"Welcome, {st.session_state.user['username']}"
    )

    if st.sidebar.button("Logout", use_container_width=True):

        st.session_state.logged_in = False
        st.session_state.user = None

        st.rerun()


# =====================================================
# 9. CHECK UPLOADED FILE
# =====================================================

if uploaded_file is None:

    st.info(
        "👈 Upload a CSV, Excel or JSON dataset "
        "from the sidebar to get started."
    )

    st.subheader("What DataGPT can do")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📊 Dashboard")
        st.write(
            "Explore dataset statistics and key information."
        )

    with col2:
        st.markdown("### 🤖 AI Assistant")
        st.write(
            "Ask questions about your dataset using Gemini."
        )

    with col3:
        st.markdown("### 📈 Visualizations")
        st.write(
            "Create interactive charts and graphs."
        )

    st.stop()



# =====================================================
# 10. READ UPLOADED DATASET
# =====================================================

if uploaded_file is not None:

    try:
        file_bytes = uploaded_file.getvalue()

        file_hash = hashlib.md5(file_bytes).hexdigest()

        dataset_id = (
            uploaded_file.name,
            file_hash
        )

        file_buffer = io.BytesIO(file_bytes)

        df = load_data(
            file_buffer,
            uploaded_file.name
        )

        if df is None or df.empty:
            st.error("Dataset is empty or could not be loaded.")
            st.stop()

        user_id = st.session_state.user["id"]

        st.session_state.history_dataset_id = history.save_dataset(
            user_id=user_id,
            file_bytes=file_bytes,
            filename=uploaded_file.name,
            df=df
        )
        st.session_state.dataset_id = st.session_state.history_dataset_id

        st.session_state.current_df = df

    except Exception as error:
        st.error(f"Unable to read file: {error}")
        st.stop()

# =====================================================
# 11. DATASET INFORMATION
# =====================================================

profile = dataset_profile(df)

st.success(
    f"Dataset loaded successfully: {uploaded_file.name}"
)


# =====================================================
# 12. DASHBOARD
# =====================================================

if page == "Dashboard":

    st.header("📊 Dashboard Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Rows",
        f"{profile['rows']:,}"
    )

    col2.metric(
        "Total Columns",
        f"{profile['columns']:,}"
    )

    col3.metric(
        "Missing Values",
        f"{profile['missing_values']:,}"
    )

    col4.metric(
        "Duplicate Rows",
        f"{profile['duplicate_rows']:,}"
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Dataset Information")

        info_df = pd.DataFrame({
            "Property": [
                "File Name",
                "Rows",
                "Columns",
                "Memory Usage",
                "Duplicate Rows"
            ],
            "Value": [
                uploaded_file.name,
                profile["rows"],
                profile["columns"],
                f"{profile['memory_usage_mb']} MB",
                profile["duplicate_rows"]
            ]
        })

        st.dataframe(
            info_df,
            use_container_width=True,
            hide_index=True
        )

    with right:

        st.subheader("Missing Values by Column")

        missing = df.isnull().sum()

        missing = missing[missing > 0]

        if not missing.empty:

            missing_df = missing.reset_index()

            missing_df.columns = [
                "Column",
                "Missing Values"
            ]

            fig = px.bar(
                missing_df,
                x="Column",
                y="Missing Values",
                title="Missing Values"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.success("No missing values found!")

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.download_button(
        "⬇️ Download Dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="dataset.csv",
        mime="text/csv"
    )


# =====================================================
# 13. DATA EXPLORER
# =====================================================

elif page == "Data Explorer":

    st.header("🔍 Data Explorer")

    st.subheader("Dataset Preview")

    st.dataframe(
        df,
        use_container_width=True,
        height=450
    )

    st.divider()

    st.subheader("Search Dataset")

    selected_column = st.selectbox(
        "Select column",
        df.columns.tolist()
    )

    search_text = st.text_input(
        "Enter search value"
    )

    if search_text:

        filtered_df = df[
            df[selected_column]
            .astype(str)
            .str.contains(
                search_text,
                case=False,
                na=False,
                regex=False
            )
        ]

        st.write(
            f"Found {len(filtered_df)} matching records."
        )

        st.dataframe(
            filtered_df,
            use_container_width=True
        )

    st.divider()

    st.subheader("Column Information")

    column_info = pd.DataFrame({
        "Column": df.columns.astype(str),
        "Data Type": [
            str(dtype) for dtype in df.dtypes
        ],
        "Non-Null Values": [
            int(df[col].notna().sum())
            for col in df.columns
        ],
        "Missing Values": [
            int(df[col].isnull().sum())
            for col in df.columns
        ],
        "Unique Values": [
            int(df[col].nunique())
            for col in df.columns
        ]
    })

    st.dataframe(
        column_info,
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# 14. VISUALIZATIONS
# =====================================================

elif page == "Visualizations":

    st.header("📈 Interactive Visualizations")

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    chart_type = st.selectbox(
        "Choose Chart Type",
        [
            "Histogram",
            "Bar Chart",
            "Scatter Plot",
            "Line Chart",
            "Box Plot",
            "Correlation Heatmap"
        ]
    )

    if chart_type == "Histogram":

        if numeric_columns:

            column = st.selectbox(
                "Select column",
                numeric_columns
            )

            fig = px.histogram(
                df,
                x=column,
                title=f"Distribution of {column}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning("No numerical columns found.")

    elif chart_type == "Bar Chart":

        if categorical_columns and numeric_columns:

            x_column = st.selectbox(
                "Category column",
                categorical_columns
            )

            y_column = st.selectbox(
                "Numerical column",
                numeric_columns
            )

            grouped = (
                df.groupby(
                    x_column,
                    dropna=False
                )[y_column]
                .sum()
                .reset_index()
                .head(30)
            )

            fig = px.bar(
                grouped,
                x=x_column,
                y=y_column,
                title=f"{y_column} by {x_column}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "A categorical and numerical column are required."
            )

    elif chart_type == "Scatter Plot":

        if len(numeric_columns) >= 2:

            x_column = st.selectbox(
                "X Axis",
                numeric_columns
            )

            y_column = st.selectbox(
                "Y Axis",
                numeric_columns,
                index=1
            )

            fig = px.scatter(
                df,
                x=x_column,
                y=y_column
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "At least two numerical columns are required."
            )

    elif chart_type == "Line Chart":

        if numeric_columns:

            x_column = st.selectbox(
                "X Axis",
                df.columns.tolist()
            )

            y_column = st.selectbox(
                "Y Axis",
                numeric_columns
            )

            fig = px.line(
                df.head(500),
                x=x_column,
                y=y_column
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning("No numerical columns found.")

    elif chart_type == "Box Plot":

        if numeric_columns:

            column = st.selectbox(
                "Select numerical column",
                numeric_columns
            )

            fig = px.box(
                df,
                y=column,
                title=f"Box Plot of {column}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning("No numerical columns found.")

    elif chart_type == "Correlation Heatmap":

        if len(numeric_columns) >= 2:

            corr = df[numeric_columns].corr()

            fig = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Correlation Matrix"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "At least two numerical columns are required."
            )


# =====================================================
# 15. AI ASSISTANT
# =====================================================

elif page == "AI Assistant":

    st.header("🤖 DataGPT AI Assistant")

    st.write(
        "Ask questions about your uploaded dataset."
    )

    st.caption(
        f"Powered by {MODEL_NAME}"
    )

    context = prepare_ai_context(df)

    suggested_questions = [
        "Give me a summary of this dataset.",
        "Find important trends in my data.",
        "Explain missing values.",
        "What insights can I get from this dataset?",
        "Suggest useful visualizations."
    ]

    st.subheader("Suggested Questions")

    cols = st.columns(2)

    for index, question in enumerate(suggested_questions):

        with cols[index % 2]:

            if st.button(
                question,
                key=f"suggestion_{index}",
                use_container_width=True
            ):

                st.session_state.pending_question = question

    st.divider()

    if "chat_history" not in st.session_state:

        st.session_state.chat_history = []

    if st.button("Clear Chat"):

        st.session_state.chat_history = []

        st.rerun()

    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    question = st.chat_input(
        "Ask anything about your dataset..."
    )

    if "pending_question" in st.session_state:

        question = st.session_state.pop(
            "pending_question"
        )

    if question:

        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):

            st.markdown(question)

        history = ""

        for message in st.session_state.chat_history[-7:-1]:

            history += (
                f"\n{message['role']}: "
                f"{message['content']}"
            )

        prompt = f"""
{context}

PREVIOUS CONVERSATION:
{history}

USER QUESTION:
{question}

Answer clearly and accurately.
Do not invent numerical results.
Mention when the supplied sample is insufficient.
"""

        with st.chat_message("assistant"):

            with st.spinner(
                "DataGPT is analyzing your dataset..."
            ):

                try:

                    answer = ask_gemini(prompt)

                    st.markdown(answer)

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as error:

                    error_message = str(error)

                    st.error(error_message)

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Error: {error_message}"
                    })

# =====================================================
# ADVANCED AI DATA ANALYSIS
# =====================================================

elif page == "Advanced AI Analysis":

    st.header("🧠 Advanced AI Data Analysis")

    st.write(
        "Generate comprehensive insights, identify patterns "
        "and receive data-driven recommendations."
    )

    st.caption(f"Powered by {MODEL_NAME}")

    profile = dataset_profile(df)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", f"{profile['rows']:,}")
    col2.metric("Total Features", f"{profile['columns']:,}")
    col3.metric(
        "Missing Values",
        f"{profile['missing_values']:,}"
    )

    st.divider()

    st.subheader("Choose your analysis")

    analysis_type = st.selectbox(
        "Analysis type",
        [
            "Complete AI Analysis",
            "Business Insights",
            "Trend and Pattern Analysis",
            "Data Quality Analysis",
            "Visualization Recommendations"
        ]
    )

    analysis_prompts = {
        "Complete AI Analysis":
            "Perform a complete analysis of this dataset.",
        "Business Insights":
            "Identify important business insights and give practical recommendations.",
        "Trend and Pattern Analysis":
            "Analyze numerical patterns, trends and relationships. Do not invent time trends.",
        "Data Quality Analysis":
            "Analyze missing values, duplicates, data quality and possible issues.",
        "Visualization Recommendations":
            "Recommend suitable charts with the exact columns to use and explain why."
    }

    if st.button(
        "🚀 Generate AI Analysis",
        type="primary",
        use_container_width=True
    ):

        prompt = f"""
{prepare_ai_context(df)}

Perform this analysis:

{analysis_prompts[analysis_type]}

Use actual supplied statistics.
Do not invent results.
Use clear headings, bullet points and simple language.
Mention limitations where necessary.
"""

        with st.spinner(
            "DataGPT is analyzing your dataset..."
        ):

            try:

                result = ask_gemini(prompt)

                st.session_state.advanced_analysis = result
                st.session_state.advanced_analysis_type = analysis_type
                st.session_state.advanced_analysis_dataset = st.session_state.dataset_id

            except requests.exceptions.HTTPError as error:

                st.error(
                    f"Gemini API error: {error}"
                )

            except requests.exceptions.Timeout:

                st.error(
                    "The AI request timed out. Please try again."
                )

            except Exception as error:

                st.error(
                    f"Analysis failed: {error}"
                )

    if (
        st.session_state.get("advanced_analysis")
        and st.session_state.get("advanced_analysis_dataset")
        == st.session_state.get("dataset_id")
    ):

        st.divider()

        st.subheader("📊 AI Analysis Results")

        st.caption(
            st.session_state.get(
                "advanced_analysis_type",
                "AI Analysis"
            )
        )

        st.markdown(
            st.session_state.advanced_analysis
        )

        st.download_button(
            "⬇️ Download Analysis",
            data=st.session_state.advanced_analysis,
            file_name="DataGPT_Advanced_Analysis.md",
            mime="text/markdown"
        )

        if st.button("Clear Analysis"):
            st.session_state.pop("advanced_analysis", None)
            st.session_state.pop("advanced_analysis_type", None)
            st.session_state.pop("advanced_analysis_dataset", None)
            st.rerun()

# =====================================================
# REPORT CENTER
# =====================================================

elif page == "Report Center":

    st.header("📄 DataGPT Report Center")

    st.write(
        "Generate and download professional reports "
        "from your uploaded dataset."
    )

    ai_analysis = st.session_state.get(
        "advanced_analysis", ""
    )

    st.subheader("Report Contents")

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset Rows", len(df))
    col2.metric("Dataset Columns", len(df.columns))
    col3.metric(
        "AI Analysis",
        "Available" if ai_analysis else "Not generated"
    )

    if not ai_analysis:

        st.info(
            "You can generate an AI analysis first "
            "from the Advanced AI Analysis page. "
            "Reports can also be created without AI insights."
        )

    st.divider()

    st.subheader("Download Reports")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 📕 PDF Report")

        if st.button(
            "Generate PDF",
            use_container_width=True
        ):

            try:

                pdf_data = generate_pdf_report(
                    df,
                    ai_analysis
                )

                st.download_button(
                    "⬇️ Download PDF",
                    data=pdf_data,
                    file_name="DataGPT_Report.pdf",
                    mime="application/pdf",
                    key="download_pdf"
                )

            except Exception as error:

                st.error(f"PDF generation failed: {error}")

    with col2:

        st.markdown("### 📗 Excel Report")

        if st.button(
            "Generate Excel",
            use_container_width=True
        ):

            try:

                excel_data = generate_excel_report(
                    df,
                    ai_analysis
                )

                st.download_button(
                    "⬇️ Download Excel",
                    data=excel_data,
                    file_name="DataGPT_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel"
                )

            except Exception as error:

                st.error(f"Excel generation failed: {error}")
elif page == "Dataset History":

    st.title("📚 Dataset History")

    user_id = st.session_state.user["id"]
    datasets = history.get_user_datasets(user_id)

    if datasets:

        for item in datasets:

            dataset_id, filename, rows, columns, uploaded_at, columns_json = item

            with st.expander(f"📄 {filename}"):

                col1, col2, col3 = st.columns(3)

                col1.metric("Rows", rows)
                col2.metric("Columns", columns)
                col3.metric("Dataset ID", dataset_id)

                st.caption(f"Uploaded: {uploaded_at}")

                st.write(", ".join(json.loads(columns_json or "[]")))

                if st.button("Delete History", key=f"delete_{dataset_id}"):
                    history.delete_dataset(user_id, dataset_id)
                    st.rerun()

    else:
        st.info("No dataset history found.")

# =====================================================
# DATASET HISTORY
# =====================================================

elif page == "Dataset History":

    st.title("📚 Dataset History")
    st.write("Your previously uploaded datasets.")

    user_id = st.session_state.user["id"]

    datasets = history.get_user_datasets(user_id)

    if datasets:

        st.success(f"You have uploaded {len(datasets)} dataset(s).")

        for item in datasets:

            dataset_id, filename, rows, columns, uploaded_at, columns_json = item

            with st.expander(f"📄 {filename}"):

                col1, col2, col3 = st.columns(3)

                col1.metric("Rows", rows)
                col2.metric("Columns", columns)
                col3.metric("Dataset ID", dataset_id)

                st.caption(f"Uploaded: {uploaded_at}")

                st.write("Column names:")
                st.write(", ".join(json.loads(columns_json or "[]")))

                if st.button(
                    "Delete History",
                    key=f"delete_{dataset_id}"
                ):
                    history.delete_dataset(user_id, dataset_id)
                    st.success("History deleted.")
                    st.rerun()

    else:
        st.info("No dataset history found. Upload a dataset first.")


# =====================================================
# 16. ADVANCED ANOMALY DETECTION
# =====================================================

elif page == "Anomaly Detection":

    st.header("🚨 AI-Powered Anomaly Detection")

    st.write(
        "Identify unusual records and outliers using "
        "Machine Learning and Isolation Forest."
    )

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if not numeric_columns:

        st.error(
            "No numerical columns found. "
            "Anomaly detection requires numerical data."
        )

    else:

        st.subheader("Configure Detection")

        selected_columns = st.multiselect(
            "Select numerical columns for analysis",
            numeric_columns,
            default=numeric_columns[
                :min(3, len(numeric_columns))
            ]
        )

        contamination = st.slider(
            "Expected anomaly proportion",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
            help="Estimated percentage of unusual records."
        )

        st.caption(
            "Tip: Select columns such as Sales, Revenue, "
            "Quantity or Amount for meaningful results."
        )

        if st.button(
            "🚀 Detect Anomalies",
            type="primary",
            use_container_width=True
        ):

            if not selected_columns:

                st.warning(
                    "Please select at least one numerical column."
                )

            else:

                try:

                    # Prepare selected data
                    analysis_df = df[
                        selected_columns
                    ].copy()

                    # Convert invalid values to NaN
                    for col in selected_columns:

                        analysis_df[col] = pd.to_numeric(
                            analysis_df[col],
                            errors="coerce"
                        )

                    # Replace infinity values
                    analysis_df = analysis_df.replace(
                        [np.inf, -np.inf],
                        np.nan
                    )

                    # Remove columns containing only missing values
                    analysis_df = analysis_df.dropna(
                        axis=1,
                        how="all"
                    )

                    if analysis_df.empty or len(analysis_df.columns) == 0:

                        st.error(
                            "Selected columns contain no usable numerical data."
                        )

                    else:

                        # Keep records with at least one available value
                        valid_rows = analysis_df.notna().any(axis=1)

                        clean_data = analysis_df.loc[
                            valid_rows
                        ].copy()

                        # Fill missing values with column medians
                        for col in clean_data.columns:

                            median_value = clean_data[col].median()

                            clean_data[col] = clean_data[col].fillna(
                                median_value
                            )

                        if len(clean_data) < 5:

                            st.error(
                                "At least 5 usable records are required."
                            )

                        else:

                            # Scale numerical data
                            scaler = StandardScaler()

                            scaled_data = scaler.fit_transform(
                                clean_data
                            )

                            # Train Isolation Forest
                            model = IsolationForest(
                                n_estimators=200,
                                contamination=contamination,
                                random_state=42,
                                n_jobs=-1
                            )

                            predictions = model.fit_predict(
                                scaled_data
                            )

                            scores = model.decision_function(
                                scaled_data
                            )

                            # Create result dataset
                            result = df.copy()

                            result["Anomaly Status"] = "Not Analyzed"
                            result["Anomaly Score"] = np.nan

                            result.loc[
                                valid_rows,
                                "Anomaly Status"
                            ] = np.where(
                                predictions == -1,
                                "Anomaly",
                                "Normal"
                            )

                            result.loc[
                                valid_rows,
                                "Anomaly Score"
                            ] = scores

                            # Save results
                            st.session_state.anomaly_results = result

                            anomaly_count = int(
                                (predictions == -1).sum()
                            )

                            normal_count = int(
                                (predictions == 1).sum()
                            )

                            st.success(
                                "Anomaly detection completed successfully!"
                            )

                            col1, col2, col3 = st.columns(3)

                            col1.metric(
                                "Analyzed Records",
                                len(clean_data)
                            )

                            col2.metric(
                                "Anomalies Detected",
                                anomaly_count
                            )

                            col3.metric(
                                "Normal Records",
                                normal_count
                            )

                except Exception as error:

                    st.error(
                        f"Anomaly detection failed: {error}"
                    )

        # Display saved results
        if "anomaly_results" in st.session_state:

            result = st.session_state.anomaly_results

            st.divider()

            st.subheader("📊 Detection Results")

            anomalies = result[
                result["Anomaly Status"] == "Anomaly"
            ]

            normal = result[
                result["Anomaly Status"] == "Normal"
            ]

            tab1, tab2, tab3 = st.tabs([
                "All Records",
                "Anomalies",
                "Normal Records"
            ])

            with tab1:

                st.dataframe(
                    result,
                    use_container_width=True
                )

            with tab2:

                st.write(
                    f"Detected {len(anomalies)} unusual records."
                )

                st.dataframe(
                    anomalies,
                    use_container_width=True
                )

            with tab3:

                st.write(
                    f"Found {len(normal)} normal records."
                )

                st.dataframe(
                    normal,
                    use_container_width=True
                )

            st.download_button(
                "⬇️ Download Anomaly Report",
                data=result.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="DataGPT_Anomaly_Report.csv",
                mime="text/csv"
            )

            if st.button("Clear Anomaly Results"):

                st.session_state.pop(
                    "anomaly_results",
                    None
                )

                st.rerun()


# =====================================================
# 17. DATA CLEANING
# =====================================================

elif page == "Data Cleaning":

    st.header("🧹 Data Cleaning")

    st.write(
        "Clean missing values and duplicate records."
    )

    cleaned_df = df.copy()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Missing Values",
        int(cleaned_df.isnull().sum().sum())
    )

    col2.metric(
        "Duplicate Rows",
        int(cleaned_df.duplicated().sum())
    )

    col3.metric(
        "Total Rows",
        len(cleaned_df)
    )

    st.divider()

    missing_option = st.selectbox(
        "Handle missing values",
        [
            "Keep missing values",
            "Remove rows with missing values",
            "Fill numerical values with mean",
            "Fill numerical values with median",
            "Fill missing values with zero"
        ]
    )

    remove_duplicates = st.checkbox(
        "Remove duplicate rows",
        value=True
    )

    if st.button(
        "Clean Dataset",
        type="primary"
    ):

        cleaned_df = df.copy()

        if missing_option == "Remove rows with missing values":

            cleaned_df = cleaned_df.dropna()

        elif missing_option == "Fill numerical values with mean":

            numeric_cols = cleaned_df.select_dtypes(
                include=np.number
            ).columns

            for col in numeric_cols:

                if cleaned_df[col].notna().any():

                    cleaned_df[col] = cleaned_df[col].fillna(
                        cleaned_df[col].mean()
                    )

        elif missing_option == "Fill numerical values with median":

            numeric_cols = cleaned_df.select_dtypes(
                include=np.number
            ).columns

            for col in numeric_cols:

                if cleaned_df[col].notna().any():

                    cleaned_df[col] = cleaned_df[col].fillna(
                        cleaned_df[col].median()
                    )

        elif missing_option == "Fill missing values with zero":

            cleaned_df = cleaned_df.fillna(0)

        if remove_duplicates:

            cleaned_df = cleaned_df.drop_duplicates()

        st.session_state.cleaned_df = cleaned_df

        st.success(
            "Dataset cleaned successfully!"
        )

    if "cleaned_df" in st.session_state:

        st.divider()

        st.subheader("Cleaned Dataset Preview")

        cleaned_df = st.session_state.cleaned_df

        st.dataframe(
            cleaned_df,
            use_container_width=True
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Remaining Rows",
            len(cleaned_df)
        )

        col2.metric(
            "Remaining Missing Values",
            int(cleaned_df.isnull().sum().sum())
        )

        st.download_button(
            "⬇️ Download Cleaned Dataset",
            data=cleaned_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )


# =====================================================
# 18. FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; opacity:0.65;">
        <p>DataGPT | Your Dedicated Analytics Assistant</p>
        <p>Built with Python, Streamlit, Plotly and Gemini AI</p>
    </div>
    """,
    unsafe_allow_html=True
)
