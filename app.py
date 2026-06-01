import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from io import BytesIO

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------

st.set_page_config(
    layout="wide",
    page_title="AI Student Intelligence Canvas"
)

# ----------------------------------------------------
# GEMINI API SETUP
# ----------------------------------------------------

GEMINI_API_KEY = "AQ.Ab8RN6Lvv3-ScuThC8-_D_vjs3k69xLlSSaob-_viYRY2ptqUQ"

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel('gemini-2.5-flash')

# ----------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #020617;
    color: #f8fafc;
}

[data-testid="stMetric"] {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
}

[data-testid="stMetricValue"] {
    color: #38bdf8 !important;
}

[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
}

.ai-card {
    background-color: #0f172a;
    color: #e2e8f0;
    padding: 25px;
    border-radius: 15px;
    border-top: 5px solid #38bdf8;
    margin: 20px 0;
    font-size: 16px;
    line-height: 1.5;
}

.ai-header {
    color: #38bdf8;
    font-weight: 800;
    font-size: 22px;
    margin-bottom: 10px;
}

.stTextInput input {
    background-color: #1e293b !important;
    color: white !important;
}

.stDataFrame {
    background-color: #1e293b;
    border-radius: 10px;
    border: 1px solid #334155;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# DATA PROCESSING FUNCTION
# ----------------------------------------------------

@st.cache_data
def process_full_project_data():

    df = pd.read_csv('student_performance_prediction.csv')

    # ------------------------------------------------
    # PREPROCESSING
    # ------------------------------------------------

    cols_to_fix = [
        'Study Hours per Week',
        'Attendance Rate',
        'Previous Grades'
    ]

    for col in cols_to_fix:
        df[col] = df[col].fillna(
            df[col].median()
        )

    df.drop_duplicates(inplace=True)

    df['Passed'] = df['Passed'].fillna('Unknown')

    # ------------------------------------------------
    # FEATURE ENGINEERING
    # ------------------------------------------------

    le = LabelEncoder()

    df["Participation_Label"] = le.fit_transform(
        df["Participation in Extracurricular Activities"]
    )

    # Engagement Score
    df["Engagement_Score"] = (
        df["Attendance Rate"] * 0.6 +
        df["Participation_Label"] * 0.4
    )

    # Performance Index
    df["Performance_Index"] = (
        df["Previous Grades"] * 0.7 +
        df["Study Hours per Week"] * 0.3
    )

    # Risk Score
    df["Risk_Score"] = (
        (100 - df["Attendance Rate"]) +
        (100 - df["Previous Grades"])
    )

    # ------------------------------------------------
    # CLUSTERING FEATURES
    # ------------------------------------------------

    clustering_features = [
        "Study Hours per Week",
        "Attendance Rate",
        "Previous Grades",
        "Engagement_Score",
        "Performance_Index",
        "Risk_Score"
    ]

    # ------------------------------------------------
    # FEATURE SCALING
    # ------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        df[clustering_features]
    )

    # ------------------------------------------------
    # KMEANS CLUSTERING
    # ------------------------------------------------

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    df['Cluster'] = kmeans.fit_predict(
        X_scaled
    )

    # ------------------------------------------------
    # SILHOUETTE SCORE
    # ------------------------------------------------

    score = silhouette_score(
        X_scaled,
        df['Cluster']
    )

    # ------------------------------------------------
    # PCA VISUALIZATION
    # ------------------------------------------------

    pca = PCA(n_components=2)

    pca_results = pca.fit_transform(
        X_scaled
    )

    df['PCA1'] = pca_results[:, 0]
    df['PCA2'] = pca_results[:, 1]

    # ------------------------------------------------
    # DYNAMIC CLUSTER LABELING
    # ------------------------------------------------

    cluster_means = df.groupby(
        "Cluster"
    )["Previous Grades"].mean().sort_values()

    cluster_mapping = {}

    cluster_mapping[
        cluster_means.index[0]
    ] = "Students At Risk"

    cluster_mapping[
        cluster_means.index[1]
    ] = "Average Students"

    cluster_mapping[
        cluster_means.index[2]
    ] = "High Achievers"

    df["Student_Segment"] = df[
        "Cluster"
    ].map(cluster_mapping)

    # Bubble size
    df['Size_Ref'] = (
        df['Study Hours per Week'].abs() + 2
    )

    return df, score

# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------

df, score = process_full_project_data()

# ----------------------------------------------------
# TITLE
# ----------------------------------------------------

st.title("🌌 Student Intelligence Canvas: Analytics & Mentorship")

st.markdown("""
AI-powered educational analytics system using:
- Unsupervised Learning
- K-Means Clustering
- PCA Visualization
- Generative AI
""")

# ----------------------------------------------------
# SEARCH SYSTEM
# ----------------------------------------------------

target_id = st.text_input(
    "🔍 Search Student ID for AI Recommendation",
    placeholder="S00001"
)

is_isolated = False
student_row = pd.DataFrame()

if target_id:

    student_row = df[
        df['Student ID'] == target_id
    ]

    if not student_row.empty:

        is_isolated = True

        row = student_row.iloc[0]

        with st.spinner("AI Mentor is analyzing student profile..."):

            prompt = f"""
            Student Academic Profile:

            Student ID:
            {target_id}

            Student Segment:
            {row['Student_Segment']}

            Study Hours:
            {row['Study Hours per Week']}

            Attendance:
            {row['Attendance Rate']}

            Previous Grades:
            {row['Previous Grades']}

            Generate:
            1. Performance analysis
            2. Improvement strategy
            3. Personalized academic recommendation

            Keep response concise and professional.
            """

            try:

                response = model.generate_content(
                    prompt
                )

                advice = response.text

                st.markdown(f"""
                <div class="ai-card">
                    <div class="ai-header">
                    💎 AI Recommendation for {target_id}
                    </div>

                    {advice}

                </div>
                """, unsafe_allow_html=True)

                # Download report

                report = student_row.copy()

                report['AI_Success_Plan'] = advice

                output_s = BytesIO()

                with pd.ExcelWriter(
                    output_s,
                    engine='xlsxwriter'
                ) as writer:

                    report.to_excel(
                        writer,
                        index=False
                    )

                st.download_button(
                    f"📥 Export {target_id} Report",
                    output_s.getvalue(),
                    f"Report_{target_id}.xlsx"
                )

            except:

                st.error("AI Service Unavailable.")

# ----------------------------------------------------
# KPI SECTION
# ----------------------------------------------------

st.write("")

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Dataset Size",
    len(df)
)

k2.metric(
    "Mean Academic Score",
    f"{df['Previous Grades'].mean():.1f}"
)

k3.metric(
    "Avg Attendance",
    f"{df['Attendance Rate'].mean():.1f}%"
)

k4.metric(
    "Silhouette Score",
    f"{score:.3f}"
)

st.markdown("---")

# ----------------------------------------------------
# CLUSTER ANALYSIS TABLE
# ----------------------------------------------------

st.subheader("📋 Cluster Center Analysis")

cluster_centers = df.groupby(
    'Student_Segment'
)[[
    'Study Hours per Week',
    'Attendance Rate',
    'Previous Grades',
    'Engagement_Score',
    'Performance_Index'
]].mean()

st.dataframe(
    cluster_centers.style
    .format("{:.2f}")
    .background_gradient(cmap='Blues'),

    use_container_width=True
)

# ----------------------------------------------------
# STUDENT LOG
# ----------------------------------------------------

st.subheader("📑 Student Dataset Log")

log_cols_frontend = [
    'Student ID',
    'Study Hours per Week',
    'Attendance Rate',
    'Previous Grades',
    'Participation in Extracurricular Activities',
    'Parent Education Level',
    'Passed'
]

st.dataframe(
    df[log_cols_frontend]
    .head(100)
    .style
    .background_gradient(
        cmap='Blues',
        subset=[
            'Previous Grades',
            'Attendance Rate'
        ]
    ),

    use_container_width=True
)

st.markdown("---")

# ----------------------------------------------------
# VISUALIZATION SECTION
# ----------------------------------------------------

st.subheader("📊 Performance Analytics Dashboard")

# ----------------------------------------------------
# ROW 1
# ----------------------------------------------------

r1_c1, r1_c2 = st.columns(2)

# PCA PLOT

with r1_c1:

    st.write("### 📍 PCA Cluster Projection")

    fig2 = px.scatter(
        df,
        x="PCA1",
        y="PCA2",
        color="Student_Segment",
        template="plotly_dark",
        opacity=0.2 if is_isolated else 0.8
    )

    if is_isolated:

        fig2.add_trace(
            go.Scatter(
                x=student_row['PCA1'],
                y=student_row['PCA2'],
                mode='markers',

                marker=dict(
                    color='white',
                    size=18,
                    symbol='star',
                    line=dict(
                        width=2,
                        color='cyan'
                    )
                ),

                name="Selected Student"
            )
        )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# ATTENDANCE VS GRADES

with r1_c2:

    st.write("### 📍 Attendance vs Grades")

    fig6 = px.scatter(
        df,
        x="Attendance Rate",
        y="Previous Grades",
        color="Student_Segment",
        size="Size_Ref",
        template="plotly_dark",
        opacity=0.2 if is_isolated else 0.7
    )

    if is_isolated:

        fig6.add_trace(
            go.Scatter(
                x=student_row['Attendance Rate'],
                y=student_row['Previous Grades'],
                mode='markers',

                marker=dict(
                    color='white',
                    size=18,
                    symbol='star',
                    line=dict(
                        width=2,
                        color='cyan'
                    )
                ),

                name="Selected Student"
            )
        )

    st.plotly_chart(
        fig6,
        use_container_width=True
    )

# ----------------------------------------------------
# ROW 2
# ----------------------------------------------------

r2_c1, r2_c2 = st.columns(2)

# BOXPLOT

with r2_c1:

    st.write("### 📍 Outlier Detection")

    fig3 = px.box(
        df,
        y=[
            "Study Hours per Week",
            "Previous Grades"
        ],
        template="plotly_dark",
        notched=True
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

# HEATMAP

with r2_c2:

    st.write("### 📍 Correlation Heatmap")

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(12,8))

    correlation = df.corr(numeric_only=True)

    sns.heatmap(
        correlation,
        annot=True,
        cmap='coolwarm',
        ax=ax
    )

    plt.title('Correlation Heatmap')

    st.pyplot(fig)

# ----------------------------------------------------
# ROW 3
# ----------------------------------------------------

st.write("### 📍 Segment Performance vs Outcome")

fig5 = px.histogram(
    df,
    x="Student_Segment",
    color="Passed",
    barmode="group",
    template="plotly_dark"
)

st.plotly_chart(
    fig5,
    use_container_width=True
)

# ----------------------------------------------------
# ADMIN SECTION
# ----------------------------------------------------

st.sidebar.title("👨‍🏫 Teacher Administration")

if st.sidebar.button("📦 Build Master Data"):

    output_t = BytesIO()

    with pd.ExcelWriter(
        output_t,
        engine='xlsxwriter'
    ) as writer:

        df.to_excel(
            writer,
            index=False
        )

    st.sidebar.download_button(
        "📥 Download Master Dataset",
        output_t.getvalue(),
        "Teacher_Master_List.xlsx"
    )