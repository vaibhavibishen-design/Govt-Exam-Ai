

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(
    page_title="Govt Exam AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
df = load_data()

try:
    model = joblib.load("topic_priority_model.pkl")

    exam_encoder = joblib.load("exam_encoder.pkl")
    subject_encoder = joblib.load("subject_encoder.pkl")
    topic_encoder = joblib.load("topic_encoder.pkl")
    priority_encoder = joblib.load("priority_encoder.pkl")

except:
    model = None

# -----------------------------
st.sidebar.title("📚 Govt Exam AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dashboard",
        "📚 PYQ Analysis",
         "📈 Trend Analysis",
        "🤖 AI Prediction",
         "📅 Study Planner",
        
        
        "ℹ About"
    ]
)

# ======================================
# HOme

if page=="🏠 Home":

    st.title("🇮🇳 Govt Exam AI")

    st.markdown("""
### AI Powered Government Exam Question Analysis

Analyze Previous Year Questions using Data Science & Machine Learning.
""")

    c1,c2,c3,c4=st.columns(4)

    c1.metric("Questions",len(df))
    c2.metric("Exams",df["exam"].nunique())
    c3.metric("Subjects",df["subject"].nunique())
    c4.metric("Topics",df["topic"].nunique())

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(df.head(10),use_container_width=True)


# ======================================
# DASHBOARD
# ======================================
elif page=="📊 Dashboard":

    st.title("Dashboard")

    exam=df["exam"].value_counts()

    fig=px.bar(
        exam,
        x=exam.index,
        y=exam.values,
        title="Questions by Exam"
    )

    st.plotly_chart(fig,use_container_width=True)

    subject=df["subject"].value_counts()

    fig2=px.pie(
        names=subject.index,
        values=subject.values,
        title="Subject Distribution"
    )

    st.plotly_chart(fig2,use_container_width=True)

    year=df["year"].value_counts().sort_index()

    fig3=px.line(
        x=year.index,
        y=year.values,
        markers=True,
        title="Year Wise Questions"
    )

    st.plotly_chart(fig3,use_container_width=True)


# ======================================
# PYQ Analysis
# ======================================

elif page == "📚 PYQ Analysis":

    st.title("📚 Previous Year Question Analysis")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_exam = st.selectbox(
            "Select Exam",
            sorted(df["exam"].unique())
        )

    with col2:
        subjects = sorted(
            df[df["exam"] == selected_exam]["subject"].unique()
        )

        selected_subject = st.selectbox(
            "Select Subject",
            subjects
        )

    with col3:
        topics = sorted(
            df[
                (df["exam"] == selected_exam) &
                (df["subject"] == selected_subject)
            ]["topic"].unique()
        )

        selected_topic = st.selectbox(
            "Select Topic",
            topics
        )

    # Filter Dataset
    filtered_df = df[
        (df["exam"] == selected_exam) &
        (df["subject"] == selected_subject) &
        (df["topic"] == selected_topic)
    ]

    st.subheader("Filtered Questions")

    st.write(f"Total Questions: {len(filtered_df)}")

    st.dataframe(
        filtered_df,
        use_container_width=True
    )

    # Download CSV
    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Filtered Questions",
        data=csv,
        file_name="filtered_questions.csv",
        mime="text/csv"
    )
    
elif page == "📈 Trend Analysis":

    st.title("📈 Topic Trend Analysis")

    exam = st.selectbox(
        "Select Exam",
        sorted(df["exam"].unique())
    )

    subject = st.selectbox(
        "Select Subject",
        sorted(
            df[df["exam"] == exam]["subject"].unique()
        )
    )

    topic = st.selectbox(
        "Select Topic",
        sorted(
            df[
                (df["exam"] == exam) &
                (df["subject"] == subject)
            ]["topic"].unique()
        )
    )

    trend = df[
        (df["exam"] == exam) &
        (df["subject"] == subject) &
        (df["topic"] == topic)
    ]

    trend = trend.groupby("year").size().reset_index(name="Questions")

    fig = px.line(
        trend,
        x="year",
        y="Questions",
        markers=True,
        title=f"{topic} Trend Over Years"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(trend, use_container_width=True)
# ======================================
elif page == "📈 Trend Analysis":

    st.title("📈 Topic Trend Analysis")

    exam = st.selectbox(
        "Select Exam",
        sorted(df["exam"].unique())
    )
    
    subject = st.selectbox(
        "Subject",
        sorted(df[df["exam"] == exam]["subject"].unique())
    )

    topic = st.selectbox(
        "Topic",
        sorted(
            df[
                (df["exam"] == exam) &
                (df["subject"] == subject)
            ]["topic"].unique()
        )
    )

    if st.button("Predict"):

        freq = len(
            df[
                (df["exam"] == exam) &
                (df["subject"] == subject) &
                (df["topic"] == topic)
            ]
        )

        latest = df[
            (df["exam"] == exam) &
            (df["subject"] == subject) &
            (df["topic"] == topic)
        ]["year"].max()

        exam_value = exam_encoder.transform([exam])[0]
        subject_value = subject_encoder.transform([subject])[0]
        topic_value = topic_encoder.transform([topic])[0]

        prediction = model.predict([[
            exam_value,
            subject_value,
            topic_value,
            freq,
            latest
        ]])

        result = priority_encoder.inverse_transform(prediction)[0]

        st.success(f"Priority : {result}")

        st.metric(
            "Question Frequency",
            freq
        )

        st.metric(
            "Latest Year",
            latest
        )

elif page == "📅 Study Planner":

    st.title("📅 AI Study Planner")

    exam = st.selectbox(
        "Select Exam",
        sorted(df["exam"].unique())
    )

    days = st.slider(
        "Days Remaining",
        7,
        365,
        60
    )

    hours = st.slider(
        "Study Hours Per Day",
        1,
        12,
        4
    )

    if st.button("Generate Study Plan"):

        topics = (
            df[df["exam"] == exam]["topic"]
            .value_counts()
            .reset_index()
        )

        topics.columns = ["Topic", "Frequency"]

        total_topics = len(topics)

        topics_per_day = max(
            1,
            total_topics // days
        )

        st.success("Study Plan Generated")

        st.write(
            f"Study approximately **{topics_per_day} topic(s)** every day."
        )

        st.subheader("High Priority Topics")

        st.dataframe(
            topics.head(10),
            use_container_width=True
        )

        st.info(
            f"""
Exam : {exam}

Days Left : {days}

Daily Study Time : {hours} Hours

Recommended Daily Topics : {topics_per_day}
"""
        )

elif page=="ℹ About":

    st.title("About")

    st.info("""
Govt Exam AI

Machine Learning Project

Built using

Python

Pandas

Scikit Learn

Streamlit

Plotly
""")


