"""
FRAUD DETECTION SYSTEM SIMULATOR
=================================
Interactive simulator showing:
- How each algorithm detects fraud
- Which features are used and why
- How scores are calculated
- How RL retraining works step-by-step

Run: streamlit run fraud_simulator.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Fraud Detection Simulator",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .algo-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .feature-box {
        background: #e7f5ff;
        border-left: 4px solid #1971c2;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    .score-box {
        background: #fff3e0;
        border-left: 4px solid #f57c00;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    .rule-box {
        background: #ffe0e0;
        border-left: 4px solid #ff6b6b;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    .ml-box {
        background: #f0f9ff;
        border-left: 4px solid #0284c7;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    .feedback-box {
        background: #dcfce7;
        border-left: 4px solid #22c55e;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    .step-number {
        display: inline-block;
        background: #667eea;
        color: white;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        text-align: center;
        line-height: 30px;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    .metric-highlight {
        background: #fef3c7;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.title("🎮 Fraud Detection Simulator")
page = st.sidebar.radio(
    "Select Demo",
    [
        "🏠 Introduction",
        "📊 Step 1: Data Generation",
        "🎯 Step 2: Rule-Based Detection",
        "🧠 Step 3: ML Feature Extraction",
        "📈 Step 4: Anomaly Detection",
        "🎲 Step 5: Supervised Learning",
        "🔀 Step 6: Ensemble Scoring",
        "👤 Step 7: Human Review",
        "🔄 Step 8: RL Retraining",
        "📊 Complete Workflow"
    ]
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def generate_normal_worker_data(minutes=100):
    """Generate normal worker behavior"""
    np.random.seed(42)
    clicks = np.random.normal(30, 5, minutes)
    keystrokes = np.random.normal(60, 10, minutes)
    scrolls = np.random.normal(10, 3, minutes)
    return np.clip(clicks, 0, 100), np.clip(keystrokes, 0, 150), np.clip(scrolls, 0, 30)

def generate_fraudulent_worker_data(minutes=100, fraud_type='robotic'):
    """Generate fraudulent worker behavior"""
    np.random.seed(123)
    
    if fraud_type == 'robotic':
        # Same values for 8+ minutes
        clicks = np.array([50] * 50 + list(np.random.normal(30, 5, 50)))
        keystrokes = np.array([100] * 50 + list(np.random.normal(60, 10, 50)))
        scrolls = np.array([15] * 50 + list(np.random.normal(10, 3, 50)))
    
    elif fraud_type == 'superhuman':
        # Impossible speeds
        clicks = np.random.normal(180, 20, minutes)
        keystrokes = np.random.normal(280, 20, minutes)
        scrolls = np.random.normal(5, 2, minutes)
    
    elif fraud_type == 'nobreak':
        # Continuous work, no breaks
        clicks = np.random.normal(50, 5, minutes)
        keystrokes = np.random.normal(80, 10, minutes)
        scrolls = np.random.normal(12, 3, minutes)
    
    return np.clip(clicks, 0, 200), np.clip(keystrokes, 0, 300), np.clip(scrolls, 0, 50)

def extract_features(clicks, keystrokes, scrolls):
    """Extract all features used by ML models"""
    features = {
        'mean_clicks': np.mean(clicks),
        'std_clicks': np.std(clicks),
        'max_clicks': np.max(clicks),
        'cv_clicks': np.std(clicks) / (np.mean(clicks) + 1e-5),
        
        'mean_keystrokes': np.mean(keystrokes),
        'std_keystrokes': np.std(keystrokes),
        'max_keystrokes': np.max(keystrokes),
        'cv_keystrokes': np.std(keystrokes) / (np.mean(keystrokes) + 1e-5),
        
        'mean_scrolls': np.mean(scrolls),
        'std_scrolls': np.std(scrolls),
        'max_scrolls': np.max(scrolls),
        'cv_scrolls': np.std(scrolls) / (np.mean(scrolls) + 1e-5),
        
        'superhuman_clicks_pct': (clicks > 150).sum() / len(clicks) * 100,
        'superhuman_keystrokes_pct': (keystrokes > 250).sum() / len(keystrokes) * 100,
        'zero_activity_pct': ((clicks == 0) & (keystrokes == 0) & (scrolls == 0)).sum() / len(clicks) * 100,
    }
    return features

def detect_rules(clicks, keystrokes, scrolls):
    """Detect fraud rules"""
    rules = []
    score = 0
    
    # Rule 1: Robotic pattern
    max_streak = 1
    current_streak = 1
    for i in range(1, len(clicks)):
        if clicks[i] == clicks[i-1] and clicks[i] > 0:
            current_streak += 1
        else:
            current_streak = 1
        max_streak = max(max_streak, current_streak)
    
    if max_streak >= 9:
        rules.append(('ROBOTIC_9MIN', 35, f'{max_streak} min identical activity'))
        score += 35
    elif max_streak >= 8:
        rules.append(('ROBOTIC_8MIN', 25, f'{max_streak} min identical activity'))
        score += 25
    
    # Rule 2: Superhuman clicks
    superhuman_clicks = (clicks > 150).sum()
    if superhuman_clicks > len(clicks) * 0.2:
        rules.append(('SUPERHUMAN_CLICKS', 15, f'{superhuman_clicks} instances > 150/min'))
        score += 15
    
    # Rule 3: Superhuman keystrokes
    superhuman_ks = (keystrokes > 250).sum()
    if superhuman_ks > len(keystrokes) * 0.15:
        rules.append(('SUPERHUMAN_KEYSTROKES', 20, f'{superhuman_ks} instances > 250/min'))
        score += 20
    
    # Rule 4: No breaks
    zero_activity = ((clicks == 0) & (keystrokes == 0) & (scrolls == 0)).sum()
    if zero_activity < len(clicks) * 0.01:
        rules.append(('NO_BREAKS', 20, f'Only {zero_activity/len(clicks)*100:.1f}% break time'))
        score += 20
    
    # Rule 5: Zero variance
    if np.std(clicks) < 0.5 and np.mean(clicks) > 20:
        rules.append(('ZERO_VARIANCE', 30, 'Impossible consistency'))
        score += 30
    
    return rules, min(100, score)

def calculate_ml_score(features):
    """Simulate ML anomaly detection"""
    # Isolation Forest logic: detect anomalies based on feature patterns
    anomaly_score = 0
    
    # High variance = normal
    cv_score = features['cv_clicks'] + features['cv_keystrokes']
    if cv_score < 0.5:
        anomaly_score += 20
    
    # Superhuman speeds
    if features['superhuman_clicks_pct'] > 10:
        anomaly_score += 25
    if features['superhuman_keystrokes_pct'] > 10:
        anomaly_score += 25
    
    # Consistency patterns
    if features['std_clicks'] < 1:
        anomaly_score += 20
    if features['std_keystrokes'] < 1:
        anomaly_score += 20
    
    # Zero activity
    if features['zero_activity_pct'] > 30:
        anomaly_score += 15
    
    return min(100, anomaly_score)

# ============================================================================
# PAGE 1: INTRODUCTION
# ============================================================================
if page == "🏠 Introduction":
    st.title("🎮 Fraud Detection System Simulator")
    
    st.markdown("""
    Welcome to the **Interactive Fraud Detection Simulator**! 
    
    This simulator walks you through every step of the fraud detection system:
    
    ### 📚 What You'll Learn:
    
    1. **Step 1: Data Generation** - How worker activity data is created
    2. **Step 2: Rule-Based Detection** - Expert rules that detect obvious fraud
    3. **Step 3: ML Feature Extraction** - 50+ features extracted from activity
    4. **Step 4: Anomaly Detection** - Isolation Forest finds unusual patterns
    5. **Step 5: Supervised Learning** - Training on human feedback
    6. **Step 6: Ensemble Scoring** - Combining multiple models
    7. **Step 7: Human Review** - You verify if the prediction is correct
    8. **Step 8: RL Retraining** - How the system learns from feedback
    9. **Complete Workflow** - See the entire process end-to-end
    
    ### 🎯 Key Concepts:
    
    **Rule-Based Detection (40% weight)**
    - Detects obvious fraud patterns
    - Uses expert rules (robotic, superhuman, no breaks)
    - Fast and explainable
    - Score: 0-100 based on rules triggered
    
    **Machine Learning (60% weight)**
    - Isolation Forest: Finds anomalies
    - Supervised Models: Learns from feedback
    - Complex pattern detection
    - Adaptive and improves over time
    
    **Ensemble Approach**
    - Combines rule-based + ML
    - Final Score = 0.4 × Rules + 0.6 × ML
    - More robust than either alone
    
    **Reinforcement Learning Loop**
    - Collect human feedback
    - Retrain models
    - Update thresholds
    - System adapts to your fraud patterns
    
    ### 🚀 Let's Get Started!
    
    Select any step from the sidebar to see how it works in detail.
    """)
    
    # Display system architecture
    st.subheader("🏗️ System Architecture")
    
    architecture = """
    ┌─────────────────────────────────────────────┐
    │         Raw Worker Activity Data            │
    │    (Clicks, Keystrokes, Scrolls)            │
    └──────────────────┬──────────────────────────┘
                       │
           ┌───────────┴────────────┐
           │                        │
    ┌──────▼──────────┐    ┌───────▼──────────┐
    │  Rule-Based     │    │  ML Pipeline     │
    │  Detection      │    │  - Feature Eng.  │
    │  - 10+ Rules    │    │  - Isolation F.  │
    │  - Score: 0-100 │    │  - Supervised    │
    └──────┬──────────┘    └───────┬──────────┘
           │                       │
           │    ┌──────────────────┘
           │    │
    ┌──────▼────▼──────────┐
    │  Ensemble Scorer     │
    │  Final Score: 0-100  │
    └──────┬───────────────┘
           │
    ┌──────▼──────────────┐
    │  Human Review       │
    │  Feedback Collect   │
    └──────┬──────────────┘
           │
    ┌──────▼──────────────┐
    │  RL Retraining      │
    │  Improve Model      │
    └─────────────────────┘
    """
    
    st.code(architecture)

# ============================================================================
# PAGE 2: STEP 1 - DATA GENERATION
# ============================================================================
elif page == "📊 Step 1: Data Generation":
    st.title("📊 Step 1: Data Generation")
    
    st.markdown("""
    The fraud detection system starts with **worker activity data**.
    
    Each minute of work generates:
    - **Clicks Per Minute (CPM)** - Mouse clicks
    - **Keystrokes Per Minute (KPM)** - Keyboard activity  
    - **Scrolls Per Minute (SPM)** - Page scrolls
    
    Let's generate and compare **Normal vs Fraudulent** worker behavior.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Normal Worker")
        clicks_n, ks_n, scrolls_n = generate_normal_worker_data(120)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=clicks_n, name='Clicks', line=dict(color='#51cf66', width=2)))
        fig.add_trace(go.Scatter(y=ks_n/5, name='Keystrokes (÷5)', line=dict(color='#667eea', width=2)))
        fig.add_trace(go.Scatter(y=scrolls_n, name='Scrolls', line=dict(color='#ffa94d', width=2)))
        
        fig.update_layout(
            title="Activity Pattern (120 minutes)",
            xaxis_title="Minute",
            yaxis_title="Activity Count",
            height=350,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="feature-box">
        <strong>✅ Normal Characteristics:</strong>
        - Varies naturally throughout day
        - No long stretches of identical activity
        - Human-like pattern with ups and downs
        - Reasonable break periods
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Fraudulent Worker (Robotic)")
        fraud_type = st.selectbox("Fraud Type", ['robotic', 'superhuman', 'nobreak'])
        
        clicks_f, ks_f, scrolls_f = generate_fraudulent_worker_data(120, fraud_type)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=clicks_f, name='Clicks', line=dict(color='#ff6b6b', width=2)))
        fig.add_trace(go.Scatter(y=ks_f/5, name='Keystrokes (÷5)', line=dict(color='#ff6b6b', width=2, dash='dash')))
        fig.add_trace(go.Scatter(y=scrolls_f, name='Scrolls', line=dict(color='#ff6b6b', width=2, dash='dot')))
        
        fig.update_layout(
            title=f"Activity Pattern ({fraud_type.upper()}) - 120 minutes",
            xaxis_title="Minute",
            yaxis_title="Activity Count",
            height=350,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if fraud_type == 'robotic':
            st.markdown("""
            <div class="rule-box">
            <strong>🚨 Robotic Characteristics:</strong>
            - IDENTICAL values for 8-9+ minutes (flat line)
            - Biologically impossible consistency
            - Suddenly returns to normal (script restart)
            - Red flag: NO human produces exact same values
            </div>
            """, unsafe_allow_html=True)
        elif fraud_type == 'superhuman':
            st.markdown("""
            <div class="rule-box">
            <strong>⚡ Superhuman Characteristics:</strong>
            - Click rate: ~180/min (humans: ~50-100)
            - Keystroke rate: ~280/min (humans: ~80-150)
            - Sustained at impossible speeds
            - Red flag: Biologically impossible for any human
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="rule-box">
            <strong>⏱️ No-Break Characteristics:</strong>
            - Continuous activity for entire 120 minutes
            - No rest periods (humans need breaks)
            - Sustained energy levels
            - Red flag: Humans need bathroom breaks, water, rest
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # Statistics comparison
    st.subheader("📊 Statistical Comparison")
    
    stats = pd.DataFrame({
        'Metric': ['Mean Clicks', 'Std Dev Clicks', 'Max Clicks', 'Min Clicks', 'CV (Variation)'],
        'Normal': [
            f'{np.mean(clicks_n):.1f}',
            f'{np.std(clicks_n):.1f}',
            f'{np.max(clicks_n):.1f}',
            f'{np.min(clicks_n):.1f}',
            f'{np.std(clicks_n)/np.mean(clicks_n):.3f}'
        ],
        'Fraudulent': [
            f'{np.mean(clicks_f):.1f}',
            f'{np.std(clicks_f):.1f}',
            f'{np.max(clicks_f):.1f}',
            f'{np.min(clicks_f):.1f}',
            f'{np.std(clicks_f)/np.mean(clicks_f):.3f}'
        ]
    })
    
    st.dataframe(stats, use_container_width=True)

# ============================================================================
# PAGE 3: STEP 2 - RULE-BASED DETECTION
# ============================================================================
elif page == "🎯 Step 2: Rule-Based Detection":
    st.title("🎯 Step 2: Rule-Based Detection")
    
    st.markdown("""
    **Rule-Based Detection** uses expert rules to catch obvious fraud patterns.
    
    Think of it as a **checklist of fraud indicators**:
    - ❌ Are clicks identical for 8+ minutes?
    - ❌ Are speeds superhuman (>150 clicks/min)?
    - ❌ Are there zero breaks for hours?
    - ❌ Is activity too consistent?
    """)
    
    # Generate sample workers
    clicks_n, ks_n, scrolls_n = generate_normal_worker_data(100)
    clicks_f, ks_f, scrolls_f = generate_fraudulent_worker_data(100, 'robotic')
    
    col1, col2 = st.columns(2)
    
    # NORMAL WORKER
    with col1:
        st.subheader("✅ Normal Worker Analysis")
        
        rules_n, score_n = detect_rules(clicks_n, ks_n, scrolls_n)
        
        st.metric("Rule-Based Score", f"{score_n:.0f}/100", "Low Risk")
        
        if rules_n:
            st.write("**Triggered Rules:**")
            for rule_name, points, description in rules_n:
                st.markdown(f"""
                <div class="rule-box">
                <strong>{rule_name}</strong> (+{points} pts)<br>
                {description}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="feedback-box">
            ✅ <strong>No rules triggered!</strong><br>
            This worker's behavior is consistent with human activity.
            </div>
            """, unsafe_allow_html=True)
    
    # FRAUDULENT WORKER
    with col2:
        st.subheader("🚨 Fraudulent Worker Analysis")
        
        rules_f, score_f = detect_rules(clicks_f, ks_f, scrolls_f)
        
        st.metric("Rule-Based Score", f"{score_f:.0f}/100", "🔴 HIGH RISK")
        
        if rules_f:
            st.write("**Triggered Rules:**")
            for rule_name, points, description in rules_f:
                st.markdown(f"""
                <div class="rule-box">
                <strong>{rule_name}</strong> (+{points} pts)<br>
                {description}
                </div>
                """, unsafe_allow_html=True)
    
    st.divider()
    
    # Rule weights explanation
    st.subheader("⚖️ Rule Weight System")
    
    rule_weights = pd.DataFrame({
        'Rule': [
            'ROBOTIC_9MIN',
            'ROBOTIC_8MIN',
            'SUPERHUMAN_KEYSTROKES',
            'ZERO_VARIANCE',
            'LONG_STREAK_8H',
            'SUPERHUMAN_CLICKS',
            'LONG_STREAK_5H',
            'NO_BREAKS',
            'LONG_STREAK_4H',
            'PERFECT_CONSISTENCY'
        ],
        'Points': [35, 25, 20, 30, 40, 15, 25, 20, 15, 20],
        'Severity': ['🔴 Critical', '🔴 Critical', '🟠 High', '🔴 Critical', '🔴 Critical', 
                     '🟡 Medium', '🟠 High', '🟠 High', '🟡 Medium', '🟠 High'],
        'What It Detects': [
            'Identical activity for 9+ consecutive minutes',
            'Identical activity for 8+ consecutive minutes',
            'Superhuman keystroke speed (>250/min)',
            'Perfectly consistent unchanging patterns',
            '8+ hours continuous work without breaks',
            'Superhuman click speed (>150/min)',
            '5+ hours continuous work without breaks',
            'Less than 1% break time entire session',
            '4+ hours continuous work without breaks',
            'Only 1-2 unique values in activity'
        ]
    })
    
    st.dataframe(rule_weights, use_container_width=True)
    
    st.markdown("""
    ### 💡 How Scoring Works:
    
    1. System checks all 10 rules
    2. For each triggered rule, add its points
    3. Cap at 100 (no overflow)
    4. Result = **Rule-Based Fraud Score (0-100)**
    
    **Key Insight:** Rules are explainable - you can see exactly WHY a worker was flagged!
    """)

# ============================================================================
# PAGE 4: STEP 3 - FEATURE EXTRACTION
# ============================================================================
elif page == "🧠 Step 3: ML Feature Extraction":
    st.title("🧠 Step 3: ML Feature Extraction")
    
    st.markdown("""
    Before ML models can work, we need to convert raw activity data into **features**.
    
    **Features** are patterns and statistics that the model learns from.
    Think of features as **"fingerprints" of behavior**.
    """)
    
    # Generate sample data
    clicks_n, ks_n, scrolls_n = generate_normal_worker_data(100)
    clicks_f, ks_f, scrolls_f = generate_fraudulent_worker_data(100, 'robotic')
    
    features_normal = extract_features(clicks_n, ks_n, scrolls_n)
    features_fraud = extract_features(clicks_f, ks_f, scrolls_f)
    
    col1, col2 = st.columns(2)
    
    # Normal worker features
    with col1:
        st.subheader("✅ Normal Worker Features")
        
        st.markdown("""
        <div class="feature-box">
        <strong>📊 Statistical Features:</strong>
        </div>
        """, unsafe_allow_html=True)
        
        for key in ['mean_clicks', 'std_clicks', 'cv_clicks', 'mean_keystrokes', 'std_keystrokes']:
            value = features_normal[key]
            st.metric(key.replace('_', ' ').title(), f"{value:.2f}")
    
    # Fraudulent worker features
    with col2:
        st.subheader("🚨 Fraudulent Worker Features")
        
        st.markdown("""
        <div class="feature-box">
        <strong>📊 Statistical Features:</strong>
        </div>
        """, unsafe_allow_html=True)
        
        for key in ['mean_clicks', 'std_clicks', 'cv_clicks', 'mean_keystrokes', 'std_keystrokes']:
            value = features_fraud[key]
            highlight = "🚨 ANOMALY" if (
                (key == 'std_clicks' and value < 1) or 
                (key == 'std_keystrokes' and value < 1) or
                (key == 'cv_clicks' and value < 0.05)
            ) else ""
            st.metric(key.replace('_', ' ').title(), f"{value:.2f}", highlight)
    
    st.divider()
    
    # Feature comparison
    st.subheader("📊 Feature Comparison")
    
    feature_comparison = pd.DataFrame({
        'Feature': list(features_normal.keys()),
        'Normal Worker': [f"{v:.2f}" for v in features_normal.values()],
        'Fraudulent Worker': [f"{v:.2f}" for v in features_fraud.values()],
        'Difference': [f"{abs(features_fraud[k] - features_normal[k]):.2f}" 
                      for k in features_normal.keys()]
    })
    
    st.dataframe(feature_comparison, use_container_width=True)
    
    # Feature importance visualization
    st.subheader("🎯 Most Important Features for Fraud Detection")
    
    feature_names = list(features_normal.keys())
    normal_vals = list(features_normal.values())
    fraud_vals = list(features_fraud.values())
    
    differences = [abs(f - n) for f, n in zip(fraud_vals, normal_vals)]
    
    # Top features
    top_indices = sorted(range(len(differences)), key=lambda i: differences[i], reverse=True)[:8]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Normal Worker',
        x=[feature_names[i] for i in top_indices],
        y=[normal_vals[i] for i in top_indices],
        marker_color='#51cf66'
    ))
    
    fig.add_trace(go.Bar(
        name='Fraudulent Worker',
        x=[feature_names[i] for i in top_indices],
        y=[fraud_vals[i] for i in top_indices],
        marker_color='#ff6b6b'
    ))
    
    fig.update_layout(
        title="Top 8 Features - Normal vs Fraudulent",
        xaxis_title="Features",
        yaxis_title="Value",
        height=400,
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### 💡 Key Feature Insights:
    
    | Feature | What It Means | Why It Matters |
    |---------|---------------|----------------|
    | **std_clicks** | Variation in clicks | Low = robotic, High = human |
    | **cv_clicks** | Coefficient of variation | Measures consistency |
    | **superhuman_clicks_pct** | % of time > 150 clicks/min | Superhuman speed |
    | **superhuman_keystrokes_pct** | % of time > 250 keystrokes/min | Impossible speed |
    | **zero_activity_pct** | % with no activity | Should have breaks |
    | **mean_clicks** | Average clicks per minute | Baseline activity |
    | **max_clicks** | Peak click rate | Maximum capability |
    | **mean_keystrokes** | Average keystrokes per minute | Baseline typing |
    
    **50+ features** are extracted in total, giving ML models rich information to learn from!
    """)

# ============================================================================
# PAGE 5: STEP 4 - ANOMALY DETECTION
# ============================================================================
elif page == "📈 Step 4: Anomaly Detection":
    st.title("📈 Step 4: Anomaly Detection (Isolation Forest)")
    
    st.markdown("""
    **Isolation Forest** is an unsupervised ML algorithm that finds **anomalies**.
    
    ### How It Works:
    
    1. **Learns normal patterns** from data
    2. **Finds outliers** (unusual patterns)
    3. **Assigns anomaly score** (0-100)
       - 0 = Normal (similar to training data)
       - 100 = Extreme anomaly (very different)
    """)
    
    # Generate sample data
    clicks_n, ks_n, scrolls_n = generate_normal_worker_data(100)
    clicks_f, ks_f, scrolls_f = generate_fraudulent_worker_data(100, 'robotic')
    
    features_normal = extract_features(clicks_n, ks_n, scrolls_n)
    features_fraud = extract_features(clicks_f, ks_f, scrolls_f)
    
    ml_score_normal = calculate_ml_score(features_normal)
    ml_score_fraud = calculate_ml_score(features_fraud)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✅ Normal Worker")
        st.metric("Anomaly Score", f"{ml_score_normal:.1f}/100", "✅ Low Risk")
        
        st.markdown("""
        <div class="ml-box">
        <strong>Why Low Anomaly?</strong><br>
        - Feature patterns match typical workers
        - Natural variation in activity
        - No extreme outliers
        - Model sees this as "normal"
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("🚨 Fraudulent Worker")
        st.metric("Anomaly Score", f"{ml_score_fraud:.1f}/100", "🚨 High Risk")
        
        st.markdown("""
        <div class="ml-box">
        <strong>Why High Anomaly?</strong><br>
        - Feature patterns don't match normal workers
        - Unusual low variation (too consistent)
        - Superhuman activity levels
        - Model sees this as "abnormal"
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Visualization
    st.subheader("🎯 How Isolation Forest Works")
    
    st.markdown("""
    **Step 1: Build Decision Trees**
    - Randomly split features
    - Create hierarchical tree structure
    - Normal points need many splits to isolate
    - Anomalies need few splits to isolate
    
    **Step 2: Count Isolation Path**
    - Short path = Anomaly (isolated quickly)
    - Long path = Normal (hard to isolate)
    
    **Step 3: Calculate Score**
    - Average across multiple trees
    - Convert to 0-100 scale
    """)
    
    # Show how anomaly score is calculated
    st.subheader("📊 Anomaly Score Calculation")
    
    score_breakdown = pd.DataFrame({
        'Feature Group': [
            'Variation (CV)',
            'Superhuman Activity',
            'Consistency Patterns',
            'Activity Gaps',
            'Total'
        ],
        'Normal Worker': ['✅ +0 pts', '✅ +0 pts', '✅ +0 pts', '✅ +0 pts', '✅ 0 pts'],
        'Fraudulent Worker': ['🚨 +20 pts', '🚨 +50 pts', '🚨 +20 pts', '🚨 +15 pts', '🚨 105 pts → 100 max'],
        'Why': [
            'Low CV = robotic',
            'High superhuman % = bot activity',
            'Perfect consistency = impossible',
            'No breaks = not human',
            'Capped at 100'
        ]
    })
    
    st.dataframe(score_breakdown, use_container_width=True)

# ============================================================================
# PAGE 6: STEP 5 - SUPERVISED LEARNING
# ============================================================================
elif page == "🎲 Step 5: Supervised Learning":
    st.title("🎲 Step 5: Supervised Learning (Training on Feedback)")
    
    st.markdown("""
    **Supervised Learning** means the model learns from **labeled examples**.
    
    When you mark a worker as FRAUD or LEGITIMATE, you're creating training data!
    
    ### How It Works:
    
    1. **Collect Feedback** - You review cases and mark verdicts
    2. **Extract Features** - Calculate features for each reviewed case
    3. **Train Model** - Teach model which features indicate fraud
    4. **Get Predictions** - Model uses learned patterns for new cases
    """)
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("📝 Sample Feedback Data")
        
        # Simulate feedback dataset
        feedback_data = {
            'Email': ['worker_1@co', 'worker_2@co', 'worker_3@co', 'worker_4@co', 'worker_5@co'],
            'Rule Score': [5, 92, 15, 85, 10],
            'ML Score': [8, 88, 12, 90, 5],
            'Human Verdict': ['LEGIT', 'FRAUD', 'LEGIT', 'FRAUD', 'LEGIT'],
            'Used for Training': ['✅', '✅', '✅', '✅', '✅']
        }
        
        feedback_df = pd.DataFrame(feedback_data)
        st.dataframe(feedback_df, use_container_width=True)
    
    with col2:
        st.subheader("🧠 Model Learning Process")
        
        st.markdown("""
        <div class="feedback-box">
        <strong><span class="step-number">1</span> Collect Feedback</strong><br>
        You review 50 cases and mark verdicts
        </div>
        
        <div class="feedback-box">
        <strong><span class="step-number">2</span> Extract Features</strong><br>
        Calculate 50+ features for each case:
        - Mean clicks, keystrokes, scrolls
        - Variation (std dev)
        - Superhuman percentages
        - Zero activity gaps
        </div>
        
        <div class="feedback-box">
        <strong><span class="step-number">3</span> Create Training Set</strong><br>
        Features → Labels mapping
        [50 features] → [FRAUD or LEGIT]
        </div>
        
        <div class="feedback-box">
        <strong><span class="step-number">4</span> Train Models</strong><br>
        Random Forest learns patterns:
        "When std_dev is low AND superhuman % high → FRAUD"
        "When variation is high → LEGITIMATE"
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("📊 What Model Learns")
    
    st.markdown("""
    ### Feature Importance (What the model learned)
    """)
    
    importance_data = pd.DataFrame({
        'Feature': [
            'superhuman_clicks_pct',
            'std_keystrokes',
            'cv_clicks',
            'zero_activity_pct',
            'std_scrolls',
            'max_keystrokes',
            'mean_clicks'
        ],
        'Importance': [0.18, 0.15, 0.14, 0.12, 0.10, 0.18, 0.13],
        'What It Means': [
            'Time spent at superhuman click speeds',
            'Consistency of keystroke patterns',
            'Coefficient of variation (changes)',
            'Percentage with zero activity',
            'Variation in scroll behavior',
            'Maximum keystrokes in single minute',
            'Average clicks per minute'
        ]
    })
    
    importance_data = importance_data.sort_values('Importance', ascending=True)
    
    fig = px.barh(
        importance_data,
        x='Importance',
        y='Feature',
        color='Importance',
        color_continuous_scale='Blues',
        title="Random Forest Feature Importance",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### 💡 Key Insights:
    
    - **Superhuman metrics** are most important (18% weight each)
    - **Variation patterns** matter (low variation = fraud)
    - **Combined features** work better than single features
    - Model learns which combinations predict fraud best
    """)

# ============================================================================
# PAGE 7: STEP 6 - ENSEMBLE SCORING
# ============================================================================
elif page == "🔀 Step 6: Ensemble Scoring":
    st.title("🔀 Step 6: Ensemble Scoring (Combining All Models)")
    
    st.markdown("""
    **Ensemble** = Combining multiple models into one super-model!
    
    ### Why Ensemble?
    - Rules catch **obvious** fraud
    - ML catches **subtle** anomalies
    - Together = **comprehensive** detection
    - More robust than any single approach
    """)
    
    # Generate sample data
    clicks_n, ks_n, scrolls_n = generate_normal_worker_data(100)
    clicks_f, ks_f, scrolls_f = generate_fraudulent_worker_data(100, 'robotic')
    
    features_normal = extract_features(clicks_n, ks_n, scrolls_n)
    features_fraud = extract_features(clicks_f, ks_f, scrolls_f)
    
    rules_n, rule_score_n = detect_rules(clicks_n, ks_n, scrolls_n)
    rules_f, rule_score_f = detect_rules(clicks_f, ks_f, scrolls_f)
    
    ml_score_n = calculate_ml_score(features_normal)
    ml_score_f = calculate_ml_score(features_fraud)
    
    final_score_n = 0.4 * rule_score_n + 0.6 * ml_score_n
    final_score_f = 0.4 * rule_score_f + 0.6 * ml_score_f
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✅ Normal Worker")
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Rule Score", f"{rule_score_n:.0f}")
        with col_b:
            st.metric("ML Score", f"{ml_score_n:.0f}")
        with col_c:
            st.metric("Final", f"{final_score_n:.0f}")
        
        fig = go.Figure(go.Scatterpolar(
            r=[rule_score_n, ml_score_n, final_score_n],
            theta=['Rule Based', 'ML Anomaly', 'Final Score'],
            fill='toself',
            name='Normal Worker',
            marker_color='#51cf66'
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Score Composition",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🚨 Fraudulent Worker")
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Rule Score", f"{rule_score_f:.0f}")
        with col_b:
            st.metric("ML Score", f"{ml_score_f:.0f}")
        with col_c:
            st.metric("Final", f"{final_score_f:.0f}")
        
        fig = go.Figure(go.Scatterpolar(
            r=[rule_score_f, ml_score_f, final_score_f],
            theta=['Rule Based', 'ML Anomaly', 'Final Score'],
            fill='toself',
            name='Fraudulent Worker',
            marker_color='#ff6b6b'
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Score Composition",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    st.subheader("📐 Ensemble Formula")
    
    st.markdown("""
    ```
    Final Score = 0.4 × Rule_Score + 0.6 × ML_Score
    
    Why these weights?
    - Rule-Based (40%): Fast, explainable, catches obvious fraud
    - ML-Based (60%): Adaptive, catches subtle patterns, improves with feedback
    ```
    
    ### Calculation Example:
    """)
    
    calculation = f"""
    **Normal Worker:**
    - Rule Score: {rule_score_n:.0f}/100
    - ML Score: {ml_score_n:.0f}/100
    - Final = (0.4 × {rule_score_n:.0f}) + (0.6 × {ml_score_n:.0f})
    - Final = {0.4 * rule_score_n:.1f} + {0.6 * ml_score_n:.1f}
    - **Final Score: {final_score_n:.1f}/100** ✅ LOW RISK
    
    **Fraudulent Worker:**
    - Rule Score: {rule_score_f:.0f}/100
    - ML Score: {ml_score_f:.0f}/100
    - Final = (0.4 × {rule_score_f:.0f}) + (0.6 × {ml_score_f:.0f})
    - Final = {0.4 * rule_score_f:.1f} + {0.6 * ml_score_f:.1f}
    - **Final Score: {final_score_f:.1f}/100** 🚨 HIGH RISK
    """
    
    st.code(calculation)
    
    st.markdown("""
    ### 🎯 Risk Categorization:
    
    | Score Range | Risk Level | Action |
    |-------------|-----------|--------|
    | 0-40 | 🟢 LOW | Legitimate user |
    | 40-60 | 🟡 MEDIUM | Monitor |
    | 60-75 | 🟠 HIGH | Review |
    | 75-100 | 🔴 CRITICAL | Immediate action |
    """)

# ============================================================================
# PAGE 8: STEP 7 - HUMAN REVIEW
# ============================================================================
elif page == "👤 Step 7: Human Review":
    st.title("👤 Step 7: Human Review & Feedback")
    
    st.markdown("""
    After the system generates fraud scores, **humans verify** if the prediction is correct!
    
    This is where the **Reinforcement Learning loop** starts.
    """)
    
    # Simulate a case
    st.subheader("Case Review Example")
    
    clicks_f, ks_f, scrolls_f = generate_fraudulent_worker_data(100, 'robotic')
    features_f = extract_features(clicks_f, ks_f, scrolls_f)
    rules_f, rule_score_f = detect_rules(clicks_f, ks_f, scrolls_f)
    ml_score_f = calculate_ml_score(features_f)
    final_score_f = 0.4 * rule_score_f + 0.6 * ml_score_f
    
    st.markdown(f"""
    <div class="critical-box">
    <h3>worker_bot_123@company.com</h3>
    <p><strong>Final Score:</strong> {final_score_f:.1f}/100 🔴 CRITICAL</p>
    <p><strong>Rule Score:</strong> {rule_score_f:.0f} | <strong>ML Score:</strong> {ml_score_f:.0f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 System Analysis")
        
        st.markdown("**Triggered Rules:**")
        if rules_f:
            for rule_name, points, description in rules_f:
                st.markdown(f"""
                <div class="rule-box">
                <strong>{rule_name}</strong><br>{description}
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("🧠 What Human Thinks")
        
        st.markdown("""
        Looking at the activity pattern:
        - ✅ Identical clicks for 50+ minutes (ROBOTIC)
        - ✅ Zero variation (impossible)
        - ✅ No breaks at all (not human)
        - ✅ Sudden return to normal (script reload)
        
        **Verdict: This IS FRAUD** 🚨
        
        The model was RIGHT!
        """)
    
    st.divider()
    
    # Show feedback process
    st.subheader("💾 Creating Training Example")
    
    st.markdown("""
    When you mark this case, the system stores:
    """)
    
    feedback_record = {
        'Email': 'worker_bot_123@company.com',
        'Rule Score': rule_score_f,
        'ML Score': ml_score_f,
        'Final Score': final_score_f,
        'Model Prediction': 'FRAUD',
        'Human Verdict': 'FRAUD',
        'Confidence': 0.95,
        'Timestamp': datetime.now()
    }
    
    st.json(feedback_record, expanded=False)
    
    st.markdown("""
    ### 🎯 What This Feedback Means:
    
    ✅ **Agreement** - Model said FRAUD, Human confirmed FRAUD
    - This is a **TRUE POSITIVE**
    - Model learned correctly
    - Reward: +15 points
    
    This creates a **training example** that shows:
    - "When features are [std=0.1, superhuman=90%, cv=0.02, ...]"
    - "And rules triggered are [ROBOTIC_9MIN, ZERO_VARIANCE, ...]"
    - "THEN verdict is FRAUD"
    """)

# ============================================================================
# PAGE 9: STEP 8 - RL RETRAINING
# ============================================================================
elif page == "🔄 Step 8: RL Retraining":
    st.title("🔄 Step 8: Reinforcement Learning Retraining")
    
    st.markdown("""
    **Reinforcement Learning (RL)** is how the system learns from feedback and improves!
    
    Think of it like training a dog:
    - Model makes a prediction
    - You give feedback (correct/incorrect)
    - Model adjusts and learns
    - Next prediction is better!
    """)
    
    st.subheader("📚 Step-by-Step RL Process")
    
    # Step 1: Collect Feedback
    st.markdown("""
    <div class="feedback-box">
    <h4><span class="step-number">1</span> Collect Feedback</h4>
    You review 50 cases, mark each as FRAUD or LEGITIMATE.
    </div>
    """, unsafe_allow_html=True)
    
    feedback_sim = pd.DataFrame({
        'Case': [f'Case {i}' for i in range(1, 11)],
        'Model Prediction': ['FRAUD', 'LEGIT', 'FRAUD', 'LEGIT', 'FRAUD', 'LEGIT', 'FRAUD', 'FRAUD', 'LEGIT', 'LEGIT'],
        'Human Verdict': ['FRAUD', 'LEGIT', 'FRAUD', 'LEGIT', 'FRAUD', 'LEGIT', 'LEGIT', 'FRAUD', 'LEGIT', 'LEGIT'],
        'Match': ['✅', '✅', '✅', '✅', '✅', '✅', '❌', '✅', '✅', '✅']
    })
    
    st.dataframe(feedback_sim, use_container_width=True)
    
    st.markdown("""
    <div class="feedback-box">
    <h4><span class="step-number">2</span> Calculate Rewards</h4>
    For each case, calculate if model was correct:
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="score-box">
        <strong>✅ TRUE POSITIVE</strong><br>
        Predicted: FRAUD<br>
        Actual: FRAUD<br>
        Reward: <strong>+15 pts</strong>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="score-box">
        <strong>✅ TRUE NEGATIVE</strong><br>
        Predicted: LEGIT<br>
        Actual: LEGIT<br>
        Reward: <strong>+2 pts</strong>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="score-box">
        <strong>❌ FALSE POSITIVE</strong><br>
        Predicted: FRAUD<br>
        Actual: LEGIT<br>
        Penalty: <strong>-8 pts</strong>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="score-box">
        <strong>❌ FALSE NEGATIVE</strong><br>
        Predicted: LEGIT<br>
        Actual: FRAUD<br>
        Penalty: <strong>-20 pts</strong>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="feedback-box">
    <h4>Reward Calculation Example:</h4>
    <pre>
    From 10 cases:
    - 7 True Positives: 7 × 15 = 105 pts
    - 2 True Negatives: 2 × 2 = 4 pts
    - 0 False Positives: 0 × (-8) = 0 pts
    - 1 False Negative: 1 × (-20) = -20 pts
    ────────────────────────────────
    Total Reward: 105 + 4 + 0 - 20 = <strong>89 pts</strong>
    </pre>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feedback-box">
    <h4><span class="step-number">3</span> Train Supervised Models</h4>
    Use feedback as training data for Random Forest:
    </div>
    """, unsafe_allow_html=True)
    
    training_viz = """
    Input Features (50+)                        Output Label
    ┌──────────────────────────┐               ┌─────────────┐
    │ mean_clicks: 32.5        │               │ FRAUD       │
    │ std_clicks: 0.1          │───────┐   ┌──▶│ (or LEGIT)  │
    │ superhuman_clicks: 85%   │       │   │   └─────────────┘
    │ cv_clicks: 0.003         │───────┼───┤
    │ zero_activity: 0%        │       │   │   ┌─────────────┐
    │ ... (45 more features)   │───────┘   └──▶│ FRAUD       │
    └──────────────────────────┘               │             │
                                               └─────────────┘
    
    Random Forest learns:
    "When std_clicks < 1 AND superhuman% > 50 AND zero_activity < 1%
     AND triggered_rules.count > 3
     THEN probability of FRAUD = 0.95"
    """
    
    st.code(training_viz)
    
    st.markdown("""
    <div class="feedback-box">
    <h4><span class="step-number">4</span> Update Weights</h4>
    Adjust the model's parameters using Q-Learning:
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Q-Learning Update Formula:**
        ```
        Q(s,a) ← Q(s,a) + lr × (r + γ × max(Q(s',a')) - Q(s,a))
        
        Where:
        - Q(s,a) = Quality of action in state
        - lr = Learning rate (0.15)
        - r = Reward (+15, -8, etc)
        - γ = Discount factor (0.9)
        - s' = New state
        ```
        """)
    
    with col2:
        st.markdown("""
        **What It Does:**
        1. Takes current model quality
        2. Adds learning rate × reward
        3. Adjusts for future predictions
        4. Updates model parameters
        
        **Effect:**
        - Increases weight on important features
        - Decreases weight on misleading features
        - Shifts decision boundary
        - Improves next predictions
        """)
    
    st.markdown("""
    <div class="feedback-box">
    <h4><span class="step-number">5</span> Optimize Threshold</h4>
    Find the best fraud score threshold using feedback:
    </div>
    """, unsafe_allow_html=True)
    
    # Simulate threshold optimization
    thresholds = np.linspace(30, 80, 10)
    accuracy_before = [0.72, 0.75, 0.78, 0.81, 0.83, 0.82, 0.80, 0.76, 0.72, 0.68]
    accuracy_after = [0.75, 0.79, 0.83, 0.87, 0.90, 0.88, 0.84, 0.80, 0.75, 0.70]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=accuracy_before,
        name='Before Retraining',
        line=dict(color='#ff6b6b', width=2, dash='dash'),
        mode='lines+markers'
    ))
    
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=accuracy_after,
        name='After Retraining',
        line=dict(color='#51cf66', width=2),
        mode='lines+markers'
    ))
    
    fig.add_vline(x=60, line_dash="dash", line_color="gray", annotation_text="Optimal Threshold")
    
    fig.update_layout(
        title="Threshold Optimization - Finding Best Cutoff",
        xaxis_title="Fraud Score Threshold",
        yaxis_title="Accuracy",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Result:** Optimal threshold moved from 55 → 60 based on feedback!
    
    This means the model now has better separation between fraud and legitimate.
    """)
    
    st.markdown("""
    <div class="feedback-box">
    <h4><span class="step-number">6</span> Deploy Updated Model</h4>
    New model is ready for production:
    </div>
    """, unsafe_allow_html=True)
    
    improvements = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'TPR', 'FPR'],
        'Before Retraining': ['78%', '75%', '82%', '0.78', '82%', '8%'],
        'After Retraining': ['87%', '88%', '90%', '0.89', '90%', '4%'],
        'Improvement': ['+9%', '+13%', '+8%', '+0.11', '+8%', '-4%']
    })
    
    st.dataframe(improvements, use_container_width=True)
    
    st.markdown("""
    ### 🎯 What Changed:
    
    1. **Better Features** - Learned which features matter most
    2. **Optimized Threshold** - Found the best fraud score cutoff
    3. **Higher Accuracy** - Catches more fraud with fewer false alarms
    4. **Reduced False Positives** - Fewer innocent users flagged
    5. **Better Recall** - Catches more actual fraudsters
    
    ### 🔄 The Loop Continues:
    
    After retraining:
    - Collect more feedback (another 50 cases)
    - Retrain again
    - Model improves further
    - Repeat indefinitely!
    """)

# ============================================================================
# PAGE 10: COMPLETE WORKFLOW
# ============================================================================
elif page == "📊 Complete Workflow":
    st.title("📊 Complete Fraud Detection Workflow")
    
    st.markdown("""
    This page shows the **entire system end-to-end** with real data flowing through!
    """)
    
    # Generate complete example
    st.subheader("📌 Complete Example: One Worker Journey")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Worker:** alice_dev@company.com")
    with col2:
        st.markdown("**Session:** 2024-12-15 9AM-5PM (480 min)")
    with col3:
        st.markdown("**Task:** Code Review")
    
    st.divider()
    
    # Generate worker data
    clicks_f, ks_f, scrolls_f = generate_fraudulent_worker_data(150, 'robotic')
    features_f = extract_features(clicks_f, ks_f, scrolls_f)
    rules_f, rule_score_f = detect_rules(clicks_f, ks_f, scrolls_f)
    ml_score_f = calculate_ml_score(features_f)
    final_score_f = 0.4 * rule_score_f + 0.6 * ml_score_f
    
    # Timeline visualization
    st.markdown("""
    ### 🔄 Workflow Timeline
    """)
    
    timeline_items = [
        ("09:00 AM", "📥 Worker Activity Logged", "System receives minute-by-minute activity data"),
        ("09:15 AM", "🎯 Rule-Based Detection", f"Rules triggered, Score: {rule_score_f:.0f}/100"),
        ("09:15 AM", "🧠 ML Feature Extraction", "50+ features extracted from activity"),
        ("09:15 AM", "📈 Anomaly Detection", f"Isolation Forest Score: {ml_score_f:.0f}/100"),
        ("09:15 AM", "🔀 Ensemble Scoring", f"Final Score: {final_score_f:.1f}/100 🔴 CRITICAL"),
        ("09:30 AM", "📤 System Alert", "Worker flagged for human review"),
        ("10:00 AM", "👤 Human Review", "Analyst reviews case, verifies fraud"),
        ("10:05 AM", "✅ Feedback Recorded", "Verdict saved: FRAUD (confidence: 95%)"),
        ("02:00 PM", "🔄 Retraining Triggered", "50 cases accumulated, retraining begins"),
        ("02:15 PM", "🧠 Model Update", "Random Forest retrained with feedback"),
        ("02:15 PM", "⚖️ Threshold Optimization", "Q-Learning finds optimal threshold"),
        ("02:30 PM", "🚀 Model Deployed", "New model goes live"),
        ("02:30 PM", "📊 Metrics Improved", "Accuracy +5%, Precision +8%, Recall +3%"),
    ]
    
    for time, event, description in timeline_items:
        st.markdown(f"""
        <div class="metric-highlight">
        <strong>{time}</strong> - {event}<br>
        <span style="color: #666; font-size: 0.9em;">{description}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Detailed breakdown
    st.subheader("🔍 Detailed Breakdown at Each Step")
    
    tabs = st.tabs([
        "1️⃣ Data In",
        "2️⃣ Rule Detection",
        "3️⃣ ML Scoring",
        "4️⃣ Ensemble",
        "5️⃣ Human Review",
        "6️⃣ RL Training"
    ])
    
    with tabs[0]:
        st.markdown("""
        ### Input Data
        """)
        
        data_df = pd.DataFrame({
            'Minute': range(1, 11),
            'Clicks': clicks_f[:10].astype(int),
            'Keystrokes': ks_f[:10].astype(int),
            'Scrolls': scrolls_f[:10].astype(int)
        })
        
        st.dataframe(data_df, use_container_width=True)
        
        st.markdown(f"""
        **Raw Statistics:**
        - Total Duration: 150 minutes
        - Mean Clicks: {np.mean(clicks_f):.1f}/min
        - Std Dev Clicks: {np.std(clicks_f):.2f}
        - Max Clicks: {np.max(clicks_f):.0f}/min
        - Zero Activity: {((clicks_f == 0) & (ks_f == 0) & (scrolls_f == 0)).sum()} minutes
        """)
    
    with tabs[1]:
        st.markdown("""
        ### Rule-Based Detection
        """)
        
        st.metric("Final Rule Score", f"{rule_score_f:.0f}/100")
        
        if rules_f:
            st.markdown("**Triggered Rules:**")
            for rule_name, points, description in rules_f:
                st.markdown(f"""
                <div class="rule-box">
                <strong>{rule_name}</strong> (+{points} pts)<br>
                {description}
                </div>
                """, unsafe_allow_html=True)
    
    with tabs[2]:
        st.markdown("""
        ### ML Scoring (Isolation Forest)
        """)
        
        st.metric("ML Anomaly Score", f"{ml_score_f:.1f}/100")
        
        st.markdown("""
        **How Model Detected:**
        - ✅ Low standard deviation (< 1.0)
        - ✅ High superhuman percentage (> 50%)
        - ✅ Coefficient of variation too low
        - ✅ No natural activity variance
        
        Model learned: "This pattern is ANOMALOUS"
        """)
        
        # Feature importance
        feature_weights = {
            'superhuman_clicks_pct': features_f['superhuman_clicks_pct'],
            'std_clicks': features_f['std_clicks'],
            'cv_clicks': features_f['cv_clicks'],
        }
        
        fig = px.bar(
            x=list(feature_weights.values()),
            y=list(feature_weights.keys()),
            orientation='h',
            title="Top Features Contributing to Anomaly",
            color=list(feature_weights.values()),
            color_continuous_scale='Reds'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        st.markdown("""
        ### Ensemble Scoring
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Rule Score", f"{rule_score_f:.0f}")
        with col2:
            st.metric("ML Score", f"{ml_score_f:.0f}")
        with col3:
            st.metric("Final Score", f"{final_score_f:.1f}")
        
        st.markdown(f"""
        **Calculation:**
        ```
        Final = 0.4 × Rule + 0.6 × ML
        Final = 0.4 × {rule_score_f:.0f} + 0.6 × {ml_score_f:.0f}
        Final = {0.4 * rule_score_f:.1f} + {0.6 * ml_score_f:.1f}
        Final = {final_score_f:.1f} 🔴 CRITICAL RISK
        ```
        
        **Decision:** Flag for immediate review
        """)
    
    with tabs[4]:
        st.markdown("""
        ### Human Review & Feedback
        """)
        
        st.markdown("""
        **System Prediction:** FRAUD (Score: 87.3/100)
        
        **Human Verdict:** FRAUD ✅
        
        **Confidence:** 95%
        
        **Reason:** 
        "The identical activity for 50+ minutes is definitely a bot.
        No human can click exactly 50 times per minute for that long.
        This is 100% a fraud case."
        
        **Feedback Record:**
        - Email: alice_dev@company.com
        - Rule Score: 75
        - ML Score: 92
        - Final Score: 87.3
        - Model Prediction: FRAUD
        - Human Verdict: FRAUD
        - Confidence: 0.95
        - Result: TRUE POSITIVE (+15 reward points)
        """)
    
    with tabs[5]:
        st.markdown("""
        ### RL Retraining
        """)
        
        st.markdown("""
        **After 50 Similar Cases Reviewed:**
        
        **Reward Calculation:**
        - 42 True Positives: 42 × 15 = 630 points
        - 5 True Negatives: 5 × 2 = 10 points
        - 2 False Positives: 2 × (-8) = -16 points
        - 1 False Negative: 1 × (-20) = -20 points
        
        **Total Reward: 604 points**
        
        **Model Improvements:**
        """)
        
        improvements_detailed = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Before': ['76%', '71%', '85%', '0.77'],
            'After': ['88%', '89%', '92%', '0.90'],
            'Change': ['+12%', '+18%', '+7%', '+0.13']
        })
        
        st.dataframe(improvements_detailed, use_container_width=True)
        
        st.markdown("""
        **What Changed in Model:**
        
        1. **Feature Weights Updated**
           - Robotic pattern detection: Weight increased 25% → 32%
           - Superhuman detection: Weight increased 18% → 28%
           - Zero variance: Weight increased 15% → 22%
        
        2. **Threshold Optimized**
           - Old threshold: 60 (caught 85% fraud, 15% false alarms)
           - New threshold: 58 (catches 92% fraud, 8% false alarms)
        
        3. **Decision Boundary Shifted**
           - More sensitive to robotic patterns
           - Better at detecting superhuman speeds
           - Reduced false positives on legitimate workers
        
        4. **Model Ready for Deployment**
           - 88% accuracy vs 76% (12% improvement!)
           - Better coverage of your fraud types
           - Fewer false alerts to analysts
        """)
    
    st.divider()
    
    st.subheader("📊 Complete System Flow Diagram")
    
    flow_diagram = """
    ┌─────────────────────────────────────────────────────────────┐
    │                    WORKER ACTIVITY DATA                      │
    │              (Clicks, Keystrokes, Scrolls)                   │
    └────────────┬────────────────────────────────────────────────┘
                 │
        ┌────────▼─────────┐
        │  FEATURE EXTRACT  │
        │   (50+ features)  │
        └────────┬─────────┘
                 │
        ┌────────┴──────────┬──────────────────┐
        │                   │                  │
    ┌───▼────┐      ┌──────▼─────┐    ┌──────▼──────┐
    │ RULES  │      │ ISOLATION   │    │ SUPERVISED  │
    │  0-100 │      │ FOREST      │    │ MODELS      │
    │        │      │  0-100      │    │  0-100      │
    └───┬────┘      └──────┬──────┘    └──────┬──────┘
        │                  │                   │
        │    ┌─────────────┴──────────────────┘
        │    │
    ┌───▼────▼───────────────┐
    │  ENSEMBLE SCORE         │
    │  0.4×Rule + 0.6×ML     │
    │  Result: 0-100          │
    └───┬────────────────────┘
        │
    ┌───▼──────────────────┐
    │  RISK CATEGORY       │
    │  LOW/MED/HIGH/CRIT  │
    └───┬──────────────────┘
        │
    ┌───▼──────────────────┐
    │  HUMAN REVIEW        │
    │  FRAUD/LEGIT         │
    └───┬──────────────────┘
        │
    ┌───▼──────────────────┐
    │  FEEDBACK LOG        │
    │  (Training Data)     │
    └───┬──────────────────┘
        │
    ┌───▼──────────────────┐
    │  RL RETRAINING       │
    │  - Calculate Rewards │
    │  - Update Weights    │
    │  - Optimize          │
    └────────────────────────┘
           │
           └──▶ Deploy Updated Model
    """
    
    st.code(flow_diagram)
    
    st.markdown("""
    ### 🎯 Key Takeaways:
    
    1. **Rule-Based (40%)** catches obvious patterns fast
    2. **ML-Based (60%)** catches subtle anomalies and learns
    3. **Ensemble** combines both for robust detection
    4. **Human Feedback** is crucial for improvement
    5. **RL Loop** continuously adapts to fraud patterns
    6. **System Improves Over Time** as feedback accumulates
    
    The longer you use it, the smarter it gets! 🚀
    """)