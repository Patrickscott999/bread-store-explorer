import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import time
from scipy import stats
from report_utils import (
    generate_pdf_report,
    export_data,
    send_email_report,
    setup_scheduled_reports
)

# Set page config with custom theme
st.set_page_config(
    page_title="Bread Store Association Rules Explorer",
    page_icon="🍞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 2rem;
        background-color: #0E1117;
    }
    
    /* Header styling */
    h1 {
        color: #FF4B4B;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    h2 {
        color: #FF4B4B;
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Metric styling */
    .stMetric {
        background-color: rgba(255, 75, 75, 0.1);
        padding: 1.5rem;
        border-radius: 1rem;
        border: 1px solid rgba(255, 75, 75, 0.2);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(255, 75, 75, 0.2);
    }
    
    /* Description box styling */
    .description-box {
        background-color: rgba(255, 75, 75, 0.05);
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 4px solid #FF4B4B;
        margin-bottom: 1.5rem;
        color: white;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    /* Data quality box styling */
    .data-quality-box {
        background-color: rgba(255, 193, 7, 0.1);
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 4px solid #FFC107;
        margin-bottom: 1.5rem;
        color: white;
    }
    
    /* Anomaly box styling */
    .anomaly-box {
        background-color: rgba(244, 67, 54, 0.1);
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 4px solid #F44336;
        margin-bottom: 1.5rem;
        color: white;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #FF6B6B;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 75, 75, 0.3);
    }
    
    /* Selectbox styling */
    .stSelectbox>div>div {
        background-color: rgba(255, 75, 75, 0.1);
        border-radius: 0.5rem;
        border: 1px solid rgba(255, 75, 75, 0.2);
    }
    
    /* Slider styling */
    .stSlider>div>div>div {
        background-color: #FF4B4B;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1E1E1E;
    }
    
    /* Table styling */
    .dataframe {
        background-color: rgba(255, 75, 75, 0.05);
        border-radius: 1rem;
        padding: 1rem;
    }
    
    /* Plot styling */
    .js-plotly-plot {
        background-color: transparent !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1E1E1E;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #FF4B4B;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #FF6B6B;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_and_process_data():
    # Read the dataset
    df = pd.read_csv('bread basket.csv')
    
    # Convert date_time to datetime
    df['date_time'] = pd.to_datetime(df['date_time'], dayfirst=True)
    
    # Group by Transaction and Item
    transactions = df.groupby(['Transaction'])['Item'].apply(list).values.tolist()
    
    # Encode transactions
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    return df_encoded, df

def check_data_quality(df):
    """Check data quality and return metrics"""
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    missing_percentage = (missing_values / df.size) * 100
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    duplicate_percentage = (duplicates / len(df)) * 100
    
    # Check for data types
    data_types = df.dtypes
    
    # Check for outliers in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers = {}
    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(df[col]))
        outliers[col] = (z_scores > 3).sum()
    
    # Check for date range
    date_range = {
        'min_date': df['date_time'].min(),
        'max_date': df['date_time'].max(),
        'date_span_days': (df['date_time'].max() - df['date_time'].min()).days
    }
    
    # Check for transaction consistency
    transaction_counts = df.groupby('Transaction').size()
    avg_items_per_transaction = transaction_counts.mean()
    max_items_per_transaction = transaction_counts.max()
    min_items_per_transaction = transaction_counts.min()
    
    return {
        'missing_values': missing_values,
        'missing_percentage': missing_percentage,
        'duplicates': duplicates,
        'duplicate_percentage': duplicate_percentage,
        'data_types': data_types,
        'outliers': outliers,
        'date_range': date_range,
        'transaction_stats': {
            'avg_items': avg_items_per_transaction,
            'max_items': max_items_per_transaction,
            'min_items': min_items_per_transaction
        }
    }

def detect_anomalies(df):
    """Detect anomalies in the data"""
    anomalies = []
    
    # Check for unusual transaction sizes
    transaction_counts = df.groupby('Transaction').size()
    z_scores = np.abs(stats.zscore(transaction_counts))
    unusual_transactions = transaction_counts[z_scores > 3]
    if not unusual_transactions.empty:
        anomalies.append({
            'type': 'Unusual Transaction Size',
            'count': len(unusual_transactions),
            'details': f"Found {len(unusual_transactions)} transactions with unusually high or low number of items"
        })
    
    # Check for unusual time patterns
    hourly_counts = df.groupby(df['date_time'].dt.hour).size()
    z_scores = np.abs(stats.zscore(hourly_counts))
    unusual_hours = hourly_counts[z_scores > 3]
    if not unusual_hours.empty:
        anomalies.append({
            'type': 'Unusual Time Pattern',
            'count': len(unusual_hours),
            'details': f"Found unusual activity in {len(unusual_hours)} hours of the day"
        })
    
    # Check for rare items
    item_counts = df['Item'].value_counts()
    rare_items = item_counts[item_counts < 5]
    if not rare_items.empty:
        anomalies.append({
            'type': 'Rare Items',
            'count': len(rare_items),
            'details': f"Found {len(rare_items)} items that appear less than 5 times"
        })
    
    return anomalies

def create_customer_segments(df):
    # Convert date_time to datetime if it's not already
    df['date_time'] = pd.to_datetime(df['date_time'], dayfirst=True)
    
    # Create customer features
    customer_features = df.groupby('Transaction').agg({
        'Item': ['count', 'nunique'],
        'date_time': ['min', 'max']
    }).reset_index()
    
    customer_features.columns = ['Transaction', 'total_items', 'unique_items', 'first_purchase', 'last_purchase']
    
    # Calculate purchase frequency in days
    customer_features['purchase_frequency'] = (customer_features['last_purchase'] - customer_features['first_purchase']).dt.total_seconds() / (24 * 60 * 60)
    customer_features['purchase_frequency'] = customer_features['purchase_frequency'].replace(0, 1)  # Avoid division by zero
    
    # Calculate average basket size
    customer_features['avg_basket_size'] = customer_features['total_items'] / customer_features['purchase_frequency']
    
    # Prepare features for clustering
    features = ['total_items', 'unique_items', 'purchase_frequency', 'avg_basket_size']
    X = customer_features[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    customer_features['segment'] = kmeans.fit_predict(X_scaled)
    
    return customer_features

def predict_future_trends(df):
    # Aggregate daily sales
    daily_sales = df.groupby(df['date_time'].dt.date)['Transaction'].count().reset_index()
    daily_sales.columns = ['date', 'sales']
    
    # Fit ARIMA model
    model = ARIMA(daily_sales['sales'], order=(1, 1, 1))
    results = model.fit()
    
    # Forecast next 30 days
    forecast = results.forecast(steps=30)
    forecast_dates = pd.date_range(start=daily_sales['date'].max(), periods=31)[1:]
    
    return pd.DataFrame({
        'date': forecast_dates,
        'forecast': forecast
    })

def generate_recommendations(rules, customer_history):
    # Get top rules for the customer's purchased items
    customer_items = set(customer_history['Item'].unique())
    relevant_rules = rules[rules['antecedents'].apply(lambda x: any(item in customer_items for item in x))]
    
    # Sort by lift and get top recommendations
    top_recommendations = relevant_rules.sort_values('lift', ascending=False).head(5)
    
    return top_recommendations

def create_network_graph(rules, top_n=20):
    """Create an enhanced network graph with better styling"""
    G = nx.DiGraph()
    
    # Add edges from top rules
    for _, rule in rules.head(top_n).iterrows():
        antecedents = list(rule['antecedents'])
        consequents = list(rule['consequents'])
        
        # Add nodes and edges
        for ant in antecedents:
            G.add_node(ant, node_type='antecedent')
        for cons in consequents:
            G.add_node(cons, node_type='consequent')
        
        # Add edge with weight based on lift
        G.add_edge(antecedents[0], consequents[0], weight=rule['lift'])
    
    # Calculate layout with better spacing
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Create edge trace with gradient colors
    edge_trace = go.Scatter(
        x=[], y=[],
        line=dict(width=1, color='rgba(255, 75, 75, 0.3)'),
        hoverinfo='none',
        mode='lines')

    # Add edges to trace
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    # Create node trace with enhanced styling
    node_trace = go.Scatter(
        x=[], y=[],
        mode='markers+text',
        hoverinfo='text',
        text=[],
        marker=dict(
            showscale=True,
            colorscale='Reds',
            size=25,
            line_width=2,
            line=dict(color='white')
        ))

    # Add nodes to trace
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['text'] += tuple([node])

    # Create figure with enhanced layout
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=dict(
                           text='Product Relationship Network',
                           font=dict(size=24, color='white'),
                           x=0.5,
                           y=0.95
                       ),
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       plot_bgcolor='rgba(0,0,0,0)',
                       paper_bgcolor='rgba(0,0,0,0)'
                   ))
    
    return fig

def main():
    st.title("🍞 Bread Store Association Rules Explorer")
    st.markdown("""
    This dashboard helps you understand customer purchasing patterns and product relationships in the bread store.
    Use the sliders in the sidebar to adjust the analysis parameters and discover interesting insights!
    """)
    
    # Add Export and Reporting section in sidebar
    st.sidebar.header("📊 Export & Reports")
    export_format = st.sidebar.selectbox(
        "Export Format",
        ["CSV", "Excel", "JSON"]
    )
    
    if st.sidebar.button("Export Data"):
        df_encoded, df = load_and_process_data()
        rules = association_rules(apriori(df_encoded, min_support=0.01, use_colnames=True), metric="confidence", min_threshold=0.2)
        
        try:
            exported_data = export_data(df, rules, format=export_format.lower())
            st.sidebar.download_button(
                label=f"Download {export_format}",
                data=exported_data,
                file_name=f"bread_store_data.{export_format.lower()}",
                mime=f"text/{export_format.lower()}"
            )
        except Exception as e:
            st.sidebar.error(f"Export failed: {str(e)}")
    
    # Manual Report Generation
    if st.sidebar.button("Generate PDF Report"):
        df_encoded, df = load_and_process_data()
        rules = association_rules(apriori(df_encoded, min_support=0.01, use_colnames=True), metric="confidence", min_threshold=0.2)
        quality_metrics = check_data_quality(df)
        anomalies = detect_anomalies(df)
        customer_segments = create_customer_segments(df)
        forecast_df = predict_future_trends(df)
        
        try:
            pdf_buffer = generate_pdf_report(
                df=df,
                rules=rules,
                quality_metrics=quality_metrics,
                anomalies=anomalies,
                customer_segments=customer_segments,
                forecast_df=forecast_df
            )
            
            st.sidebar.download_button(
                label="Download PDF Report",
                data=pdf_buffer,
                file_name="bread_store_report.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.sidebar.error(f"Report generation failed: {str(e)}")
    
    # Email Report Configuration
    st.sidebar.header("📧 Email Reports")
    email_recipient = st.sidebar.text_input("Email Recipient")
    if st.sidebar.button("Send Report via Email"):
        if email_recipient:
            try:
                df_encoded, df = load_and_process_data()
                rules = association_rules(apriori(df_encoded, min_support=0.01, use_colnames=True), metric="confidence", min_threshold=0.2)
                quality_metrics = check_data_quality(df)
                anomalies = detect_anomalies(df)
                customer_segments = create_customer_segments(df)
                forecast_df = predict_future_trends(df)
                
                pdf_buffer = generate_pdf_report(
                    df=df,
                    rules=rules,
                    quality_metrics=quality_metrics,
                    anomalies=anomalies,
                    customer_segments=customer_segments,
                    forecast_df=forecast_df
                )
                
                success = send_email_report(
                    recipient=email_recipient,
                    subject="Bread Store Analysis Report",
                    body="Please find attached the Bread Store Analysis Report.",
                    attachments={'bread_store_report.pdf': pdf_buffer.getvalue()}
                )
                
                if success:
                    st.sidebar.success("Report sent successfully!")
                else:
                    st.sidebar.error("Failed to send report. Please check your email configuration.")
            except Exception as e:
                st.sidebar.error(f"Failed to send report: {str(e)}")
        else:
            st.sidebar.warning("Please enter an email address.")
    
    # Scheduled Reports Configuration
    st.sidebar.header("🕒 Scheduled Reports")
    if st.sidebar.button("Setup Scheduled Reports"):
        try:
            scheduler = setup_scheduled_reports()
            st.sidebar.success("Scheduled reports configured successfully!")
            st.sidebar.info("Reports will be sent daily at 11 PM and weekly on Mondays at 9 AM.")
        except Exception as e:
            st.sidebar.error(f"Failed to setup scheduled reports: {str(e)}")
    
    # Data Quality Section
    st.header("🔍 Data Quality Dashboard")
    
    # Data refresh controls
    st.sidebar.header("🔄 Data Controls")
    refresh_data = st.sidebar.button("Refresh Data")
    if refresh_data:
        st.cache_data.clear()
        st.success("Data refreshed successfully!")
        time.sleep(1)
        st.experimental_rerun()
    
    # Load and process data
    df_encoded, df = load_and_process_data()
    
    # Check data quality
    quality_metrics = check_data_quality(df)
    
    # Display data completeness metrics
    st.subheader("📊 Data Completeness")
    st.markdown("""
    <div class="data-quality-box">
    These metrics show how complete and reliable your data is. High completeness and low anomaly rates indicate more reliable insights.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Missing Values", f"{quality_metrics['missing_values']} ({quality_metrics['missing_percentage']:.2f}%)")
    with col2:
        st.metric("Duplicate Records", f"{quality_metrics['duplicates']} ({quality_metrics['duplicate_percentage']:.2f}%)")
    with col3:
        st.metric("Date Range", f"{quality_metrics['date_range']['date_span_days']} days")
    with col4:
        st.metric("Transactions", f"{len(df['Transaction'].unique())}")
    
    # Display data validation checks
    st.subheader("✅ Data Validation")
    st.markdown("""
    <div class="data-quality-box">
    These checks verify that your data meets expected formats and contains valid values.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Data Types:")
        st.dataframe(quality_metrics['data_types'])
    
    with col2:
        st.write("Transaction Statistics:")
        st.write(f"- Average items per transaction: {quality_metrics['transaction_stats']['avg_items']:.2f}")
        st.write(f"- Maximum items in a transaction: {quality_metrics['transaction_stats']['max_items']}")
        st.write(f"- Minimum items in a transaction: {quality_metrics['transaction_stats']['min_items']}")
    
    # Detect and display anomalies
    st.subheader("⚠️ Data Anomalies")
    st.markdown("""
    <div class="anomaly-box">
    These anomalies may indicate data quality issues or unusual patterns that require attention.
    </div>
    """, unsafe_allow_html=True)
    
    anomalies = detect_anomalies(df)
    if anomalies:
        for anomaly in anomalies:
            st.warning(f"**{anomaly['type']}**: {anomaly['details']}")
    else:
        st.success("No significant anomalies detected in the data.")
    
    # Sidebar for parameters
    st.sidebar.header("📊 Analysis Parameters")
    st.sidebar.markdown("""
    - **Support**: The frequency of items appearing together in transactions
    - **Confidence**: The reliability of the rule
    """)
    min_support = st.sidebar.slider("Minimum Support", 0.001, 0.1, 0.01, 0.001)
    min_confidence = st.sidebar.slider("Minimum Confidence", 0.1, 1.0, 0.2, 0.1)
    
    # Generate frequent itemsets and rules
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    # Customer Segmentation
    st.header("👥 Customer Segments")
    st.markdown("""
    <div class="description-box">
    This analysis groups customers based on their purchasing behavior, helping identify different customer types
    and their characteristics. Use this to tailor marketing strategies and improve customer experience.
    </div>
    """, unsafe_allow_html=True)
    
    customer_segments = create_customer_segments(df)
    
    # Display segment characteristics
    segment_stats = customer_segments.groupby('segment').agg({
        'total_items': 'mean',
        'unique_items': 'mean',
        'purchase_frequency': 'mean',
        'avg_basket_size': 'mean'
    }).round(2)
    
    st.write("Segment Characteristics:")
    st.dataframe(segment_stats)
    
    # Visualize segments
    fig_segments = px.scatter(
        customer_segments,
        x='total_items',
        y='unique_items',
        color='segment',
        title='Customer Segments by Purchase Behavior',
        labels={'total_items': 'Total Items Purchased', 'unique_items': 'Unique Items Purchased'}
    )
    st.plotly_chart(fig_segments, use_container_width=True)
    
    # Predictive Analytics
    st.header("🔮 Sales Forecast")
    st.markdown("""
    <div class="description-box">
    This forecast uses time series analysis to predict future sales trends, helping with inventory planning
    and resource allocation.
    </div>
    """, unsafe_allow_html=True)
    
    forecast_df = predict_future_trends(df)
    
    fig_forecast = px.line(
        forecast_df,
        x='date',
        y='forecast',
        title='30-Day Sales Forecast',
        labels={'date': 'Date', 'forecast': 'Predicted Sales'}
    )
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Personalized Recommendations
    st.header("🎯 Personalized Recommendations")
    st.markdown("""
    <div class="description-box">
    Get personalized product recommendations based on customer purchase history and association rules.
    </div>
    """, unsafe_allow_html=True)
    
    # Select a customer
    customer_id = st.selectbox(
        "Select Customer Transaction ID",
        options=df['Transaction'].unique()
    )
    
    # Get customer history and recommendations
    customer_history = df[df['Transaction'] == customer_id]
    recommendations = generate_recommendations(rules, customer_history)
    
    st.write("Customer Purchase History:")
    st.write(customer_history['Item'].tolist())
    
    st.write("Recommended Products:")
    for _, rec in recommendations.iterrows():
        st.write(f"- If you bought {', '.join(list(rec['antecedents']))}, you might like {', '.join(list(rec['consequents']))} (Confidence: {rec['confidence']:.2f})")
    
    # Convert frozensets to strings for display
    rules_display = rules.copy()
    rules_display['antecedents'] = rules_display['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules_display['consequents'] = rules_display['consequents'].apply(lambda x: ', '.join(list(x)))
    
    # Enhanced visualizations
    st.header("📊 Key Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Total Rules Found",
            len(rules),
            delta=f"{len(rules) - 100} from baseline"
        )
    with col2:
        st.metric(
            "Average Rule Strength",
            f"{rules['lift'].mean():.3f}",
            delta=f"{rules['lift'].mean() - 1:.3f} from baseline"
        )
    with col3:
        st.metric(
            "Strongest Rule",
            f"{rules['lift'].max():.3f}",
            delta="New high"
        )
    
    # Enhanced scatter plot
    st.header("📈 Support vs Confidence Analysis")
    fig = px.scatter(
        rules_display,
        x='support',
        y='confidence',
        size='lift',
        color='lift',
        hover_data={
            'antecedents': True,
            'consequents': True,
            'support': ':.3f',
            'confidence': ':.3f',
            'lift': ':.3f'
        },
        title='Support vs Confidence (Bubble size and color represent Lift)',
        labels={
            'support': 'Support',
            'confidence': 'Confidence',
            'lift': 'Lift',
            'antecedents': 'Antecedents',
            'consequents': 'Consequents'
        },
        color_continuous_scale='Reds',
        template='plotly_dark'
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(size=24)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Enhanced bar chart
    st.header("📊 Top Items by Frequency")
    item_counts = df['Item'].value_counts().head(20)
    fig_bar = px.bar(
        x=item_counts.index,
        y=item_counts.values,
        title='Top 20 Most Frequent Items',
        labels={'x': 'Item', 'y': 'Frequency'},
        text=item_counts.values,
        template='plotly_dark'
    )
    fig_bar.update_traces(
        texttemplate='%{text}',
        textposition='outside',
        marker_color='#FF4B4B'
    )
    fig_bar.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(size=24),
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Enhanced network graph
    st.header("🕸️ Product Relationship Network")
    network_fig = create_network_graph(rules)
    st.plotly_chart(network_fig, use_container_width=True)
    
    # Business Recommendations
    st.header("💡 Business Recommendations")
    st.markdown("""
    <div class="description-box">
    Based on the analysis above, here are some actionable insights for your bread store:
    
    1. **Product Placement**: Place frequently co-purchased items near each other to increase basket size
    2. **Bundle Opportunities**: Create bundles for items with high lift values
    3. **Inventory Management**: Stock more of the high-frequency items shown in the bar chart
    4. **Promotional Strategy**: Use the network graph to identify which items to promote together
    5. **Cross-selling**: Train staff to suggest complementary items based on the association rules
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 