# Bread Store Association Rules Explorer

A comprehensive Streamlit application for analyzing customer purchasing patterns and product relationships in a bread store using association rule mining, customer segmentation, and predictive analytics.

## Features

- **Association Rule Mining**: Discover which products are frequently purchased together
- **Customer Segmentation**: Group customers based on purchasing behavior
- **Predictive Analytics**: Forecast future sales trends
- **Data Quality Dashboard**: Monitor data completeness and detect anomalies
- **Export & Reporting**: Generate PDF reports and export data in various formats
- **Email Reports**: Send automated reports via email
- **Scheduled Reports**: Configure daily and weekly report generation

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/bread-store-explorer.git
cd bread-store-explorer
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Configure email settings in `.env` file:
```
EMAIL_USER=your_email@example.com
EMAIL_PASSWORD=your_app_password
REPORT_RECIPIENTS=recipient1@example.com,recipient2@example.com
```

## Usage

Run the application:
```
streamlit run app.py
```

The application will be available at http://localhost:8501

## Project Structure

- `app.py`: Main application file
- `report_utils.py`: Utility functions for report generation and data export
- `bread basket.csv`: Sample dataset
- `requirements.txt`: Project dependencies
- `.env`: Email configuration

## Technologies Used

- Streamlit
- Pandas
- NumPy
- MLxtend
- Plotly
- NetworkX
- Scikit-learn
- Statsmodels
- ReportLab
- APScheduler

## License

MIT

## Author

Your Name 