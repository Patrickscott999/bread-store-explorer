import pandas as pd
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import plotly.io as pio
import io
import base64
from datetime import datetime
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import smtplib
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

load_dotenv()

def generate_pdf_report(df, rules, quality_metrics, anomalies, customer_segments, forecast_df):
    """Generate a comprehensive PDF report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    elements.append(Paragraph("Bread Store Analysis Report", title_style))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Data Quality Section
    elements.append(Paragraph("Data Quality Metrics", styles['Heading2']))
    quality_data = [
        ["Metric", "Value"],
        ["Missing Values", f"{quality_metrics['missing_values']} ({quality_metrics['missing_percentage']:.2f}%)"],
        ["Duplicate Records", f"{quality_metrics['duplicates']} ({quality_metrics['duplicate_percentage']:.2f}%)"],
        ["Date Range", f"{quality_metrics['date_range']['date_span_days']} days"],
        ["Total Transactions", str(len(df['Transaction'].unique()))]
    ]
    quality_table = Table(quality_data)
    quality_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(quality_table)
    elements.append(Spacer(1, 20))
    
    # Anomalies Section
    if anomalies:
        elements.append(Paragraph("Detected Anomalies", styles['Heading2']))
        for anomaly in anomalies:
            elements.append(Paragraph(f"• {anomaly['type']}: {anomaly['details']}", styles['Normal']))
        elements.append(Spacer(1, 20))
    
    # Top Association Rules
    elements.append(Paragraph("Top Association Rules", styles['Heading2']))
    rules_data = [["Antecedents", "Consequents", "Support", "Confidence", "Lift"]]
    for _, rule in rules.head(10).iterrows():
        rules_data.append([
            ', '.join(list(rule['antecedents'])),
            ', '.join(list(rule['consequents'])),
            f"{rule['support']:.3f}",
            f"{rule['confidence']:.3f}",
            f"{rule['lift']:.3f}"
        ])
    rules_table = Table(rules_data)
    rules_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(rules_table)
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

def export_data(df, rules, format='csv'):
    """Export data in various formats"""
    if format == 'csv':
        return df.to_csv(index=False)
    elif format == 'excel':
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Transactions', index=False)
            rules.to_excel(writer, sheet_name='Association Rules', index=False)
        output.seek(0)
        return output
    elif format == 'json':
        return df.to_json(orient='records')
    else:
        raise ValueError(f"Unsupported format: {format}")

def send_email_report(recipient, subject, body, attachments=None):
    """Send email with report attachments"""
    sender = os.getenv('EMAIL_USER')
    password = os.getenv('EMAIL_PASSWORD')
    
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = recipient
    msg['Subject'] = subject
    
    msg.attach(MIMEText(body, 'plain'))
    
    if attachments:
        for filename, content in attachments.items():
            part = MIMEApplication(content)
            part.add_header('Content-Disposition', 'attachment', filename=filename)
            msg.attach(part)
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Failed to send email: {str(e)}")
        return False

def schedule_report_generation(scheduler, report_type, schedule, recipients):
    """Schedule automated report generation and email delivery"""
    def generate_and_send_report():
        # Load data
        df = pd.read_csv('bread basket.csv')
        df['date_time'] = pd.to_datetime(df['date_time'], dayfirst=True)
        
        # Generate report based on type
        if report_type == 'daily':
            # Daily summary report
            daily_stats = df.groupby(df['date_time'].dt.date).agg({
                'Transaction': 'count',
                'Item': 'count'
            }).reset_index()
            
            # Create PDF report
            pdf_buffer = generate_pdf_report(
                df=df,
                rules=pd.DataFrame(),  # Empty for daily report
                quality_metrics={'missing_values': 0, 'missing_percentage': 0, 'duplicates': 0, 'duplicate_percentage': 0, 'date_range': {'date_span_days': 1}},
                anomalies=[],
                customer_segments=pd.DataFrame(),
                forecast_df=pd.DataFrame()
            )
            
            # Send email
            attachments = {
                'daily_report.pdf': pdf_buffer.getvalue(),
                'daily_stats.csv': daily_stats.to_csv(index=False)
            }
            
            send_email_report(
                recipient=recipients,
                subject='Daily Sales Report',
                body='Please find attached the daily sales report.',
                attachments=attachments
            )
    
    # Add job to scheduler
    scheduler.add_job(
        generate_and_send_report,
        trigger='cron',
        **schedule
    )

def setup_scheduled_reports():
    """Setup all scheduled reports"""
    scheduler = BackgroundScheduler()
    
    # Schedule daily report
    schedule_report_generation(
        scheduler=scheduler,
        report_type='daily',
        schedule={'hour': 23, 'minute': 0},  # Run at 11 PM daily
        recipients=os.getenv('REPORT_RECIPIENTS', '').split(',')
    )
    
    # Schedule weekly report
    schedule_report_generation(
        scheduler=scheduler,
        report_type='weekly',
        schedule={'day_of_week': 'mon', 'hour': 9, 'minute': 0},  # Run at 9 AM every Monday
        recipients=os.getenv('REPORT_RECIPIENTS', '').split(',')
    )
    
    scheduler.start()
    return scheduler 