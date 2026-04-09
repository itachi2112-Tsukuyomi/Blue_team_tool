import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email_alert(sender_email: str, sender_password: str, recipient_email: str, events: list):
    """
    Sends an email alert containing the details of the network events detected.
    Assumes Gmail SMTP for simplicity, can be adjusted for others.
    """
    if not events:
        return False, "No events to alert on."
        
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = f"🚨 Phishlink Alert: Phishing Access Detected! ({len(events)} events)"
    
    body = "<h3>Phishing Site Access Log</h3>"
    body += "<p>The following accesses to known phishing sites were detected on your network:</p>"
    
    body += "<table border='1' cellpadding='5' cellspacing='0'>"
    body += "<tr><th>Timestamp</th><th>Source IP</th><th>Destination IP</th><th>Domain Accessed</th><th>Risk Band</th><th>AI Phishing Probability</th></tr>"
    
    for event in events:
        body += f"<tr>"
        body += f"<td>{event.get('timestamp', 'N/A')}</td>"
        body += f"<td>{event.get('source_ip', 'N/A')}</td>"
        body += f"<td>{event.get('destination_ip', 'N/A')}</td>"
        body += f"<td>{event.get('domain', 'N/A')}</td>"
        
        # In manual mode, we might not have AI scores attached, so we fallback
        risk = event.get('risk_band', 'Manual Target Match')
        prob = event.get('ai_risk_score', 'N/A')
        prob_str = f"{prob:.2f}%" if isinstance(prob, float) else prob
        
        body += f"<td>{risk}</td>"
        body += f"<td>{prob_str}</td>"
        body += f"</tr>"
        
    body += "</table>"
    body += "<br><p>Stay safe,<br>Phishlink AI</p>"
    
    msg.attach(MIMEText(body, 'html'))
    
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        # Login
        server.login(sender_email, sender_password)
        # Send
        server.send_message(msg)
        server.quit()
        return True, "Email sent successfully."
    except Exception as e:
        return False, f"Failed to send email: {str(e)}"
