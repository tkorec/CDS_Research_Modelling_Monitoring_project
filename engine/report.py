import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import requests
import config

class Report():

    def __init__(self):
        self.telegram_bot_token = config.TELEGRAM_BOT_TOKEN
        self.telegram_chat_id = config.TELEGRAM_CHAT_ID
        self.telegram_bot_url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"

    def send_telegram_message(self, message):
        payload = {"chat_id": self.telegram_chat_id, "text": message, "parse_mode": "HTML"}
        requests.post(self.telegram_bot_url, data=payload)


    def send_email(self, subject, body, to_email):
        smtp_server = "smtp.gmail.com"
        smtp_port = 587

        sender_email = "tk.korec@gmail.com"
        sender_password = config.SENDER_PASSWORD

        message = MIMEMultipart()
        message["Subject"] = subject
        message["From"] = sender_email
        message["To"] = to_email
        message.attach(MIMEText(body, "html"))

        try:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.sendmail(message["From"], message["To"], message.as_string())
                self.send_telegram_message("Email regarding possible trading options sent.")
        except Exception as e:
            self.send_telegram_message(f"Sending Mail Failure Because Of Error: {e}")
            print(f"""---❌ Message from send_email() function Report() class ---\nSending Mail Failure Because Of Error: {e}""")


    def create_notification_email_possible_trade(self, spreads, asset):
        spreads_html = spreads.to_html(index=False, border=1, justify="center")
        subject = "Suitable Call Debit Spreads for " + asset
        to_email = "tk.korec@gmail.com"
        html_content = f"""
        <html>
            <body>
                <h2>{subject}</h2>
                <br>
                {spreads_html}  <!-- Embed DataFrame as HTML -->
            </body>
        </html>
        """
        self.send_email(subject, html_content, to_email)


