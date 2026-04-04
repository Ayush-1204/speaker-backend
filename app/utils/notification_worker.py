"""Async notification worker for alert escalation (FCM → SMS → Email)."""

import os
import time
import json
from typing import Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

FCM_TIMEOUT_SEC = int(os.environ.get("SAFEEAR_FCM_TIMEOUT_SEC", "30"))
SMS_TIMEOUT_SEC = int(os.environ.get("SAFEEAR_SMS_TIMEOUT_SEC", "60"))
EMAIL_TIMEOUT_SEC = int(os.environ.get("SAFEEAR_EMAIL_TIMEOUT_SEC", "30"))

FIREBASE_CREDENTIALS = os.environ.get("FIREBASE_CREDENTIALS_PATH")
FIREBASE_CREDENTIALS_JSON = os.environ.get("FIREBASE_CREDENTIALS_JSON")
FIREBASE_PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
ANDROID_NOTIFICATION_CHANNEL_ID = os.environ.get("SAFEEAR_ANDROID_NOTIFICATION_CHANNEL_ID", "safeear_alerts")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.environ.get("TWILIO_FROM_NUMBER")
SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY")
SENDGRID_FROM_EMAIL = os.environ.get("SENDGRID_FROM_EMAIL", "alerts@safeear.app")


def _init_firebase():
    """Initialize Firebase Admin SDK."""
    try:
        import firebase_admin
        from firebase_admin import credentials, messaging

        init_options = {"projectId": FIREBASE_PROJECT_ID} if FIREBASE_PROJECT_ID else None
        
        if not firebase_admin._apps:
            if FIREBASE_CREDENTIALS:
                cred = credentials.Certificate(FIREBASE_CREDENTIALS)
                firebase_admin.initialize_app(cred, options=init_options)
            elif FIREBASE_CREDENTIALS_JSON:
                cred = credentials.Certificate(json.loads(FIREBASE_CREDENTIALS_JSON))
                firebase_admin.initialize_app(cred, options=init_options)
            else:
                # Works on Render if GOOGLE_APPLICATION_CREDENTIALS is set.
                firebase_admin.initialize_app(credentials.ApplicationDefault(), options=init_options)
        return messaging
    except Exception as e:
        logger.error(f"[firebase] Init failed: {e}")
        return None


def send_fcm_push(
    fcm_token: str,
    alert_id: str,
    lat: Optional[float],
    lon: Optional[float],
    confidence: float
) -> bool:
    """Send FCM push notification to parent device."""
    messaging = _init_firebase()
    if not messaging or not fcm_token:
        return False

    try:
        maps_url = (
            f"https://www.google.com/maps?q={lat},{lon}"
            if lat is not None and lon is not None
            else "https://www.google.com/maps"
        )
        message = messaging.Message(
            notification=messaging.Notification(
                title="SafeEar Alert",
                body=f"Unknown speaker detected (confidence: {confidence:.1%})"
            ),
            data={
                "alert_id": alert_id,
                "type": "stranger_detected",
                "maps_url": maps_url,
                "confidence": str(confidence)
            },
            android=messaging.AndroidConfig(
                priority="high",
                ttl=timedelta(minutes=15),
                notification=messaging.AndroidNotification(
                    channel_id=ANDROID_NOTIFICATION_CHANNEL_ID,
                    sound="default",
                    click_action="SAFEEAR_ALERT",
                ),
            ),
            token=fcm_token,
        )
        response = messaging.send(message, dry_run=False)
        logger.info("[fcm] Sent | token_suffix=%s | response=%s", fcm_token[-8:], response)
        return True
    except Exception as e:
        logger.error("[fcm] Send failed | token_suffix=%s | error=%s", (fcm_token[-8:] if fcm_token else "none"), str(e))
        return False


def send_sms(phone: str, alert_id: str, lat: Optional[float], lon: Optional[float]) -> bool:
    """Send SMS via Twilio escalation."""
    if not TWILIO_ACCOUNT_SID or not phone:
        return False

    try:
        from twilio.rest import Client

        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        maps_url = f"https://www.google.com/maps?q={lat},{lon}" if lat and lon else "https://www.google.com/maps"
        body = f"SafeEar: Unknown speaker detected near your child. View location: {maps_url}"

        message = client.messages.create(
            from_=TWILIO_FROM_NUMBER,
            to=phone,
            body=body
        )
        logger.info(f"[sms] Sent to {phone}: {message.sid}")
        return True
    except Exception as e:
        logger.error(f"[sms] Send failed: {e}")
        return False


def send_email(email: str, alert_id: str, lat: Optional[float], lon: Optional[float], audio_url: str) -> bool:
    """Send email via SendGrid escalation."""
    if not SENDGRID_API_KEY or not email:
        return False

    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail

        sg = SendGridAPIClient(SENDGRID_API_KEY)
        maps_url = f"https://www.google.com/maps?q={lat},{lon}" if lat and lon else "https://www.google.com/maps"

        html_content = f"""
        <h2>SafeEar Security Alert</h2>
        <p>An unknown speaker was detected near your child at {datetime.utcnow().isoformat()}Z.</p>
        <p><a href="{maps_url}">View Location on Map</a></p>
        <p><a href="{audio_url}">Listen to Audio Clip</a></p>
        <p>Alert ID: {alert_id}</p>
        """

        message = Mail(
            from_email=SENDGRID_FROM_EMAIL,
            to_emails=email,
            subject="SafeEar: Unknown Speaker Detected",
            html_content=html_content
        )

        response = sg.send(message)
        logger.info(f"[email] Sent to {email}: {response.status_code}")
        return response.status_code in (200, 202)
    except Exception as e:
        logger.error(f"[email] Send failed: {e}")
        return False


def escalate_alert(
    parent_email: Optional[str],
    parent_phone: Optional[str],
    fcm_token: Optional[str],
    alert_id: str,
    lat: Optional[float],
    lon: Optional[float],
    audio_url: str,
    confidence: float,
) -> dict:
    """
    Escalation chain: FCM (T+0) -> SMS (T+30s) -> Email (T+90s).
    Returns summary of each attempt.
    """
    result = {
        "fcm": {"sent": False, "at_ms": 0},
        "sms": {"sent": False, "at_ms": 0},
        "email": {"sent": False, "at_ms": 0}
    }

    start_ms = time.time() * 1000

    # T+0: Try FCM
    if fcm_token:
        result["fcm"]["sent"] = send_fcm_push(fcm_token, alert_id, lat, lon, confidence)
        if result["fcm"]["sent"]:
            result["fcm"]["at_ms"] = int(time.time() * 1000 - start_ms)
            logger.info(f"[escalate] FCM sent at {result['fcm']['at_ms']}ms")
            return result

    # T+30s: Try SMS if FCM failed
    time.sleep(FCM_TIMEOUT_SEC)
    if parent_phone:
        result["sms"]["sent"] = send_sms(parent_phone, alert_id, lat, lon)
        if result["sms"]["sent"]:
            result["sms"]["at_ms"] = int(time.time() * 1000 - start_ms)
            logger.info(f"[escalate] SMS sent at {result['sms']['at_ms']}ms")
            return result

    # T+90s: Try Email if SMS failed
    time.sleep(SMS_TIMEOUT_SEC)
    if parent_email:
        result["email"]["sent"] = send_email(parent_email, alert_id, lat, lon, audio_url)
        if result["email"]["sent"]:
            result["email"]["at_ms"] = int(time.time() * 1000 - start_ms)
            logger.info(f"[escalate] Email sent at {result['email']['at_ms']}ms")

    return result


# Placeholder for Celery task (if async worker is enabled)
"""
from celery import Celery

celery_app = Celery('safeear_notifications', broker=os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379'))

@celery_app.task
def notify_parent_async(parent_email, parent_phone, fcm_token, alert_id, lat, lon, audio_url, confidence):
    return escalate_alert(parent_email, parent_phone, fcm_token, alert_id, lat, lon, audio_url, confidence)
"""
