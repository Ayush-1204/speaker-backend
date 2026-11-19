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
ANDROID_SOS_CHANNEL_ID = os.environ.get("SAFEEAR_ANDROID_SOS_CHANNEL_ID", "safeear_sos")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.environ.get("TWILIO_FROM_NUMBER")
SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY")
SENDGRID_FROM_EMAIL = os.environ.get("SENDGRID_FROM_EMAIL", "alerts@safeear.app")


def _resolve_redis_url() -> str:
    return (
        (os.environ.get("REDIS_URL") or "").strip()
        or (os.environ.get("SAFEEAR_REDIS_URL") or "").strip()
        or "redis://localhost:6379"
    )


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
            elif os.path.exists("firebase-adminsdk.json"):
                cred = credentials.Certificate("firebase-adminsdk.json")
                firebase_admin.initialize_app(cred, options=init_options)
            else:
                # Works on Render if GOOGLE_APPLICATION_CREDENTIALS is set.
                firebase_admin.initialize_app(credentials.ApplicationDefault(), options=init_options)
        return messaging
    except Exception as e:
        logger.error(f"[firebase] Init failed: {e}")
        return None


def alert_text(alert_type: str, device_name: Optional[str] = None, confidence: Optional[float] = None) -> dict:
    """Title / body / SMS / email subject for each alert type."""
    who = (device_name or "your child's device").strip()
    if alert_type == "sos":
        return {
            "title": "SOS Emergency",
            "body": f"Your child pressed SOS on {who}. Tap to see the location and listen.",
            "sms": f"SafeEar SOS Emergency: your child pressed SOS on {who}.",
            "subject": "SafeEar: SOS Emergency",
            "html": f"<p>Your child pressed <b>SOS</b> on {who}.</p>",
        }
    body = "Unknown speaker detected"
    if confidence is not None:
        body += f" (confidence: {confidence:.1%})"
    return {
        "title": "SafeEar Alert",
        "body": body,
        "sms": "SafeEar: Unknown speaker detected near your child.",
        "subject": "SafeEar: Unknown Speaker Detected",
        "html": "<p>An unknown speaker was detected near your child.</p>",
    }


def send_fcm_push(
    fcm_token: str,
    alert_id: str,
    device_id: str,
    timestamp_ms: int,
    lat: Optional[float],
    lon: Optional[float],
    confidence: float,
    recipient_role: Optional[str] = None,
    alert_type: str = "unenrolled_contact",
    device_name: Optional[str] = None,
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
        is_sos = alert_type == "sos"
        text = alert_text(alert_type, device_name, None if is_sos else confidence)
        message = messaging.Message(
            notification=messaging.Notification(title=text["title"], body=text["body"]),
            data={
                "alert_id": alert_id,
                "device_id": device_id,
                "timestamp_ms": str(timestamp_ms),
                "type": "sos" if is_sos else "stranger_detected",
                "alert_type": alert_type,
                "title": text["title"],
                "body": text["body"],
                "maps_url": maps_url,
                "confidence": str(confidence),
                "target_role": recipient_role or "parent_device",
            },
            android=messaging.AndroidConfig(
                priority="high",
                ttl=timedelta(minutes=15),
                notification=messaging.AndroidNotification(
                    channel_id=ANDROID_SOS_CHANNEL_ID if is_sos else ANDROID_NOTIFICATION_CHANNEL_ID,
                    sound="default",
                    click_action="SAFEEAR_ALERT",
                    # One notification per alert: a new alert never replaces an
                    # older one, and the app reuses this tag so the same alert is
                    # never shown twice (push + background refresh).
                    tag=f"alert:{alert_id}",
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


def send_status_push(fcm_token: str, title: str, body: str, data: Optional[dict] = None) -> bool:
    """High-priority informational push to a parent device (e.g. protection degraded)."""
    messaging = _init_firebase()
    if not messaging or not fcm_token:
        return False
    try:
        payload = {k: str(v) for k, v in (data or {}).items()}
        payload.setdefault("target_role", "parent_device")
        message = messaging.Message(
            notification=messaging.Notification(title=title, body=body),
            data=payload,
            android=messaging.AndroidConfig(
                priority="high",
                ttl=timedelta(hours=1),
                notification=messaging.AndroidNotification(channel_id=ANDROID_NOTIFICATION_CHANNEL_ID, sound="default"),
            ),
            token=fcm_token,
        )
        messaging.send(message, dry_run=False)
        return True
    except Exception as e:
        logger.error("[fcm] Status push failed | token_suffix=%s | error=%s", fcm_token[-8:], str(e))
        return False


def send_sms(phone: str, alert_id: str, lat: Optional[float], lon: Optional[float],
             alert_type: str = "unenrolled_contact", device_name: Optional[str] = None) -> bool:
    """Send SMS via Twilio escalation."""
    if not TWILIO_ACCOUNT_SID or not phone:
        return False

    try:
        from twilio.rest import Client

        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        maps_url = f"https://www.google.com/maps?q={lat},{lon}" if lat and lon else "https://www.google.com/maps"
        body = f"{alert_text(alert_type, device_name)['sms']} View location: {maps_url}"

        message = client.messages.create(
            from_=TWILIO_FROM_NUMBER,
            to=phone,
            body=body
        )
        logger.info("[sms] Sent | sid=%s", message.sid)
        return True
    except Exception as e:
        logger.error(f"[sms] Send failed: {e}")
        return False


def send_email(email: str, alert_id: str, lat: Optional[float], lon: Optional[float], audio_url: str,
               alert_type: str = "unenrolled_contact", device_name: Optional[str] = None) -> bool:
    """Send email via SendGrid escalation."""
    if not SENDGRID_API_KEY or not email:
        return False

    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail

        sg = SendGridAPIClient(SENDGRID_API_KEY)
        maps_url = f"https://www.google.com/maps?q={lat},{lon}" if lat and lon else "https://www.google.com/maps"

        text = alert_text(alert_type, device_name)
        html_content = f"""
        <h2>SafeEar Security Alert</h2>
        {text['html']}
        <p>Time: {datetime.utcnow().isoformat()}Z</p>
        <p><a href="{maps_url}">View Location on Map</a></p>
        <p><a href="{audio_url}">Listen to Audio Clip</a></p>
        <p>Alert ID: {alert_id}</p>
        """

        message = Mail(
            from_email=SENDGRID_FROM_EMAIL,
            to_emails=email,
            subject=text["subject"],
            html_content=html_content
        )

        response = sg.send(message)
        logger.info("[email] Sent | status=%s", response.status_code)
        return response.status_code in (200, 202)
    except Exception as e:
        logger.error(f"[email] Send failed: {e}")
        return False


def escalate_alert(
    parent_email: Optional[str],
    parent_phone: Optional[str],
    fcm_token: Optional[str],
    alert_id: str,
    device_id: str,
    timestamp_ms: int,
    lat: Optional[float],
    lon: Optional[float],
    audio_url: str,
    confidence: float,
    recipient_role: Optional[str] = None,
    alert_type: str = "unenrolled_contact",
    device_name: Optional[str] = None,
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
        result["fcm"]["sent"] = send_fcm_push(
            fcm_token, alert_id, device_id, timestamp_ms, lat, lon, confidence,
            recipient_role=recipient_role, alert_type=alert_type, device_name=device_name,
        )
        if result["fcm"]["sent"]:
            result["fcm"]["at_ms"] = int(time.time() * 1000 - start_ms)
            logger.info(f"[escalate] FCM sent at {result['fcm']['at_ms']}ms")
            return result

    # T+30s: Try SMS if FCM failed
    time.sleep(FCM_TIMEOUT_SEC)
    if parent_phone:
        result["sms"]["sent"] = send_sms(parent_phone, alert_id, lat, lon, alert_type, device_name)
        if result["sms"]["sent"]:
            result["sms"]["at_ms"] = int(time.time() * 1000 - start_ms)
            logger.info(f"[escalate] SMS sent at {result['sms']['at_ms']}ms")
            return result

    # T+90s: Try Email if SMS failed
    time.sleep(SMS_TIMEOUT_SEC)
    if parent_email:
        result["email"]["sent"] = send_email(parent_email, alert_id, lat, lon, audio_url, alert_type, device_name)
        if result["email"]["sent"]:
            result["email"]["at_ms"] = int(time.time() * 1000 - start_ms)
            logger.info(f"[escalate] Email sent at {result['email']['at_ms']}ms")

    return result
