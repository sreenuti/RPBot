"""LLM client with OpenAI, Gemini, and deterministic mock support."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from src.schemas import InputRecord

load_dotenv()

CONSENT_FIELD_MAP = {
    "sms": "sms_opt_in",
    "email": "email_opt_in",
    "push": "push_opt_in",
    "voice": "voice_opt_in",
}


class LLMError(Exception):
    """Raised when LLM generation fails."""


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)
    return json.loads(cleaned)


def _profile_extras(record: InputRecord) -> dict[str, Any]:
    extra = getattr(record.input.profile, "model_extra", None) or {}
    return dict(extra)


def _get_extra(record: InputRecord, *keys: str) -> Any:
    for key in keys:
        if hasattr(record.input, key):
            value = getattr(record.input, key)
            if value is not None:
                return value
        extra = getattr(record.input, "model_extra", None) or {}
        if key in extra:
            return extra[key]
    constraints_extra = getattr(record.assertions.constraints, "model_extra", None) or {}
    for key in keys:
        if key in constraints_extra:
            return constraints_extra[key]
    return None


def _is_channel_eligible(record: InputRecord, channel: str) -> bool:
    consent_field = CONSENT_FIELD_MAP.get(channel)
    if consent_field is None:
        extra = getattr(record.consent, "model_extra", None) or {}
        return bool(extra.get(f"{channel}_opt_in", False))
    return bool(getattr(record.consent, consent_field, False))


def _select_channel(record: InputRecord) -> str | None:
    for channel in record.channel_preferences:
        normalized = channel.lower()
        if normalized in ("sms", "email", "push") and _is_channel_eligible(record, normalized):
            return normalized
    return None


def _parse_datetime(value: str, tz: ZoneInfo) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(tz)


def _compute_send_at(record: InputRecord, channel: str) -> str:
    tz_name = record.input.timezone or "UTC"
    tz = ZoneInfo(tz_name)
    constraints = record.assertions.constraints

    explicit = _get_extra(record, "scheduled_send_at") or constraints.send_at
    if explicit:
        return _parse_datetime(str(explicit), tz).isoformat(timespec="seconds")

    last_interaction_raw = record.input.last_interaction
    default_hours = {"sms": 9, "email": 10, "push": 10}
    if not last_interaction_raw:
        now_local = datetime.now(tz)
        send_dt = now_local.replace(
            hour=default_hours.get(channel, 9),
            minute=0,
            second=0,
            microsecond=0,
        )
        return send_dt.isoformat(timespec="seconds")

    last_dt = _parse_datetime(last_interaction_raw, tz)
    follow_up_days = _get_extra(record, "follow_up_days")
    preferred_hour = _get_extra(record, "preferred_send_hour")
    lifecycle = (record.lifecycle_stage or "").lower()

    if follow_up_days is not None:
        target_date = last_dt.date() + timedelta(days=int(follow_up_days))
        hour = int(preferred_hour) if preferred_hour is not None else default_hours.get(channel, 10)
    elif lifecycle == "new":
        target_date = last_dt.date() + timedelta(days=1)
        hour = int(preferred_hour) if preferred_hour is not None else default_hours.get(channel, 9)
    elif lifecycle == "open":
        target_date = last_dt.date() + timedelta(days=3)
        hour = int(preferred_hour) if preferred_hour is not None else default_hours.get(channel, 10)
    else:
        target_date = last_dt.date() + timedelta(days=1)
        hour = int(preferred_hour) if preferred_hour is not None else default_hours.get(channel, 9)

    send_dt = datetime(
        target_date.year,
        target_date.month,
        target_date.day,
        hour,
        0,
        0,
        tzinfo=tz,
    )
    return send_dt.isoformat(timespec="seconds")


def _mock_autonomous_decision(record: InputRecord) -> dict[str, Any]:
    """Offline test double that simulates an LLM inferring decisions from the record."""
    constraints = record.assertions.constraints
    include_opt_out = bool(constraints.include_opt_out_instructions)
    primary_cta = constraints.primary_cta or "engage"
    profile = record.input.profile
    first_name = profile.first_name or "there"
    property_name = record.input.property_name or "our community"
    extras = _profile_extras(record)
    amenities = extras.get("amenity_interest") or []
    city = extras.get("city_interest")
    move_date = record.input.move_date_target or ""

    extra_consent = getattr(record.consent, "model_extra", None) or {}
    if record.consent.global_opt_out or extra_consent.get("global_opt_out"):
        return {
            "should_send": False,
            "next_message": {
                "channel": "none",
                "send_at": None,
                "subject": None,
                "body": None,
                "cta": None,
            },
            "next_action": {"type": "suppress", "details": {"reason": "global_opt_out"}},
            "reasoning": "User has globally opted out; no communication should be sent.",
        }

    channel = _select_channel(record)
    if channel is None:
        return {
            "should_send": False,
            "next_message": {
                "channel": "none",
                "send_at": None,
                "subject": None,
                "body": None,
                "cta": None,
            },
            "next_action": {"type": "suppress", "details": {"reason": "no_eligible_channel"}},
            "reasoning": "No consented channel matches channel preferences; suppressing send.",
        }

    send_at = _compute_send_at(record, channel)
    lifecycle = (record.lifecycle_stage or "").lower()
    persona = record.persona or "user"
    if lifecycle == "new":
        next_action = {
            "type": "start_cadence",
            "details": {"name": f"{persona}_welcome_short_horizon"},
        }
    else:
        follow_up_days = _get_extra(record, "follow_up_days")
        next_action = {
            "type": "follow_up_in_days",
            "details": {"value": int(follow_up_days) if follow_up_days is not None else 3},
        }

    cta_type = "schedule_tour" if primary_cta in ("book_tour", "schedule_tour") else primary_cta
    if channel == "sms":
        cta = {"type": cta_type, "options": ["Thu", "Fri"]}
    elif channel == "email":
        cta = {"type": cta_type, "link": "https://property.example/tour"}
    else:
        cta = {"type": cta_type}

    if channel == "sms":
        city_part = f" in {city}" if city else ""
        move_part = f" Your target move date is {move_date}." if move_date else ""
        opt_out = " Reply STOP to opt out." if include_opt_out else ""
        body = (
            f"Hi {first_name} - welcome to {property_name}{city_part}!{move_part} "
            f"Tours are available this week. Would you like to book a time on Thursday or Friday? "
            f"Reply 1 for Thu, 2 for Fri.{opt_out}"
        )
        return {
            "should_send": True,
            "next_message": {
                "channel": "sms",
                "send_at": send_at,
                "subject": None,
                "body": body,
                "cta": cta,
            },
            "next_action": next_action,
            "reasoning": (
                f"SMS is the highest-priority consented channel. Scheduled for {send_at} "
                f"based on lifecycle and last interaction. Personalized welcome with tour CTA."
            ),
        }

    amenity_text = ""
    if amenities:
        amenity_text = f" See the {' & '.join(amenities)} you asked about."
    move_text = ""
    if move_date:
        move_text = f" Since you are planning a move around {move_date},"
    opt_out = (
        "\nTo opt out of emails, click here or reply STOP."
        if include_opt_out
        else ""
    )
    subject = f"Tour {property_name}{amenity_text}".replace("  ", " ").strip()
    body = (
        f"Hi {first_name},{move_text} here is a quick look at {property_name}.{amenity_text} "
        f"Book a visit this week to compare floor plans.\n"
        f"Book now -> https://property.example/tour{opt_out}"
    )
    return {
        "should_send": True,
        "next_message": {
            "channel": "email",
            "send_at": send_at,
            "subject": subject[:120],
            "body": body,
            "cta": cta,
        },
        "next_action": next_action,
        "reasoning": (
            f"Email selected as highest-priority consented channel. Scheduled for {send_at}. "
            f"Highlights amenities and move context with tour CTA."
        ),
    }


class LLMClient:
    """Generate full autonomous agent decisions via LLM or mock simulation."""

    def __init__(self, mock: bool = False) -> None:
        self.mock = mock
        self.provider = os.getenv("LLM_PROVIDER", "openai").lower()
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    def generate(self, prompt: str, record: InputRecord | None = None) -> dict[str, Any]:
        if self.mock:
            if record is None:
                raise LLMError("Mock mode requires record context.")
            return _mock_autonomous_decision(record)
        return self._generate_api(prompt)

    def _generate_api(self, prompt: str) -> dict[str, Any]:
        text = self._call_provider(prompt)
        try:
            return _extract_json(text)
        except (json.JSONDecodeError, TypeError):
            retry_prompt = (
                f"{prompt}\n\nYour previous response was invalid JSON. "
                "Respond with valid JSON only containing should_send, next_message, "
                "next_action, and reasoning."
            )
            text = self._call_provider(retry_prompt)
            try:
                return _extract_json(text)
            except (json.JSONDecodeError, TypeError) as exc:
                raise LLMError(f"LLM returned invalid JSON after retry: {exc}") from exc

    def _call_provider(self, prompt: str) -> str:
        if self.provider == "gemini":
            return self._call_gemini(prompt)
        return self._call_openai(prompt)

    def _call_openai(self, prompt: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError("openai package is not installed.") from exc

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=self.openai_model,
            temperature=0,
            messages=[
                {"role": "system", "content": "Respond with JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or "{}"

    def _call_gemini(self, prompt: str) -> str:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set.")
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise LLMError("google-generativeai package is not installed.") from exc

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(self.gemini_model)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0, "response_mime_type": "application/json"},
        )
        return response.text or "{}"
