#!/usr/bin/env python3
"""Generate synthetic train/validation/test datasets for LoRA fine-tuning."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydantic import BaseModel, Field, ValidationError  # noqa: E402

from src.output_parser import parse_agent_output  # noqa: E402
from src.schemas import AgentOutput, InputRecord  # noqa: E402
from src.validator import validate  # noqa: E402

FIRST_NAMES = [
    "Taylor", "Alex", "Sam", "Jordan", "Casey", "Riley", "Morgan", "Avery",
    "Dana", "Pat", "Chris", "Maria", "Quinn", "Elena", "Robin", "Lee",
    "Jamie", "Cameron", "Drew", "Skyler", "Blake", "Harper", "Reese", "Sage",
]

PROPERTY_NAMES = [
    "Oak Ridge Apartments", "Lakeside Commons", "Harbor View Residences",
    "Parkside Flats", "Summit Heights", "Riverwalk Lofts", "Cedar Point Apartments",
    "Willow Creek Homes", "Metro Square", "Greenfield Estates", "Downtown Lofts",
    "Northgate Apartments", "Skyline Towers", "Bayview Residences", "Heritage Oaks",
    "Maple Grove Villas", "Silver Creek Townhomes", "Pinehurst Place",
]

CITIES = [
    ("Richardson", "TX"), ("Atlanta", "GA"), ("Chicago", "IL"), ("Austin", "TX"),
    ("Dallas", "TX"), ("Denver", "CO"), ("Seattle", "WA"), ("Phoenix", "AZ"),
    ("Nashville", "TN"), ("Charlotte", "NC"), ("Portland", "OR"), ("Miami", "FL"),
]

TIMEZONES = [
    "America/Chicago", "America/New_York", "America/Los_Angeles",
    "America/Denver", "America/Phoenix", "America/Detroit",
]

LEAD_SOURCES = [
    "website", "apartments_com", "zillow", "referral", "walk_in",
    "google_ads", "social_media", "property_sign",
]

AMENITIES = [
    "pool", "fitness center", "dog park", "EV charging", "rooftop deck",
    "coworking lounge", "package lockers", "grill area", "playground",
]

CTA_TYPES = [
    "schedule_tour", "book_tour", "renew_lease", "pay_rent", "rsvp_event",
    "view_announcement", "apply_online", "confirm_maintenance", "pickup_package",
]

SUPPRESSION_REASONS = [
    "global_opt_out",
    "no_eligible_channel",
    "missing_contact_info",
    "quiet_hours_no_send_window",
    "sensitive_compliance_risk",
]

CHANNEL_SEND_HOURS = {"sms": 9, "email": 10, "push": 11}
CHANNEL_WEIGHTS = {"sms": 0.45, "email": 0.40, "push": 0.15}

SCENARIO_SLUGS = [
    "prospect_welcome",
    "tour_scheduling",
    "tour_reminder",
    "tour_follow_up",
    "long_move_in_horizon",
    "short_move_in_horizon",
    "global_opt_out_suppress",
    "sms_opt_out_email_allowed",
    "email_opt_out_sms_allowed",
    "all_channels_opted_out",
    "quiet_hours_adjustment",
    "invalid_missing_phone",
    "invalid_missing_email",
    "lease_renewal_reminder",
    "maintenance_appointment_update",
    "rent_reminder",
    "package_notification",
    "community_event",
    "amenity_promotion",
    "emergency_maintenance_notice",
]

EDGE_SCENARIO_SLUGS = {
    "global_opt_out_suppress",
    "all_channels_opted_out",
    "quiet_hours_adjustment",
    "invalid_missing_phone",
    "invalid_missing_email",
}

SPLIT_COUNTS = {
    "train": 3000,
    "validation": 500,
    "test": 500,
    "edge_cases": 300,
}


class GeneratedTrainingRow(BaseModel):
    input: dict[str, Any]
    expected: dict[str, Any]


@dataclass
class ScenarioContext:
    slug: str
    split: str
    index: int
    rng: random.Random
    force_send: bool | None = None
    force_channel: str | None = None
    force_suppress_reason: str | None = None


def _slug_to_property_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _pick(rng: random.Random, items: list[Any]) -> Any:
    return items[rng.randrange(len(items))]


def _property_link(property_name: str, path: str) -> str:
    return f"https://{_slug_to_property_slug(property_name)}.example{path}"


def _offset_for_timezone(tz_name: str, dt: datetime) -> str:
    aware = dt.replace(tzinfo=ZoneInfo(tz_name))
    return aware.isoformat(timespec="seconds")


def _next_send_at(
    *,
    channel: str,
    timezone: str,
    base_dt: datetime,
    preferred_hour: int | None = None,
    override_send_at: str | None = None,
    immediate: bool = False,
) -> str:
    if override_send_at:
        return override_send_at
    if immediate:
        return _offset_for_timezone(timezone, base_dt)
    hour = preferred_hour if preferred_hour is not None else CHANNEL_SEND_HOURS[channel]
    send_dt = (base_dt + timedelta(days=1)).replace(
        hour=hour, minute=0, second=0, microsecond=0
    )
    return _offset_for_timezone(timezone, send_dt)


def _eligible_channels(record: InputRecord) -> list[str]:
    if record.consent.global_opt_out:
        return []
    contact = getattr(record, "model_extra", None) or {}
    contact_info = contact.get("contact") or {}
    phone_ok = bool(contact_info.get("phone")) and contact_info.get("phone_valid", True)
    email_ok = bool(contact_info.get("email")) and contact_info.get("email_valid", True)

    eligible: list[str] = []
    if record.consent.sms_opt_in and phone_ok:
        eligible.append("sms")
    if record.consent.email_opt_in and email_ok:
        eligible.append("email")
    if record.consent.push_opt_in:
        eligible.append("push")
    return eligible


def _select_channel(record: InputRecord, preferences: list[str], rng: random.Random) -> str | None:
    eligible = set(_eligible_channels(record))
    for channel in preferences:
        if channel in eligible:
            return channel
    return None


def _opt_out_suffix(channel: str, include_opt_out: bool) -> str:
    if not include_opt_out:
        return ""
    if channel == "sms":
        return " Reply STOP to opt out."
    if channel == "email":
        return "\nTo opt out, reply STOP or unsubscribe."
    if channel == "push":
        return " Reply STOP to opt out."
    return ""


def _personalization_score(body: str | None, record: InputRecord) -> float:
    if not body:
        return 0.0
    body_lower = body.lower()
    hits = 0
    total = 0
    profile = record.input.profile
    for value in (
        profile.first_name,
        record.input.property_name,
        record.input.move_date_target,
        getattr(profile, "model_extra", None) or {},
    ):
        if isinstance(value, dict):
            for nested in value.values():
                if isinstance(nested, str) and nested.lower() in body_lower:
                    hits += 1
                if isinstance(nested, str):
                    total += 1
            continue
        if value:
            total += 1
            if str(value).lower() in body_lower:
                hits += 1
    extra = getattr(record.input.profile, "model_extra", None) or {}
    for key in ("city_interest", "amenity_interest"):
        val = extra.get(key)
        if val:
            total += 1
            if isinstance(val, list):
                if any(str(item).lower() in body_lower for item in val):
                    hits += 1
            elif str(val).lower() in body_lower:
                hits += 1
    if total == 0:
        return 0.75
    return round(min(0.98, 0.7 + (hits / total) * 0.28), 2)


def _build_suppress_expected(
    task_id: str,
    reason: str,
    rng: random.Random,
    reasoning: str | None = None,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "should_send": False,
        "next_message": {
            "channel": "none",
            "send_at": None,
            "subject": None,
            "body": None,
            "cta": None,
        },
        "next_action": {"type": "suppress", "details": {"reason": reason}},
        "reasoning": reasoning or f"Send suppressed due to {reason.replace('_', ' ')}.",
        "quality": {
            "personalization_score": 0.0,
            "safety_violations": 0,
            "latency_ms": rng.randint(120, 450),
        },
    }


def _scenario_consent(ctx: ScenarioContext) -> dict[str, bool]:
    slug = ctx.slug
    if slug == "global_opt_out_suppress":
        return {
            "email_opt_in": True, "sms_opt_in": True, "voice_opt_in": False,
            "push_opt_in": True, "global_opt_out": True,
        }
    if slug == "sms_opt_out_email_allowed":
        return {
            "email_opt_in": True, "sms_opt_in": False, "voice_opt_in": False,
            "push_opt_in": False, "global_opt_out": False,
        }
    if slug == "email_opt_out_sms_allowed":
        return {
            "email_opt_in": False, "sms_opt_in": True, "voice_opt_in": False,
            "push_opt_in": False, "global_opt_out": False,
        }
    if slug == "all_channels_opted_out":
        return {
            "email_opt_in": False, "sms_opt_in": False, "voice_opt_in": False,
            "push_opt_in": False, "global_opt_out": False,
        }
    if slug in ("invalid_missing_phone", "invalid_missing_email"):
        return {
            "email_opt_in": True, "sms_opt_in": True, "voice_opt_in": False,
            "push_opt_in": False, "global_opt_out": False,
        }
    if slug == "community_event" and ctx.rng.random() < 0.3:
        return {
            "email_opt_in": False, "sms_opt_in": False, "voice_opt_in": False,
            "push_opt_in": True, "global_opt_out": False,
        }
    return {
        "email_opt_in": ctx.rng.random() < 0.85,
        "sms_opt_in": ctx.rng.random() < 0.8,
        "voice_opt_in": False,
        "push_opt_in": ctx.rng.random() < 0.35,
        "global_opt_out": False,
    }


def _scenario_contact(ctx: ScenarioContext) -> dict[str, Any]:
    first = _pick(ctx.rng, FIRST_NAMES).lower()
    phone = f"+1{ctx.rng.randint(200, 999)}{ctx.rng.randint(200, 999)}{ctx.rng.randint(1000, 9999)}"
    email = f"{first}.{ctx.rng.randint(10, 99)}@example.com"
    contact: dict[str, Any] = {
        "phone": phone,
        "email": email,
        "phone_valid": True,
        "email_valid": True,
    }
    if ctx.slug == "invalid_missing_phone":
        if ctx.rng.random() < 0.5:
            contact["phone"] = None
        else:
            contact["phone_valid"] = False
    if ctx.slug == "invalid_missing_email":
        if ctx.rng.random() < 0.5:
            contact["email"] = None
        else:
            contact["email_valid"] = False
    return contact


def _scenario_persona(ctx: ScenarioContext) -> tuple[str, str]:
    resident_scenarios = {
        "lease_renewal_reminder", "maintenance_appointment_update", "rent_reminder",
        "package_notification", "community_event", "emergency_maintenance_notice",
    }
    if ctx.slug in resident_scenarios:
        lifecycle = _pick(ctx.rng, ["active", "renewal", "active"])
        return "resident", lifecycle
    lifecycle = _pick(ctx.rng, ["new", "open", "nurture", "hot", "application"])
    return "prospect", lifecycle


def _primary_cta(slug: str) -> str:
    mapping = {
        "prospect_welcome": "book_tour",
        "tour_scheduling": "schedule_tour",
        "tour_reminder": "schedule_tour",
        "tour_follow_up": "book_tour",
        "long_move_in_horizon": "book_tour",
        "short_move_in_horizon": "book_tour",
        "lease_renewal_reminder": "renew_lease",
        "maintenance_appointment_update": "confirm_maintenance",
        "rent_reminder": "pay_rent",
        "package_notification": "pickup_package",
        "community_event": "rsvp_event",
        "amenity_promotion": "book_tour",
        "emergency_maintenance_notice": "view_announcement",
    }
    return mapping.get(slug, "book_tour")


def _message_copy(
    *,
    slug: str,
    channel: str,
    first_name: str,
    property_name: str,
    city_state: str | None,
    amenities: list[str] | None,
    move_date: str | None,
    include_opt_out: bool,
    primary_cta: str,
) -> tuple[str | None, str | None, dict[str, Any] | None]:
    city_part = f" in {city_state}" if city_state else ""
    amenity_part = ""
    if amenities:
        amenity_part = f" featuring our {amenities[0]} and {amenities[1] if len(amenities) > 1 else 'community amenities'}"
    move_part = ""
    if move_date:
        move_part = f" for your move around {move_date}"

    templates: dict[str, dict[str, Any]] = {
        "prospect_welcome": {
            "sms": (
                None,
                f"Hi {first_name}—welcome to {property_name}{city_part}! "
                f"Tours are available this week. Book a visit on our website."
            ),
            "email": (
                f"Welcome to {property_name}",
                f"Hi {first_name},\nWelcome to {property_name}{city_part}! "
                f"We would love to show you available homes{move_part}. Book a tour this week.",
            ),
        },
        "tour_scheduling": {
            "sms": (
                None,
                f"Hi {first_name}—let's schedule your tour at {property_name}. "
                f"Reply with a day that works or book online.",
            ),
            "email": (
                f"Schedule your {property_name} tour",
                f"Hi {first_name},\nReady to tour {property_name}? "
                f"Pick a time this week and we will confirm your visit.",
            ),
        },
        "tour_reminder": {
            "sms": (
                None,
                f"Hi {first_name}—reminder: your tour at {property_name} is tomorrow. "
                f"Reply if you need to reschedule.",
            ),
            "email": (
                f"Tour reminder — {property_name}",
                f"Hi {first_name},\nThis is a friendly reminder about your upcoming tour at {property_name}.",
            ),
        },
        "tour_follow_up": {
            "sms": (
                None,
                f"Hi {first_name}—thanks for visiting {property_name}. "
                f"Book a follow-up tour or apply online when you are ready.",
            ),
            "email": (
                f"Thanks for touring {property_name}",
                f"Hi {first_name},\nThanks for visiting {property_name}. "
                f"Let us know if you would like to compare floor plans or apply.",
            ),
        },
        "long_move_in_horizon": {
            "email": (
                f"Plan your move to {property_name}",
                f"Hi {first_name},\nSince you are planning ahead{move_part}, "
                f"explore floor plans{amenity_part} and book a tour at your pace.",
            ),
            "sms": (
                None,
                f"Hi {first_name}—planning a future move to {property_name}{city_part}? "
                f"Tour now to compare options.",
            ),
        },
        "short_move_in_horizon": {
            "sms": (
                None,
                f"Hi {first_name}—your move to {property_name}{city_part} is coming up soon. "
                f"Homes are available now. Book a tour today.",
            ),
            "email": (
                f"Homes ready now at {property_name}",
                f"Hi {first_name},\nYour timeline is approaching{move_part}. "
                f"Tour {property_name} this week to secure your home.",
            ),
        },
        "lease_renewal_reminder": {
            "email": (
                f"Your {property_name} renewal options",
                f"Hi {first_name},\nYour lease renewal window is open at {property_name}. "
                f"Review updated rates and terms online.",
            ),
            "sms": (
                None,
                f"Hi {first_name}—your lease renewal at {property_name} is ready to review. "
                f"Log in to see options.",
            ),
        },
        "maintenance_appointment_update": {
            "sms": (
                None,
                f"Hi {first_name}—your maintenance visit at {property_name} has been updated. "
                f"Tap to confirm the new time.",
            ),
            "email": (
                f"Maintenance update — {property_name}",
                f"Hi {first_name},\nYour maintenance appointment at {property_name} has a new time. "
                f"Please confirm or request changes.",
            ),
        },
        "rent_reminder": {
            "email": (
                f"Rent reminder — {property_name}",
                f"Hi {first_name},\nYour rent is due soon. Pay online through the resident portal at any time.",
            ),
        },
        "package_notification": {
            "sms": (
                None,
                f"Hi {first_name}—a package arrived for you at {property_name}. "
                f"Pick it up from the locker room during office hours.",
            ),
            "push": (
                None,
                f"Hi {first_name}—package ready for pickup at {property_name}. Tap for locker details.",
            ),
        },
        "community_event": {
            "email": (
                f"Community event at {property_name}",
                f"Hi {first_name},\nJoin neighbors for an upcoming community event at {property_name}. RSVP online.",
            ),
            "push": (
                None,
                f"Hi {first_name}—community event this week at {property_name}. Tap to RSVP.",
            ),
        },
        "amenity_promotion": {
            "email": (
                f"Explore amenities at {property_name}",
                f"Hi {first_name},\nDiscover{amenity_part} at {property_name}. Book a tour to see everything in person.",
            ),
            "sms": (
                None,
                f"Hi {first_name}—see{amenity_part} at {property_name}. Book a tour this week.",
            ),
        },
        "emergency_maintenance_notice": {
            "sms": (
                None,
                f"Hi {first_name}—urgent maintenance notice for {property_name}: "
                f"water will be temporarily shut off today. See portal for details.",
            ),
            "email": (
                f"Urgent maintenance notice — {property_name}",
                f"Hi {first_name},\nUrgent maintenance is scheduled today at {property_name}. "
                f"See the resident portal for timing and updates.",
            ),
            "push": (
                None,
                f"Urgent maintenance today at {property_name}. Tap for details.",
            ),
        },
        "sms_opt_out_email_allowed": {
            "email": (
                f"Tour invitation — {property_name}",
                f"Hi {first_name},\nWe would love to show you {property_name}{city_part}. Book a tour this week.",
            ),
        },
        "email_opt_out_sms_allowed": {
            "sms": (
                None,
                f"Hi {first_name}—welcome to {property_name}{city_part}! Book a tour this week.",
            ),
        },
        "quiet_hours_adjustment": {
            "sms": (
                None,
                f"Hi {first_name}—welcome to {property_name}. Book a tour this week at a time that works for you.",
            ),
            "email": (
                f"Welcome to {property_name}",
                f"Hi {first_name},\nThanks for your interest in {property_name}. Book a tour this week.",
            ),
        },
    }

    template = templates.get(slug, templates["prospect_welcome"])
    channel_template = template.get(channel)
    if not channel_template:
        cta_hint = "book a tour" if primary_cta in ("book_tour", "schedule_tour") else "see details"
        if channel == "sms":
            channel_template = (
                None,
                f"Hi {first_name}—update from {property_name}{city_part}. {cta_hint.capitalize()} on our website.",
            )
        elif channel == "email":
            channel_template = (
                f"Update from {property_name}",
                f"Hi {first_name},\nWe have an update from {property_name}{city_part}. {cta_hint.capitalize()} at your convenience.",
            )
        elif channel == "push":
            channel_template = (
                None,
                f"Hi {first_name}—update from {property_name}. Tap to {cta_hint}.",
            )
        else:
            return None, None, None

    subject, body = channel_template[0], channel_template[1]
    if channel in ("sms", "push"):
        subject = None
    body = body + _opt_out_suffix(channel, include_opt_out)
    cta_type = _primary_cta(slug)
    cta: dict[str, Any] = {"type": cta_type}
    if channel == "email":
        paths = {
            "book_tour": "/tour", "schedule_tour": "/tour", "renew_lease": "/renew",
            "pay_rent": "/pay", "rsvp_event": "/events", "view_announcement": "/notice",
            "confirm_maintenance": "/maintenance", "pickup_package": "/packages",
        }
        cta["link"] = _property_link(property_name, paths.get(cta_type, "/tour"))
    elif channel == "sms" and cta_type in ("book_tour", "schedule_tour"):
        cta["options"] = ["Thu", "Fri", "Today", "Tomorrow"][:2]
    elif channel == "push":
        cta["link"] = _property_link(property_name, "/app")
    return subject, body, cta


def _should_suppress_scenario(slug: str) -> bool:
    return slug in {
        "global_opt_out_suppress",
        "all_channels_opted_out",
        "invalid_missing_phone",
        "invalid_missing_email",
    }


def _build_input_record(ctx: ScenarioContext, task_id: str) -> dict[str, Any]:
    rng = ctx.rng
    first_name = _pick(rng, FIRST_NAMES)
    property_name = _pick(rng, PROPERTY_NAMES)
    city, state = _pick(rng, CITIES)
    timezone = _pick(rng, TIMEZONES)
    persona, lifecycle = _scenario_persona(ctx)
    consent = _scenario_consent(ctx)
    contact = _scenario_contact(ctx)
    amenities = rng.sample(AMENITIES, k=rng.randint(1, 3)) if rng.random() < 0.6 else []

    days_ahead = 14
    if ctx.slug == "long_move_in_horizon":
        days_ahead = rng.randint(45, 120)
    elif ctx.slug == "short_move_in_horizon":
        days_ahead = rng.randint(3, 14)

    base_dt = datetime(2025, 12, 8, rng.randint(8, 20), rng.randint(0, 59))
    move_date = (base_dt + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    channel_prefs = ["sms", "email", "push"]
    rng.shuffle(channel_prefs)
    if ctx.slug == "sms_opt_out_email_allowed":
        channel_prefs = ["sms", "email"]
    elif ctx.slug == "email_opt_out_sms_allowed":
        channel_prefs = ["email", "sms"]
    elif ctx.slug == "community_event" and consent["push_opt_in"]:
        channel_prefs = ["push", "email", "sms"]

    profile: dict[str, Any] = {"first_name": first_name}
    if rng.random() < 0.7:
        profile["city_interest"] = f"{city}, {state}"
    if amenities:
        profile["amenity_interest"] = amenities

    constraints: dict[str, Any] = {
        "no_pii_leak": True,
        "no_sensitive_discrimination": True,
        "include_opt_out_instructions": ctx.slug not in ("rent_reminder",),
        "primary_cta": _primary_cta(ctx.slug),
    }

    quiet_hours: dict[str, Any] | None = None
    compliance_hold = False
    if ctx.slug == "quiet_hours_adjustment":
        if ctx.force_suppress_reason == "quiet_hours_no_send_window":
            quiet_hours = {"start": "00:00", "end": "23:59", "timezone": timezone}
        else:
            quiet_hours = {"start": "22:00", "end": "08:00", "timezone": timezone}
        constraints["quiet_hours"] = quiet_hours
    elif ctx.slug == "emergency_maintenance_notice" and quiet_hours:
        pass
    elif rng.random() < 0.15 and ctx.slug not in EDGE_SCENARIO_SLUGS:
        quiet_hours = {"start": "21:00", "end": "08:00", "timezone": timezone}
        constraints["quiet_hours"] = quiet_hours

    if ctx.force_suppress_reason == "sensitive_compliance_risk":
        compliance_hold = True

    record: dict[str, Any] = {
        "task_id": task_id,
        "persona": persona,
        "lifecycle_stage": lifecycle,
        "consent": consent,
        "channel_preferences": channel_prefs,
        "lead_source": _pick(rng, LEAD_SOURCES),
        "contact": contact,
        "input": {
            "property_name": property_name,
            "move_date_target": move_date if persona == "prospect" else None,
            "last_interaction": base_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "timezone": timezone,
            "language": "en",
            "profile": profile,
        },
        "assertions": {
            "required_states": ["consent_verified", "fair_housing_check_passed"],
            "constraints": constraints,
        },
        "thresholds": {
            "p95_latency_ms": 2000,
            "personalization_score_min": round(rng.uniform(0.7, 0.9), 2),
            "reply_classification_f1_min": 0.9,
            "safety_violations_max": 0,
        },
    }

    if persona == "resident" and ctx.slug == "lease_renewal_reminder":
        record["input"]["lease_end_date"] = (base_dt + timedelta(days=rng.randint(30, 90))).strftime("%Y-%m-%d")

    if compliance_hold:
        record["compliance_hold"] = True
        record["fair_housing_review_required"] = True

    if rng.random() < 0.1:
        record["preferred_send_hour"] = rng.choice([8, 9, 10, 11, 17])

    return record


def _resolve_send_channel(
    record: InputRecord,
    ctx: ScenarioContext,
    rng: random.Random,
) -> str | None:
    if ctx.force_channel:
        eligible = _eligible_channels(record)
        if ctx.force_channel in eligible:
            return ctx.force_channel
    channel = _select_channel(record, record.channel_preferences, rng)
    if channel:
        return channel
    return None


def _build_send_expected(
    record: InputRecord,
    ctx: ScenarioContext,
    channel: str,
) -> dict[str, Any]:
    rng = ctx.rng
    profile_extra = getattr(record.input.profile, "model_extra", None) or {}
    city_state = profile_extra.get("city_interest")
    amenities = profile_extra.get("amenity_interest")
    include_opt_out = record.assertions.constraints.include_opt_out_instructions

    subject, body, cta = _message_copy(
        slug=ctx.slug,
        channel=channel,
        first_name=record.input.profile.first_name or "there",
        property_name=record.input.property_name or "the community",
        city_state=city_state,
        amenities=amenities if isinstance(amenities, list) else None,
        move_date=record.input.move_date_target,
        include_opt_out=include_opt_out,
        primary_cta=record.assertions.constraints.primary_cta or "book_tour",
    )

    base_dt = datetime.fromisoformat(
        (record.input.last_interaction or "2025-12-08T12:00:00Z").replace("Z", "+00:00")
    ).replace(tzinfo=None)

    preferred_hour = getattr(record, "model_extra", None) or {}
    preferred_hour_val = preferred_hour.get("preferred_send_hour")
    override = record.assertions.constraints.send_at
    immediate = ctx.slug == "emergency_maintenance_notice" and rng.random() < 0.7

    quiet = record.assertions.constraints.quiet_hours
    if quiet and quiet.get("start") == "00:00" and quiet.get("end") == "23:59":
        raise ValueError("no send window")

    send_at = _next_send_at(
        channel=channel,
        timezone=record.input.timezone,
        base_dt=base_dt,
        preferred_hour=preferred_hour_val,
        override_send_at=override,
        immediate=immediate,
    )

    next_action_type = "follow_up_in_days"
    next_action_details: dict[str, Any] = {"value": rng.choice([1, 2, 3, 5, 7, 14])}
    if ctx.slug in ("prospect_welcome", "quiet_hours_adjustment") or record.lifecycle_stage == "new":
        next_action_type = "start_cadence"
        next_action_details = {"name": "prospect_welcome_short_horizon"}

    expected = {
        "task_id": record.task_id,
        "should_send": True,
        "next_message": {
            "channel": channel,
            "send_at": send_at,
            "subject": subject,
            "body": body,
            "cta": cta,
        },
        "next_action": {"type": next_action_type, "details": next_action_details},
        "reasoning": (
            f"Send a personalized {channel} message for {record.input.property_name} "
            f"honoring consent, channel preferences, and constraints."
        ),
        "quality": {
            "personalization_score": _personalization_score(body, record),
            "safety_violations": 0,
            "latency_ms": rng.randint(450, 1900),
        },
    }
    return expected


def _build_expected_output(record: InputRecord, ctx: ScenarioContext) -> dict[str, Any]:
    if ctx.force_suppress_reason:
        return _build_suppress_expected(
            record.task_id,
            ctx.force_suppress_reason,
            ctx.rng,
        )

    if _should_suppress_scenario(ctx.slug):
        reason_map = {
            "global_opt_out_suppress": "global_opt_out",
            "all_channels_opted_out": "no_eligible_channel",
            "invalid_missing_phone": "missing_contact_info",
            "invalid_missing_email": "missing_contact_info",
        }
        return _build_suppress_expected(record.task_id, reason_map[ctx.slug], ctx.rng)

    extra = getattr(record, "model_extra", None) or {}
    if extra.get("compliance_hold"):
        return _build_suppress_expected(
            record.task_id,
            "sensitive_compliance_risk",
            ctx.rng,
            reasoning="Send suppressed pending fair housing compliance review.",
        )

    quiet = record.assertions.constraints.quiet_hours
    if quiet and quiet.get("start") == "00:00" and quiet.get("end") == "23:59":
        return _build_suppress_expected(
            record.task_id,
            "quiet_hours_no_send_window",
            ctx.rng,
            reasoning="Send suppressed because quiet-hour rules leave no compliant send window.",
        )

    channel = _resolve_send_channel(record, ctx, ctx.rng)
    if not channel:
        return _build_suppress_expected(
            record.task_id,
            "no_eligible_channel",
            ctx.rng,
            reasoning="Send suppressed because no consented channel has valid contact info.",
        )

    return _build_send_expected(record, ctx, channel)


SEND_SCENARIOS = [s for s in SCENARIO_SLUGS if not _should_suppress_scenario(s)]

SUPPRESS_SCENARIOS = [
    "global_opt_out_suppress",
    "all_channels_opted_out",
    "invalid_missing_phone",
    "invalid_missing_email",
    "quiet_hours_adjustment",
]


def _build_allocation(count: int, ratio: float) -> list[bool]:
    send_count = int(round(count * ratio))
    flags = [True] * send_count + [False] * (count - send_count)
    return flags


def _build_channel_allocation(send_count: int) -> list[str]:
    sms_count = int(round(send_count * CHANNEL_WEIGHTS["sms"]))
    email_count = int(round(send_count * CHANNEL_WEIGHTS["email"]))
    push_count = send_count - sms_count - email_count
    return ["sms"] * sms_count + ["email"] * email_count + ["push"] * push_count


def _pick_send_scenario(rng: random.Random, channel: str) -> str:
    if channel == "push":
        push_scenarios = [
            "community_event", "package_notification", "emergency_maintenance_notice",
        ]
        return _pick(rng, push_scenarios)
    if channel == "email":
        email_friendly = [
            "long_move_in_horizon", "lease_renewal_reminder", "rent_reminder",
            "amenity_promotion", "tour_follow_up", "sms_opt_out_email_allowed",
        ]
        if rng.random() < 0.45:
            return _pick(rng, email_friendly)
    return _pick(rng, SEND_SCENARIOS)


def _pick_suppress_scenario(rng: random.Random, split: str, reason: str) -> str:
    if split == "edge_cases":
        reason_map = {
            "global_opt_out": "global_opt_out_suppress",
            "no_eligible_channel": "all_channels_opted_out",
            "missing_contact_info": _pick(rng, ["invalid_missing_phone", "invalid_missing_email"]),
            "quiet_hours_no_send_window": "quiet_hours_adjustment",
            "sensitive_compliance_risk": _pick(rng, SEND_SCENARIOS),
        }
        return reason_map.get(reason, _pick(rng, list(EDGE_SCENARIO_SLUGS)))
    reason_map = {
        "global_opt_out": "global_opt_out_suppress",
        "no_eligible_channel": "all_channels_opted_out",
        "missing_contact_info": _pick(rng, ["invalid_missing_phone", "invalid_missing_email"]),
        "quiet_hours_no_send_window": "quiet_hours_adjustment",
        "sensitive_compliance_risk": _pick(rng, SEND_SCENARIOS),
    }
    return reason_map.get(reason, _pick(rng, SUPPRESS_SCENARIOS))


def _configure_input_for_channel(input_data: dict[str, Any], channel: str) -> None:
    consent = input_data.setdefault("consent", {})
    contact = input_data.setdefault("contact", {})
    consent["global_opt_out"] = False
    consent.setdefault("voice_opt_in", False)
    consent["sms_opt_in"] = channel == "sms"
    consent["email_opt_in"] = channel == "email"
    consent["push_opt_in"] = channel == "push"
    contact["phone_valid"] = True
    contact["email_valid"] = True
    contact["phone"] = contact.get("phone") or "+12145550199"
    contact["email"] = contact.get("email") or "guest@example.com"
    prefs = {"sms": ["sms", "email", "push"], "email": ["email", "sms", "push"], "push": ["push", "email", "sms"]}
    input_data["channel_preferences"] = prefs[channel]
    input_data.pop("compliance_hold", None)
    input_data.pop("fair_housing_review_required", None)
    constraints = input_data.setdefault("assertions", {}).setdefault("constraints", {})
    if constraints.get("quiet_hours", {}).get("start") == "00:00":
        constraints.pop("quiet_hours", None)


def _apply_suppress_reason(input_data: dict[str, Any], reason: str, slug: str) -> None:
    if reason == "global_opt_out" or slug == "global_opt_out_suppress":
        input_data.setdefault("consent", {})["global_opt_out"] = True
    elif reason == "no_eligible_channel" or slug == "all_channels_opted_out":
        input_data["consent"] = {
            "email_opt_in": False, "sms_opt_in": False,
            "voice_opt_in": False, "push_opt_in": False, "global_opt_out": False,
        }
    elif reason == "missing_contact_info" or slug in ("invalid_missing_phone", "invalid_missing_email"):
        if slug == "invalid_missing_email":
            input_data.setdefault("contact", {})["email"] = None
            input_data["contact"]["email_valid"] = False
        elif slug == "invalid_missing_phone":
            input_data.setdefault("contact", {})["phone"] = None
            input_data["contact"]["phone_valid"] = False
        else:
            input_data["contact"] = {
                "phone": None, "email": None, "phone_valid": False, "email_valid": False,
            }
    elif reason == "quiet_hours_no_send_window" or (
        slug == "quiet_hours_adjustment" and reason == "quiet_hours_no_send_window"
    ):
        tz = input_data.get("input", {}).get("timezone", "America/Chicago")
        input_data.setdefault("assertions", {}).setdefault("constraints", {})["quiet_hours"] = {
            "start": "00:00", "end": "23:59", "timezone": tz,
        }
    elif reason == "sensitive_compliance_risk":
        input_data["compliance_hold"] = True
        input_data["fair_housing_review_required"] = True


def generate_row(
    split: str,
    index: int,
    rng: random.Random,
    *,
    should_send: bool,
    force_channel: str | None = None,
    force_suppress_reason: str | None = None,
) -> GeneratedTrainingRow:
    if should_send:
        channel = force_channel or "sms"
        slug = _pick_send_scenario(rng, channel)
        force_suppress = None
    else:
        channel = None
        reason = force_suppress_reason or _pick(rng, SUPPRESSION_REASONS)
        slug = _pick_suppress_scenario(rng, split, reason)
        force_suppress = reason

    task_id = f"{slug}_{split}_{index:05d}"
    ctx = ScenarioContext(
        slug=slug,
        split=split,
        index=index,
        rng=rng,
        force_send=should_send,
        force_channel=channel,
        force_suppress_reason=force_suppress,
    )

    input_data = _build_input_record(ctx, task_id)

    if should_send:
        _configure_input_for_channel(input_data, channel)
    else:
        _apply_suppress_reason(input_data, force_suppress or "no_eligible_channel", slug)

    record = InputRecord.model_validate(input_data)
    expected = _build_expected_output(record, ctx)

    if should_send and not expected["should_send"]:
        _configure_input_for_channel(input_data, channel)
        record = InputRecord.model_validate(input_data)
        expected = _build_send_expected(record, ctx, channel)
    elif not should_send and expected["should_send"]:
        expected = _build_suppress_expected(
            task_id,
            force_suppress or "no_eligible_channel",
            rng,
        )

    return GeneratedTrainingRow(input=input_data, expected=expected)


def validate_generated_row(row: GeneratedTrainingRow) -> list[str]:
    errors: list[str] = []
    try:
        record = InputRecord.model_validate(row.input)
    except ValidationError as exc:
        return [f"input schema invalid: {exc}"]

    try:
        output = parse_agent_output(record.task_id, row.expected)
    except Exception as exc:  # noqa: BLE001
        return [f"expected schema invalid: {exc}"]

    if output.task_id != record.task_id:
        errors.append("task_id mismatch between input and expected")

    if output.quality.safety_violations != 0:
        errors.append("safety_violations must be 0")

    if not output.should_send:
        msg = output.next_message
        if msg.body is not None:
            errors.append("no body for suppressed records")
        if msg.channel not in (None, "none"):
            errors.append("channel must be none when should_send is false")
        if msg.send_at is not None or msg.subject is not None or msg.cta is not None:
            errors.append("suppressed next_message fields must be null")
    else:
        channel = output.next_message.channel
        if channel == "email" and not output.next_message.subject:
            errors.append("email requires non-null subject")
        if channel in ("sms", "push") and output.next_message.subject is not None:
            errors.append(f"{channel} subject must be null")

    errors.extend(validate(output, record))
    return errors


def validate_dataset(rows: list[GeneratedTrainingRow]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for idx, row in enumerate(rows):
        task_id = row.input.get("task_id") or row.expected.get("task_id")
        if task_id in seen:
            errors.append(f"duplicate task_id: {task_id}")
        seen.add(task_id)
        row_errors = validate_generated_row(row)
        for err in row_errors:
            errors.append(f"row {idx} ({task_id}): {err}")
    return errors


def _distribution_stats(rows: list[GeneratedTrainingRow]) -> dict[str, Any]:
    sends = [r for r in rows if r.expected.get("should_send")]
    suppress = [r for r in rows if not r.expected.get("should_send")]
    channel_counts: dict[str, int] = {}
    for row in sends:
        ch = row.expected.get("next_message", {}).get("channel", "unknown")
        channel_counts[ch] = channel_counts.get(ch, 0) + 1
    return {
        "total": len(rows),
        "should_send_true": len(sends),
        "should_send_false": len(suppress),
        "channel_counts": channel_counts,
    }


def generate_split(split: str, count: int, seed: int) -> list[GeneratedTrainingRow]:
    rng = random.Random(seed + hash(split) % 10000)
    send_ratio = 0.65 if split != "edge_cases" else 0.35
    send_flags = _build_allocation(count, send_ratio)
    rng.shuffle(send_flags)

    send_count = sum(send_flags)
    channels = _build_channel_allocation(send_count)
    rng.shuffle(channels)

    suppress_reasons = [_pick(rng, SUPPRESSION_REASONS) for _ in range(count - send_count)]
    channel_idx = 0
    suppress_idx = 0

    rows: list[GeneratedTrainingRow] = []
    attempts = 0
    max_attempts = count * 10
    index = 1
    flag_idx = 0
    while len(rows) < count and attempts < max_attempts:
        attempts += 1
        should_send = send_flags[flag_idx]
        force_channel = channels[channel_idx] if should_send else None
        force_suppress = suppress_reasons[suppress_idx] if not should_send else None
        try:
            row = generate_row(
                split,
                index,
                rng,
                should_send=should_send,
                force_channel=force_channel,
                force_suppress_reason=force_suppress,
            )
            row_errors = validate_generated_row(row)
            if row_errors:
                continue
            rows.append(row)
            if should_send:
                channel_idx += 1
            else:
                suppress_idx += 1
            flag_idx += 1
            index += 1
        except (ValidationError, ValueError):
            continue
    if len(rows) < count:
        raise RuntimeError(f"Generated only {len(rows)}/{count} valid rows for {split}")
    return rows


def write_jsonl(path: Path, rows: list[GeneratedTrainingRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.model_dump(), ensure_ascii=False))
            handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic LoRA training datasets")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "generated"),
        help="Directory for generated JSONL files (default: data/generated)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    all_rows: list[GeneratedTrainingRow] = []
    summaries: dict[str, Any] = {}

    for split, count in SPLIT_COUNTS.items():
        rows = generate_split(split, count, args.seed)
        errors = validate_dataset(rows)
        if errors:
            print(f"Validation failed for {split}:", file=sys.stderr)
            for err in errors[:20]:
                print(f"  - {err}", file=sys.stderr)
            return 1
        path = output_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        stats = _distribution_stats(rows)
        summaries[split] = stats
        all_rows.extend(rows)
        print(f"Wrote {path} ({stats['total']} rows, send={stats['should_send_true']}, suppress={stats['should_send_false']})")

    global_errors = validate_dataset(all_rows)
    if global_errors:
        print("Global validation failed:", file=sys.stderr)
        for err in global_errors[:20]:
            print(f"  - {err}", file=sys.stderr)
        return 1

    summary_path = output_dir / "generation_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
