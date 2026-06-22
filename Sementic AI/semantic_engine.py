import json
import os
import socket
import google.generativeai as genai

# ---------------------------------------------------------------------------
# API key — read from environment variable, never hardcoded in source
# ---------------------------------------------------------------------------
_api_key = os.environ.get("GEMINI_API_KEY", "")
if not _api_key:
    raise EnvironmentError(
        "GEMINI_API_KEY environment variable is not set. "
        "Run: export GEMINI_API_KEY='your-key-here'"
    )

genai.configure(api_key=_api_key)
_model = genai.GenerativeModel('gemini-2.5-flash')

# ---------------------------------------------------------------------------
# Valid values — single source of truth shared with inference.py
# ---------------------------------------------------------------------------
VALID_METRICS = [
    "latency_ms",
    "packet_loss_percent",
    "jitter_ms",
    "bandwidth_used_mbps",
]

# Per-application SLA defaults — used by inference.py danger limits
# Add more applications here as your network grows
APP_SLA_DEFAULTS = {
    "gaming":        {"latency_ms": 30.0,  "packet_loss_percent": 0.5,  "jitter_ms": 5.0},
    "video_stream":  {"latency_ms": 50.0,  "packet_loss_percent": 0.1,  "jitter_ms": 10.0},
    "video_call":    {"latency_ms": 20.0,  "packet_loss_percent": 0.1,  "jitter_ms": 2.0},
    "voip":          {"latency_ms": 10.0,  "packet_loss_percent": 0.1,  "jitter_ms": 2.0},
    "backup_sync":   {"latency_ms": 500.0, "packet_loss_percent": 2.0,  "jitter_ms": 50.0},
    "web_browsing":  {"latency_ms": 100.0, "packet_loss_percent": 1.0,  "jitter_ms": 20.0},
}
DEFAULT_SLA = {"latency_ms": 50.0, "packet_loss_percent": 1.0, "jitter_ms": 10.0}


def get_device_name(ip_address):
    """Resolve IP to hostname. Falls back gracefully."""
    if ip_address in ("127.0.0.1", "localhost", "::1"):
        return socket.gethostname()
    try:
        hostname, _, _ = socket.gethostbyaddr(ip_address)
        return hostname
    except (socket.herror, socket.gaierror):
        return f"Device ({ip_address})"


def _validate_and_clean(ai_data, source_ip):
    """
    Validates the LLM output and fills in any missing fields.
    Returns a clean intent dict or raises ValueError with a clear message.
    """
    requirements = ai_data.get("requirements", {})

    # 1. application must be present
    app = requirements.get("application", "").strip().lower()
    if not app:
        raise ValueError("LLM response missing 'requirements.application'.")
    requirements["application"] = app

    # 2. critical_metrics must be a non-empty list of known metrics
    metrics = requirements.get("critical_metrics", [])
    if not isinstance(metrics, list) or not metrics:
        # Fall back: choose the most critical metric for this app from SLA table
        sla = APP_SLA_DEFAULTS.get(app, DEFAULT_SLA)
        metrics = [min(sla, key=sla.get)]  # metric with tightest limit
        print(f"   [WARN] LLM gave no critical_metrics — defaulting to {metrics}")
    else:
        # Filter out any metric the LLM hallucinated
        metrics = [m for m in metrics if m in VALID_METRICS]
        if not metrics:
            metrics = ["latency_ms"]
            print(f"   [WARN] LLM gave unknown metrics — defaulting to {metrics}")
    requirements["critical_metrics"] = metrics[:2]  # cap at 2 as per prompt

    # 3. bandwidth_guarantee must parse to a float
    constraints = requirements.get("constraints", {})
    bw_str = constraints.get("bandwidth_guarantee", "0Mbps")
    try:
        float(bw_str.replace("Mbps", "").replace("mbps", "").strip())
    except (ValueError, AttributeError):
        constraints["bandwidth_guarantee"] = "0Mbps"
        print(f"   [WARN] Could not parse bandwidth_guarantee '{bw_str}' — set to 0Mbps")
    requirements["constraints"] = constraints

    # 4. Attach SLA thresholds so inference.py can use them without re-lookup
    requirements["sla_thresholds"] = APP_SLA_DEFAULTS.get(app, DEFAULT_SLA)

    # 5. Ensure target block is complete
    target = ai_data.get("target", {})
    target["ip_address"]   = target.get("ip_address", source_ip)
    target["device_name"]  = get_device_name(source_ip)
    ai_data["target"]      = target

    ai_data["requirements"] = requirements
    ai_data["status"]       = "Ready for Causal Analysis"

    return ai_data


def parse_intent(user_text, source_ip="127.0.0.1"):
    """
    Calls Gemini to translate free-form user text into a structured network intent.

    Returns a validated intent dict, or {"error": "..."} on failure.

    Fixes vs original:
      - API key read from environment variable (not hardcoded).
      - Prompt explicitly lists valid metric names so LLM cannot hallucinate.
      - Output validated and sanitised before returning — bad LLM output is
        caught here, not buried as a crash in inference.py.
      - SLA thresholds attached to the intent so inference.py uses per-app
        limits instead of magic numbers.
      - Specific exception types caught (not bare except).
      - File-level _model instance reused across calls (not re-created each time).
    """
    print(f"\n   -> [Semantic Engine] Parsing intent for: '{user_text}'")

    prompt = f"""
You are the Semantic Engine for an Intent-Based Network (IBN).
Translate the user's request into a strict JSON network intent.

Rules:
- 'application' must be a single lowercase string (e.g. "gaming", "video_stream", "voip").
- 'critical_metrics' must be an array of 1 or 2 values chosen ONLY from:
  {json.dumps(VALID_METRICS)}
  Pick the metrics most critical for the application's quality of experience.
- 'bandwidth_guarantee' must be a string like "10Mbps".
- Output ONLY valid JSON. No markdown, no explanation, no extra text.

Output format:
{{
    "intent_category": "Reactive",
    "action": "prioritize",
    "target": {{
        "ip_address": "{source_ip}"
    }},
    "requirements": {{
        "application": "<app_name>",
        "critical_metrics": ["<metric_1>"],
        "constraints": {{
            "bandwidth_guarantee": "<number>Mbps"
        }}
    }}
}}

User request: "{user_text}"
"""

    try:
        response  = _model.generate_content(prompt)
        raw_text  = response.text.strip()

        # Strip markdown code fences if Gemini wraps the JSON
        raw_text = raw_text.removeprefix("```json").removeprefix("```")
        raw_text = raw_text.removesuffix("```").strip()

        ai_data = json.loads(raw_text)

    except json.JSONDecodeError as e:
        print(f"   [!] Gemini returned invalid JSON: {e}")
        print(f"   [!] Raw output was: {response.text[:300]}")
        return {"error": "Semantic engine returned malformed JSON."}

    except Exception as e:
        print(f"   [!] Gemini API error: {e}")
        return {"error": f"Semantic engine failed: {e}"}

    # Validate and enrich before returning
    try:
        clean_intent = _validate_and_clean(ai_data, source_ip)
        print(f"   -> Intent parsed: app={clean_intent['requirements']['application']}, "
              f"metrics={clean_intent['requirements']['critical_metrics']}, "
              f"bw={clean_intent['requirements']['constraints']['bandwidth_guarantee']}")
        return clean_intent

    except ValueError as e:
        print(f"   [!] Intent validation failed: {e}")
        return {"error": str(e)}
