import os
import sys
from flask import Flask, request, jsonify, render_template

# ---------------------------------------------------------------------------
# Path bridge: connect Semantic AI folder to Causal AI src folder
# ---------------------------------------------------------------------------
CURRENT_DIR     = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR      = os.path.dirname(CURRENT_DIR)
CAUSAL_SRC_PATH = os.path.join(PARENT_DIR, 'causal-ibn-project', 'src')

if CAUSAL_SRC_PATH not in sys.path:
    sys.path.insert(0, CAUSAL_SRC_PATH)

# ---------------------------------------------------------------------------
# Local imports — after path is set up
# ---------------------------------------------------------------------------
from semantic_engine import parse_intent       # fixed spelling
from orchestrator import process_network_intent

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder='templates')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/translate', methods=['POST'])
def translate():
    """
    Full pipeline endpoint:
      1. Receive raw user text
      2. Semantic engine → structured intent JSON
      3. Orchestrator → causal safety check + conflict resolution
      4. Return unified response to frontend

    Fixes vs original:
      - process_network_intent() now receives the raw intent dict directly
        (orchestrator.py was updated to accept dict OR string).
      - Response uses the new status field from fixed orchestrator.py instead
        of the ambiguous is_safe boolean.
      - Semantic errors and orchestrator errors both return proper HTTP 4xx/5xx
        so the frontend can distinguish them from successful (but blocked) intents.
      - causal_analysis key always present in response, even on clean deployment,
        so frontend never has to branch on key existence.
    """
    data      = request.get_json(silent=True)
    user_text = (data or {}).get('text', '').strip()
    source_ip = request.remote_addr

    if not user_text:
        return jsonify({"error": "No text provided."}), 400

    # --- Phase 1: Semantic translation ---
    print(f"\n{'='*52}")
    print(f"  NEW REQUEST from {source_ip}")
    print(f"  Input: \"{user_text}\"")
    print(f"{'='*52}")

    intent_json = parse_intent(user_text, source_ip)

    if "error" in intent_json:
        return jsonify({
            "error":            intent_json["error"],
            "translated_intent": None,
            "causal_analysis":  None,
        }), 422

    # --- Phase 2: Causal inference + conflict resolution ---
    orchestrator_response = process_network_intent(intent_json)

    # --- Phase 3: Build unified frontend response ---
    # Map internal status to a human-readable outcome for the UI
    status = orchestrator_response.get("status", "unknown")
    STATUS_MESSAGES = {
        "deployed_clean":              "Intent deployed successfully — no conflicts detected.",
        "deployed_after_resolution":   "Intent deployed after resolving a bandwidth conflict.",
        "blocked":                     "Intent blocked — network cannot safely support this request.",
        "unknown":                     "Unknown outcome.",
    }

    # Pull counterfactual_details up to the top level for the frontend
    causal = orchestrator_response.get("causal_analysis", {})
    cf_details = (causal.get("safety_report", {}).get("counterfactual_details")
                  or orchestrator_response.get("counterfactual_details", {}))

    response_payload = {
        "translated_intent":      intent_json,
        "causal_analysis":        causal,
        "counterfactual_details": cf_details,
        "is_safe":                orchestrator_response.get("is_safe", False),
        "status":                 status,
        "message":                orchestrator_response.get(
                                      "message",
                                      STATUS_MESSAGES.get(status, "")
                                  ),
    }

    # Return 200 even for blocked intents — blocking is a valid business outcome,
    # not a server error. The frontend checks response.is_safe to decide UI state.
    return jsonify(response_payload), 200


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error.", "detail": str(e)}), 500


if __name__ == '__main__':
    # Never run debug=True in production
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)