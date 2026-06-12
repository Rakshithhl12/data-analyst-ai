"""
Autonomous Data Analyst — Flask App with Authentication
========================================================
"""
import json, logging, os, uuid, secrets
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path

from flask import (Flask, request, jsonify, render_template,
                   send_from_directory, Response, redirect, url_for,
                   session, flash)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Simple file-based user store (no DB dependency for portability) ──────────
USERS_FILE = Path("instance/users.json")

def _load_users():
    if USERS_FILE.exists():
        return json.loads(USERS_FILE.read_text())
    return {}

def _save_users(users):
    USERS_FILE.parent.mkdir(exist_ok=True)
    USERS_FILE.write_text(json.dumps(users, indent=2))

def _get_user(email):
    return _load_users().get(email.lower())

def _create_user(name, email, password):
    users = _load_users()
    if email.lower() in users:
        return False, "Email already registered"
    users[email.lower()] = {
        "id": str(uuid.uuid4()),
        "name": name,
        "email": email.lower(),
        "password": generate_password_hash(password),
        "created_at": datetime.utcnow().isoformat(),
        "sessions": []
    }
    _save_users(users)
    return True, "OK"

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_email" not in session:
            if request.is_json or request.path.startswith("/upload") or \
               request.path.startswith("/analyze") or request.path.startswith("/chat"):
                return jsonify({"error": "Authentication required"}), 401
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated

# ─────────────────────────────────────────────────────────────────────────────

def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"]              = os.getenv("SECRET_KEY", "ada-super-secret-2024")
    app.config["MAX_CONTENT_LENGTH"]      = 50 * 1024 * 1024
    app.config["UPLOAD_FOLDER"]           = Path("uploads")
    app.config["REPORTS_FOLDER"]          = Path("reports")
    app.config["CHARTS_FOLDER"]           = Path("charts")
    app.config["VECTOR_STORE_FOLDER"]     = Path("vector_store")
    app.config["ALLOWED_EXTENSIONS"]      = {"csv", "xlsx", "xls"}
    # Session settings — required for cookies to persist on production hosts
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["SESSION_COOKIE_SECURE"]   = os.getenv("FLASK_ENV") == "production"
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=7)
    for k in ("UPLOAD_FOLDER","REPORTS_FOLDER","CHARTS_FOLDER","VECTOR_STORE_FOLDER"):
        app.config[k].mkdir(parents=True, exist_ok=True)

    from workflow import run_analysis_workflow
    from agents.chat_agent import ChatAgent
    _chat = ChatAgent()

    def allowed(fn):
        return "." in fn and fn.rsplit(".",1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

    # ── Auth pages ────────────────────────────────────────────────────────────
    @app.route("/login")
    def login_page():
        if "user_email" in session:
            return redirect(url_for("index"))
        return render_template("login.html")

    @app.route("/register")
    def register_page():
        if "user_email" in session:
            return redirect(url_for("index"))
        return render_template("register.html")

    @app.route("/api/register", methods=["POST"])
    def api_register():
        d = request.get_json(silent=True) or {}
        name  = (d.get("name","")).strip()
        email = (d.get("email","")).strip()
        pwd   = d.get("password","")
        if not name or not email or not pwd:
            return jsonify({"error":"All fields required"}), 400
        if len(pwd) < 6:
            return jsonify({"error":"Password must be at least 6 characters"}), 400
        ok, msg = _create_user(name, email, pwd)
        if not ok:
            return jsonify({"error": msg}), 409
        session.permanent     = True
        session["user_email"] = email.lower()
        session["user_name"]  = name
        return jsonify({"message":"Registered successfully"}), 200

    @app.route("/api/login", methods=["POST"])
    def api_login():
        d = request.get_json(silent=True) or {}
        email = (d.get("email","")).strip().lower()
        pwd   = d.get("password","")
        user  = _get_user(email)
        if not user or not check_password_hash(user["password"], pwd):
            return jsonify({"error":"Invalid email or password"}), 401
        session.permanent     = True
        session["user_email"] = email
        session["user_name"]  = user["name"]
        return jsonify({"message":"Login successful", "name": user["name"]}), 200

    @app.route("/api/logout", methods=["POST"])
    def api_logout():
        session.clear()
        return jsonify({"message":"Logged out"}), 200

    @app.route("/api/me")
    def api_me():
        if "user_email" not in session:
            return jsonify({"authenticated": False}), 401
        return jsonify({"authenticated": True,
                        "email": session["user_email"],
                        "name": session.get("user_name","User")}), 200

    # ── Main pages ────────────────────────────────────────────────────────────
    @app.route("/")
    @login_required
    def index():
        return render_template("index.html")

    @app.route("/dashboard")
    @login_required
    def dashboard():
        return render_template("dashboard.html")

    @app.route("/report-page")
    @login_required
    def report_page():
        return render_template("report.html")

    @app.route("/history")
    @login_required
    def history_page():
        return render_template("history.html")

    # ── POST /upload ──────────────────────────────────────────────────────────
    @app.route("/upload", methods=["POST"])
    @login_required
    def upload_file():
        if "file" not in request.files:
            return jsonify({"error":"No file part"}), 400
        f = request.files["file"]
        if not f.filename:
            return jsonify({"error":"No file selected"}), 400
        if not allowed(f.filename):
            return jsonify({"error":"Only CSV and Excel files accepted"}), 400
        sid      = str(uuid.uuid4())
        filename = secure_filename(f.filename)
        dest     = app.config["UPLOAD_FOLDER"] / sid
        dest.mkdir(parents=True, exist_ok=True)
        f.save(str(dest / filename))
        # Track in user history
        _add_session_to_user(session["user_email"], sid, filename)
        logger.info("Uploaded: %s  session=%s", filename, sid)
        return jsonify({"message":"Uploaded","session_id":sid,"filename":filename}), 200

    def _add_session_to_user(email, sid, filename):
        try:
            users = _load_users()
            if email in users:
                if "sessions" not in users[email]:
                    users[email]["sessions"] = []
                users[email]["sessions"].insert(0, {
                    "session_id": sid, "filename": filename,
                    "created_at": datetime.utcnow().isoformat(),
                    "status": "pending"
                })
                users[email]["sessions"] = users[email]["sessions"][:20]
                _save_users(users)
        except Exception as e:
            logger.warning("Could not save session to user: %s", e)

    def _update_session_status(email, sid, status):
        try:
            users = _load_users()
            if email in users:
                for s in users[email].get("sessions", []):
                    if s["session_id"] == sid:
                        s["status"] = status
                        break
                _save_users(users)
        except Exception as e:
            logger.warning("Could not update session status: %s", e)

    # ── POST /analyze ─────────────────────────────────────────────────────────
    @app.route("/analyze", methods=["POST"])
    @login_required
    def analyze():
        d   = request.get_json(silent=True) or {}
        sid = d.get("session_id","")
        if not sid:
            return jsonify({"error":"session_id required"}), 400
        upload_dir = app.config["UPLOAD_FOLDER"] / sid
        if not upload_dir.exists():
            return jsonify({"error":"Session not found"}), 404
        files = list(upload_dir.iterdir())
        if not files:
            return jsonify({"error":"No file found for session"}), 404
        try:
            result = run_analysis_workflow(
                filepath         = str(files[0]),
                session_id       = sid,
                charts_dir       = str(app.config["CHARTS_FOLDER"]       / sid),
                reports_dir      = str(app.config["REPORTS_FOLDER"]      / sid),
                vector_store_dir = str(app.config["VECTOR_STORE_FOLDER"] / sid),
            )
            _update_session_status(session["user_email"], sid,
                                   result.get("status","complete"))
            return jsonify(result), 200
        except Exception as e:
            logger.exception("Analysis failed session=%s", sid)
            _update_session_status(session["user_email"], sid, "failed")
            return jsonify({"error": str(e)}), 500

    # ── POST /chat ────────────────────────────────────────────────────────────
    @app.route("/chat", methods=["POST"])
    @login_required
    def chat():
        d        = request.get_json(silent=True) or {}
        sid      = d.get("session_id","")
        question = d.get("question","").strip()
        if not sid or not question:
            return jsonify({"error":"session_id and question required"}), 400
        vs_path = str(app.config["VECTOR_STORE_FOLDER"] / sid)
        try:
            answer = _chat.answer(question=question,
                                  vector_store_path=vs_path,
                                  session_id=sid)
            return jsonify({"answer": answer}), 200
        except Exception as e:
            logger.exception("Chat failed session=%s", sid)
            return jsonify({"error": str(e)}), 500

    # ── GET /report/<sid> ─────────────────────────────────────────────────────
    @app.route("/report/<sid>")
    @login_required
    def get_report(sid):
        f = app.config["REPORTS_FOLDER"] / sid / "report.html"
        if not f.exists():
            return ("<html><body style='background:#0f0f23;color:#94a3b8;"
                    "font-family:system-ui;text-align:center;padding:4rem'>"
                    "<h2>Report not found</h2><p>Run analysis first.</p>"
                    "</body></html>", 404, {"Content-Type":"text/html"})
        return Response(f.read_text(encoding="utf-8"), mimetype="text/html")

    # ── GET /report/pdf/<sid> ─────────────────────────────────────────────────
    @app.route("/report/pdf/<sid>")
    @login_required
    def get_pdf(sid):
        f = app.config["REPORTS_FOLDER"] / sid / "report.pdf"
        if not f.exists():
            return jsonify({"error":"PDF not found"}), 404
        return send_from_directory(str(app.config["REPORTS_FOLDER"] / sid),
                                   "report.pdf", as_attachment=True,
                                   download_name=f"report_{sid[:8]}.pdf")

    # ── GET /charts ───────────────────────────────────────────────────────────
    @app.route("/charts/<sid>")
    @login_required
    def list_charts(sid):
        d = app.config["CHARTS_FOLDER"] / sid
        if not d.exists():
            return jsonify({"charts":[]}), 200
        charts = sorted(f.name for f in d.iterdir() if f.suffix in (".png",".html"))
        return jsonify({"charts": charts}), 200

    @app.route("/charts/<sid>/<path:fname>")
    @login_required
    def serve_chart(sid, fname):
        resp = send_from_directory(str(app.config["CHARTS_FOLDER"] / sid), fname)
        resp.headers["X-Frame-Options"] = "SAMEORIGIN"
        return resp

    # ── GET /status/<sid> ────────────────────────────────────────────────────
    @app.route("/status/<sid>")
    @login_required
    def get_status(sid):
        f = app.config["REPORTS_FOLDER"] / sid / "status.json"
        if f.exists():
            return Response(f.read_text(), mimetype="application/json")
        return jsonify({"status":"pending"}), 200

    # ── GET /history ──────────────────────────────────────────────────────────
    @app.route("/api/history")
    @login_required
    def api_history():
        user = _get_user(session["user_email"])
        if not user:
            return jsonify({"sessions":[]}), 200
        return jsonify({"sessions": user.get("sessions",[])[:20]}), 200

    # ── DELETE /session/<sid> ─────────────────────────────────────────────────
    @app.route("/session/<sid>", methods=["DELETE"])
    @login_required
    def delete_session(sid):
        import shutil
        for folder_key in ("UPLOAD_FOLDER","REPORTS_FOLDER","CHARTS_FOLDER","VECTOR_STORE_FOLDER"):
            p = app.config[folder_key] / sid
            if p.exists():
                shutil.rmtree(str(p))
        # Remove from user history
        try:
            users = _load_users()
            email = session["user_email"]
            if email in users:
                users[email]["sessions"] = [
                    s for s in users[email].get("sessions",[])
                    if s["session_id"] != sid
                ]
                _save_users(users)
        except Exception: pass
        return jsonify({"message":"Deleted"}), 200

    # ── Forgot / Reset password ──────────────────────────────────────────────
    @app.route("/forgot-password")
    def forgot_page():
        return render_template("forgot_password.html")

    @app.route("/reset-password")
    def reset_page():
        return render_template("reset_password.html")

    @app.route("/api/forgot-password", methods=["POST"])
    def api_forgot_password():
        d     = request.get_json(silent=True) or {}
        email = d.get("email", "").strip().lower()
        user  = _get_user(email)
        # Always return 200 to prevent email enumeration
        if user:
            token = secrets.token_urlsafe(32)
            expiry = (datetime.utcnow() + timedelta(hours=1)).isoformat()
            users = _load_users()
            users[email]["reset_token"]  = token
            users[email]["reset_expiry"] = expiry
            _save_users(users)
            reset_url = f"{request.host_url}reset-password?token={token}"
            # Log for development (replace with real email in production)
            logger.info("PASSWORD RESET LINK for %s: %s", email, reset_url)
            # TODO: send actual email via SendGrid/Mailgun/SES
        return jsonify({"message": "If that email exists, a reset link has been sent."}), 200

    @app.route("/api/reset-password", methods=["POST"])
    def api_reset_password():
        d     = request.get_json(silent=True) or {}
        token = d.get("token", "").strip()
        pwd   = d.get("password", "")
        if not token or not pwd or len(pwd) < 6:
            return jsonify({"error": "Invalid request"}), 400
        users = _load_users()
        matched_email = None
        for email, user in users.items():
            if user.get("reset_token") == token:
                expiry = user.get("reset_expiry", "")
                if expiry and datetime.fromisoformat(expiry) > datetime.utcnow():
                    matched_email = email
                break
        if not matched_email:
            return jsonify({"error": "Reset link is invalid or has expired."}), 400
        users[matched_email]["password"]    = generate_password_hash(pwd)
        users[matched_email]["reset_token"]  = None
        users[matched_email]["reset_expiry"] = None
        _save_users(users)
        return jsonify({"message": "Password updated successfully"}), 200

    # ── Google OAuth ──────────────────────────────────────────────────────────
    @app.route("/auth/google")
    def google_oauth():
        google_client_id = os.getenv("GOOGLE_CLIENT_ID", "")
        if not google_client_id:
            # OAuth not configured — redirect to login with message
            return redirect(url_for("login_page") + "?msg=google_not_configured")
        callback_url = url_for("google_callback", _external=True)
        import urllib.parse
        params = urllib.parse.urlencode({
            "client_id":     google_client_id,
            "redirect_uri":  callback_url,
            "response_type": "code",
            "scope":         "openid email profile",
            "access_type":   "offline",
            "state":         secrets.token_urlsafe(16),
        })
        return redirect(f"https://accounts.google.com/o/oauth2/v2/auth?{params}")

    @app.route("/auth/google/callback")
    def google_callback():
        import urllib.parse, urllib.request
        code = request.args.get("code", "")
        if not code:
            return redirect(url_for("login_page"))
        google_client_id     = os.getenv("GOOGLE_CLIENT_ID", "")
        google_client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "")
        callback_url         = url_for("google_callback", _external=True)
        try:
            # Exchange code for token
            token_data = urllib.parse.urlencode({
                "code":          code,
                "client_id":     google_client_id,
                "client_secret": google_client_secret,
                "redirect_uri":  callback_url,
                "grant_type":    "authorization_code",
            }).encode()
            req = urllib.request.Request(
                "https://oauth2.googleapis.com/token",
                data=token_data,
                method="POST"
            )
            with urllib.request.urlopen(req) as r:
                token_resp = json.loads(r.read())
            access_token = token_resp.get("access_token", "")
            # Get user info
            uinfo_req = urllib.request.Request(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            with urllib.request.urlopen(uinfo_req) as r:
                uinfo = json.loads(r.read())
            email = uinfo.get("email", "").lower()
            name  = uinfo.get("name", email.split("@")[0])
            if not email:
                return redirect(url_for("login_page"))
            # Create user if not exists
            users = _load_users()
            if email not in users:
                users[email] = {
                    "id":         str(uuid.uuid4()),
                    "name":       name,
                    "email":      email,
                    "password":   "",  # Google users have no local password
                    "provider":   "google",
                    "created_at": datetime.utcnow().isoformat(),
                    "sessions":   []
                }
                _save_users(users)
            session["user_email"] = email
            session["user_name"]  = name
            return redirect(url_for("index"))
        except Exception as e:
            logger.exception("Google OAuth failed: %s", e)
            return redirect(url_for("login_page"))

    # ── Health ────────────────────────────────────────────────────────────────
    @app.route("/health")
    def health():
        return jsonify({"status":"ok","gemini":bool(os.getenv("GEMINI_API_KEY"))}), 200

    @app.errorhandler(413)
    def too_large(_): return jsonify({"error":"File too large. Max 50 MB."}), 413
    @app.errorhandler(404)
    def not_found(_): return jsonify({"error":"Not found"}), 404

    return app

app = create_app()

if __name__ == "__main__":
    port  = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG","false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)