import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { DragEvent, FormEvent } from "react";
import { Link, Navigate, NavLink, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import "./App.css";

const DEFAULT_API_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";
const HISTORY_LIMIT = 200;

type ViewId =
  | "dashboard"
  | "image-detection"
  | "video-detection"
  | "audio-detection"
  | "history"
  | "connectivity"
  | "settings";

type HealthResponse = {
  status: string;
  application: string;
  version: string;
  detection_models: {
    image: string;
    video: string;
    audio: string;
  };
};

type Score = {
  label: string;
  score: number;
  score_percent: number;
};

type ImageResult = {
  status: string;
  filename: string;
  prediction: string;
  confidence: number;
  scores: Score[];
  model: string;
  device: string;
  note: string;
};

type VideoFrameResult = {
  frame_number: number;
  timestamp_seconds: number;
  prediction: string;
  confidence: number;
  scores: Score[];
};

type VideoResult = {
  status: string;
  filename: string;
  duration_seconds: number;
  frames_analyzed: number;
  average_fake_frame_score: number;
  frame_results: VideoFrameResult[];
  model: string;
  note: string;
};

type AudioResult = {
  status: string;
  filename: string;
  prediction: string;
  confidence: number;
  model_label: string;
  scores: Score[];
  duration_seconds: number;
  sample_rate: number;
  model: string;
  device: string;
  note: string;
};

type HistoryEntry = {
  id: number;
  filename: string;
  media_type: "image" | "video" | "audio" | string;
  prediction: string;
  score: number | null;
  timestamp: string;
};

type HistoryResponse = {
  items: HistoryEntry[];
  total: number;
  limit: number;
  offset: number;
};

function clampPercent(value: number | string | null | undefined): number {
  const numericValue = Number(value ?? 0);

  if (!Number.isFinite(numericValue)) {
    return 0;
  }

  return Math.min(100, Math.max(0, numericValue));
}

function getVerdictLabel(prediction?: string | null): "REAL" | "FAKE" | "UNKNOWN" {
  const normalized = String(prediction ?? "").trim().toUpperCase();

  if (
    normalized.includes("FAKE") ||
    normalized.includes("SPOOF") ||
    normalized.includes("DEEPFAKE") ||
    normalized.includes("GENERATED") ||
    normalized.includes("MANIPULATED")
  ) {
    return "FAKE";
  }

  if (
    normalized.includes("REAL") ||
    normalized.includes("AUTHENTIC") ||
    normalized.includes("GENUINE") ||
    normalized.includes("BONAFIDE")
  ) {
    return "REAL";
  }

  return "UNKNOWN";
}

function verdictClass(verdict: "REAL" | "FAKE" | "UNKNOWN"): string {
  if (verdict === "FAKE") return "is-fake";
  if (verdict === "REAL") return "is-real";
  return "is-unknown";
}

function getVideoVerdict(score: number | string | null | undefined): "REAL" | "FAKE" {
  return clampPercent(score) >= 50 ? "FAKE" : "REAL";
}

function formatTimestamp(value: string): string {
  if (!value) {
    return "Unknown time";
  }

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleString([], {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatHistoryScore(entry: HistoryEntry): string {
  if (entry.score == null || Number.isNaN(Number(entry.score))) {
    return "No score available";
  }
  return `${clampPercent(entry.score).toFixed(2)}%`;
}

function formatFileSize(size: number): string {
  if (size < 1024 * 1024) {
    return `${(size / 1024).toFixed(1)} KB`;
  }
  return `${(size / (1024 * 1024)).toFixed(2)} MB`;
}

function localDateKey(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function NeuralNetworkIllustration() {
  return (
    <svg className="neural-illustration" viewBox="0 0 440 310" role="img" aria-label="Abstract neural network and media signal visualization">
      <defs>
        <linearGradient id="networkLine" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0" stopColor="#5dddea" stopOpacity=".08" />
          <stop offset=".52" stopColor="#78c8ff" stopOpacity=".76" />
          <stop offset="1" stopColor="#858eff" stopOpacity=".16" />
        </linearGradient>
        <radialGradient id="networkGlow">
          <stop offset="0" stopColor="#3db8ec" stopOpacity=".24" />
          <stop offset="1" stopColor="#3db8ec" stopOpacity="0" />
        </radialGradient>
        <pattern id="networkGrid" width="24" height="24" patternUnits="userSpaceOnUse">
          <path d="M24 0H0v24" fill="none" stroke="currentColor" strokeOpacity=".09" strokeWidth=".6" />
        </pattern>
        <filter id="nodeGlow" x="-100%" y="-100%" width="300%" height="300%">
          <feGaussianBlur stdDeviation="4" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
      </defs>
      <rect x="35" y="17" width="370" height="276" rx="28" fill="url(#networkGrid)" opacity=".6" />
      <circle cx="229" cy="155" r="137" fill="url(#networkGlow)" />
      <circle cx="229" cy="155" r="103" fill="none" stroke="currentColor" strokeOpacity=".11" />
      <circle cx="229" cy="155" r="76" fill="none" stroke="currentColor" strokeOpacity=".14" strokeDasharray="2 9" />
      <g fill="none" stroke="url(#networkLine)" strokeWidth="1.4">
        <path d="m91 93 75-39 65 44 89-53M91 93l50 80 90-75 73 82 16-155M141 173l-38 70 92 25 109-88 44 48M166 54l27 214m38-170 19 169m70-222-59 142M91 93l210 177m-160-97L320 45M103 243l118-145 67 145" />
      </g>
      <g fill="#101e34" stroke="#73dce9" strokeWidth="1.7" filter="url(#nodeGlow)">
        <circle cx="91" cy="93" r="7" /><circle cx="166" cy="54" r="5" />
        <circle cx="231" cy="98" r="9" /><circle cx="320" cy="45" r="6" />
        <circle cx="141" cy="173" r="8" /><circle cx="287" cy="182" r="7" />
        <circle cx="103" cy="243" r="5" /><circle cx="195" cy="268" r="7" />
        <circle cx="304" cy="180" r="5" /><circle cx="348" cy="228" r="6" />
      </g>
      <g fill="#a5f4f4">
        <circle cx="91" cy="93" r="2" /><circle cx="231" cy="98" r="3" />
        <circle cx="141" cy="173" r="2.5" /><circle cx="195" cy="268" r="2" />
        <circle cx="304" cy="180" r="2" />
      </g>
      <path d="M64 205h51l13-24 21 49 17-35 17 18h74l17-26 19 42 18-24h68" fill="none" stroke="#7fe5ed" strokeOpacity=".65" strokeWidth="1.4" />
      <circle cx="231" cy="98" r="18" fill="none" stroke="#9bb4ff" strokeOpacity=".48" />
    </svg>
  );
}

function MediaForensicsIllustration({ media }: { media: "image" | "video" | "audio" }) {
  return (
    <svg
      className={`media-forensics-art ${media}-forensics-art`}
      viewBox="0 0 360 180"
      role="img"
      aria-label={`Decorative ${media} forensics visualization`}
    >
      <defs>
        <linearGradient id={`art-${media}`} x1="0" x2="1" y1="0" y2="1">
          <stop offset="0" stopColor="#55e0e9" stopOpacity=".95" />
          <stop offset="1" stopColor="#668bff" stopOpacity=".85" />
        </linearGradient>
        <radialGradient id={`art-glow-${media}`}>
          <stop offset="0" stopColor="#278fdb" stopOpacity=".32" />
          <stop offset="1" stopColor="#278fdb" stopOpacity="0" />
        </radialGradient>
        <pattern id={`art-grid-${media}`} width="18" height="18" patternUnits="userSpaceOnUse">
          <path d="M18 0H0v18" fill="none" stroke="#a6d9ff" strokeOpacity=".12" strokeWidth=".6" />
        </pattern>
      </defs>
      <rect x="14" y="12" width="332" height="156" rx="18" fill={`url(#art-glow-${media})`} />
      <rect x="14" y="12" width="332" height="156" rx="18" fill={`url(#art-grid-${media})`} opacity=".68" />
      {media === "image" && (
        <g fill="none" stroke={`url(#art-${media})`} strokeWidth="2">
          <rect x="97" y="37" width="166" height="106" rx="10" fill="#102643" fillOpacity=".68" />
          <circle cx="143" cy="69" r="10" fill="#91eaff" fillOpacity=".42" />
          <path d="m108 126 42-39 28 22 26-31 47 48" />
          <path d="M82 48V31h20M278 31h20v20M82 132v18h20m176 0h20v-18" strokeOpacity=".8" />
          <path d="M100 91h160" stroke="#a6e9ff" strokeOpacity=".28" strokeDasharray="3 6" />
          <path d="M180 36v108" stroke="#80e7ee" strokeOpacity=".35" />
        </g>
      )}
      {media === "video" && (
        <g fill="none" stroke={`url(#art-${media})`} strokeWidth="2">
          <rect x="65" y="42" width="230" height="96" rx="10" fill="#102643" fillOpacity=".7" />
          <path d="M65 63h230M65 117h230" strokeOpacity=".27" />
          <path d="m107 104 43-38 33 26 29-19 49 36" />
          <path d="m164 76 31 18-31 18z" fill="#62d9ed" fillOpacity=".26" />
          <path d="M48 49h9m-9 17h9m-9 17h9m-9 17h9m-9 17h9m246-68h9m-9 17h9m-9 17h9m-9 17h9m-9 17h9" strokeOpacity=".6" />
          <path d="M92 32v10m36-10v10m36-10v10m36-10v10m36-10v10m-144 96v10m36-10v10m36-10v10m36-10v10m36-10v10" strokeOpacity=".5" />
        </g>
      )}
      {media === "audio" && (
        <g fill="none" stroke={`url(#art-${media})`} strokeLinecap="round">
          <circle cx="180" cy="90" r="56" strokeOpacity=".16" strokeWidth="1.5" />
          <circle cx="180" cy="90" r="41" strokeOpacity=".26" strokeWidth="1.5" strokeDasharray="2 7" />
          <path d="M45 90h30l9-17 13 35 12-52 16 75 14-59 13 39 14-28 13 14h19l12-23 13 43 12-61 13 45 11-11h26" strokeWidth="3" />
          <path d="M57 48v-9m25 9V32m25 16v-7m186 7V32m25 16v-7m-261 97v-9m25 9v-16m25 16v-7m186 7v-16m25 16v-7" strokeOpacity=".55" strokeWidth="2" />
          <circle cx="180" cy="90" r="6" fill="#82eaf0" stroke="none" />
        </g>
      )}
      <g fill="#9beaf1">
        <circle cx="36" cy="30" r="2" /><circle cx="324" cy="42" r="2" />
        <circle cx="45" cy="151" r="1.5" /><circle cx="315" cy="145" r="1.5" />
      </g>
    </svg>
  );
}

function SecurityIcon({ name }: { name: ViewId }) {
  const common = {
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.7,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
  };

  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      {name === "dashboard" && (
        <g {...common}>
          <rect x="3.5" y="3.5" width="7" height="7" rx="1.5" />
          <rect x="13.5" y="3.5" width="7" height="4" rx="1.5" />
          <rect x="13.5" y="10.5" width="7" height="10" rx="1.5" />
          <rect x="3.5" y="13.5" width="7" height="7" rx="1.5" />
        </g>
      )}
      {name === "image-detection" && (
        <g {...common}>
          <rect x="3" y="4" width="18" height="16" rx="2.5" />
          <circle cx="8.5" cy="9" r="1.5" />
          <path d="m4 17 5-5 3.5 3 2.5-2 5 4" />
        </g>
      )}
      {name === "video-detection" && (
        <g {...common}>
          <rect x="3" y="5" width="18" height="14" rx="2.5" />
          <path d="m10 9 5 3-5 3z" />
        </g>
      )}
      {name === "audio-detection" && (
        <g {...common}>
          <path d="M3 10v4m4-8v12m4-15v18m4-14v10m4-8v6m3-9v12" />
        </g>
      )}
      {name === "history" && (
        <g {...common}>
          <circle cx="12" cy="12" r="8.5" />
          <path d="M12 7v5l3.5 2M4.5 5.5 3 7m0-3v3h3" />
        </g>
      )}
      {name === "connectivity" && (
        <g {...common}>
          <path d="M5 12a10 10 0 0 1 14 0m-11 3a6 6 0 0 1 8 0m-5 3a2 2 0 0 1 2 0" />
        </g>
      )}
      {name === "settings" && (
        <g {...common}>
          <circle cx="12" cy="12" r="3" />
          <path d="m19 13.5 1.3 1-.9 1.6-1.6-.4a7 7 0 0 1-1.4.8l-.4 1.7h-2l-.5-1.6a7 7 0 0 1-1.5 0l-.9 1.4-1.7-.8.4-1.7a7 7 0 0 1-1-1.3l-1.7.1-.5-1.9 1.5-.8a7 7 0 0 1 0-1.6l-1.3-1 .9-1.6 1.6.4a7 7 0 0 1 1.4-.8l.4-1.7h2l.5 1.6a7 7 0 0 1 1.5 0l.9-1.4 1.7.8-.4 1.7a7 7 0 0 1 1 1.3l1.7-.1.5 1.9-1.5.8a7 7 0 0 1 0 1.6Z" />
        </g>
      )}
    </svg>
  );
}

function normalizeApiUrl(value: string): string | null {
  try {
    const parsed = new URL(value.trim());
    if (!["http:", "https:"].includes(parsed.protocol)) {
      return null;
    }
    return parsed.toString().replace(/\/+$/, "");
  } catch {
    return null;
  }
}

function isHealthResponse(value: unknown): value is HealthResponse {
  if (!value || typeof value !== "object") {
    return false;
  }

  const response = value as Partial<HealthResponse>;
  const models = response.detection_models;
  return (
    typeof response.status === "string" &&
    typeof response.application === "string" &&
    typeof response.version === "string" &&
    !!models &&
    typeof models.image === "string" &&
    typeof models.video === "string" &&
    typeof models.audio === "string"
  );
}

const DEMO_SESSION_KEY = "deepfakeguard-demo-user";

function AuthVisual() {
  return (
    <aside className="login-visual" aria-label="Abstract cybersecurity illustration">
      <div className="login-visual-orbit" aria-hidden="true" />
      <p className="login-visual-kicker">AI MEDIA FORENSICS</p>
      <h2>See Beyond<br /><span>the Surface.</span></h2>
      <p>Screen images, videos, and audio for signals that may indicate manipulated media.</p>
      <NeuralNetworkIllustration />
      <span className="login-visual-caption">AI-assisted screening · Results are estimates</span>
    </aside>
  );
}

function LoginPage({ onLogin }: { onLogin: (email: string) => void }) {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");

  function submitLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");

    const normalizedEmail = email.trim().toLowerCase();
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(normalizedEmail)) {
      setError("Enter a valid email address.");
      return;
    }
    if (password.length < 8) {
      setError("For this demo sign-in, enter a password with at least 8 characters.");
      return;
    }

    onLogin(normalizedEmail);
    navigate("/dashboard", { replace: true });
  }

  return (
    <main className="login-shell auth-shell">
      <AuthVisual />
      <section className="login-card" aria-labelledby="login-title">
        <Link className="login-brand" to="/login">
          <span className="brand-mark" aria-hidden="true">D</span>
          <span>
            <strong>DeepfakeGuard</strong>
            <small>AI Screening</small>
          </span>
        </Link>
        <p className="section-label">DEMO AUTHENTICATION</p>
        <h1 id="login-title">Welcome back</h1>
        <p className="login-description">
          Sign in to continue to your dashboard. This demo session is stored locally
          in this browser and is not production authentication.
        </p>
        <form className="login-form" onSubmit={submitLogin} noValidate>
          <label htmlFor="login-email">Email</label>
          <input
            id="login-email"
            type="email"
            autoComplete="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            required
          />
          <label htmlFor="login-password">Password</label>
          <div className="password-field">
            <input
              id="login-password"
              type={showPassword ? "text" : "password"}
              autoComplete="current-password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              minLength={8}
              required
            />
            <button
              type="button"
              className="password-toggle"
              onClick={() => setShowPassword((visible) => !visible)}
              aria-label={showPassword ? "Hide password" : "Show password"}
            >
              {showPassword ? "Hide" : "Show"}
            </button>
          </div>
          {error && <p className="error-message" role="alert">{error}</p>}
          <button type="submit" className="primary-button login-submit">Sign in</button>
        </form>
        <p className="demo-auth-note">
          Demo only: any valid email and password of at least 8 characters will create a local session.
        </p>
        <p className="auth-switch">
          Don&apos;t have an account? <Link to="/signup">Sign up</Link>
        </p>
      </section>
    </main>
  );
}

function SignupPage({
  onContinue,
}: {
  onContinue: (email: string) => void;
}) {
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState(false);
  const submissionPending = useRef(false);

  async function submitSignup(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (submissionPending.current) return;
    submissionPending.current = true;
    setIsSubmitting(true);
    setError("");

    await new Promise<void>((resolve) => window.setTimeout(resolve, 0));

    const normalizedName = fullName.trim();
    const normalizedEmail = email.trim().toLowerCase();
    if (!normalizedName) {
      setError("Enter your full name.");
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(normalizedEmail)) {
      setError("Enter a valid email address.");
    } else if (password.length < 8) {
      setError("Use a password with at least 8 characters for this demo session.");
    } else if (password !== confirmPassword) {
      setError("The passwords do not match.");
    } else {
      setEmail(normalizedEmail);
      setPassword("");
      setConfirmPassword("");
      setSuccess(true);
    }

    submissionPending.current = false;
    setIsSubmitting(false);
  }

  return (
    <main className="login-shell auth-shell signup-shell">
      <AuthVisual />
      <section className="login-card signup-card" aria-labelledby="signup-title">
        <Link className="login-brand" to="/login">
          <span className="brand-mark" aria-hidden="true">D</span>
          <span>
            <strong>DeepfakeGuard</strong>
            <small>AI Screening</small>
          </span>
        </Link>
        <p className="section-label">DEMO AUTHENTICATION</p>
        {success ? (
          <div className="signup-success" role="status">
            <span className="signup-success-mark" aria-hidden="true">✓</span>
            <h1 id="signup-title">You&apos;re ready to continue</h1>
            <p>
              Your details passed local validation. No account or password was sent to or
              stored by a server; continuing creates a demo session in this browser.
            </p>
            <button
              type="button"
              className="primary-button login-submit"
              onClick={() => onContinue(email)}
            >
              Continue to dashboard <span aria-hidden="true">→</span>
            </button>
          </div>
        ) : (
          <>
            <h1 id="signup-title">Create your demo session</h1>
            <p className="login-description">
              This project has no account-registration backend. Enter details to validate
              the form and continue with a local demo session.
            </p>
            <form className="login-form signup-form" onSubmit={submitSignup} noValidate>
              <label htmlFor="signup-name">Full name</label>
              <input
                id="signup-name"
                type="text"
                autoComplete="name"
                value={fullName}
                onChange={(event) => setFullName(event.target.value)}
                required
                disabled={isSubmitting}
              />
              <label htmlFor="signup-email">Email address</label>
              <input
                id="signup-email"
                type="email"
                autoComplete="email"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                required
                disabled={isSubmitting}
              />
              <label htmlFor="signup-password">Password</label>
              <div className="password-field">
                <input
                  id="signup-password"
                  type={showPassword ? "text" : "password"}
                  autoComplete="new-password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  minLength={8}
                  required
                  disabled={isSubmitting}
                />
                <button
                  type="button"
                  className="password-toggle"
                  onClick={() => setShowPassword((visible) => !visible)}
                  aria-label={showPassword ? "Hide password" : "Show password"}
                  disabled={isSubmitting}
                >
                  {showPassword ? "Hide" : "Show"}
                </button>
              </div>
              <label htmlFor="signup-confirm-password">Confirm password</label>
              <div className="password-field">
                <input
                  id="signup-confirm-password"
                  type={showConfirmPassword ? "text" : "password"}
                  autoComplete="new-password"
                  value={confirmPassword}
                  onChange={(event) => setConfirmPassword(event.target.value)}
                  required
                  disabled={isSubmitting}
                />
                <button
                  type="button"
                  className="password-toggle"
                  onClick={() => setShowConfirmPassword((visible) => !visible)}
                  aria-label={showConfirmPassword ? "Hide confirmation" : "Show confirmation"}
                  disabled={isSubmitting}
                >
                  {showConfirmPassword ? "Hide" : "Show"}
                </button>
              </div>
              {error && <p className="error-message" role="alert">{error}</p>}
              <button type="submit" className="primary-button login-submit" disabled={isSubmitting}>
                {isSubmitting ? "Preparing demo session…" : "Sign up"}
                {!isSubmitting && <span aria-hidden="true">→</span>}
              </button>
            </form>
            <p className="demo-auth-note">
              Demo only. Passwords are used for this form check and are never saved.
            </p>
          </>
        )}
        <p className="auth-switch">
          Already have an account? <Link to="/login">Sign in</Link>
        </p>
      </section>
    </main>
  );
}

function App() {
  const location = useLocation();
  const navigate = useNavigate();
  const [currentUser, setCurrentUser] = useState<string | null>(() => {
    try {
      return localStorage.getItem(DEMO_SESSION_KEY);
    } catch {
      return null;
    }
  });
  const imageInputRef = useRef<HTMLInputElement>(null);
  const videoInputRef = useRef<HTMLInputElement>(null);
  const audioInputRef = useRef<HTMLInputElement>(null);

  const [apiUrl, setApiUrl] = useState(() => {
    const savedUrl = localStorage.getItem("deepfakeguard-api-url");
    return savedUrl ? normalizeApiUrl(savedUrl) ?? DEFAULT_API_URL : DEFAULT_API_URL;
  });
  const [apiUrlDraft, setApiUrlDraft] = useState(apiUrl);
  const [settingsMessage, setSettingsMessage] = useState("");
  const [settingsError, setSettingsError] = useState("");

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState(true);
  const [healthError, setHealthError] = useState("");
  const [healthCheckedAt, setHealthCheckedAt] = useState<string | null>(null);

  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState("");
  const [imageResult, setImageResult] = useState<ImageResult | null>(null);
  const [imageLoading, setImageLoading] = useState(false);
  const [imageError, setImageError] = useState("");

  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoPreview, setVideoPreview] = useState("");
  const [videoResult, setVideoResult] = useState<VideoResult | null>(null);
  const [videoLoading, setVideoLoading] = useState(false);
  const [videoError, setVideoError] = useState("");

  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioPreview, setAudioPreview] = useState("");
  const [audioResult, setAudioResult] = useState<AudioResult | null>(null);
  const [audioLoading, setAudioLoading] = useState(false);
  const [audioError, setAudioError] = useState("");

  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const [historyClearing, setHistoryClearing] = useState(false);
  const [draggingMedia, setDraggingMedia] = useState<"image" | "video" | "audio" | null>(null);
  const [mobileNavigationOpen, setMobileNavigationOpen] = useState(false);

  const navigation: { id: ViewId; label: string; path: string; icon: ViewId }[] = [
    { id: "dashboard", label: "Dashboard", path: "/dashboard", icon: "dashboard" },
    { id: "image-detection", label: "Image Detection", path: "/image-detection", icon: "image-detection" },
    { id: "video-detection", label: "Video Detection", path: "/video-detection", icon: "video-detection" },
    { id: "audio-detection", label: "Audio Detection", path: "/audio-detection", icon: "audio-detection" },
    { id: "history", label: "Analysis History", path: "/history", icon: "history" },
  ] as const;

  const activeView = navigation.find((item) => item.path === location.pathname)?.id;
  const currentPage = navigation.find((item) => item.id === activeView);
  const pageTitle =
    currentPage?.label ??
    (location.pathname === "/connectivity" ? "System Connectivity" : "Settings");
  const isAuthenticated = currentUser !== null;

  const overview = useMemo(() => {
    const total = history.length;
    const image = history.filter((entry) => entry.media_type === "image").length;
    const video = history.filter((entry) => entry.media_type === "video").length;
    const audio = history.filter((entry) => entry.media_type === "audio").length;

    return { total, image, video, audio };
  }, [history]);

  const analytics = useMemo(() => {
    const mediaCounts = {
      image: history.filter((entry) => entry.media_type === "image").length,
      video: history.filter((entry) => entry.media_type === "video").length,
      audio: history.filter((entry) => entry.media_type === "audio").length,
    };
    const predictions = history.reduce(
      (counts, entry) => {
        const verdict = getVerdictLabel(entry.prediction);
        counts[verdict.toLowerCase() as "real" | "fake" | "unknown"] += 1;
        return counts;
      },
      { real: 0, fake: 0, unknown: 0 }
    );
    const today = new Date();
    const dailyActivity = Array.from({ length: 7 }, (_, index) => {
      const day = new Date(today);
      day.setHours(0, 0, 0, 0);
      day.setDate(day.getDate() - (6 - index));
      const key = localDateKey(day);
      return {
        key,
        label: day.toLocaleDateString([], { weekday: "short" }),
        count: history.filter((entry) => {
          const timestamp = new Date(entry.timestamp);
          return !Number.isNaN(timestamp.getTime()) && localDateKey(timestamp) === key;
        }).length,
      };
    });
    return { mediaCounts, predictions, dailyActivity };
  }, [history]);

  function handleFileDrop(
    event: DragEvent<HTMLDivElement>,
    media: "image" | "video" | "audio"
  ) {
    event.preventDefault();
    setDraggingMedia(null);
    const file = event.dataTransfer.files[0];
    if (media === "image") selectImage(file);
    if (media === "video") selectVideo(file);
    if (media === "audio") selectAudio(file);
  }

  function handleDragLeave(event: DragEvent<HTMLDivElement>) {
    if (!(event.relatedTarget instanceof Node) || !event.currentTarget.contains(event.relatedTarget)) {
      setDraggingMedia(null);
    }
  }

  function login(email: string) {
    try {
      localStorage.setItem(DEMO_SESSION_KEY, email);
    } catch {
      // Keep the session for the current app lifetime when storage is unavailable.
    }
    setCurrentUser(email);
  }

  function logout() {
    try {
      localStorage.removeItem(DEMO_SESSION_KEY);
    } catch {
      // The in-memory session is still cleared below.
    }
    setCurrentUser(null);
    navigate("/login", { replace: true });
  }

  const checkBackend = useCallback(async () => {
    setHealthLoading(true);
    setHealthError("");
    setHealth(null);

    try {
      const response = await fetch(`${apiUrl}/api/health`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data: unknown = await response.json();
      if (!isHealthResponse(data)) {
        throw new Error("The backend returned an invalid health response.");
      }
      setHealth(data);
    } catch (error) {
      setHealth(null);
      setHealthError(error instanceof Error ? error.message : "Unable to reach the backend.");
    } finally {
      setHealthCheckedAt(new Date().toISOString());
      setHealthLoading(false);
    }
  }, [apiUrl]);

  const fetchHistory = useCallback(async () => {
    setHistoryLoading(true);
    setHistoryError("");

    try {
      const response = await fetch(
        `${apiUrl}/api/history?limit=${HISTORY_LIMIT}&offset=0`
      );

      if (!response.ok) {
        throw new Error(`Request failed: ${response.status}`);
      }

      const data = (await response.json()) as HistoryResponse;
      const firstPage = Array.isArray(data.items) ? data.items : [];
      const pages = [firstPage];

      for (let offset = HISTORY_LIMIT; offset < data.total; offset += HISTORY_LIMIT) {
        const pageResponse = await fetch(
          `${apiUrl}/api/history?limit=${HISTORY_LIMIT}&offset=${offset}`
        );
        if (!pageResponse.ok) {
          throw new Error(`Request failed: ${pageResponse.status}`);
        }
        const pageData = (await pageResponse.json()) as HistoryResponse;
        pages.push(Array.isArray(pageData.items) ? pageData.items : []);
      }

      setHistory(pages.flat());
    } catch (error) {
      setHistory([]);
      setHistoryError(
        error instanceof Error ? error.message : "Unable to load analysis history."
      );
    } finally {
      setHistoryLoading(false);
    }
  }, [apiUrl]);

  async function clearHistory() {
    if (!window.confirm("Clear the saved analysis history? This action cannot be undone.")) {
      return;
    }

    setHistoryClearing(true);
    setHistoryError("");

    try {
      const response = await fetch(`${apiUrl}/api/history/clear`, {
        method: "DELETE",
      });

      if (!response.ok) {
        throw new Error(`Request failed: ${response.status}`);
      }

      await fetchHistory();
    } catch (error) {
      setHistoryError(
        error instanceof Error ? error.message : "Unable to clear analysis history."
      );
    } finally {
      setHistoryClearing(false);
    }
  }

  useEffect(() => {
    if (!isAuthenticated) {
      return;
    }
    const requestId = window.setTimeout(() => {
      void checkBackend();
      void fetchHistory();
    }, 0);
    return () => window.clearTimeout(requestId);
  }, [checkBackend, fetchHistory, isAuthenticated]);

  function saveApiUrl(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const normalizedUrl = normalizeApiUrl(apiUrlDraft);
    if (!normalizedUrl) {
      setSettingsError("Enter a valid http:// or https:// API base URL.");
      setSettingsMessage("");
      return;
    }

    localStorage.setItem("deepfakeguard-api-url", normalizedUrl);
    setApiUrl(normalizedUrl);
    setApiUrlDraft(normalizedUrl);
    setSettingsError("");
    setSettingsMessage("API URL saved. Backend requests now use this address.");
  }

  function selectImage(file?: File) {
    setImageError("");
    setImageResult(null);
    if (!file) return;

    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
      setImageError("Choose a JPG, PNG, or WEBP image.");
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setImageError("Image size must not exceed 10 MB.");
      return;
    }

    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
  }

  async function analyzeImage() {
    if (!imageFile) {
      setImageError("Please select an image first.");
      return;
    }

    setImageLoading(true);
    setImageError("");
    setImageResult(null);

    try {
      const formData = new FormData();
      formData.append("file", imageFile);

      const response = await fetch(`${apiUrl}/api/analysis/image`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail ?? `Request failed: ${response.status}`);
      }

      setImageResult(data as ImageResult);
      await fetchHistory();
    } catch (error) {
      setImageError(
        error instanceof Error ? error.message : "Image analysis failed."
      );
    } finally {
      setImageLoading(false);
    }
  }

  function selectVideo(file?: File) {
    setVideoError("");
    setVideoResult(null);
    if (!file) return;

    if (!/\.(mp4|m4v)$/i.test(file.name)) {
      setVideoError("Choose an MP4 or M4V video.");
      return;
    }

    if (file.size > 50 * 1024 * 1024) {
      setVideoError("Video size must not exceed 50 MB.");
      return;
    }

    setVideoFile(file);
    setVideoPreview(URL.createObjectURL(file));
  }

  async function analyzeVideo() {
    if (!videoFile) {
      setVideoError("Please select a video first.");
      return;
    }

    setVideoLoading(true);
    setVideoError("");
    setVideoResult(null);

    try {
      const formData = new FormData();
      formData.append("file", videoFile);

      const response = await fetch(`${apiUrl}/api/analysis/video`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail ?? `Request failed: ${response.status}`);
      }

      setVideoResult(data as VideoResult);
      await fetchHistory();
    } catch (error) {
      setVideoError(
        error instanceof Error ? error.message : "Video analysis failed."
      );
    } finally {
      setVideoLoading(false);
    }
  }

  function selectAudio(file?: File) {
    setAudioError("");
    setAudioResult(null);
    if (!file) return;

    if (!/\.(wav|mp3|flac|ogg)$/i.test(file.name)) {
      setAudioError("Choose a WAV, MP3, FLAC, or OGG audio file.");
      return;
    }

    if (file.size > 20 * 1024 * 1024) {
      setAudioError("Audio size must not exceed 20 MB.");
      return;
    }

    setAudioFile(file);
    setAudioPreview(URL.createObjectURL(file));
  }

  async function analyzeAudio() {
    if (!audioFile) {
      setAudioError("Please select an audio file first.");
      return;
    }

    setAudioLoading(true);
    setAudioError("");
    setAudioResult(null);

    try {
      const formData = new FormData();
      formData.append("file", audioFile);

      const response = await fetch(`${apiUrl}/api/analysis/audio`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail ?? `Request failed: ${response.status}`);
      }

      setAudioResult(data as AudioResult);
      await fetchHistory();
    } catch (error) {
      setAudioError(
        error instanceof Error ? error.message : "Audio analysis failed."
      );
    } finally {
      setAudioLoading(false);
    }
  }

  if (location.pathname === "/login") {
    if (isAuthenticated) {
      return <Navigate to="/dashboard" replace />;
    }
    return <LoginPage onLogin={login} />;
  }

  if (location.pathname === "/signup") {
    if (isAuthenticated) {
      return <Navigate to="/dashboard" replace />;
    }
    return (
      <SignupPage
        onContinue={(email) => {
          login(email);
          navigate("/dashboard", { replace: true });
        }}
      />
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  }

  return (
    <main className="dashboard-shell">
      <aside className={`sidebar ${mobileNavigationOpen ? "mobile-nav-open" : ""}`}>
        <div className="sidebar-top-row">
          <Link className="sidebar-brand" to="/dashboard" aria-label="DeepfakeGuard dashboard">
            <div className="brand-mark">D</div>
            <div>
              <div className="brand-label">DeepfakeGuard</div>
              <div className="brand-subtitle">AI Forensics Platform</div>
            </div>
          </Link>
          <button
            type="button"
            className="mobile-nav-toggle"
            aria-label={mobileNavigationOpen ? "Close navigation menu" : "Open navigation menu"}
            aria-expanded={mobileNavigationOpen}
            aria-controls="primary-navigation"
            onClick={() => setMobileNavigationOpen((isOpen) => !isOpen)}
          >
            <span aria-hidden="true">{mobileNavigationOpen ? "×" : "☰"}</span>
          </button>
        </div>

        <nav className="sidebar-nav" id="primary-navigation" aria-label="Main navigation">
          {navigation.map((item) => (
            <NavLink
              key={item.id}
              to={item.path}
              className={`nav-button ${activeView === item.id ? "active" : ""}`}
              onClick={() => setMobileNavigationOpen(false)}
            >
              <span className="nav-icon"><SecurityIcon name={item.icon} /></span>
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-account">
          <span className="sidebar-user" title={currentUser ?? ""}>{currentUser}</span>
          <span className="demo-auth-label">Demo session</span>
          <button type="button" className="logout-button" onClick={logout}>
            Log out
          </button>
        </div>
      </aside>

      <div className="content-pane">
        <header className={`content-header ${activeView === "dashboard" ? "dashboard-hero" : ""}`}>
          <div>
            <p className="eyebrow">{activeView === "dashboard" ? "AI MEDIA FORENSICS" : "DEEPFAKEGUARD AI"}</p>
            <h1>{activeView === "dashboard" ? "See Beyond the Surface." : pageTitle}</h1>
            {activeView === "dashboard" && (
              <>
                <p className="header-subtitle">
                  Analyze images, videos, and audio to identify potential signs of manipulated media.
                </p>
                <div className="hero-actions">
                  <Link to="/image-detection" className="hero-action primary">Start Image Analysis <span aria-hidden="true">→</span></Link>
                  <Link to="/video-detection" className="hero-action">Analyze Video</Link>
                  <Link to="/audio-detection" className="hero-action">Analyze Audio</Link>
                </div>
              </>
            )}
          </div>
          {activeView === "dashboard" && <NeuralNetworkIllustration />}
        </header>

        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={
            <>
        <Link
          to="/connectivity"
          className={`connection-pill ${health ? "online" : "offline"}`}
          aria-label="View backend connectivity"
        >
          <span className="connection-dot" aria-hidden="true" />
          {healthLoading ? "Checking backend..." : health ? "Connected" : "Disconnected"}
        </Link>
        <section className="overview-grid" id="dashboard">
          <article className="overview-card">
            <span className="overview-icon total-icon"><SecurityIcon name="dashboard" /></span>
            <span className="badge">Total saved analyses</span>
            <strong>{overview.total}</strong>
            <small>Saved detections</small>
          </article>
          <article className="overview-card blue">
            <span className="overview-icon image-icon"><SecurityIcon name="image-detection" /></span>
            <span className="badge">Image scans</span>
            <strong>{overview.image}</strong>
            <small>Image screening history</small>
          </article>
          <article className="overview-card cyan">
            <span className="overview-icon video-icon"><SecurityIcon name="video-detection" /></span>
            <span className="badge">Video scans</span>
            <strong>{overview.video}</strong>
            <small>Video frame screening</small>
          </article>
          <article className="overview-card purple">
            <span className="overview-icon audio-icon"><SecurityIcon name="audio-detection" /></span>
            <span className="badge">Audio scans</span>
            <strong>{overview.audio}</strong>
            <small>Audio classification</small>
          </article>
        </section>

        <section className="panel-section">
          <div className="panel-header">
            <div>
              <p className="section-label">Analytics</p>
              <h2>Saved analysis patterns</h2>
            </div>
          </div>

          {historyLoading ? (
            <div className="analytics-grid"><p className="muted-text">Loading saved analysis records…</p></div>
          ) : historyError ? (
            <div className="analytics-grid"><p className="error-message" role="alert">{historyError}</p></div>
          ) : history.length === 0 ? (
            <div className="empty-state analytics-empty">
              <div className="empty-history-illustration"><MediaForensicsIllustration media="image" /></div>
              <span className="empty-state-icon" aria-hidden="true">⌁</span>
              <strong>No saved analyses yet</strong>
              <p>Complete an image, video, or audio screening to populate these charts.</p>
            </div>
          ) : (
            <div className="analytics-grid">
              <article className="chart-card">
                <div className="chart-heading">
                  <div><span className="chart-kicker">BY MEDIA</span><h3>Media comparison</h3></div>
                  <span className="chart-total">{history.length} records</span>
                </div>
                <div className="horizontal-chart">
                  {([
                    ["Image", analytics.mediaCounts.image, "image"],
                    ["Video", analytics.mediaCounts.video, "video"],
                    ["Audio", analytics.mediaCounts.audio, "audio"],
                  ] as const).map(([label, count, key]) => (
                    <div className="horizontal-bar-row" key={key}>
                      <div className="bar-label"><span>{label}</span><strong>{count}</strong></div>
                      <div className="bar-track" role="img" aria-label={`${label}: ${count} analyses`}>
                        <div className={`bar-fill ${key}`} style={{ width: `${overview.total ? (count / overview.total) * 100 : 0}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </article>

              <article className="chart-card">
                <div className="chart-heading">
                  <div><span className="chart-kicker">LAST 7 DAYS</span><h3>Daily activity</h3></div>
                  <span className="chart-total">{analytics.dailyActivity.reduce((sum, day) => sum + day.count, 0)} records</span>
                </div>
                <div className="daily-chart" role="img" aria-label="Saved analysis records per day over the last seven calendar days">
                  {analytics.dailyActivity.map((day) => {
                    const max = Math.max(1, ...analytics.dailyActivity.map((item) => item.count));
                    return (
                      <div className="daily-column" key={day.key} title={`${day.label}: ${day.count}`}>
                        <strong>{day.count || ""}</strong>
                        <div className="daily-bar-rail"><div className="daily-bar" style={{ height: `${day.count ? Math.max(8, (day.count / max) * 100) : 0}%` }} /></div>
                        <span>{day.label}</span>
                      </div>
                    );
                  })}
                </div>
              </article>

              <article className="chart-card prediction-card">
                <div className="chart-heading">
                  <div><span className="chart-kicker">CLASSIFICATION</span><h3>Prediction distribution</h3></div>
                  <span className="chart-total">{history.length} records</span>
                </div>
                <div className="distribution-layout">
                  <div
                    className="prediction-donut"
                    role="img"
                    aria-label={`Predictions: ${analytics.predictions.real} real, ${analytics.predictions.fake} fake, ${analytics.predictions.unknown} unknown`}
                    style={{
                      background: `conic-gradient(#59d8cf 0 ${(analytics.predictions.real / history.length) * 100}%, #a78bfa ${(analytics.predictions.real / history.length) * 100}% ${((analytics.predictions.real + analytics.predictions.fake) / history.length) * 100}%, #506078 ${((analytics.predictions.real + analytics.predictions.fake) / history.length) * 100}% 100%)`,
                    }}
                  ><span>{history.length}<small>scans</small></span></div>
                  <div className="distribution-legend">
                    <div><i className="legend-dot real" /><span>REAL</span><strong>{analytics.predictions.real}</strong></div>
                    <div><i className="legend-dot fake" /><span>FAKE</span><strong>{analytics.predictions.fake}</strong></div>
                    <div><i className="legend-dot unknown" /><span>UNKNOWN</span><strong>{analytics.predictions.unknown}</strong></div>
                  </div>
                </div>
                <p className="chart-footnote">Unrecognized prediction labels are counted as unknown, not REAL.</p>
              </article>

              <article className="chart-card quick-links-card">
                <div className="chart-heading"><div><span className="chart-kicker">WORKSPACE</span><h3>Quick navigation</h3></div></div>
                <div className="quick-link-grid">
                  <Link to="/image-detection" className="quick-link"><span aria-hidden="true"><SecurityIcon name="image-detection" /></span><strong>Image</strong><small>Screen an image</small></Link>
                  <Link to="/video-detection" className="quick-link"><span aria-hidden="true"><SecurityIcon name="video-detection" /></span><strong>Video</strong><small>Analyze sampled frames</small></Link>
                  <Link to="/audio-detection" className="quick-link"><span aria-hidden="true"><SecurityIcon name="audio-detection" /></span><strong>Audio</strong><small>Screen an audio file</small></Link>
                  <Link to="/history" className="quick-link"><span aria-hidden="true"><SecurityIcon name="history" /></span><strong>History</strong><small>Review saved records</small></Link>
                </div>
              </article>
            </div>
          )}
        </section>

        <section className="panel-section recent-table-section">
          <div className="panel-header">
            <div><p className="section-label">LATEST RECORDS</p><h2>Recent analysis</h2></div>
            <Link to="/history" className="text-link">View all history →</Link>
          </div>
          {historyLoading ? (
            <div className="table-card muted-text">Loading recent records…</div>
          ) : historyError ? (
            <div className="table-card"><p className="error-message" role="alert">{historyError}</p></div>
          ) : history.length === 0 ? (
            <div className="table-card empty-state"><strong>No saved records</strong><p>Recent analyses will appear here after a successful scan.</p></div>
          ) : (
            <div className="table-card table-scroll">
              <table className="analysis-table">
                <thead><tr><th>Filename</th><th>Media</th><th>Prediction</th><th>Score</th><th>Timestamp</th></tr></thead>
                <tbody>
                  {history.slice(0, 8).map((entry) => {
                    const verdict = getVerdictLabel(entry.prediction);
                    return <tr key={entry.id}>
                      <td className="table-filename" title={entry.filename}>{entry.filename}</td>
                      <td><span className={`media-chip ${entry.media_type}`}>{entry.media_type}</span></td>
                      <td><span className={`history-verdict ${verdict === "FAKE" ? "is-fake" : verdict === "REAL" ? "is-real" : "is-unknown"}`}>{verdict}</span></td>
                      <td>{entry.score == null ? "—" : entry.media_type === "video" ? `${formatHistoryScore(entry)} avg fake-frame` : formatHistoryScore(entry)}</td>
                      <td>{formatTimestamp(entry.timestamp)}</td>
                    </tr>;
                  })}
                </tbody>
              </table>
            </div>
          )}
        </section>
            </>
          } />

          <Route path="/image-detection" element={
        <section className="panel-section" id="image">
          <div className="panel-header detection-heading">
            <div>
              <p className="section-label">IMAGE FORENSICS</p>
              <h2>Inspect an image for authenticity signals</h2>
              <p className="module-description">Screen a supported image with the configured model and review its returned prediction and scores.</p>
            </div>
            <span className="module-count">MAX 10 MB</span>
          </div>

          <div
            className={`upload-panel ${draggingMedia === "image" ? "is-dragging" : ""}`}
            onDragEnter={(event) => { event.preventDefault(); setDraggingMedia("image"); }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={handleDragLeave}
            onDrop={(event) => handleFileDrop(event, "image")}
          >
            <input
              ref={imageInputRef}
              className="file-input"
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={(event) => selectImage(event.target.files?.[0])}
            />

            <div className="upload-visual-banner image-banner">
              <MediaForensicsIllustration media="image" />
              <div><span>IMAGE SCAN</span><strong>Visual signal review</strong><small>Your selected image is previewed below before analysis.</small></div>
            </div>
            <div className="drop-prompt">
              <span className="drop-icon" aria-hidden="true">↑</span>
              <div><strong>Drop an image here</strong><small>JPG, PNG, or WEBP · Up to 10 MB</small></div>
            </div>
            <button
              type="button"
              className="upload-button"
              onClick={() => imageInputRef.current?.click()}
              disabled={imageLoading}
            >
              Choose image
            </button>

            {imageFile && <p className="selected-file">{imageFile.name} <span>{imageFile.type || "Image"} · {formatFileSize(imageFile.size)}</span></p>}

            {imagePreview && (
              <div className="image-preview">
                <img src={imagePreview} alt="Selected image preview" />
              </div>
            )}

            <button
              type="button"
              className="primary-button analyze-button"
              onClick={analyzeImage}
              disabled={!imageFile || imageLoading}
            >
              {imageLoading ? "Analyzing image..." : "Analyze image"}
              {!imageLoading && <span aria-hidden="true">→</span>}
            </button>
            {imageLoading && <div className="analysis-progress" role="status"><span />Analyzing image with the configured model…</div>}

            {imageError && <p className="error-message" role="alert">{imageError}</p>}

            {imageResult && (
              <div className="result-panel" role="status">
                <p className="section-label">MODEL RESULT</p>
                <div className="result-top-row">
                  <h3>{imageResult.prediction}</h3>
                  <span
                    className={`verdict-badge ${verdictClass(getVerdictLabel(imageResult.prediction))}`}
                  >
                    {getVerdictLabel(imageResult.prediction)}
                  </span>
                </div>

                <p>Confidence: {clampPercent(imageResult.confidence).toFixed(2)}%</p>

                <div className="score-row">
                  <div className="score-label">
                    <span>Confidence</span>
                    <span>{clampPercent(imageResult.confidence).toFixed(2)}%</span>
                  </div>
                  <div className="score-track">
                    <div
                      className="score-fill"
                      style={{ width: `${clampPercent(imageResult.confidence)}%` }}
                    />
                  </div>
                </div>

                {imageResult.scores.map((score) => (
                  <div className="score-row" key={score.label}>
                    <div className="score-label">
                      <span>{score.label}</span>
                      <span>{clampPercent(score.score_percent).toFixed(2)}%</span>
                    </div>
                    <div className="score-track">
                      <div
                        className="score-fill"
                        style={{ width: `${clampPercent(score.score_percent)}%` }}
                      />
                    </div>
                  </div>
                ))}

                <p className="result-note">{imageResult.note}</p>
                <p className="result-meta">Model: {imageResult.model} · Device: {imageResult.device}</p>
                <p className="result-note danger-note">
                  Warning: AI screening results are estimates, not definitive proof.
                </p>
              </div>
            )}
          </div>
        </section>
          } />

          <Route path="/video-detection" element={
        <section className="panel-section" id="video">
          <div className="panel-header detection-heading">
            <div>
              <p className="section-label">VIDEO FORENSICS</p>
              <h2>Review sampled frames from a video</h2>
              <p className="module-description">The current workflow samples frames and reports an average fake-frame score; it does not perform temporal analysis.</p>
            </div>
            <span className="module-count">MAX 50 MB</span>
          </div>

          <div
            className={`upload-panel ${draggingMedia === "video" ? "is-dragging" : ""}`}
            onDragEnter={(event) => { event.preventDefault(); setDraggingMedia("video"); }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={handleDragLeave}
            onDrop={(event) => handleFileDrop(event, "video")}
          >
            <input
              ref={videoInputRef}
              className="file-input"
              type="file"
              accept=".mp4,.m4v,video/mp4,video/x-m4v"
              onChange={(event) => selectVideo(event.target.files?.[0])}
            />

            <div className="upload-visual-banner video-banner">
              <MediaForensicsIllustration media="video" />
              <div><span>FRAME REVIEW</span><strong>Sampled visual sequence</strong><small>View the selected video preview and returned frame details.</small></div>
            </div>
            <div className="drop-prompt">
              <span className="drop-icon" aria-hidden="true">↑</span>
              <div><strong>Drop a video here</strong><small>MP4 or M4V · Up to 50 MB</small></div>
            </div>
            <button
              type="button"
              className="upload-button"
              onClick={() => videoInputRef.current?.click()}
              disabled={videoLoading}
            >
              Choose video
            </button>

            {videoFile && <p className="selected-file">{videoFile.name} <span>{videoFile.type || "Video"} · {formatFileSize(videoFile.size)}</span></p>}

            {videoPreview && (
              <div className="video-preview">
                <video src={videoPreview} controls preload="metadata" />
              </div>
            )}

            <button
              type="button"
              className="primary-button analyze-button"
              onClick={analyzeVideo}
              disabled={!videoFile || videoLoading}
            >
              {videoLoading ? "Analyzing video frames..." : "Analyze video"}
              {!videoLoading && <span aria-hidden="true">→</span>}
            </button>
            {videoLoading && <div className="analysis-progress" role="status"><span />Analyzing sampled video frames…</div>}

            {videoError && <p className="error-message" role="alert">{videoError}</p>}

            {videoResult && (
              <div className="result-panel" role="status">
                <p className="section-label">VIDEO SCREENING RESULT</p>
                <div className="result-top-row">
                  <h3>{videoResult.average_fake_frame_score.toFixed(2)}% average fake-frame score</h3>
                  <span
                    className={`verdict-badge ${getVideoVerdict(videoResult.average_fake_frame_score) === "FAKE" ? "is-fake" : "is-real"}`}
                  >
                    {getVideoVerdict(videoResult.average_fake_frame_score)}
                  </span>
                </div>

                <p>Duration: {videoResult.duration_seconds.toFixed(2)} seconds</p>
                <p>Frames analyzed: {videoResult.frames_analyzed}</p>

                <div className="score-row">
                  <div className="score-label">
                    <span>Average fake-frame score</span>
                    <span>{clampPercent(videoResult.average_fake_frame_score).toFixed(2)}%</span>
                  </div>
                  <div className="score-track">
                    <div
                      className="score-fill"
                      style={{
                        width: `${clampPercent(videoResult.average_fake_frame_score)}%`,
                      }}
                    />
                  </div>
                </div>

                <h4>Sampled frame results</h4>
                {videoResult.frame_results.map((frame) => (
                  <div className="frame-result" key={frame.frame_number}>
                    <strong>Frame {frame.frame_number}</strong>
                    <span>{frame.timestamp_seconds.toFixed(2)} s</span>
                    <span>{frame.prediction}</span>
                    <span>{frame.confidence.toFixed(2)}%</span>
                  </div>
                ))}

                <p className="result-note">{videoResult.note}</p>
                <p className="result-meta">Model: {videoResult.model}</p>
                <p className="result-note danger-note">
                  Warning: AI screening results are estimates, not definitive proof.
                </p>
              </div>
            )}
          </div>
        </section>
          } />

          <Route path="/audio-detection" element={
        <section className="panel-section" id="audio">
          <div className="panel-header detection-heading">
            <div>
              <p className="section-label">AUDIO FORENSICS</p>
              <h2>Screen an audio recording</h2>
              <p className="module-description">Submit a supported audio file and review the model's returned prediction, scores, and metadata.</p>
            </div>
            <span className="module-count">MAX 20 MB · 30 SEC</span>
          </div>

          <div
            className={`upload-panel ${draggingMedia === "audio" ? "is-dragging" : ""}`}
            onDragEnter={(event) => { event.preventDefault(); setDraggingMedia("audio"); }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={handleDragLeave}
            onDrop={(event) => handleFileDrop(event, "audio")}
          >
            <input
              ref={audioInputRef}
              className="file-input"
              type="file"
              accept=".wav,.mp3,.flac,.ogg,audio/*"
              onChange={(event) => selectAudio(event.target.files?.[0])}
            />

            <div className="upload-visual-banner audio-banner">
              <MediaForensicsIllustration media="audio" />
              <div><span>AUDIO SIGNAL</span><strong>Audio screening workspace</strong><small>Choose a file to preview and submit for analysis.</small></div>
            </div>
            <div className="drop-prompt">
              <span className="drop-icon" aria-hidden="true">↑</span>
              <div><strong>Drop an audio file here</strong><small>WAV, MP3, FLAC, or OGG · Up to 20 MB</small></div>
            </div>
            <button
              type="button"
              className="upload-button"
              onClick={() => audioInputRef.current?.click()}
              disabled={audioLoading}
            >
              Choose audio
            </button>

            {audioFile && <p className="selected-file">{audioFile.name} <span>{audioFile.type || "Audio"} · {formatFileSize(audioFile.size)}</span></p>}

            {audioPreview && (
              <div className="audio-preview">
                <audio src={audioPreview} controls preload="metadata" />
              </div>
            )}

            <button
              type="button"
              className="primary-button analyze-button"
              onClick={analyzeAudio}
              disabled={!audioFile || audioLoading}
            >
              {audioLoading ? "Analyzing audio..." : "Analyze audio"}
              {!audioLoading && <span aria-hidden="true">→</span>}
            </button>
            {audioLoading && <div className="analysis-progress" role="status"><span />Analyzing audio with the configured model…</div>}

            {audioError && <p className="error-message" role="alert">{audioError}</p>}

            {audioResult && (
              <div className="result-panel" role="status">
                <p className="section-label">AUDIO SCREENING RESULT</p>
                <div className="result-top-row">
                  <h3>{audioResult.prediction}</h3>
                  <span
                    className={`verdict-badge ${verdictClass(getVerdictLabel(audioResult.prediction))}`}
                  >
                    {getVerdictLabel(audioResult.prediction)}
                  </span>
                </div>

                <p>Top model score: {clampPercent(audioResult.confidence).toFixed(2)}%</p>
                <p>Duration: {audioResult.duration_seconds.toFixed(2)} seconds</p>
                <p>Sample rate: {audioResult.sample_rate} Hz</p>
                <p>Model label: {audioResult.model_label}</p>

                <div className="score-row">
                  <div className="score-label">
                    <span>Confidence</span>
                    <span>{clampPercent(audioResult.confidence).toFixed(2)}%</span>
                  </div>
                  <div className="score-track">
                    <div
                      className="score-fill"
                      style={{ width: `${clampPercent(audioResult.confidence)}%` }}
                    />
                  </div>
                </div>

                {audioResult.scores.map((score) => (
                  <div className="score-row" key={score.label}>
                    <div className="score-label">
                      <span>{score.label}</span>
                      <span>{clampPercent(score.score_percent).toFixed(2)}%</span>
                    </div>
                    <div className="score-track">
                      <div
                        className="score-fill"
                        style={{ width: `${clampPercent(score.score_percent)}%` }}
                      />
                    </div>
                  </div>
                ))}

                <p className="result-note">{audioResult.note}</p>
                <p className="result-meta">Model: {audioResult.model} · Device: {audioResult.device}</p>
                <p className="result-note danger-note">
                  Warning: AI screening results are estimates, not definitive proof.
                </p>
              </div>
            )}
          </div>
        </section>
          } />

          <Route path="/history" element={
        <section className="panel-section" id="history">
          <div className="panel-header">
            <div>
              <p className="section-label">Saved detections</p>
              <h2>Analysis history</h2>
              <p className="history-total">{history.length} saved {history.length === 1 ? "record" : "records"}</p>
            </div>
            <div className="history-actions">
              <button
                type="button"
                className="upload-button"
                onClick={() => void fetchHistory()}
                disabled={historyLoading || historyClearing}
              >
                {historyLoading ? "Refreshing..." : "Refresh"}
              </button>
              <button
                type="button"
                className="clear-history-button"
                onClick={clearHistory}
                disabled={historyClearing || historyLoading}
              >
                {historyClearing ? "Clearing..." : "Clear History"}
              </button>
            </div>
          </div>

          {historyLoading ? (
            <div className="history-panel">
              <p className="muted-text">Loading analysis history...</p>
            </div>
          ) : historyError ? (
            <div className="history-panel">
              <p className="error-message">{historyError}</p>
            </div>
          ) : history.length === 0 ? (
            <div className="history-panel">
              <div className="empty-history-illustration"><MediaForensicsIllustration media="image" /></div>
              <p className="history-empty">
                No saved analysis history yet. Scan an image, video, or audio clip to populate the dashboard.
              </p>
              <div className="empty-actions">
                <Link to="/image-detection" className="hero-action primary">Start an image analysis <span aria-hidden="true">→</span></Link>
                <Link to="/video-detection" className="hero-action">Analyze video</Link>
                <Link to="/audio-detection" className="hero-action">Analyze audio</Link>
              </div>
            </div>
          ) : (
            <div className="history-panel">
              <ul className="history-list">
                {history.map((entry) => (
                  <li key={entry.id} className="history-item">
                    <div className="history-main">
                      <div className="history-meta-row">
                        <span className="history-file">{entry.filename}</span>
                        <span className={`history-badge ${entry.media_type}`}>{entry.media_type}</span>
                      </div>
                      <div className="history-result-row">
                        <strong>{entry.prediction}</strong>
                        <span className={`history-verdict ${verdictClass(getVerdictLabel(entry.prediction))}`}>
                          {getVerdictLabel(entry.prediction)}
                        </span>
                      </div>
                      <div className="history-score-row">
                        <span>
                          {entry.media_type === "video"
                            ? "Average fake-frame score"
                            : "Confidence"}
                        </span>
                        <span>{entry.score != null ? formatHistoryScore(entry) : "No score"}</span>
                      </div>
                    </div>
                    <div className="history-time">{formatTimestamp(entry.timestamp)}</div>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>
          } />

          <Route path="/connectivity" element={
        <section className="panel-section" id="connectivity">
          <div className="panel-header">
            <div>
              <p className="section-label">Backend service</p>
              <h2>System connectivity</h2>
            </div>
            <button
              type="button"
              className="upload-button"
              onClick={() => void checkBackend()}
              disabled={healthLoading}
            >
              {healthLoading ? "Checking..." : "Refresh connection"}
            </button>
          </div>
          <div className="connectivity-panel">
            <div className={`connectivity-status ${health ? "online" : "offline"}`}>
              <span className="connection-dot" aria-hidden="true" />
              <strong>{healthLoading ? "Checking" : health ? "Connected" : "Disconnected / Unavailable"}</strong>
            </div>
            <dl className="connectivity-details">
              <div><dt>API URL</dt><dd>{apiUrl}/api/health</dd></div>
              <div><dt>Last checked</dt><dd>{healthCheckedAt ? formatTimestamp(healthCheckedAt) : "Not checked yet"}</dd></div>
              {health && (
                <>
                  <div><dt>Status</dt><dd>{health.status}</dd></div>
                  <div><dt>Application</dt><dd>{health.application}</dd></div>
                  <div><dt>Version</dt><dd>{health.version}</dd></div>
                  <div><dt>Image model</dt><dd>{health.detection_models.image}</dd></div>
                  <div><dt>Video model</dt><dd>{health.detection_models.video}</dd></div>
                  <div><dt>Audio model</dt><dd>{health.detection_models.audio}</dd></div>
                </>
              )}
            </dl>
            {healthError && <p className="error-message" role="alert">{healthError}</p>}
          </div>
        </section>
          } />

          <Route path="/settings" element={
        <section className="panel-section" id="settings">
          <div className="panel-header">
            <div>
              <p className="section-label">System settings</p>
              <h2>Detection configuration</h2>
            </div>
          </div>

          <div className="settings-panel">
            <div className="settings-card">
              <span>Model notes</span>
              <strong>Output is model-based screening and should be treated as an estimate, not proof.</strong>
            </div>
            <form className="settings-card settings-form" onSubmit={saveApiUrl}>
              <label htmlFor="api-url">Backend API base URL</label>
              <input
                id="api-url"
                name="api-url"
                type="url"
                value={apiUrlDraft}
                onChange={(event) => setApiUrlDraft(event.target.value)}
                placeholder="http://127.0.0.1:8000"
                required
              />
              <small>Used for detection, history, and health-check requests.</small>
              <button type="submit" className="primary-button">Save API URL</button>
              {settingsMessage && <p className="success-message" role="status">{settingsMessage}</p>}
              {settingsError && <p className="error-message" role="alert">{settingsError}</p>}
            </form>
          </div>
        </section>
          } />
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </div>
    </main>
  );
}

export default App;
