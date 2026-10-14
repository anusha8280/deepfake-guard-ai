import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const API_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";
const HISTORY_LIMIT = 8;

type ViewId = "dashboard" | "image" | "video" | "audio" | "history" | "settings";

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

function getVerdictLabel(prediction?: string | null): "REAL" | "FAKE" {
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

  return "REAL";
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

function App() {
  const imageInputRef = useRef<HTMLInputElement>(null);
  const videoInputRef = useRef<HTMLInputElement>(null);
  const audioInputRef = useRef<HTMLInputElement>(null);

  const [activeView, setActiveView] = useState<ViewId>("dashboard");

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState(false);
  const [healthError, setHealthError] = useState("");

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

  const navigation = [
    { id: "dashboard", label: "Dashboard" },
    { id: "image", label: "Image Detection" },
    { id: "video", label: "Video Detection" },
    { id: "audio", label: "Audio Detection" },
    { id: "history", label: "Analysis History" },
    { id: "settings", label: "Settings" },
  ] as const;

  const overview = useMemo(() => {
    const total = history.length;
    const image = history.filter((entry) => entry.media_type === "image").length;
    const video = history.filter((entry) => entry.media_type === "video").length;
    const audio = history.filter((entry) => entry.media_type === "audio").length;

    return { total, image, video, audio };
  }, [history]);

  const recentActivity = history.slice(0, 5);

  async function checkBackend() {
    setHealthLoading(true);
    setHealthError("");

    try {
      const response = await fetch(`${API_URL}/api/health`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = (await response.json()) as HealthResponse;
      setHealth(data);
    } catch {
      setHealth(null);
      setHealthError("Backend offline. Start the FastAPI service on http://127.0.0.1:8000.");
    } finally {
      setHealthLoading(false);
    }
  }

  async function fetchHistory() {
    setHistoryLoading(true);
    setHistoryError("");

    try {
      const response = await fetch(
        `${API_URL}/api/history?limit=${HISTORY_LIMIT}&offset=0`
      );

      if (!response.ok) {
        throw new Error(`Request failed: ${response.status}`);
      }

      const data = (await response.json()) as HistoryResponse;
      setHistory(Array.isArray(data.items) ? data.items : []);
    } catch (error) {
      setHistory([]);
      setHistoryError(
        error instanceof Error ? error.message : "Unable to load analysis history."
      );
    } finally {
      setHistoryLoading(false);
    }
  }

  async function clearHistory() {
    if (!window.confirm("Clear the saved analysis history? This action cannot be undone.")) {
      return;
    }

    setHistoryClearing(true);
    setHistoryError("");

    try {
      const response = await fetch(`${API_URL}/api/history/clear`, {
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
    void checkBackend();
    void fetchHistory();
  }, []);

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

      const response = await fetch(`${API_URL}/api/analysis/image`, {
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

      const response = await fetch(`${API_URL}/api/analysis/video`, {
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

      const response = await fetch(`${API_URL}/api/analysis/audio`, {
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

  return (
    <main className="dashboard-shell">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="brand-mark">D</div>
          <div>
            <div className="brand-label">DeepfakeGuard</div>
            <div className="brand-subtitle">AI Screening</div>
          </div>
        </div>

        <nav className="sidebar-nav" aria-label="Main navigation">
          {navigation.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`nav-button ${activeView === item.id ? "active" : ""}`}
              onClick={() => setActiveView(item.id)}
            >
              <span aria-hidden="true">{item.id === "dashboard" ? "◉" : item.id === "image" ? "◈" : item.id === "video" ? "▣" : item.id === "audio" ? "♫" : item.id === "history" ? "▤" : "⚙"}</span>
              {item.label}
            </button>
          ))}
        </nav>

        <div className="sidebar-status">
          <span className={`connection-dot ${health ? "online" : "offline"}`} aria-hidden="true" />
          <div>
            <strong>{healthLoading ? "Checking" : health ? "Connected" : "Offline"}</strong>
            <small>{health ? `${health.status} • ${health.application}` : "Backend unavailable"}</small>
          </div>
        </div>
      </aside>

      <div className="content-pane">
        <header className="content-header">
          <div>
            <p className="eyebrow">Threat intelligence dashboard</p>
            <h1>AI deepfake detection overview</h1>
          </div>
          <div className={`connection-pill ${health ? "online" : "offline"}`}>
            <span className="connection-dot" aria-hidden="true" />
            {healthLoading ? "Checking backend..." : health ? "Backend connected" : "Backend offline"}
          </div>
        </header>

        {healthError && <p className="error-message" role="alert">{healthError}</p>}

        <section className="overview-grid">
          <article className="overview-card">
            <span className="badge">Total scans</span>
            <strong>{overview.total}</strong>
            <small>Saved detections</small>
          </article>
          <article className="overview-card blue">
            <span className="badge">Image scans</span>
            <strong>{overview.image}</strong>
            <small>Image screening history</small>
          </article>
          <article className="overview-card cyan">
            <span className="badge">Video scans</span>
            <strong>{overview.video}</strong>
            <small>Video frame screening</small>
          </article>
          <article className="overview-card purple">
            <span className="badge">Audio scans</span>
            <strong>{overview.audio}</strong>
            <small>Audio classification</small>
          </article>
        </section>

        <section className="panel-section" id="dashboard">
          <div className="panel-header">
            <div>
              <p className="section-label">Activity feed</p>
              <h2>Recent analysis activity</h2>
            </div>
          </div>

          <div className="dashboard-grid">
            <div className="activity-panel">
              {historyLoading ? (
                <p className="muted-text">Loading saved analysis history...</p>
              ) : historyError ? (
                <p className="error-message">{historyError}</p>
              ) : recentActivity.length > 0 ? (
                <ul className="activity-list">
                  {recentActivity.map((entry) => (
                    <li key={entry.id} className="activity-item">
                      <div>
                        <strong>{entry.filename}</strong>
                        <span>{entry.media_type.toUpperCase()} • {entry.prediction}</span>
                      </div>
                      <div className="activity-meta">
                        <span>{entry.score != null ? formatHistoryScore(entry) : "No score"}</span>
                        <small>{formatTimestamp(entry.timestamp)}</small>
                      </div>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="muted-text">
                  No saved analysis history yet. Run an image, video, or audio scan to populate this dashboard.
                </p>
              )}
            </div>

            <div className="shortcut-panel">
              <p className="section-label">Quick access</p>
              <div className="shortcut-grid">
                <button type="button" className="shortcut-card" onClick={() => setActiveView("image")}>
                  <span>▧</span>
                  <strong>Image</strong>
                  <small>Analyze stills</small>
                </button>
                <button type="button" className="shortcut-card" onClick={() => setActiveView("video")}>
                  <span>▶</span>
                  <strong>Video</strong>
                  <small>Frame screening</small>
                </button>
                <button type="button" className="shortcut-card" onClick={() => setActiveView("audio")}>
                  <span>♫</span>
                  <strong>Audio</strong>
                  <small>Speech checks</small>
                </button>
              </div>
            </div>
          </div>
        </section>

        <section className="panel-section" id="image">
          <div className="panel-header">
            <div>
              <p className="section-label">AI image analysis</p>
              <h2>Analyze an image</h2>
            </div>
            <span className="module-count">MAX 10 MB</span>
          </div>

          <div className="upload-panel">
            <input
              ref={imageInputRef}
              className="file-input"
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={(event) => selectImage(event.target.files?.[0])}
            />

            <button
              type="button"
              className="upload-button"
              onClick={() => imageInputRef.current?.click()}
              disabled={imageLoading}
            >
              Choose image
            </button>

            {imageFile && <p className="selected-file">{imageFile.name}</p>}

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

            {imageError && <p className="error-message" role="alert">{imageError}</p>}

            {imageResult && (
              <div className="result-panel" role="status">
                <p className="section-label">MODEL RESULT</p>
                <div className="result-top-row">
                  <h3>{imageResult.prediction}</h3>
                  <span
                    className={`verdict-badge ${getVerdictLabel(imageResult.prediction) === "FAKE" ? "is-fake" : "is-real"}`}
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

        <section className="panel-section" id="video">
          <div className="panel-header">
            <div>
              <p className="section-label">AI video analysis</p>
              <h2>Analyze a video</h2>
            </div>
            <span className="module-count">MAX 50 MB</span>
          </div>

          <div className="upload-panel">
            <input
              ref={videoInputRef}
              className="file-input"
              type="file"
              accept=".mp4,.m4v,video/mp4,video/x-m4v"
              onChange={(event) => selectVideo(event.target.files?.[0])}
            />

            <button
              type="button"
              className="upload-button"
              onClick={() => videoInputRef.current?.click()}
              disabled={videoLoading}
            >
              Choose video
            </button>

            {videoFile && <p className="selected-file">{videoFile.name}</p>}

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

        <section className="panel-section" id="audio">
          <div className="panel-header">
            <div>
              <p className="section-label">AI audio analysis</p>
              <h2>Analyze an audio recording</h2>
            </div>
            <span className="module-count">MAX 20 MB · 30 SEC</span>
          </div>

          <div className="upload-panel">
            <input
              ref={audioInputRef}
              className="file-input"
              type="file"
              accept=".wav,.mp3,.flac,.ogg,audio/*"
              onChange={(event) => selectAudio(event.target.files?.[0])}
            />

            <button
              type="button"
              className="upload-button"
              onClick={() => audioInputRef.current?.click()}
              disabled={audioLoading}
            >
              Choose audio
            </button>

            {audioFile && <p className="selected-file">{audioFile.name}</p>}

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

            {audioError && <p className="error-message" role="alert">{audioError}</p>}

            {audioResult && (
              <div className="result-panel" role="status">
                <p className="section-label">AUDIO SCREENING RESULT</p>
                <div className="result-top-row">
                  <h3>{audioResult.prediction}</h3>
                  <span
                    className={`verdict-badge ${getVerdictLabel(audioResult.prediction) === "FAKE" ? "is-fake" : "is-real"}`}
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

        <section className="panel-section" id="history">
          <div className="panel-header">
            <div>
              <p className="section-label">Saved detections</p>
              <h2>Analysis history</h2>
            </div>
            <div className="history-actions">
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
              <p className="history-empty">
                No saved analysis history yet. Scan an image, video, or audio clip to populate the dashboard.
              </p>
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
                        <span className={`history-verdict ${getVerdictLabel(entry.prediction) === "FAKE" ? "is-fake" : "is-real"}`}>
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

        <section className="panel-section" id="settings">
          <div className="panel-header">
            <div>
              <p className="section-label">System settings</p>
              <h2>Detection configuration</h2>
            </div>
          </div>

          <div className="settings-panel">
            <div className="settings-card">
              <span>Backend URL</span>
              <strong>{API_URL}</strong>
            </div>
            <div className="settings-card">
              <span>Model notes</span>
              <strong>Output is model-based screening and should be treated as an estimate, not proof.</strong>
            </div>
            <div className="settings-card">
              <span>Current status</span>
              <strong>{health ? `Healthy (${health.status})` : "Unavailable"}</strong>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}

export default App;
