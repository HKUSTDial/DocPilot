import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { pdfjs } from "react-pdf";
import ReactMarkdown from "react-markdown";
import { fetchWorkspaceFiles, streamChat, toAbsoluteUrl, uploadWorkspaceFiles } from "../api";
import type { ModelConfig, WorkspaceFile } from "../types";
import { DEMO_BEATS, DEMO_DOCUMENTS } from "./demo";
import { FLIGHT_TIMING, crossBookRoute, flightBeatsForEvent, flightPageAt, flightTravelTime, pageKey, resolveFlightTheme, supportingEvidenceIds, type FlightBeat, type FlightBookRoute, type FlightEvidence, type FlightLocation } from "./flight";
import { FlightOverlay } from "./FlightOverlay";
import { FlightBook } from "./FlightBook";

pdfjs.GlobalWorkerOptions.workerSrc = new URL("pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url).toString();

function initialConfig(): ModelConfig {
  const fallback: ModelConfig = { provider: "vllm", base_url: "http://127.0.0.1:8000/v1", api_key: "", model: "", reasoning: false, timeout: 180 };
  try {
    const saved = JSON.parse(localStorage.getItem("docpilot-chat-config-v2") ?? "null");
    if (saved && typeof saved === "object") return {
      ...fallback, provider: saved.provider === "custom" ? "custom" : "vllm",
      base_url: typeof saved.base_url === "string" ? saved.base_url : fallback.base_url,
      model: typeof saved.model === "string" ? saved.model : "",
    };
  } catch { return fallback; }
  return fallback;
}

function PlaneIcon() {
  return <svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="m21 3-7 18-4-8-8-4 19-6ZM10 13 21 3" /></svg>;
}

const isPdf = (file: WorkspaceFile) => file.extension.toLowerCase() === ".pdf";
const THEME_KEY = "docpilot-flight-theme-v1";
type BookTransfer = FlightBookRoute & { phase: "closing" | "opening"; duration: number };

function initialTheme() {
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  try { return resolveFlightTheme(localStorage.getItem(THEME_KEY), prefersDark); }
  catch { return resolveFlightTheme(null, prefersDark); }
}

export default function FlightApp() {
  const [mode, setMode] = useState<"live" | "demo">(() => new URLSearchParams(location.search).get("demo") === "1" ? "demo" : "live");
  const [files, setFiles] = useState<WorkspaceFile[]>([]);
  const [selectedNames, setSelectedNames] = useState<string[]>([]);
  const [config, setConfig] = useState(initialConfig);
  const [theme, setTheme] = useState(initialTheme);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [question, setQuestion] = useState("");
  const [mission, setMission] = useState("");
  const [beats, setBeats] = useState<FlightBeat[]>([]);
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [paused, setPaused] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [calm, setCalm] = useState(() => window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  const [focus, setFocus] = useState<FlightLocation | null>(null);
  const [visited, setVisited] = useState<Record<string, "seen" | "candidate">>({});
  const [landed, setLanded] = useState<FlightEvidence[]>([]);
  const [collected, setCollected] = useState<FlightEvidence[]>([]);
  const [selectedEvidence, setSelectedEvidence] = useState<FlightEvidence | null>(null);
  const [answer, setAnswer] = useState("");
  const [supportingIds, setSupportingIds] = useState<string[]>([]);
  const [phase, setPhase] = useState<"idle" | "turning" | "travelling" | "delivering" | "complete" | "stopped">("idle");
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [loadingFiles, setLoadingFiles] = useState(false);
  const [bookReadyKey, setBookReadyKey] = useState("");
  const [bookTransfer, setBookTransfer] = useState<BookTransfer | null>(null);
  const sceneRef = useRef<HTMLDivElement>(null);
  const evidenceListRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const controllerRef = useRef<AbortController | null>(null);
  const runRef = useRef(0);
  const initializedSelectionRef = useRef(false);
  const pausedRef = useRef(paused);
  const speedRef = useRef(speed);
  const calmRef = useRef(calm);
  pausedRef.current = paused;
  speedRef.current = speed;
  calmRef.current = calm;

  const visibleFiles = mode === "demo" ? DEMO_DOCUMENTS : files;
  const selectedFiles = mode === "demo" ? DEMO_DOCUMENTS : files.filter((file) => selectedNames.includes(file.name) && isPdf(file));
  const activeFile = focus ? visibleFiles.find((file) => file.doc_id === focus.docId) : selectedFiles[0] ?? visibleFiles.find(isPdf);
  const current = beats[cursor];
  const busy = playing || streaming;
  const preparing = selectedFiles.some((file) => file.processing_status !== "ready");
  const currentEvidence = current?.kind === "evidence" ? current.evidence : null;
  const flightLocation = current?.kind === "visit" ? current.location : currentEvidence;
  const planeLocation = playing ? flightPageAt(beats, cursor) : focus;
  const waitingForBook = Boolean(playing && (bookTransfer || (flightLocation && bookReadyKey !== pageKey(flightLocation))));
  const transferRoute = bookTransfer ?? (playing ? crossBookRoute(flightPageAt(beats, cursor - 1), flightLocation) : null);
  const transferLabel = transferRoute ? [transferRoute.from, transferRoute.to].map((location) => `${visibleFiles.find((file) => file.doc_id === location.docId)?.name ?? location.docId} · p.${location.page}`).join(" → ") : null;
  const scanning = Boolean(playing && currentEvidence && !waitingForBook && phase === "travelling");
  const status = error ? "Flight interrupted" : paused ? "View paused" : bookTransfer ? "Switching documents"
    : phase === "complete" ? "Mission complete" : phase === "stopped" ? "Flight stopped" : !playing ? "Ready for takeoff"
      : waitingForBook ? `Opening page ${flightLocation?.page}` : !current ? "Waiting for the agent" : current.kind === "stage" ? current.title
        : current.kind === "visit" ? `Exploring page ${current.location.page}`
          : current.kind === "evidence" ? phase === "delivering" ? "Evidence recorded" : "Scanning evidence" : "Answer assembled";

  useEffect(() => () => { runRef.current += 1; controllerRef.current?.abort(); }, []);

  const handleBookReady = useCallback((key: string) => {
    setBookReadyKey(key);
    setBookTransfer((transfer) => transfer?.phase === "opening" && pageKey(transfer.to) === key ? null : transfer);
  }, []);

  function navigateTo(location: FlightLocation) {
    if (bookTransfer && pageKey(bookTransfer.to) === pageKey(location)) return;
    const from = focus ?? (activeFile ? { docId: activeFile.doc_id, page: 1 } : null);
    const route = crossBookRoute(from, location);
    if (!route || calmRef.current || !activeFile || !visibleFiles.some((file) => file.doc_id === location.docId)) {
      setBookTransfer(null);
      setFocus(location);
      return;
    }
    setFocus(route.from);
    setBookTransfer({ ...route, phase: "closing", duration: Math.max(120, FLIGHT_TIMING.bookClose / speedRef.current) });
  }

  useEffect(() => {
    if (!bookTransfer || bookTransfer.phase !== "closing") return;
    const departure = bookTransfer;
    let elapsed = 0;
    let previous = performance.now();
    let frame = 0;
    function closeBook(now: number) {
      elapsed += pausedRef.current ? 0 : Math.min(now - previous, 100);
      previous = now;
      if (calm || elapsed >= departure.duration) {
        setFocus(departure.to);
        setBookTransfer((transfer) => transfer === departure ? { ...transfer, phase: "opening" } : transfer);
      } else frame = requestAnimationFrame(closeBook);
    }
    frame = requestAnimationFrame(closeBook);
    return () => cancelAnimationFrame(frame);
  }, [bookTransfer, calm]);

  useLayoutEffect(() => {
    document.documentElement.dataset.flightTheme = theme;
    document.querySelector('meta[name="theme-color"]')?.setAttribute("content", theme === "day" ? "#f1f4f1" : "#111d28");
    try { localStorage.setItem(THEME_KEY, theme); } catch {}
  }, [theme]);

  useEffect(() => {
    const list = evidenceListRef.current;
    if (list) list.scrollTo({ top: list.scrollHeight, behavior: calm ? "auto" : "smooth" });
  }, [collected.length, calm]);

  useEffect(() => {
    if (mode !== "live") return;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    async function refresh() {
      setLoadingFiles(true);
      try {
        const updated = await fetchWorkspaceFiles("default");
        if (!active) return;
        setFiles(updated);
        if (!initializedSelectionRef.current) {
          initializedSelectionRef.current = true;
          setSelectedNames(updated.filter((file) => isPdf(file) && file.processing_status !== "failed").slice(0, 1).map((file) => file.name));
        }
      } catch (failure) {
        if (active) setError(String(failure));
      } finally {
        if (active) {
          setLoadingFiles(false);
          timer = setTimeout(refresh, 4000);
        }
      }
    }
    void refresh();
    return () => { active = false; clearTimeout(timer); };
  }, [mode]);

  useEffect(() => {
    if (!playing || !flightLocation) return;
    navigateTo(flightLocation);
    if (!visibleFiles.some((file) => file.doc_id === flightLocation.docId)) setBookReadyKey(pageKey(flightLocation));
    else if (bookReadyKey !== pageKey(flightLocation)) setPhase("turning");
  }, [playing, current, bookReadyKey]);

  useEffect(() => {
    if (!playing || !current || bookTransfer) return;
    if (current.kind === "answer") {
      setAnswer(current.answer);
      setSupportingIds(current.supportingIds);
      setPhase("complete");
      setPlaying(false);
      return;
    }
    const destination = current.kind === "visit" ? current.location : current.kind === "evidence" ? current.evidence : null;
    if (destination && bookReadyKey !== pageKey(destination)) return;
    setPhase("travelling");
    const travel = flightTravelTime(beats, cursor);
    let elapsed = 0;
    let previous = performance.now();
    let arrived = false;
    let frame = 0;
    function tick(now: number) {
      const delta = now - previous;
      previous = now;
      if (!pausedRef.current) elapsed += Math.min(delta, 100) * speedRef.current;
      const landing = calmRef.current ? 100 : travel + (current?.kind === "evidence" ? FLIGHT_TIMING.scan : 0);
      const duration = current?.kind === "stage" ? calmRef.current ? 120 : 450 : current?.kind === "evidence" ? landing + (calmRef.current ? 120 : FLIGHT_TIMING.delivery) : landing + 180;
      if (!arrived && elapsed >= landing) {
        arrived = true;
        if (current?.kind === "visit") setVisited((previousVisited) => ({ ...previousVisited, [pageKey(current.location)]: current.candidate ? "candidate" : "seen" }));
        if (current?.kind === "evidence") {
          setVisited((previousVisited) => ({ ...previousVisited, [pageKey(current.evidence)]: "candidate" }));
          setLanded((previousEvidence) => [...previousEvidence.filter((item) => item.id !== current.evidence.id), current.evidence]);
          setPhase("delivering");
        }
      }
      if (elapsed >= duration) {
        if (current?.kind === "evidence") setCollected((previousEvidence) => [...previousEvidence.filter((item) => item.id !== current.evidence.id), current.evidence]);
        setCursor((previousCursor) => previousCursor + 1);
      } else frame = requestAnimationFrame(tick);
    }
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, current, bookReadyKey, bookTransfer]);

  useEffect(() => {
    if (playing && !streaming && cursor >= beats.length) {
      setPlaying(false);
      setPhase("stopped");
    }
  }, [playing, streaming, cursor, beats.length]);

  function resetVisuals(initialFocus: FlightLocation | null = null) {
    setCursor(0); setFocus(initialFocus); setVisited({}); setLanded([]); setCollected([]);
    setSelectedEvidence(null);
    setBookTransfer(null);
    setAnswer(""); setSupportingIds([]); setError(""); setPhase("idle"); setPaused(false);
  }

  function locateEvidence(item: FlightEvidence) {
    if (busy || bookTransfer) return;
    setSelectedEvidence({ ...item });
    navigateTo(item);
  }

  function stop() {
    runRef.current += 1;
    controllerRef.current?.abort();
    controllerRef.current = null;
    setBookTransfer(null);
    setStreaming(false); setPlaying(false); setPaused(false); setPhase("stopped");
  }

  function startDemo() {
    stop(); resetVisuals();
    setMode("demo"); setQuestion(""); setMission("Sample flight");
    setBeats([...DEMO_BEATS]); setPlaying(true);
  }

  async function launch() {
    if (busy || uploading || bookTransfer) return;
    if (mode === "demo") { startDemo(); return; }
    if (!question.trim() || !selectedFiles.length || preparing) return;
    if (!config.model.trim() || !config.base_url.trim()) {
      setSettingsOpen(true); setError("Add your model name and API base URL before takeoff."); return;
    }
    resetVisuals(activeFile && selectedFiles.includes(activeFile) ? focus ?? { docId: activeFile.doc_id, page: 1 } : { docId: selectedFiles[0].doc_id, page: 1 }); setBeats([]); setMission(question.trim()); setPlaying(true); setStreaming(true);
    const run = ++runRef.current;
    const controller = new AbortController();
    controllerRef.current = controller;
    const seen = new Set<string>();
    let result = "";
    let finalState: Record<string, unknown> | null = null;
    let completed = false;
    const session = `flight-${crypto.randomUUID()}`;
    try {
      await streamChat(session, "default", session, config,
        [{ id: session, role: "user", content: question.trim() }], selectedFiles.map((file) => file.name), "default", (event) => {
          if (run !== runRef.current || controller.signal.aborted) return;
          if (event.type === "error") throw new Error(event.message);
          const additions = flightBeatsForEvent(event, seen);
          if (additions.length) setBeats((previousBeats) => [...previousBeats, ...additions]);
          if (event.type === "delta") {
            result += event.content;
            if (event.state) finalState = event.state;
          }
          if (event.type === "done") {
            if (!result.trim()) throw new Error("The stream completed without an answer.");
            completed = true;
            setBeats((previousBeats) => [...previousBeats, { kind: "answer", answer: result, supportingIds: supportingEvidenceIds(finalState) }]);
          }
        }, controller.signal);
      if (!completed && !controller.signal.aborted) throw new Error("The connection ended before completion. Collected evidence is preserved; no final answer was received.");
    } catch (failure) {
      if (run === runRef.current && !controller.signal.aborted) { setError(String(failure)); setPlaying(false); setPhase("stopped"); }
    } finally {
      if (run === runRef.current) { setStreaming(false); controllerRef.current = null; }
    }
  }

  async function importFiles(chosen: File[]) {
    if (!chosen.length) return;
    if (chosen.some((file) => !file.name.toLowerCase().endsWith(".pdf"))) { setError("Flight Lab currently visualizes PDF documents. Other formats remain available in Classic Chat."); return; }
    setUploading(true); setError("");
    try {
      const updated = await uploadWorkspaceFiles("default", chosen);
      initializedSelectionRef.current = true;
      setFiles(updated);
      setSelectedNames(chosen.map((file) => file.name));
      const imported = updated.find((file) => file.name === chosen[0].name);
      if (imported) setFocus({ docId: imported.doc_id, page: 1 });
    } catch (failure) { setError(String(failure)); }
    finally { setUploading(false); }
  }

  return <div className={`flight-app ${calm ? "calm" : ""} ${mission ? "flight-in-mission" : ""} ${answer ? "flight-finished" : ""}`} data-paused={paused}>
    <header className="flight-header">
      <a className="flight-brand" href="./"><span><img src={`${import.meta.env.BASE_URL}flight-logo-transparent.png`} alt="" aria-hidden="true" /></span><strong>DocPilot <em>Flight Lab</em></strong></a>
      <div className="flight-header-links"><span className={`flight-mode-badge ${mode}`}>{mode === "demo" ? "DEMO REPLAY · FICTIONAL SOURCES" : "LIVE AGENT EXPERIENCE"}</span><a href="./">Classic Chat ↗</a><button className="flight-theme-toggle" type="button" role="switch" aria-label="Day mode" aria-checked={theme === "day"} onClick={() => setTheme(theme === "day" ? "night" : "day")}><svg viewBox="0 0 24 24" aria-hidden="true">{theme === "day" ? <><circle cx="12" cy="12" r="4" /><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5" /></> : <path d="M20 14a8 8 0 0 1-10-10A8.5 8.5 0 1 0 20 14Z" />}</svg><span>{theme === "day" ? "Day" : "Night"}</span></button><button type="button" aria-expanded={settingsOpen} onClick={() => setSettingsOpen(!settingsOpen)}>Model settings</button></div>
    </header>

    <section className="flight-intro"><div><h1>Let your question <em>take flight.</em></h1></div><button className="flight-demo-button" type="button" disabled={busy || uploading || Boolean(bookTransfer)} onClick={mode === "live" ? startDemo : () => { resetVisuals(); setBeats([]); setMode("live"); setQuestion(""); }}>{mode === "live" ? "↗ Preview a sample flight" : "← Use my documents"}</button></section>

    {settingsOpen ? <section className="flight-settings" aria-label="Model settings"><label>Provider<select value={config.provider} disabled={busy} onChange={(event) => setConfig({ ...config, provider: event.target.value === "custom" ? "custom" : "vllm" })}><option value="vllm">vLLM</option><option value="custom">OpenAI-compatible</option></select></label><label>API base URL<input value={config.base_url} disabled={busy} onChange={(event) => setConfig({ ...config, base_url: event.target.value })} /></label><label>Model<input value={config.model} disabled={busy} onChange={(event) => setConfig({ ...config, model: event.target.value })} placeholder="Your model name" /></label><label>API key<input type="password" autoComplete="off" value={config.api_key} disabled={busy} onChange={(event) => setConfig({ ...config, api_key: event.target.value })} placeholder="Optional · memory only" /></label><p>Separate flight sessions. No changes to Classic Chat settings or conversation history. API keys stay in memory.</p></section> : null}

    <form className="flight-question" onSubmit={(event) => { event.preventDefault(); void launch(); }}><span className="flight-question-icon"><PlaneIcon /></span><label className="flight-sr-only" htmlFor="flight-question">Your question</label><input id="flight-question" value={question} onChange={(event) => setQuestion(event.target.value)} disabled={busy || mode === "demo"} placeholder={mode === "demo" ? "Preview the sample flight" : "Ask a question. Watch your documents come alive…"} /><button className="flight-primary" type="submit" disabled={busy || uploading || Boolean(bookTransfer) || (mode === "live" && (!question.trim() || !selectedFiles.length || preparing))}>{preparing && mode === "live" ? "Preparing documents…" : "Launch flight"}<span aria-hidden="true">↗</span></button></form>
    {error ? <div className="flight-error" role="alert">{error}<button type="button" aria-label="Dismiss error" onClick={() => setError("")}>×</button></div> : null}

    <div className="flight-controls"><div className={`flight-live-status ${busy && !paused ? "active" : ""}`} role="status"><i />{status}<small>{mode === "demo" ? "Illustrative replay · no API calls" : streaming ? "Receiving live agent events" : playing ? "Playing received events" : "Driven by real document evidence"}</small></div><div className="flight-playback"><label><input type="checkbox" checked={calm} onChange={(event) => setCalm(event.target.checked)} />Calm motion</label><button type="button" aria-label="Playback speed" onClick={() => setSpeed(speed === 1 ? 2 : speed === 2 ? 4 : 1)}>{speed}×</button>{playing ? <button type="button" onClick={() => setPaused(!paused)}>{paused ? "Resume view" : "Pause view"}</button> : null}{busy ? <button type="button" onClick={stop}>Stop flight</button> : <button type="button" disabled={!beats.length} onClick={() => { resetVisuals(); setPlaying(true); }}>↻ Replay</button>}</div></div>

    <main className="flight-scene" ref={sceneRef} aria-label="Document exploration">
      <section className="flight-atlas">
        <div className="flight-section-heading"><span>01 / THE READING ROOM</span><div className="flight-volume-actions"><small>{Object.keys(visited).length} pages inspected</small>{mode === "live" ? <button type="button" className="flight-import" disabled={busy || uploading || Boolean(bookTransfer)} onClick={() => fileInputRef.current?.click()}>{uploading ? "Uploading…" : "+ Import PDFs"}</button> : <small>FICTIONAL SAMPLE VOLUMES</small>}</div></div>
        <input className="flight-sr-only" ref={fileInputRef} type="file" multiple accept=".pdf,application/pdf" aria-label="Import PDF documents" onChange={(event) => { const chosen = Array.from(event.target.files ?? []); event.target.value = ""; void importFiles(chosen); }} />
        {mode === "live" ? <div className="flight-qa-scope" aria-label="QA sources"><strong>Ask about: {selectedFiles.length ? selectedFiles.map((file) => file.name).join(", ") : "No PDFs selected"}</strong><span>Click books for QA</span></div> : null}
        <div className="flight-volumes" aria-label="Document books">{visibleFiles.map((file, index) => <div className={`flight-volume ${activeFile?.doc_id === file.doc_id ? "active" : ""} ${bookTransfer?.to.docId === file.doc_id ? "incoming" : ""} ${mode === "live" && selectedNames.includes(file.name) ? "selected" : ""}`} key={file.name}>
          <button className={`flight-book-cover tone-${index % 3}`} type="button" disabled={busy || Boolean(bookTransfer) || !isPdf(file)} onClick={() => { setSelectedEvidence(null); navigateTo({ docId: file.doc_id, page: 1 }); if (mode === "live") setSelectedNames([file.name]); }} aria-label={`Open ${file.name}`} aria-pressed={activeFile?.doc_id === file.doc_id}><span className="flight-book-spine" /><small>{activeFile?.doc_id === file.doc_id ? "NOW OPEN" : `VOLUME ${String(index + 1).padStart(2, "0")}`}</small><strong>{file.name.replace(/\.pdf$/i, "").replaceAll("_", " ")}</strong><span className="flight-book-art">✳</span><footer>DOC PILOT <span>↗</span></footer></button>
          <label className="flight-book-choice"><input type="checkbox" aria-label={`Use ${file.name}`} checked={mode === "demo" || selectedNames.includes(file.name)} disabled={busy || mode === "demo" || !isPdf(file) || (file.processing_status === "failed" && !selectedNames.includes(file.name))} onChange={() => setSelectedNames((previousNames) => previousNames.includes(file.name) ? previousNames.filter((name) => name !== file.name) : [...previousNames, file.name])} /><span>{mode === "demo" ? "Sample volume" : !isPdf(file) ? "PDF only" : `${selectedNames.includes(file.name) ? "QA · " : ""}${file.processing_status === "ready" ? `${(file.size / 1024 / 1024).toFixed(1)} MB` : file.processing_status}`}</span></label>{file.processing_status === "failed" ? <small className="flight-file-error">{file.processing_error || "Retry in Classic Chat."}</small> : null}
        </div>)}</div>
        <div className="flight-atlas-title"><div><h2>{activeFile ? activeFile.name.replace(/\.pdf$/i, "").replaceAll("_", " ") : "Your next chapter awaits"}</h2></div><span className="flight-atlas-coordinate">{visibleFiles.length} VOLUMES</span><span className="flight-takeoff-point" aria-hidden="true" /></div>
        {activeFile ? <FlightBook key={`${mode}:${activeFile.doc_id}`} file={activeFile} mode={mode} focus={focus} visited={visited} evidence={landed} selectedEvidence={selectedEvidence} scanningEvidence={playing ? currentEvidence : null} scanning={scanning} speed={speed} calm={calm} paused={paused} disabled={busy || Boolean(bookTransfer)} departing={bookTransfer?.phase === "closing"} departureDuration={bookTransfer?.duration ?? FLIGHT_TIMING.bookClose} onFocus={(location) => { setSelectedEvidence(null); navigateTo(location); }} onReady={handleBookReady} />
          : <div className="flight-empty-atlas"><div className="flight-empty-volume"><PlaneIcon /></div><h3>{focus ? "Source volume unavailable" : "Every answer has a flight path."}</h3><p>{focus ? `The agent referenced ${focus.docId}, page ${focus.page}. No matching source is available; a different document will not be substituted.` : loadingFiles ? "Opening your books…" : "Import a PDF to open your first book, or explore the sample flight."}</p><button type="button" onClick={startDemo} disabled={busy}>Explore the sample flight ↗</button></div>}
      </section>

      <aside className="flight-results">
        <section className="flight-log-panel" aria-label="Flight log"><div className="flight-section-heading"><span>02 / LOG</span><small>{beats.length ? `EVENT ${Math.min(cursor + 1, beats.length).toString().padStart(2, "0")}` : "STANDBY"}</small></div>{transferLabel ? <p className="flight-book-route-log">{transferLabel}</p> : null}<p className="flight-log-message">{bookTransfer ? bookTransfer.phase === "closing" ? "Closing the source volume. The plane is holding its position." : "Opening the destination page. Waiting for the PDF to render." : current?.kind === "stage" ? current.detail : current?.kind === "visit" ? `The agent inspected ${current.location.docId} · page ${current.location.page}. Candidate pages are not yet evidence.` : currentEvidence ? `${currentEvidence.modality} evidence · ${currentEvidence.docId} · page ${currentEvidence.page}` : phase === "complete" ? "Evidence recorded. The final answer is ready." : "Waiting for agent events. Waiting never counts as a page visit."}</p></section>
        <div className="flight-section-heading"><span>03 / DISCOVERIES</span><small>{collected.length.toString().padStart(2, "0")} collected</small></div><div className={`flight-inbox-dock ${currentEvidence && phase === "delivering" ? "receiving" : ""}`}><span>↓</span><div><h2>Evidence, arriving.</h2><p>Little discoveries. A grounded answer.</p></div></div>
        <div className="flight-evidence-list" ref={evidenceListRef} aria-live="polite" aria-relevant="additions">{collected.map((item, index) => {
          const source = visibleFiles.find((file) => file.doc_id === item.docId);
          return <article className={`flight-evidence-card ${selectedEvidence?.id === item.id ? "active" : ""}`} key={item.id} data-flight-evidence={item.id} onClick={(event) => { if (event.target instanceof Element && !event.target.closest("a, button")) locateEvidence(item); }}><header><button className="flight-evidence-locate" type="button" aria-label={`Locate E${index + 1} in book`} aria-pressed={selectedEvidence?.id === item.id} disabled={busy} onClick={() => locateEvidence(item)}>E{index + 1} <small>{item.modality}</small><span aria-hidden="true">↗</span></button><span>{answer && supportingIds.includes(item.id) ? "✓ In answer" : "Collected"}</span></header><details><summary>{item.text.slice(0, 180)}{item.text.length > 180 ? "…" : ""}</summary><p>{item.text}</p></details><footer><button type="button" disabled={busy} onClick={() => locateEvidence(item)} title={`Locate evidence in ${source?.name ?? item.docId}`}>{source?.name ?? item.docId} · p.{item.page} ↗</button>{mode === "live" && source ? <a href={`${toAbsoluteUrl(source.download_url)}#page=${item.page}`} target="_blank" rel="noreferrer">PDF ↗</a> : null}</footer>{!item.box ? <small className="flight-page-only">Page-level evidence · no exact region supplied</small> : null}</article>;
        })}{!collected.length ? <div className="flight-evidence-empty"><span>✧</span><p>No discoveries yet.</p><small>Evidence cards land here only after the agent extracts a source.</small></div> : null}</div>
        <section className={`flight-answer ${answer ? "ready" : ""}`} aria-label="Final answer"><div className="flight-answer-heading"><span>THE ANSWER</span><span>{answer ? "✦" : "· · ·"}</span></div>{answer ? <div aria-live="polite" role="region" aria-label="Answer content" tabIndex={0}><ReactMarkdown>{answer}</ReactMarkdown></div> : <p>The pieces come together here.<br />First, we follow the evidence.</p>}{answer && !supportingIds.length ? <small>The agent returned no supporting evidence IDs. Do not assume collected items support every claim.</small> : null}</section>
      </aside>
      <FlightOverlay sceneRef={sceneRef} location={planeLocation} evidence={playing ? currentEvidence : null} delivering={Boolean(playing && currentEvidence && phase === "delivering")} complete={phase === "complete"} supportingIds={supportingIds} scanning={scanning} evidenceLabel={`E${collected.length + 1}`} routeKey={`${cursor}:${phase}`} layoutKey={`${activeFile?.doc_id}:${bookReadyKey}`} paused={paused || waitingForBook || Boolean(bookTransfer)} parked={!playing} holding={playing && (!current || current.kind === "stage" || waitingForBook)} duration={FLIGHT_TIMING.travel / speed} calm={calm} />
    </main>
    <footer className="flight-footer"><span>{mode === "demo" ? "SAMPLE SOURCES · NOT A LIVE MODEL RUN" : "Live trace → observed pages → extracted evidence → final answer"}</span></footer>
  </div>;
}
