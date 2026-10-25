import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Document, Page } from "react-pdf";
import { toAbsoluteUrl } from "../api";
import type { WorkspaceFile } from "../types";
import { DEMO_PAGES } from "./demo";
import { bookSpread, bookZoom, pageKey, type FlightEvidence, type FlightLocation } from "./flight";

type PageTurn = { from: number; to: number; direction: "forward" | "backward"; duration: number };

export function FlightBook({ file, mode, focus, visited, evidence, selectedEvidence, scanningEvidence, scanning, speed, calm, paused, disabled, departing, departureDuration, onFocus, onReady }: {
  file: WorkspaceFile;
  mode: "live" | "demo";
  focus: FlightLocation | null;
  visited: Record<string, "seen" | "candidate">;
  evidence: FlightEvidence[];
  selectedEvidence: FlightEvidence | null;
  scanningEvidence: FlightEvidence | null;
  scanning: boolean;
  speed: number;
  calm: boolean;
  paused: boolean;
  disabled: boolean;
  departing: boolean;
  departureDuration: number;
  onFocus: (location: FlightLocation) => void;
  onReady: (key: string) => void;
}) {
  const targetPage = focus?.docId === file.doc_id ? focus.page : 1;
  const requestedSpread = bookSpread(targetPage);
  const [spread, setSpread] = useState(requestedSpread);
  const [turn, setTurn] = useState<PageTurn | null>(null);
  const [opening, setOpening] = useState(!calm);
  const [count, setCount] = useState<number | null>(mode === "demo" ? DEMO_PAGES[file.doc_id]?.length ?? 0 : null);
  const [error, setError] = useState("");
  const [renderedPages, setRenderedPages] = useState<Record<number, boolean>>({});
  const [ratios, setRatios] = useState<Record<number, number>>({});
  const [paperSize, setPaperSize] = useState({ width: 300, height: 400 });
  const [pageInput, setPageInput] = useState(String(targetPage));
  const [zoom, setZoom] = useState(1);
  const bookRef = useRef<HTMLDivElement>(null);
  const paperSizeRef = useRef(paperSize);
  const spreadRef = useRef(spread);
  const pauseRef = useRef(paused);
  const speedRef = useRef(speed);
  pauseRef.current = paused;
  speedRef.current = speed;

  useEffect(() => { setPageInput(String(targetPage)); }, [targetPage]);

  useEffect(() => {
    const book = bookRef.current;
    if (!book) return;
    function handleWheel(event: WheelEvent) {
      if (!event.ctrlKey) return;
      event.preventDefault();
      if (turn || departing) return;
      const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? book!.clientHeight : 1);
      setZoom((value) => bookZoom(value, delta));
    }
    function handleKey(event: KeyboardEvent) {
      if (!event.ctrlKey || !["+", "=", "-", "0"].includes(event.key)) return;
      event.preventDefault();
      if (!turn && !departing) setZoom((value) => event.key === "0" ? 1 : bookZoom(value, event.key === "-" ? 100 : -100));
    }
    book.addEventListener("wheel", handleWheel, { passive: false });
    book.addEventListener("keydown", handleKey);
    return () => { book.removeEventListener("wheel", handleWheel); book.removeEventListener("keydown", handleKey); };
  }, [count, error, turn, departing]);

  useEffect(() => {
    if (!opening || paused) return;
    const timer = setTimeout(() => setOpening(false), calm ? 0 : Math.max(180, 420 / speed));
    return () => clearTimeout(timer);
  }, [opening, paused, calm, speed]);

  useLayoutEffect(() => {
    const from = spreadRef.current;
    spreadRef.current = requestedSpread;
    setSpread(requestedSpread);
    if (from !== requestedSpread) setRenderedPages({});
    if (calm || from === requestedSpread) { setTurn(null); return; }
    const duration = Math.max(180, 620 / speedRef.current);
    setTurn({ from, to: requestedSpread, direction: requestedSpread > from ? "forward" : "backward", duration });
    let elapsed = 0;
    let previous = performance.now();
    let frame = 0;
    function animate(now: number) {
      if (!pauseRef.current) elapsed += Math.min(now - previous, 100);
      previous = now;
      if (elapsed >= duration) setTurn(null);
      else frame = requestAnimationFrame(animate);
    }
    frame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frame);
  }, [requestedSpread, calm]);

  useLayoutEffect(() => {
    const book = bookRef.current;
    if (!book) return;
    function resize() {
      const paper = book?.querySelector<HTMLElement>(".flight-book-paper");
      if (!paper) return;
      const styles = getComputedStyle(paper);
      const nextSize = {
        width: Math.max(30, paper.clientWidth - parseFloat(styles.paddingLeft) - parseFloat(styles.paddingRight)),
        height: Math.max(40, paper.clientHeight - parseFloat(styles.paddingTop) - parseFloat(styles.paddingBottom)),
      };
      if (nextSize.width !== paperSizeRef.current.width || nextSize.height !== paperSizeRef.current.height) {
        paperSizeRef.current = nextSize;
        setRenderedPages({});
        setPaperSize(nextSize);
      }
    }
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(book);
    return () => observer.disconnect();
  }, [count, error]);

  const invalidPage = count !== null && targetPage > count;
  useEffect(() => {
    if (!departing && !opening && !turn && (mode === "demo" || error || invalidPage || renderedPages[targetPage])) {
      onReady(pageKey({ docId: file.doc_id, page: targetPage }));
    }
  }, [departing, opening, turn, mode, error, invalidPage, renderedPages, file.doc_id, targetPage, onReady]);

  useEffect(() => {
    if (!selectedEvidence || selectedEvidence.docId !== file.doc_id || selectedEvidence.page !== targetPage || departing || opening || turn || (mode !== "demo" && !renderedPages[targetPage])) return;
    const book = bookRef.current;
    const region = Array.from(book?.querySelectorAll<HTMLElement>("[data-flight-evidence-region]") ?? []).find((element) => element.dataset.flightEvidenceRegion === selectedEvidence.id);
    const viewport = region?.closest<HTMLElement>(".flight-page-viewport");
    const page = region?.closest<HTMLElement>(".flight-source-page");
    if (!region || !viewport || !page) return;
    book?.scrollIntoView({ block: "nearest", inline: "nearest", behavior: calm ? "auto" : "smooth" });
    viewport.scrollTo({
      left: page.offsetLeft + region.offsetLeft + Math.min(region.offsetWidth, viewport.clientWidth) / 2 - viewport.clientWidth / 2,
      top: page.offsetTop + region.offsetTop + Math.min(region.offsetHeight, viewport.clientHeight) / 2 - viewport.clientHeight / 2,
      behavior: "auto",
    });
  }, [selectedEvidence, file.doc_id, targetPage, departing, opening, turn, mode, renderedPages[targetPage], zoom, calm]);

  function renderPage(page: number, locatable = true) {
    const key = pageKey({ docId: file.doc_id, page });
    const sample = mode === "demo" ? DEMO_PAGES[file.doc_id]?.[page - 1] : null;
    const available = count !== null && page <= count;
    const ready = available && !error && (mode === "demo" || renderedPages[page]);
    const pageEvidence = ready && locatable ? evidence.filter((item) => pageKey(item) === key) : [];
    const inspection = ready && locatable && scanningEvidence && pageKey(scanningEvidence) === key ? scanningEvidence : null;
    const ratio = ratios[page] ?? 0.75;
    const width = Math.max(1, Math.floor(Math.min(paperSize.width, paperSize.height * ratio) * zoom));
    return <div key={key} className={`flight-book-paper ${page === targetPage ? "targeted" : ""} ${visited[key] ?? ""}`}>
      {available && !error ? <div className="flight-page-viewport" tabIndex={locatable ? 0 : -1} role="region" aria-label={`Page ${page}. Ctrl + scroll to zoom; Ctrl + 0 to reset.`}>
        <div className="flight-page-stage" style={{ width, height: width / ratio }}><div
        className="flight-source-page"
        data-flight-page={locatable && ready ? key : undefined}
        style={{ width, height: width / ratio }}
      >
        {sample ? <div className="flight-sample-surface" style={{ width: width / zoom, height: width / ratio / zoom, transform: `scale(${zoom})` }}><div className="flight-sample-page">
          <span>FLIGHT LAB / SAMPLE {String(page).padStart(2, "0")}</span>
          <h3>{sample.title}</h3><i />
          {sample.text.split("\n").map((paragraph, index) => <p key={index}>{paragraph}</p>)}
          <footer>HARBOR CITY · ILLUSTRATIVE MATERIAL</footer>
        </div></div> : <Page
          pageNumber={page}
          width={width}
          devicePixelRatio={Math.min(3, Math.max(2, window.devicePixelRatio || 1))}
          renderAnnotationLayer={false}
          renderTextLayer={false}
          loading={<span className="flight-page-loading">Opening page {page}…</span>}
          onLoadSuccess={(pdfPage) => {
            const viewport = pdfPage.getViewport({ scale: 1 });
            const nextRatio = viewport.width / viewport.height;
            setRatios((previous) => previous[page] === nextRatio ? previous : { ...previous, [page]: nextRatio });
          }}
          onRenderSuccess={() => {
            if (locatable) setRenderedPages((previous) => previous[page] ? previous : { ...previous, [page]: true });
          }}
          onRenderError={(failure) => setError(failure.message)}
          onLoadError={(failure) => setError(failure.message)}
        />}
        {inspection ? <span className="flight-source-anchor" data-flight-anchor={inspection.id} style={{ left: `${inspection.box ? inspection.box.left + inspection.box.width / 2 : 50}%`, top: `${inspection.box ? inspection.box.top + inspection.box.height / 2 : 50}%` }} /> : null}
        {inspection && scanning ? <span className="flight-scan-area" style={inspection.box ? { left: `${inspection.box.left}%`, top: `${inspection.box.top}%`, width: `${inspection.box.width}%`, height: `${inspection.box.height}%` } : { inset: "8px" }}><i /><b>SCANNING SOURCE</b></span> : null}
        {pageEvidence.map((item) => <span
          className={`flight-evidence-region ${item.box ? "" : "page-level"} ${selectedEvidence?.id === item.id ? "active" : ""}`}
          key={item.id}
          data-flight-evidence-region={item.id}
          style={item.box ? { left: `${item.box.left}%`, top: `${item.box.top}%`, width: `${item.box.width}%`, height: `${item.box.height}%` } : { inset: "3px" }}
        ><b>E{evidence.findIndex((entry) => entry.id === item.id) + 1}</b></span>)}
      </div></div></div> : <div className="flight-book-endpaper"><span>✧</span><p>{error ? "Preview unavailable" : count === null ? "Opening this volume…" : "End of this volume"}</p></div>}
      <div className="flight-book-folio"><span>{available ? String(page).padStart(2, "0") : "—"}</span><span>{pageEvidence.length ? "Evidence found" : visited[key] === "candidate" ? "Candidate" : visited[key] === "seen" ? "Inspected" : "DOC PILOT"}</span></div>
    </div>;
  }

  const leftPage = turn?.direction === "forward" ? turn.from : spread;
  const rightPage = turn?.direction === "backward" ? turn.from + 1 : spread + 1;
  const spreadLabel = invalidPage ? `page ${targetPage} unavailable` : count !== null && spread + 1 > count ? `page ${spread}` : `pages ${spread} and ${spread + 1}`;
  const book = <div className="flight-book-perspective">
    <div className="flight-bound-book" key={file.doc_id} style={{ animationName: departing ? "flight-volume-close" : opening ? undefined : "none", animationDuration: `${departing ? departureDuration : Math.max(180, 420 / speed)}ms`, animationPlayState: paused ? "paused" : "running", transitionDuration: `${Math.max(80, 280 / speed)}ms` }}>
      <div className="flight-open-book" ref={bookRef} data-spread={spread} aria-label={`${file.name}, ${spreadLabel}`}>
        <div className="flight-book-leaf left" style={departing ? { animationDuration: `${departureDuration}ms` } : undefined}>{renderPage(leftPage)}</div>
        <div className="flight-book-leaf right" style={departing ? { animationDuration: `${departureDuration}ms` } : undefined}>{renderPage(rightPage)}</div>
        <div className="flight-book-binding" aria-hidden="true" />
        <div className="flight-book-ribbon" aria-hidden="true" />
        {turn ? <div
          key={`${turn.from}:${turn.to}`}
          className={`flight-turn-leaf ${turn.direction}`}
          aria-hidden="true"
          style={{ animationDuration: `${turn.duration}ms`, animationPlayState: paused ? "paused" : "running" }}
        >
          <div className="flight-turn-face front">{renderPage(turn.direction === "forward" ? turn.from + 1 : turn.from, false)}</div>
          <div className="flight-turn-face back">{renderPage(turn.direction === "forward" ? turn.to : turn.to + 1, false)}</div>
        </div> : null}
      </div>
      <div className="flight-book-edge" aria-hidden="true" />
    </div>
  </div>;

  return <section className={`flight-book-reader ${departing ? "departing" : ""} ${turn ? "turning" : ""} ${scanning && !turn && !departing ? "cinematic-focus" : ""}`} data-focus-side={targetPage === spread ? "left" : "right"} aria-label="Open document book">
    <div className="flight-book-route"><span>{turn ? `Turning to page ${targetPage}` : `PAGE ${targetPage}`}</span><span>{count === null ? "Opening volume" : `${count} physical pages`} · {mode === "demo" ? "SAMPLE EDITION" : "ORIGINAL PDF"}</span></div>
    <div className="flight-book-viewport" role="region" aria-label="Scrollable book">
    {mode === "demo" ? book : <Document
      file={toAbsoluteUrl(file.viewer_url ?? file.download_url)}
      className="flight-pdf-document"
      loading={book}
      error={book}
      onLoadSuccess={({ numPages }) => setCount(numPages)}
      onLoadError={(failure) => setError(failure.message)}
    >{book}</Document>}
    </div>
    {invalidPage || error ? <p className="flight-atlas-notice" role="alert">{invalidPage ? `Page ${targetPage} is outside this ${count}-page volume. No other page will be substituted.` : `PDF preview unavailable: ${error}`}</p> : null}
    <div className="flight-book-navigation">
      <button type="button" aria-label="Previous spread" disabled={disabled || spread <= 1 || Boolean(turn)} onClick={() => onFocus({ docId: file.doc_id, page: Math.max(1, spread - 2) })}>← Previous</button>
      <form onSubmit={(event) => {
        event.preventDefault();
        const page = Number(pageInput);
        if (!disabled && Number.isInteger(page) && page >= 1 && count !== null && page <= count) onFocus({ docId: file.doc_id, page });
      }}><label htmlFor="flight-book-page">Page</label><input id="flight-book-page" type="number" min="1" max={count ?? undefined} value={pageInput} disabled={disabled || count === null} onChange={(event) => setPageInput(event.target.value)} /><span>of {count ?? "…"}</span><button type="submit" disabled={disabled || count === null}>Go</button></form>
      <button type="button" aria-label="Next spread" disabled={disabled || count === null || spread + 2 > count || Boolean(turn)} onClick={() => onFocus({ docId: file.doc_id, page: spread + 2 })}>Next →</button>
    </div>
  </section>;
}
