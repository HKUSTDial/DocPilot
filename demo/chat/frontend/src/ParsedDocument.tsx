import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { marked, type Tokens } from "marked";
import { fetchParsedOutput, toAbsoluteUrl } from "./api";
import { normalizeEvidenceBox } from "./evidence";
import { findParsedTextMatches, ocrCategory, parsedImageUrl } from "./parse";
import { WorkspaceSelect } from "./WorkspaceSelect";
import type { OcrLayout, ParsedOutput, ParsedOutputFormat, WorkspaceFile } from "./types";

export function OcrOverlay({ layout, page }: { layout: OcrLayout | null; page: number }) {
  if (!layout || layout.page !== page) return null;
  return (
    <div className="pdf-ocr-overlay" aria-hidden="true">
      {layout.blocks.map((block, index) => {
        const box = normalizeEvidenceBox(block.bbox);
        return box ? (
          <span
            key={index}
            className={`pdf-ocr-box ocr-${ocrCategory(block.type)}`}
            title={block.type.replaceAll("_", " ")}
            style={{ left: `${box.left}%`, top: `${box.top}%`, width: `${box.width}%`, height: `${box.height}%` }}
          >
            <span>{block.type.replaceAll("_", " ")}</span>
          </span>
        ) : null;
      })}
    </div>
  );
}

function ParsedMarkdown({ content, imageBaseUrl }: { content: string; imageBaseUrl: string }) {
  const tokens = useMemo(() => marked.lexer(content), [content]);
  const components = {
    img: ({ src, alt }: { src?: string; alt?: string }) => {
      const url = parsedImageUrl(src ?? "", imageBaseUrl);
      return url ? <img src={url} alt={alt || "Parsed document image"} loading="lazy" /> : <span>{alt}</span>;
    },
  };
  return tokens.map((token, index) => {
    if (token.type !== "table") {
      return <ReactMarkdown key={index} components={components}>{token.raw}</ReactMarkdown>;
    }
    const table = token as Tokens.Table;
    return (
    <div className="parsed-table" key={index}>
      <table>
        <thead><tr>{table.header.map((cell, column) => (
          <th key={column}><ReactMarkdown components={components}>{cell.text}</ReactMarkdown></th>
        ))}</tr></thead>
        <tbody>{table.rows.map((row, rowIndex) => <tr key={rowIndex}>
          {row.map((cell, column) => <td key={column}>
            <ReactMarkdown components={components}>{cell.text}</ReactMarkdown>
          </td>)}
        </tr>)}</tbody>
      </table>
    </div>
    );
  });
}

export function ParsedDocumentOutput({ file, sessionId }: { file: WorkspaceFile; sessionId: string }) {
  const [expanded, setExpanded] = useState(false);
  const [format, setFormat] = useState<ParsedOutputFormat>("markdown");
  const [output, setOutput] = useState<(ParsedOutput & { format: ParsedOutputFormat }) | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [actionLoading, setActionLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [matches, setMatches] = useState<Range[]>([]);
  const [activeMatch, setActiveMatch] = useState(0);
  const contentRef = useRef<HTMLDivElement | null>(null);
  const searchRef = useRef<HTMLInputElement | null>(null);
  const actionControllerRef = useRef<AbortController | null>(null);
  const ready = file.processing_status === "ready";
  const currentOutput = output?.format === format ? output : null;
  const busy = loading || actionLoading;

  useEffect(() => {
    if (!expanded || !ready || currentOutput) {
      setLoading(false);
      return;
    }
    const controller = new AbortController();
    setLoading(true);
    setError("");
    void fetchParsedOutput(sessionId, file.name, format, controller.signal)
      .then((result) => {
        if (!controller.signal.aborted) setOutput({ ...result, format });
      })
      .catch(() => {
        if (!controller.signal.aborted) setError("Parsed output is not available for this document.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [expanded, ready, file.name, sessionId, format, currentOutput]);

  useEffect(() => () => actionControllerRef.current?.abort(), []);

  useEffect(() => {
    setCopied(false);
  }, [format]);

  useEffect(() => {
    if (expanded && searchOpen) searchRef.current?.focus();
  }, [expanded, searchOpen]);

  useEffect(() => {
    const ranges: Range[] = [];
    if (expanded && searchOpen && query && currentOutput && !loading && contentRef.current) {
      const walker = document.createTreeWalker(contentRef.current, NodeFilter.SHOW_TEXT);
      while (walker.nextNode()) {
        const node = walker.currentNode;
        for (const match of findParsedTextMatches(node.textContent ?? "", query)) {
          const range = document.createRange();
          range.setStart(node, match.start);
          range.setEnd(node, match.end);
          ranges.push(range);
        }
      }
    }
    setMatches(ranges);
    setActiveMatch(0);
    if (typeof Highlight !== "undefined" && CSS.highlights) {
      const highlights = new Highlight();
      ranges.forEach((range) => highlights.add(range));
      CSS.highlights.set("docpilot-parsed-matches", highlights);
    }
    return () => {
      CSS.highlights?.delete("docpilot-parsed-matches");
      CSS.highlights?.delete("docpilot-parsed-active");
    };
  }, [expanded, searchOpen, query, currentOutput, loading]);

  useEffect(() => {
    const range = matches[activeMatch];
    const container = contentRef.current;
    if (!range || !container) return;
    if (typeof Highlight !== "undefined" && CSS.highlights) {
      CSS.highlights.set("docpilot-parsed-active", new Highlight(range));
    } else {
      const selection = window.getSelection();
      selection?.removeAllRanges();
      selection?.addRange(range);
    }
    const bounds = range.getBoundingClientRect();
    container.scrollTop += bounds.top - container.getBoundingClientRect().top - container.clientHeight / 2;
  }, [matches, activeMatch]);

  function moveMatch(direction: number) {
    if (matches.length) setActiveMatch((current) => (current + direction + matches.length) % matches.length);
  }

  async function handleOutputAction(action: "copy" | "download") {
    const controller = new AbortController();
    actionControllerRef.current = controller;
    setActionLoading(true);
    setCopied(false);
    setError("");
    try {
      const result = currentOutput ?? await fetchParsedOutput(sessionId, file.name, format, controller.signal);
      if (controller.signal.aborted) return;
      setOutput({ ...result, format });
      if (action === "copy") {
        await navigator.clipboard.writeText(result.content);
        if (!controller.signal.aborted) setCopied(true);
      } else {
        const url = URL.createObjectURL(new Blob([result.content], {
          type: format === "json" ? "application/json;charset=utf-8" : "text/markdown;charset=utf-8",
        }));
        const link = document.createElement("a");
        link.href = url;
        link.download = `${file.name.replace(/\.pdf$/i, "")}.${format === "json" ? "json" : "md"}`;
        link.click();
        window.setTimeout(() => URL.revokeObjectURL(url), 0);
      }
    } catch {
      if (!controller.signal.aborted) {
        setError(action === "copy" ? "Could not copy output. Allow clipboard access and try again." : "Could not download parsed output. Please try again.");
        setExpanded(true);
      }
    } finally {
      if (!controller.signal.aborted) setActionLoading(false);
    }
  }

  return (
    <section className={`parsed-output ${expanded ? "expanded" : ""}`} aria-label="Parsed document output">
      <div className="parsed-output-toolbar">
        <button
          className="parsed-output-toggle secondary viewer-icon-button"
          type="button"
          disabled={!ready}
          aria-label={expanded ? "Collapse parsed output" : "Expand parsed output"}
          title={ready ? expanded ? "Collapse parsed output" : "Expand parsed output" : "Available when ready"}
          aria-expanded={expanded}
          aria-controls="parsed-output-body"
          onClick={() => setExpanded((current) => !current)}
        ><svg viewBox="0 0 24 24" aria-hidden="true"><path d={expanded ? "m6 9 6 6 6-6" : "m9 6 6 6-6 6"} /></svg></button>
        <WorkspaceSelect
          className="parsed-output-format"
          label="Parsed output format"
          value={format}
          disabled={!ready || busy}
          options={[
            { value: "markdown", label: "Markdown", description: "Formatted document" },
            { value: "json", label: "JSON", description: "Structured data" },
          ]}
          onChange={(value) => {
            setFormat(value === "json" ? "json" : "markdown");
            setExpanded(true);
          }}
        />
        <div className="parsed-output-actions">
          <button
            className={`secondary viewer-icon-button ${searchOpen && expanded ? "active" : ""}`}
            type="button"
            disabled={!ready}
            aria-label="Search parsed output"
            title="Search parsed output"
            aria-expanded={expanded && searchOpen}
            onClick={() => {
              setSearchOpen(!expanded || !searchOpen);
              setExpanded(true);
            }}
          ><svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5" /><path d="m16 16 5 5" /></svg></button>
          <button
            className="secondary viewer-icon-button"
            type="button"
            disabled={!ready || busy}
            aria-label="Download parsed output"
            title="Download parsed output"
            onClick={() => void handleOutputAction("download")}
          ><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5" /></svg></button>
          <button
            className={`secondary viewer-icon-button ${copied ? "active" : ""}`}
            type="button"
            disabled={!ready || busy}
            aria-label={copied ? "Parsed output copied" : "Copy parsed output"}
            title={copied ? "Copied" : "Copy parsed output"}
            onClick={() => void handleOutputAction("copy")}
          ><svg viewBox="0 0 24 24" aria-hidden="true">{copied ? <path d="m5 12 4 4L19 6" /> : <>
            <rect x="8" y="8" width="12" height="13" rx="2" /><path d="M15 8V3H3v13h5" />
          </>}</svg></button>
        </div>
      </div>
      {expanded ? <div id="parsed-output-body" className="parsed-output-body">
        {searchOpen ? <div className="parsed-output-search">
          <input
            ref={searchRef}
            type="search"
            value={query}
            placeholder="Search parsed output"
            aria-label="Find in parsed output"
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") { event.preventDefault(); moveMatch(event.shiftKey ? -1 : 1); }
              if (event.key === "Escape") setSearchOpen(false);
            }}
          />
          <span role="status">{matches.length ? activeMatch + 1 : 0}/{matches.length}</span>
          <button className="secondary viewer-icon-button" type="button" disabled={!matches.length} aria-label="Previous parsed match" onClick={() => moveMatch(-1)}>
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m7 14 5-5 5 5" /></svg>
          </button>
          <button className="secondary viewer-icon-button" type="button" disabled={!matches.length} aria-label="Next parsed match" onClick={() => moveMatch(1)}>
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m7 10 5 5 5-5" /></svg>
          </button>
        </div> : null}
        <div ref={contentRef} className="parsed-output-content rich-viewer" aria-busy={loading || actionLoading}>
          {loading ? <p role="status">Loading parsed output...</p> : null}
          {error ? <p role="status">{error}</p> : null}
          {!loading && currentOutput ? format === "json"
            ? <pre>{currentOutput.content}</pre>
            : <ParsedMarkdown content={currentOutput.content} imageBaseUrl={toAbsoluteUrl(currentOutput.image_base_url)} />
            : null}
        </div>
      </div> : null}
    </section>
  );
}
