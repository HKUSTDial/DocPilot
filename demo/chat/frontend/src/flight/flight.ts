import { normalizeEvidenceBox, type EvidenceBox } from "../evidence";
import type { ChatEvent } from "../types";

export type FlightLocation = { docId: string; page: number };
export type FlightBookRoute = { from: FlightLocation; to: FlightLocation };
export type FlightEvidence = FlightLocation & {
  id: string;
  modality: string;
  text: string;
  box: EvidenceBox | null;
};
export type FlightBeat =
  | { kind: "stage"; title: string; detail: string }
  | { kind: "visit"; location: FlightLocation; candidate: boolean }
  | { kind: "evidence"; evidence: FlightEvidence }
  | { kind: "answer"; answer: string; supportingIds: string[] };

export const pageKey = (location: FlightLocation) => `${location.docId}:${location.page}`;

export const FLIGHT_TIMING = { travel: 950, scan: 850, delivery: 450, bookClose: 360 };

export function crossBookRoute(from: FlightLocation | null, to: FlightLocation | null): FlightBookRoute | null {
  return from && to && from.docId !== to.docId ? { from, to } : null;
}

export function flightPageAt(beats: FlightBeat[], cursor: number): FlightLocation | null {
  for (let index = Math.min(cursor, beats.length - 1); index >= 0; index -= 1) {
    const beat = beats[index];
    if (beat.kind === "visit") return beat.location;
    if (beat.kind === "evidence") return beat.evidence;
  }
  return null;
}

export function flightTravelTime(beats: FlightBeat[], cursor: number): number {
  const current = beats[cursor];
  const location = current?.kind === "visit" ? current.location : current?.kind === "evidence" ? current.evidence : null;
  if (!location) return 0;
  const previous = flightPageAt(beats, cursor - 1);
  return previous && pageKey(previous) === pageKey(location) ? 0 : FLIGHT_TIMING.travel;
}

export function resolveFlightTheme(saved: unknown, prefersDark: boolean): "day" | "night" {
  return saved === "day" || saved === "night" ? saved : prefersDark ? "night" : "day";
}

export function bookSpread(page: number): number {
  return Math.floor((page - 1) / 2) * 2 + 1;
}

export function bookZoom(current: number, delta: number): number {
  return Math.min(2, Math.max(0.5, current * Math.exp(-delta * 0.0015)));
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

export function flightPages(value: unknown, defaultDocId = ""): FlightLocation[] {
  const items = Array.isArray(value) ? value : value == null ? [] : [value];
  return items.flatMap((item) => {
    const entry = record(item);
    const reference = typeof item === "string" ? /^(.*):(\d+)$/.exec(item) : null;
    const docId = String(entry.doc_id ?? reference?.[1] ?? defaultDocId).trim();
    const page = Number(entry.page ?? reference?.[2] ?? item);
    return docId && Number.isInteger(page) && page > 0 ? [{ docId, page }] : [];
  });
}

export function flightEvidence(state: Record<string, unknown>): FlightEvidence[] {
  return Object.entries(record(state.evidence_store)).flatMap(([id, value]) => {
    const entry = record(value);
    const provenance = record(entry.provenance);
    const location = flightPages({
      doc_id: provenance.doc_id || String(entry.node_id ?? "").split("::")[0],
      page: entry.page,
    })[0];
    if (!location) return [];
    const content = entry.content;
    return [{
      ...location, id, modality: String(entry.modality ?? "mixed"),
      text: typeof content === "string" ? content : content == null ? "No text preview available." : JSON.stringify(content, null, 2),
      box: normalizeEvidenceBox(provenance.bbox),
    }];
  });
}

export function flightBeatsForEvent(event: ChatEvent, seen: Set<string>): FlightBeat[] {
  if (event.type !== "step" && event.type !== "delta") return [];
  const beats: FlightBeat[] = [];
  const state = record(event.state);
  const history = event.history ?? [];
  const last = record(history[history.length - 1]);
  const action = record(event.type === "step" ? event.action ?? last.action : null);
  const observation = record(event.type === "step" ? event.observation ?? last.observation : null);
  const payload = record(observation.payload);
  const metadata = record(state.metadata);
  const navigation = record(metadata.navigation);
  const documentIds = Array.isArray(state.document_ids) ? state.document_ids.map(String) : [];
  const defaultDocId = String(record(action.args).doc_id ?? navigation.active_document ?? (documentIds.length === 1 ? documentIds[0] : ""));
  const step = event.type === "step" ? event.step_index ?? history.length - 1 : -1;
  const actionType = String(action.action_type ?? "");
  function add(key: string, beat: FlightBeat) {
    if (!seen.has(key)) {
      seen.add(key);
      beats.push(beat);
    }
  }
  if (payload.phase === "action" && actionType !== "thinking") {
    const titles: Record<string, string> = {
      navigate: "Charting a route", extract: "Looking closer", focus: "Focusing on a region",
      analyze: "Checking the evidence", solver: "Connecting the dots", execute: "Working through the numbers",
      answer: "Preparing the answer", finish: "Finishing the flight",
    };
    add(`stage:${step}`, { kind: "stage", title: titles[actionType] ?? "Following the agent", detail: String(action.intent ?? actionType) });
  }
  if (actionType === "navigate" && observation.success === true && !payload.phase) {
    const candidatePages = flightPages(navigation.candidate_pages ?? payload.candidate_pages, defaultDocId);
    const candidates = new Set(candidatePages.map(pageKey));
    const pages = flightPages(payload.page_set, defaultDocId);
    for (const location of pages.length ? pages : candidatePages) {
      add(`visit:${step}:${pageKey(location)}`, { kind: "visit", location, candidate: candidates.has(pageKey(location)) });
    }
  }
  for (const evidence of flightEvidence(state)) {
    add(`evidence:${evidence.id}`, { kind: "evidence", evidence });
  }
  return beats;
}

export function supportingEvidenceIds(state: Record<string, unknown> | null): string[] {
  const ids = record(state?.candidate_answer).supporting_evidence_ids;
  return Array.isArray(ids) ? [...new Set(ids.map(String))] : [];
}
