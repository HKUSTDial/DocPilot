import { FLIGHT_TIMING, bookSpread, bookZoom, crossBookRoute, flightBeatsForEvent, flightEvidence, flightPages, flightPageAt, flightTravelTime, pageKey, resolveFlightTheme, supportingEvidenceIds, type FlightBeat } from "../src/flight/flight";
import type { ChatEvent } from "../src/types";
import { DEMO_BEATS, DEMO_PAGES } from "../src/flight/demo";

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

assert(bookSpread(1) === 1 && bookSpread(2) === 1, "opens physical pages 1 and 2 as the first spread");
assert(bookSpread(3) === 3 && bookSpread(4) === 3, "turns to the spread containing the exact navigation page");
assert(bookSpread(23) === 23 && bookSpread(109) === 109, "keeps the last odd-numbered page without inventing a next page");
const departure = { docId: "report", page: 3 };
const destination = { docId: "other", page: 12 };
assert(crossBookRoute(null, destination) === null && crossBookRoute(departure, null) === null, "missing sources or destinations do not close a book");
assert(crossBookRoute(departure, { ...departure, page: 12 }) === null, "same-document navigation retains normal page turns");
const transfer = crossBookRoute(departure, destination);
assert(transfer?.from === departure && transfer.to === destination, "cross-document transitions preserve exact source and destination pages");
assert(FLIGHT_TIMING.bookClose > 0, "cross-document transitions reserve time to close the source volume");
assert(bookZoom(1, -100) > 1 && bookZoom(1, 100) < 1, "Ctrl wheel adjusts only the document zoom in the scroll direction");
assert(bookZoom(1, 0) === 1 && Math.abs(bookZoom(bookZoom(1, -100), 100) - 1) < 0.000001, "opposite wheel movements restore the original scale");
assert(bookZoom(1, -10000) === 2 && bookZoom(1, 10000) === 0.5, "wheel zoom stays within 50 to 200 percent");
assert(resolveFlightTheme("day", true) === "day", "explicit day preference overrides the system dark theme");
assert(resolveFlightTheme("night", false) === "night", "explicit night preference overrides the system light theme");
assert(resolveFlightTheme(null, true) === "night" && resolveFlightTheme("invalid", false) === "day", "uses the system appearance when there is no valid saved theme");
assert(FLIGHT_TIMING.travel > 0 && FLIGHT_TIMING.scan > 0 && FLIGHT_TIMING.delivery > 0, "reserves separate time for travel, inspection and evidence delivery");
const motionBeats: FlightBeat[] = [
  { kind: "stage", title: "Navigate", detail: "Finding a page" },
  { kind: "visit", location: { docId: "book", page: 3 }, candidate: true },
  { kind: "stage", title: "Extract", detail: "Waiting for extraction" },
  { kind: "evidence", evidence: { id: "one", docId: "book", page: 3, text: "First passage", modality: "text", box: null } },
  { kind: "stage", title: "Focus", detail: "Inspecting the same page" },
  { kind: "evidence", evidence: { id: "two", docId: "book", page: 3, text: "Second passage", modality: "text", box: null } },
  { kind: "visit", location: { docId: "other", page: 3 }, candidate: true },
  { kind: "visit", location: { docId: "other", page: 4 }, candidate: true },
];
assert(flightTravelTime(motionBeats, 0) === 0 && flightTravelTime(motionBeats, 1) === FLIGHT_TIMING.travel, "waiting does not fly; the first target page does");
assert(flightTravelTime(motionBeats, 3) === 0 && flightTravelTime(motionBeats, 5) === 0, "same-page evidence reuses the landed position across extract and focus stages");
assert(flightTravelTime(motionBeats, 6) === FLIGHT_TIMING.travel && flightTravelTime(motionBeats, 7) === FLIGHT_TIMING.travel, "a different document or physical page gets a new route");
assert(flightPageAt([], 0) === null && flightPageAt(motionBeats, 0) === null, "does not fly to the preview document before an observed target page");
assert(flightPageAt(motionBeats, 2)?.page === 3 && flightPageAt(motionBeats, 4)?.docId === "book", "keeps the observed source across pending extract and focus stages");
assert(flightPageAt(motionBeats, motionBeats.length)?.page === 4, "waits at the latest observed page when the next event has not arrived");

assert(flightPages([1, { page: 3 }], "book").map(pageKey).join() === "book:1,book:3", "maps single-document physical pages");
assert(flightPages(["book:4", "other:2"]).map(pageKey).join() === "book:4,other:2", "keeps multi-document pages separate");
assert(flightPages([1, "2"]).length === 0, "does not guess a document for ambiguous pages");
assert(flightPages([-1, 0, 1.5, "NaN", { doc_id: "book", page: "bad" }], "book").length === 0, "rejects invalid pages");

const state = {
  document_ids: ["book"],
  metadata: { navigation: { candidate_pages: [{ doc_id: "book", page: 2 }] } },
  evidence_store: {
    first: { node_id: "book::page::2", page: 2, content: "A real passage", provenance: { bbox: [100, 200, 400, 500] } },
    second: { node_id: "other::page::2", page: 2, content: { value: 12 }, provenance: { doc_id: "other" } },
    invalid: { page: -1 },
  },
  candidate_answer: { supporting_evidence_ids: ["second", "second"] },
};
const evidence = flightEvidence(state);
assert(evidence.length === 2 && evidence[0].box?.left === 10, "reuses normalized evidence boxes");
assert(evidence[1].box === null && evidence[1].docId === "other", "does not invent regions for page-level evidence");
assert(supportingEvidenceIds(state).join() === "second", "distinguishes collected and answer-supporting evidence");
const seen = new Set<string>();
const navigation: ChatEvent = {
  type: "step", step_index: 0, state,
  action: { action_type: "navigate" },
  observation: { success: true, payload: { page_set: [1, 2, 3] } },
};
const beats = flightBeatsForEvent(navigation, seen);
assert(beats.length === 5 && beats[1].kind === "visit" && beats[1].candidate, "visits observed pages before collecting actual evidence");
assert(flightBeatsForEvent(navigation, seen).length === 0, "does not replay repeated event snapshots");
assert(flightBeatsForEvent({ ...navigation, observation: { success: false, payload: { page_set: [8] } } }, new Set()).every((beat) => beat.kind !== "visit"), "never marks failed navigation as inspected");
assert(flightBeatsForEvent({ ...navigation, observation: { success: false, payload: { phase: "action", page_set: [8] } } }, new Set()).filter((beat) => beat.kind === "visit").length === 0, "pending actions are not evidence");
assert(flightBeatsForEvent({ type: "delta", content: "answer", state }, seen).length === 0, "final snapshots do not duplicate evidence");
assert(flightBeatsForEvent({ type: "delta", content: "answer", state }, new Set()).length === 2, "collects evidence supplied only by the final snapshot");
assert(flightBeatsForEvent({ ...navigation, state: { document_ids: ["book", "other"] } }, new Set()).length === 0, "does not assign ambiguous numeric pages to a random document");
assert(flightBeatsForEvent({ type: "step", history: [{ action: { action_type: "thinking" }, observation: { payload: { phase: "thinking" } } }] }, new Set()).length === 0, "waiting does not manufacture progress");

const sampleIds = new Set(DEMO_BEATS.flatMap((beat) => beat.kind === "evidence" ? [beat.evidence.id] : []));
for (const beat of DEMO_BEATS) {
  const location = beat.kind === "visit" ? beat.location : beat.kind === "evidence" ? beat.evidence : null;
  if (location) assert(Boolean(DEMO_PAGES[location.docId]?.[location.page - 1]), "sample flights only visit existing sample pages");
  if (beat.kind === "answer") assert(beat.supportingIds.every((id) => sampleIds.has(id)), "sample answer references collected sample evidence");
}
