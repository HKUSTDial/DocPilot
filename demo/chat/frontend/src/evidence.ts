export type EvidenceBox = {
  left: number;
  top: number;
  width: number;
  height: number;
};

export type EvidenceLocator = {
  id: string;
  docId: string;
  page: number;
  modality: string;
  box: EvidenceBox | null;
};

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

export function normalizeEvidenceBox(value: unknown): EvidenceBox | null {
  if (!Array.isArray(value) || value.length !== 4) {
    return null;
  }
  const coordinates = value.map(Number);
  if (coordinates.some((coordinate) => !Number.isFinite(coordinate))) {
    return null;
  }
  const [rawX1, rawY1, rawX2, rawY2] = coordinates;
  const maxCoordinate = Math.max(...coordinates.map(Math.abs));
  // shortcut: MinerU content-list bboxes use 0-1000; add coordinate metadata for other parsers.
  const scale = maxCoordinate <= 1 ? 100 : maxCoordinate <= 1000 ? 0.1 : 0;
  if (!scale) {
    return null;
  }
  const x1 = Math.max(0, Math.min(100, Math.min(rawX1, rawX2) * scale));
  const y1 = Math.max(0, Math.min(100, Math.min(rawY1, rawY2) * scale));
  const x2 = Math.max(0, Math.min(100, Math.max(rawX1, rawX2) * scale));
  const y2 = Math.max(0, Math.min(100, Math.max(rawY1, rawY2) * scale));
  if (x2 <= x1 || y2 <= y1) {
    return null;
  }
  return { left: x1, top: y1, width: x2 - x1, height: y2 - y1 };
}

export function collectAnswerEvidenceLocators(
  state: Record<string, unknown> | null,
): EvidenceLocator[] {
  if (!state) {
    return [];
  }
  const candidateAnswer = asRecord(state.candidate_answer);
  const evidenceStore = asRecord(state.evidence_store);
  if (!candidateAnswer || !evidenceStore) {
    return [];
  }
  const supportingIds = Array.isArray(candidateAnswer.supporting_evidence_ids)
    ? candidateAnswer.supporting_evidence_ids.map(String).filter(Boolean)
    : [];

  return [...new Set(supportingIds)].flatMap((id) => {
    const evidence = asRecord(evidenceStore[id]);
    const provenance = asRecord(evidence?.provenance);
    const page = Number(evidence?.page);
    if (!evidence || !Number.isInteger(page) || page < 1) {
      return [];
    }
    const nodeId = String(evidence.node_id ?? "");
    const docId = String(provenance?.doc_id ?? nodeId.split("::")[0] ?? "").trim();
    if (!docId) {
      return [];
    }
    return [{
      id,
      docId,
      page,
      modality: String(evidence.modality ?? "mixed"),
      box: normalizeEvidenceBox(provenance?.bbox),
    }];
  });
}
