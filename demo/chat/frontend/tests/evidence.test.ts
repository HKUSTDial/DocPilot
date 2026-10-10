import { collectAnswerEvidenceLocators, normalizeEvidenceBox } from "../src/evidence";

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) {
    throw new Error(message);
  }
}

const box = normalizeEvidenceBox([598, 290, 843, 479]);
assert(
  box && Math.abs(box.left - 59.8) < 1e-9 && Math.abs(box.width - 24.5) < 1e-9,
  "normalizes MinerU coordinates",
);
assert(normalizeEvidenceBox([0.1, 0.2, 0.4, 0.5])?.top === 20, "normalizes fractional coordinates");
assert(normalizeEvidenceBox([0, 0, 0, 0]) === null, "rejects empty boxes");
assert(normalizeEvidenceBox([0, 0, 1224, 1584]) === null, "rejects unknown coordinate spaces");

const locators = collectAnswerEvidenceLocators({
  evidence_store: {
    first: {
      node_id: "report::page::5::block::1",
      page: 5,
      modality: "text",
      provenance: { doc_id: "report", bbox: [100, 200, 300, 400] },
    },
    second: {
      node_id: "report::page::6",
      page: 6,
      modality: "image",
      provenance: { doc_id: "report", bbox: null },
    },
  },
  candidate_answer: { supporting_evidence_ids: ["second", "second"] },
});
assert(locators.length === 1 && locators[0]?.id === "second", "uses unique answer evidence only");
assert(locators[0]?.box === null, "keeps page-only evidence");
assert(collectAnswerEvidenceLocators({ evidence_store: {} }).length === 0, "requires a final answer");
