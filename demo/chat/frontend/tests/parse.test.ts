import { findParsedTextMatches, ocrCategory, parsedImageUrl } from "../src/parse";

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

assert(ocrCategory("doc_title") === "title", "groups document titles");
assert(ocrCategory("table") === "table", "preserves table regions");
assert(ocrCategory("logo") === "logo", "preserves a logo only when the parser emits one");
assert(ocrCategory("page_number") === "marginalia", "groups page furniture");
assert(ocrCategory("caption") === "text", "keeps captions visible");

const base = "https://example.com/analyze/api/action-images/report/auto/";
assert(parsedImageUrl("images/chart 1.jpg", base) === `${base}images/chart%201.jpg`, "resolves parsed images through the API prefix");
assert(parsedImageUrl("https://other.example/image.jpg", base) === undefined, "rejects external images");
assert(parsedImageUrl("//other.example/image.jpg", base) === undefined, "rejects protocol-relative images");
assert(parsedImageUrl("../private/image.jpg", base) === undefined, "rejects parent paths");
assert(parsedImageUrl("%2e%2e%2fprivate/image.jpg", base) === undefined, "rejects encoded parent paths");
assert(parsedImageUrl("javascript:alert(1)", base) === undefined, "rejects script URLs");
assert(parsedImageUrl("%broken", base) === undefined, "rejects malformed image URLs");

const matches = findParsedTextMatches("Report report REPORT", "report");
assert(matches.length === 3 && matches[1].start === 7 && matches[1].end === 13, "finds case-insensitive literal matches");
assert(findParsedTextMatches("[a+b] aab", "[a+b]").length === 1, "escapes search pattern syntax");
assert(findParsedTextMatches("document", "").length === 0, "ignores empty searches");
assert(findParsedTextMatches("document", "missing").length === 0, "handles no results");
assert(findParsedTextMatches("文档分析，文档解析", "文档")[1].start === 5, "preserves Unicode match positions");
