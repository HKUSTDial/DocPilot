export const OCR_CATEGORIES = {
  text: "Text",
  title: "Title",
  table: "Table",
  chart: "Chart",
  image: "Image",
  logo: "Logo",
  formula: "Formula",
  marginalia: "Header / footer",
};

export function ocrCategory(type: string): keyof typeof OCR_CATEGORIES {
  if (["doc_title", "paragraph_title", "title"].includes(type)) return "title";
  if (["header", "footer", "page_number"].includes(type)) return "marginalia";
  if (type === "figure") return "image";
  if (["table", "chart", "image", "logo", "formula"].includes(type)) {
    return type as keyof typeof OCR_CATEGORIES;
  }
  return "text";
}

export function parsedImageUrl(source: string, baseUrl: string): string | undefined {
  try {
    if (!source) return undefined;
    const base = new URL(baseUrl);
    const url = new URL(source, base);
    if (decodeURIComponent(url.pathname).split(/[\\/]/).includes("..")) return undefined;
    return url.origin === base.origin && url.pathname.startsWith(base.pathname)
      ? url.href : undefined;
  } catch {
    return undefined;
  }
}

export function findParsedTextMatches(text: string, query: string): Array<{ start: number; end: number }> {
  if (!query) return [];
  const pattern = new RegExp(query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi");
  return Array.from(text.matchAll(pattern), (match) => ({
    start: match.index!,
    end: match.index! + match[0].length,
  }));
}
