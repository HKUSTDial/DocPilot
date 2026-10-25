import type { WorkspaceFile } from "../types";
import type { FlightBeat } from "./flight";

export const DEMO_DOCUMENTS: WorkspaceFile[] = [
  ["city", "The greener city.pdf"],
  ["mobility", "Mobility field notes.pdf"],
  ["research", "Research appendix.pdf"],
].map(([doc_id, name]) => ({
  doc_id, name, size: 0, modified_at: "", content_type: "application/pdf", download_url: "",
  extension: ".pdf", processing_status: "ready", viewer_type: "pdf",
}));

export const DEMO_PAGES: Record<string, Array<{ title: string; text: string }>> = {
  city: [
    { title: "A greener city", text: "HARBOR CITY · FIELD REPORT\nA small collection of ideas for a more connected, lower-carbon city.\nAn illustrative document created for the Flight Lab demo." },
    { title: "The daily journey", text: "Every morning, thousands of journeys connect homes, schools and workplaces. This report looks at what changed when the city invested in alternatives to driving alone." },
    { title: "A quieter kind of progress", text: "Commuter transport emissions fell by 18% in 2024 compared with the 2022 baseline.\nThe report attributes the reduction primarily to electric buses and improved cycling connections.\nChanges in weather and population were not evaluated in this illustrative study." },
    { title: "Electric, together", text: "The city replaced diesel vehicles on its busiest bus routes with electric buses. More frequent departures made public transport a practical option for everyday travel." },
    { title: "Streets for everyone", text: "Protected cycling lanes connected residential neighborhoods to transit stops. The most useful improvements closed short but important gaps in the network." },
    { title: "Measuring change", text: "The same baseline and measurement boundary were used for the annual comparison. Commuter journeys, rather than all city emissions, are the subject of this report." },
    { title: "What comes next", text: "Future work includes better access outside the city center and a longer observation period. The report does not claim that every neighborhood benefited equally." },
    { title: "Source notes", text: "Flight Lab sample materials. All names, measurements and findings in these demonstration documents are fictional. No model or external source was used." },
  ],
  mobility: [
    { title: "Mobility field notes", text: "OBSERVATIONS FROM THE NETWORK\nHow small connections change a daily journey.\nIllustrative material for the Flight Lab demo." },
    { title: "The missing last mile", text: "Protected bike lanes linked residential neighborhoods to major bus stops.\nThese connections made it easier to combine cycling and public transit instead of driving the entire journey.\nThe notes describe an enabling mechanism, not a separate emissions estimate." },
    { title: "Waiting less", text: "More reliable bus frequency reduced the uncertainty of transferring between routes. Travelers could plan a journey without a long buffer between connections." },
    { title: "At the station", text: "Clear wayfinding and bicycle parking made the transfer between cycling and public transport more comfortable. These were supporting improvements." },
    { title: "Listening to riders", text: "Field interviews describe convenience as well as environmental motivation. The sample is illustrative and should not be read as a representative survey." },
    { title: "A connected network", text: "A cycling route is only as useful as its most difficult crossing. Small, targeted connections can make a larger network usable." },
    { title: "Open questions", text: "Which neighborhoods still lack access? How do travel patterns change in winter? These questions require further observation." },
    { title: "About these notes", text: "A fictional companion to the Flight Lab sample report. This page is not a source for a real-world claim." },
  ],
  research: [
    { title: "Research appendix", text: "METHODS & DEFINITIONS\nSupporting notes for the illustrative Harbor City report." },
    { title: "Scope", text: "The measurement concerns commuter transport emissions within a consistent boundary. It is not a measure of total city emissions." },
    { title: "Baseline", text: "The comparison year is 2022. The observation year is 2024. All figures in this demo are synthetic." },
    { title: "Travel modes", text: "The sample compares private car journeys, public transport and cycling. It does not assign a separate numerical contribution to each intervention." },
    { title: "Attribution, carefully", text: "Electric buses and protected cycling connections are the two main interventions identified in the report.\nThe appendix does not quantify their individual shares of the 18% reduction.\nA careful answer should name both without inventing a breakdown." },
    { title: "Limitations", text: "This illustrative evidence does not establish a controlled causal estimate. The demo shows how a grounded answer preserves the limits of its sources." },
    { title: "Reading the result", text: "Keep the headline measurement separate from the explanation. Connect claims to their source pages and preserve the distinction between observation and inference." },
    { title: "Demo notice", text: "All Flight Lab sample content is fictional and is clearly separated from live document analysis." },
  ],
};

export const DEMO_BEATS: FlightBeat[] = [
  { kind: "stage", title: "Charting a route", detail: "Demo replay · locating the report and its supporting notes." },
  ...[1, 2, 3, 4].map((page): FlightBeat => ({ kind: "visit", location: { docId: "city", page }, candidate: page === 3 || page === 4 })),
  { kind: "evidence", evidence: { id: "demo-emissions", docId: "city", page: 3, modality: "text", box: { left: 10, top: 31, width: 80, height: 31 }, text: "Commuter transport emissions fell by 18% in 2024 compared with the 2022 baseline. The report attributes the reduction primarily to electric buses and improved cycling connections." } },
  { kind: "visit", location: { docId: "mobility", page: 1 }, candidate: false },
  { kind: "visit", location: { docId: "mobility", page: 2 }, candidate: true },
  { kind: "evidence", evidence: { id: "demo-connections", docId: "mobility", page: 2, modality: "text", box: { left: 10, top: 31, width: 80, height: 36 }, text: "Protected bike lanes linked residential neighborhoods to major bus stops, making it easier to combine cycling and public transit instead of driving the entire journey." } },
  { kind: "visit", location: { docId: "research", page: 5 }, candidate: true },
  { kind: "evidence", evidence: { id: "demo-limits", docId: "research", page: 5, modality: "text", box: null, text: "Electric buses and protected cycling connections are the two main interventions. The appendix does not quantify their individual shares of the reduction." } },
  { kind: "stage", title: "Connecting the dots", detail: "Bringing the findings together without inventing an attribution breakdown." },
  { kind: "answer", supportingIds: ["demo-emissions", "demo-connections", "demo-limits"], answer: "**Electric buses and better cycling connections** were the two main improvements. Protected bike lanes connected neighborhoods to bus stops, helping commuters combine cycling with public transit instead of driving.\n\nThe sample report records an **18% reduction in commuter transport emissions in 2024**, relative to 2022. It does **not** provide a separate percentage contribution for either intervention.\n\n*Illustrative answer from fictional demo documents, not a live model response.*" },
];
