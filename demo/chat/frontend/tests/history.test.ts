import {
  HISTORY_LIMIT,
  getConversationTitle,
  parseConversationHistory,
  removeConversationHistory,
  updateConversationHistoryInPlace,
  upsertConversationHistory,
  type ConversationHistoryItem,
} from "../src/history";

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) {
    throw new Error(message);
  }
}

const item = (id: string): ConversationHistoryItem => ({
  id,
  title: id,
  messages: [{ id: `${id}-message`, role: "user", content: id }],
  selectedFiles: [],
  updatedAt: "2026-10-08T00:00:00.000Z",
});

assert(getConversationTitle(item("First question").messages) === "First question", "uses first user message");
assert(parseConversationHistory("not json").length === 0, "rejects malformed cache");
assert(
  parseConversationHistory(
    JSON.stringify([{ ...item("trace"), agentTrace: [{ action: {}, observation: {} }], responseMode: "docpilot" }]),
  ).length === 1,
  "keeps persisted agent traces",
);
assert(
  parseConversationHistory(JSON.stringify([{ ...item("bad-trace"), agentTrace: ["bad"] }])).length === 0,
  "rejects malformed agent traces",
);
assert(
  parseConversationHistory(JSON.stringify([{ ...item("chat"), agent: "chat" }]))[0]?.agent === "chat",
  "keeps the conversation agent",
);
assert(
  parseConversationHistory(
    JSON.stringify([
      {
        ...item("message-trace"),
        messages: [
          {
            id: "assistant-1",
            role: "assistant",
            content: "answer",
            agentTrace: [{ action: {}, observation: {} }],
            agentState: { done: true },
            responseMode: "docpilot",
          },
        ],
      },
    ]),
  )[0]?.messages[0]?.agentTrace?.length === 1,
  "keeps per-assistant agent traces",
);
assert(
  parseConversationHistory(JSON.stringify([{ ...item("bad-agent"), agent: "other" }])).length === 0,
  "rejects an unknown conversation agent",
);

const deduplicated = upsertConversationHistory([item("same")], { ...item("same"), title: "updated" });
assert(deduplicated.length === 1 && deduplicated[0]?.title === "updated", "updates duplicate ids");
const stable = updateConversationHistoryInPlace(
  [item("first"), item("active")],
  { ...item("active"), title: "updated in place" },
);
assert(stable[0]?.id === "first" && stable[1]?.title === "updated in place", "updates without reordering");
assert(removeConversationHistory([item("keep"), item("remove")], "remove").length === 1, "clears one conversation");

const limited = Array.from({ length: HISTORY_LIMIT + 1 }, (_, index) => item(String(index))).reduce(
  (history, entry) => upsertConversationHistory(history, entry),
  [] as ConversationHistoryItem[],
);
assert(limited.length === HISTORY_LIMIT, "caps stored history");
