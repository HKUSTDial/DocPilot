import type { AgentType, Message } from "./types";

export const HISTORY_LIMIT = 50;

export type ConversationHistoryItem = {
  id: string;
  title: string;
  messages: Message[];
  selectedFiles: string[];
  agentTrace?: Array<Record<string, unknown>>;
  agentState?: Record<string, unknown> | null;
  responseMode?: "chat" | "docpilot";
  agent?: AgentType;
  updatedAt: string;
};

export function getConversationTitle(messages: Message[]): string {
  return (
    messages.find((message) => message.role === "user" && message.content.trim())?.content.trim() ??
    "New conversation"
  );
}

export function parseConversationHistory(raw: string | null): ConversationHistoryItem[] {
  if (!raw) {
    return [];
  }

  try {
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed.filter(isConversationHistoryItem).slice(0, HISTORY_LIMIT);
  } catch {
    return [];
  }
}

export function upsertConversationHistory(
  history: ConversationHistoryItem[],
  item: ConversationHistoryItem,
): ConversationHistoryItem[] {
  return [item, ...history.filter((entry) => entry.id !== item.id)].slice(0, HISTORY_LIMIT);
}

export function updateConversationHistoryInPlace(
  history: ConversationHistoryItem[],
  item: ConversationHistoryItem,
): ConversationHistoryItem[] {
  return history.some((entry) => entry.id === item.id)
    ? history.map((entry) => entry.id === item.id ? item : entry)
    : upsertConversationHistory(history, item);
}

export function removeConversationHistory(
  history: ConversationHistoryItem[],
  id: string | null,
): ConversationHistoryItem[] {
  return id ? history.filter((entry) => entry.id !== id) : history;
}

function isConversationHistoryItem(value: unknown): value is ConversationHistoryItem {
  if (!value || typeof value !== "object") {
    return false;
  }
  const item = value as Record<string, unknown>;
  return (
    typeof item.id === "string" &&
    typeof item.title === "string" &&
    typeof item.updatedAt === "string" &&
    !Number.isNaN(Date.parse(item.updatedAt)) &&
    Array.isArray(item.selectedFiles) &&
    item.selectedFiles.every((file) => typeof file === "string") &&
    Array.isArray(item.messages) &&
    item.messages.every(isMessage) &&
    (item.agentTrace === undefined ||
      (Array.isArray(item.agentTrace) && item.agentTrace.every(isRecord))) &&
    (item.agentState === undefined || item.agentState === null || isRecord(item.agentState)) &&
    (item.responseMode === undefined ||
      item.responseMode === "chat" ||
      item.responseMode === "docpilot") &&
    (item.agent === undefined || item.agent === "default" || item.agent === "chat")
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function isMessage(value: unknown): value is Message {
  if (!value || typeof value !== "object") {
    return false;
  }
  const message = value as Record<string, unknown>;
  return (
    typeof message.id === "string" &&
    (message.role === "user" || message.role === "assistant") &&
    typeof message.content === "string" &&
    (message.agentTrace === undefined ||
      (Array.isArray(message.agentTrace) && message.agentTrace.every(isRecord))) &&
    (message.agentState === undefined || message.agentState === null || isRecord(message.agentState)) &&
    (message.responseMode === undefined ||
      message.responseMode === "chat" ||
      message.responseMode === "docpilot")
  );
}
