import type { AgentType, ChatEvent, Message, ModelConfig, OcrLayout, ParsedOutput, ParsedOutputFormat, WorkspaceFile } from "./types";
import type { ConversationHistoryItem } from "./history";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8008";

function normalizeError(error: unknown): Error {
  if (error instanceof Error) {
    if (error.message === "Failed to fetch" || error.message === "Load failed") {
      return new Error(
        `无法连接后端 ${API_BASE}。请确认 FastAPI 服务已经启动，并且浏览器可以访问它。`,
      );
    }
    return error;
  }
  return new Error(String(error));
}

async function expectJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function toAbsoluteUrl(relativeUrl: string): string {
  return `${API_BASE}${relativeUrl}`;
}

export type ConversationHistoryStore = {
  active_id: string | null;
  conversations: ConversationHistoryItem[];
};

export async function fetchConversationHistory(): Promise<ConversationHistoryStore> {
  try {
    const response = await fetch(`${API_BASE}/api/history`);
    return expectJson<ConversationHistoryStore>(response);
  } catch (error) {
    throw normalizeError(error);
  }
}

export async function saveConversationHistory(
  history: ConversationHistoryStore,
): Promise<ConversationHistoryStore> {
  try {
    const response = await fetch(`${API_BASE}/api/history`, {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(history),
    });
    return expectJson<ConversationHistoryStore>(response);
  } catch (error) {
    throw normalizeError(error);
  }
}

export async function deleteConversationHistory(
  conversationId: string,
): Promise<ConversationHistoryStore> {
  try {
    const response = await fetch(`${API_BASE}/api/history/${encodeURIComponent(conversationId)}`, {
      method: "DELETE",
    });
    return expectJson<ConversationHistoryStore>(response);
  } catch (error) {
    throw normalizeError(error);
  }
}

export async function fetchPresets(): Promise<Record<string, ModelConfig>> {
  try {
    const response = await fetch(`${API_BASE}/api/presets`);
    return expectJson<Record<string, ModelConfig>>(response);
  } catch (error) {
    throw normalizeError(error);
  }
}

export async function fetchWorkspaceFiles(sessionId: string): Promise<WorkspaceFile[]> {
  try {
    const url = new URL(`${API_BASE}/api/files`);
    url.searchParams.set("session_id", sessionId);
    const response = await fetch(url);
    const payload = await expectJson<{ files: WorkspaceFile[] }>(response);
    return payload.files;
  } catch (error) {
    throw normalizeError(error);
  }
}

export async function fetchFileLayout(
  sessionId: string,
  fileName: string,
  page: number,
  signal: AbortSignal,
): Promise<OcrLayout> {
  const url = new URL(`${API_BASE}/api/files/${encodeURIComponent(fileName)}/layout`);
  url.searchParams.set("session_id", sessionId);
  url.searchParams.set("page", String(page));
  return expectJson<OcrLayout>(await fetch(url, { signal }));
}

export async function fetchParsedOutput(
  sessionId: string,
  fileName: string,
  format: ParsedOutputFormat,
  signal: AbortSignal,
): Promise<ParsedOutput> {
  const url = new URL(`${API_BASE}/api/files/${encodeURIComponent(fileName)}/parsed`);
  url.searchParams.set("session_id", sessionId);
  url.searchParams.set("format", format);
  return expectJson<ParsedOutput>(await fetch(url, { signal }));
}

export async function uploadWorkspaceFiles(
  sessionId: string,
  files: File[],
): Promise<WorkspaceFile[]> {
  try {
    const formData = new FormData();
    for (const file of files) {
      formData.append("files", file);
    }
    const url = new URL(`${API_BASE}/api/files`);
    url.searchParams.set("session_id", sessionId);
    const response = await fetch(url, {
      method: "POST",
      body: formData,
    });
    const payload = await expectJson<{ files: WorkspaceFile[] }>(response);
    return payload.files;
  } catch (error) {
    throw normalizeError(error);
  }
}

export async function deleteWorkspaceFile(
  sessionId: string,
  fileName: string,
): Promise<WorkspaceFile[]> {
  try {
    const url = new URL(`${API_BASE}/api/files/${encodeURIComponent(fileName)}`);
    url.searchParams.set("session_id", sessionId);
    const response = await fetch(url, { method: "DELETE" });
    const payload = await expectJson<{ files: WorkspaceFile[] }>(response);
    return payload.files;
  } catch (error) {
    throw normalizeError(error);
  }
}

export async function retryWorkspaceFile(
  sessionId: string,
  fileName: string,
): Promise<WorkspaceFile[]> {
  try {
    const url = new URL(`${API_BASE}/api/files/${encodeURIComponent(fileName)}/retry`);
    url.searchParams.set("session_id", sessionId);
    const response = await fetch(url, { method: "POST" });
    const payload = await expectJson<{ files: WorkspaceFile[] }>(response);
    return payload.files;
  } catch (error) {
    throw normalizeError(error);
  }
}

export async function discoverModels(config: ModelConfig): Promise<string[]> {
  try {
    const url = new URL(`${API_BASE}/api/models/discover`);
    url.searchParams.set("provider", config.provider);
    url.searchParams.set("base_url", config.base_url);
    url.searchParams.set("api_key", config.api_key);
    const response = await fetch(url);
    const payload = await expectJson<{ models: string[] }>(response);
    return payload.models;
  } catch (error) {
    throw normalizeError(error);
  }
}

export async function fetchFilePreview(
  sessionId: string,
  fileName: string,
): Promise<string> {
  try {
    const url = new URL(
      `${API_BASE}/api/files/content/${encodeURIComponent(fileName)}`,
    );
    url.searchParams.set("session_id", sessionId);
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(await response.text());
    }
    return response.text();
  } catch (error) {
    throw normalizeError(error);
  }
}

export async function streamChat(
  sessionId: string,
  workspaceSessionId: string,
  conversationId: string,
  config: ModelConfig,
  messages: Message[],
  selectedFiles: string[],
  agent: AgentType,
  onEvent: (event: ChatEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/api/chat/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      signal,
      body: JSON.stringify({
        session_id: sessionId,
        workspace_session_id: workspaceSessionId,
        conversation_id: conversationId,
        config,
        agent,
        selected_files: selectedFiles,
        messages: messages.map((item) => ({
          role: item.role,
          content: item.content,
        })),
      }),
    });
  } catch (error) {
    throw normalizeError(error);
  }

  if (!response.ok || !response.body) {
    throw new Error(await response.text());
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) {
        continue;
      }
      onEvent(JSON.parse(trimmed) as ChatEvent);
    }
  }

  if (buffer.trim()) {
    onEvent(JSON.parse(buffer.trim()) as ChatEvent);
  }
}
