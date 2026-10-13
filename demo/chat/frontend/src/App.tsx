import {
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type KeyboardEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { Document, Page, pdfjs } from "react-pdf";
import mammoth from "mammoth";
import { marked } from "marked";
import ReactMarkdown from "react-markdown";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";
import {
  deleteWorkspaceFile,
  deleteConversationHistory,
  discoverModels,
  fetchConversationHistory,
  fetchWorkspaceFiles,
  importWorkspaceFilesFromPaths,
  retryWorkspaceFile,
  saveConversationHistory,
  streamChat,
  toAbsoluteUrl,
  uploadWorkspaceFiles,
} from "./api";
import {
  getConversationTitle,
  parseConversationHistory,
  updateConversationHistoryInPlace,
  upsertConversationHistory,
  type ConversationHistoryItem,
} from "./history";
import { collectAnswerEvidenceLocators } from "./evidence";
import type { AgentType, Message, ModelConfig, WorkspaceFile } from "./types";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).toString();

const CONFIG_KEY = "docpilot-chat-config-v2";
const LEGACY_CONFIG_KEY = "docpilot-chat-config";
const LEGACY_HISTORY_KEY = "docpilot-chat-history-v1";
const LEGACY_ACTIVE_HISTORY_KEY = "docpilot-chat-active-history-v1";
const AGENT_KEY = "docpilot-chat-agent-v1";

const FALLBACK_CONFIG: ModelConfig = {
  provider: "vllm",
  base_url: "http://127.0.0.1:8000/v1",
  api_key: "",
  model: "",
  reasoning: false,
  timeout: 180,
};

const PRESET_CONFIGS: Record<"vllm" | "custom", ModelConfig> = {
  vllm: {
    provider: "vllm",
    base_url: "http://127.0.0.1:8000/v1",
    api_key: "",
    model: "",
    reasoning: false,
    timeout: 180,
  },
  custom: {
    provider: "custom",
    base_url: "https://api.openai.com/v1",
    api_key: "",
    model: "",
    reasoning: false,
    timeout: 180,
  },
};

function loadStoredConfig(): ModelConfig {
  const storedConfig = localStorage.getItem(CONFIG_KEY);
  if (!storedConfig) {
    return { ...FALLBACK_CONFIG };
  }

  try {
    const parsed = JSON.parse(storedConfig) as ModelConfig & {
      temperature?: unknown;
      max_tokens?: unknown;
    };
    delete parsed.temperature;
    delete parsed.max_tokens;
    return { ...parsed, reasoning: parsed.reasoning === true };
  } catch {
    localStorage.removeItem(CONFIG_KEY);
    return { ...FALLBACK_CONFIG };
  }
}

type ViewerMode =
  | "empty"
  | "pdf"
  | "image"
  | "docx"
  | "markdown"
  | "text"
  | "unsupported";

type ViewerState = {
  mode: ViewerMode;
  title: string;
  sourceUrl: string;
  htmlContent: string;
  textContent: string;
};

type PdfSearchMatch = {
  pageNumber: number;
  occurrenceInPage: number;
  left: number;
  top: number;
  width: number;
  height: number;
  rotation: number;
};

type AgentStepView = {
  index: number;
  key: string;
  title: string;
  thought: string;
  reasoning: string;
  result: string;
  images: Array<{ url: string; label: string }>;
  success: boolean;
  done: boolean;
  phase: "action" | "result";
};

const EMPTY_VIEWER: ViewerState = {
  mode: "empty",
  title: "",
  sourceUrl: "",
  htmlContent: "",
  textContent: "",
};

const TEXT_EXTENSIONS = new Set([
  "txt",
  "json",
  "yaml",
  "yml",
  "csv",
  "tsv",
  "xml",
  "html",
  "py",
  "log",
  "ini",
  "toml",
]);
const CHAT_ATTACHMENT_EXTENSIONS = new Set(["pdf", "png", "jpg", "jpeg", "webp", "gif"]);

function createId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function formatBytes(size: number): string {
  if (size < 1024) {
    return `${size} B`;
  }
  if (size < 1024 * 1024) {
    return `${(size / 1024).toFixed(1)} KB`;
  }
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function getFileExtension(name: string): string {
  return name.split(".").pop()?.toLowerCase() ?? "";
}

function escapeHtml(content: string): string {
  return content
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function highlightHtml(content: string, query: string): string {
  if (!query.trim()) {
    return content;
  }
  const pattern = buildHighlightPattern(query);
  return content.replace(pattern, "<mark>$1</mark>");
}

function buildHighlightPattern(query: string): RegExp {
  const escaped = query.trim().replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`(${escaped})`, "gi");
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function isPendingAgentStep(item: Record<string, unknown>): boolean {
  const observation = asRecord(item.observation);
  const payload = asRecord(observation?.payload);
  return (
    observation?.message === "Running..." ||
    payload?.phase === "thinking" ||
    payload?.phase === "action"
  );
}

function isAgentThinkingPlaceholder(item: Record<string, unknown>): boolean {
  const action = asRecord(item.action);
  const observation = asRecord(item.observation);
  const payload = asRecord(observation?.payload);
  return action?.action_type === "thinking" && payload?.phase === "thinking";
}

function nonEmptyText(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function formatAgentStepTitle(value: string): string {
  return (value === "solver" ? "solve" : value)
    .split(/[_-]+/)
    .filter(Boolean)
    .map((part) => `${part.charAt(0).toUpperCase()}${part.slice(1)}`)
    .join(" ");
}

function formatAgentResult(
  actionType: string,
  observation: Record<string, unknown>,
): string {
  const message = String(observation.message ?? "").trim();
  const payload = asRecord(observation.payload);
  if (!payload || Object.keys(payload).length === 0) {
    return message;
  }

  const displayResult = nonEmptyText(payload.display_result);
  if (displayResult) {
    return displayResult;
  }

  const rawValue =
    payload.raw_action_response ?? payload.raw_model_response ?? payload.raw_model_payload;
  const rawText = nonEmptyText(rawValue);
  if (rawText) {
    return rawText;
  }
  const raw = asRecord(rawValue);
  const candidateAnswer = asRecord(payload.candidate_answer);
  const verification = asRecord(payload.verification);
  const planner = asRecord(payload.planner);
  const directText = [
    raw?.reason,
    raw?.summary,
    raw?.rationale,
    payload.reason,
    payload.summary,
    payload.rationale,
    planner?.reason,
    candidateAnswer?.rationale,
    verification?.summary,
  ].map(nonEmptyText).find(Boolean);
  if (directText) {
    return directText;
  }

  if (actionType === "extract" && Array.isArray(payload.candidate_selection_decisions)) {
    const reasons = payload.candidate_selection_decisions
      .map(asRecord)
      .map((decision) => nonEmptyText(decision?.reason))
      .filter((reason, index, values) => reason && values.indexOf(reason) === index);
    if (reasons.length > 0) {
      return reasons.join("\n\n");
    }
  }

  return nonEmptyText(payload.result) || nonEmptyText(payload.answer) || message;
}

function buildAgentStepViews(
  history: Array<Record<string, unknown>>,
  done: boolean,
): AgentStepView[] {
  return history.map((item, index) => {
    if (isAgentThinkingPlaceholder(item)) {
      return null;
    }
    const action = (item.action ?? {}) as Record<string, unknown>;
    const observation = (item.observation ?? {}) as Record<string, unknown>;
    const args = ((action.args as Record<string, unknown> | undefined) ?? {});
    const baseType = String(action.action_type ?? observation.action_type ?? "step");
    const toolName = String(args.tool_name ?? "");
    const payload = asRecord(observation.payload);
    const rawPhase = payload?.phase;
    const title = formatAgentStepTitle(toolName || baseType);
    const phase = rawPhase === "action"
      ? "action"
      : observation.message === "Running..."
        ? "action"
        : "result";
    const result = phase === "result" ? formatAgentResult(baseType, observation) : "";
    const images = Array.isArray(item.display_images)
      ? item.display_images.flatMap((image) => {
          if (!image || typeof image !== "object") {
            return [];
          }
          const record = image as Record<string, unknown>;
          const url = String(record.url ?? "").trim();
          return url ? [{ url, label: String(record.label ?? "Action image") }] : [];
        })
      : [];
    return {
      index,
      key: `${title}-${index}`,
      title,
      thought: String(action.intent ?? "").trim(),
      reasoning: nonEmptyText(payload?.reasoning_content),
      result,
      images,
      success: Boolean(observation.success),
      done: done || phase === "result" || index < history.length - 1,
      phase,
    };
  }).filter((step): step is AgentStepView => step !== null);
}

function attachProcessToLatestAssistant(
  messages: Message[],
  agentTrace: Array<Record<string, unknown>>,
  agentState: Record<string, unknown> | null,
  responseMode: "chat" | "docpilot",
): Message[] {
  const assistantIndex = latestAssistantIndex(messages);
  if (assistantIndex < 0) {
    return messages;
  }
  return messages.map((message, index) =>
    index === assistantIndex
      ? { ...message, agentTrace, agentState, responseMode }
      : message,
  );
}

function latestAssistantIndex(messages: Message[]): number {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === "assistant") {
      return index;
    }
  }
  return -1;
}

function latestAssistantProcess(messages: Message[]) {
  const index = latestAssistantIndex(messages);
  const message = index >= 0 ? messages[index] : undefined;
  return {
    agentTrace: message?.agentTrace,
    agentState: message?.agentState,
    responseMode: message?.responseMode,
  };
}

function statusLabel(file: WorkspaceFile): string {
  switch (file.processing_status) {
    case "queued":
      return "queued";
    case "processing":
      return "mineru";
    case "ready":
      return "ready";
    case "ready_image":
      return "image";
    case "ready_text":
      return "text";
    case "failed":
      return "failed";
    default:
      return "skip";
  }
}

function scrollElementWithinContainer(container: HTMLElement, target: HTMLElement) {
  const containerRect = container.getBoundingClientRect();
  const targetRect = target.getBoundingClientRect();
  const desiredTop =
    container.scrollTop +
    (targetRect.top - containerRect.top) -
    (container.clientHeight - targetRect.height) / 2;
  container.scrollTo({ top: Math.max(0, desiredTop), behavior: "smooth" });
}

function restoreWindowScroll(x: number, y: number) {
  window.scrollTo({ left: x, top: y, behavior: "auto" });
}

export default function App() {
  const [config, setConfig] = useState<ModelConfig>(loadStoredConfig);
  const [currentAgent, setCurrentAgent] = useState<AgentType>(() =>
    (localStorage.getItem(AGENT_KEY) === "chat" ? "chat" : "default"),
  );
  const usesVllmTags = currentAgent === "default" && config.provider === "vllm";
  const conversationSessionId = currentAgent === "chat" ? "chat" : "default";
  const workspaceSessionId = "default";
  const [messages, setMessages] = useState<Message[]>([]);
  const [conversationHistory, setConversationHistory] = useState<ConversationHistoryItem[]>([]);
  const [activeHistoryId, setActiveHistoryId] = useState<string | null>(null);
  const [isHistoryLoaded, setIsHistoryLoaded] = useState(false);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [input, setInput] = useState("");
  const [workspaceFiles, setWorkspaceFiles] = useState<WorkspaceFile[]>([]);
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [isModelMenuOpen, setIsModelMenuOpen] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [isUploadingAttachments, setIsUploadingAttachments] = useState(false);
  const [isDraggingAttachment, setIsDraggingAttachment] = useState(false);
  const [isLoadingFiles, setIsLoadingFiles] = useState(false);
  const [isViewerLoading, setIsViewerLoading] = useState(false);
  const [errorText, setErrorText] = useState("");
  const [activeFile, setActiveFile] = useState<WorkspaceFile | null>(null);
  const [viewer, setViewer] = useState<ViewerState>(EMPTY_VIEWER);
  const [viewerQuery, setViewerQuery] = useState("");
  const [activeViewerQuery, setActiveViewerQuery] = useState("");
  const [isSearchingDocument, setIsSearchingDocument] = useState(false);
  const [searchResultCount, setSearchResultCount] = useState(0);
  const [activeSearchIndex, setActiveSearchIndex] = useState(-1);
  const [viewerZoom, setViewerZoom] = useState(1);
  const [pdfPageNumber, setPdfPageNumber] = useState(1);
  const [pdfPageInput, setPdfPageInput] = useState("1");
  const [isPdfPageInputFocused, setIsPdfPageInputFocused] = useState(false);
  const [pdfPageCount, setPdfPageCount] = useState(0);
  const [pdfDisplayedPageNumber, setPdfDisplayedPageNumber] = useState(1);
  const [pdfPendingPageNumber, setPdfPendingPageNumber] = useState<number | null>(null);
  const [pdfPageBaseSize, setPdfPageBaseSize] = useState<{ width: number; height: number } | null>(
    null,
  );
  const [pdfViewportWidth, setPdfViewportWidth] = useState(0);
  const [pdfSearchMatches, setPdfSearchMatches] = useState<PdfSearchMatch[]>([]);
  const [activeEvidenceIndex, setActiveEvidenceIndex] = useState(-1);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isAgentMenuOpen, setIsAgentMenuOpen] = useState(false);
  const [isImportOpen, setIsImportOpen] = useState(false);
  const [chatPanelWidth, setChatPanelWidth] = useState<number | null>(null);
  const [hasManualResize, setHasManualResize] = useState(false);
  const [isResizing, setIsResizing] = useState(false);
  const [uploadMode, setUploadMode] = useState<"local" | "server">("local");
  const [serverPathInput, setServerPathInput] = useState("");
  const [workspaceSelection, setWorkspaceSelection] = useState("");
  const [lastAgentTrace, setLastAgentTrace] = useState<Array<Record<string, unknown>>>([]);
  const [lastAgentState, setLastAgentState] = useState<Record<string, unknown> | null>(null);
  const [lastResponseMode, setLastResponseMode] = useState<"chat" | "docpilot">("chat");
  const [activeAgentStepIndex, setActiveAgentStepIndex] = useState(0);
  const [expandedAgentStepIndexes, setExpandedAgentStepIndexes] = useState<Set<number>>(new Set());
  const [expandedArchivedAgentSteps, setExpandedArchivedAgentSteps] = useState<Set<string>>(new Set());
  const [copiedAgentStepKey, setCopiedAgentStepKey] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const messageListRef = useRef<HTMLDivElement | null>(null);
  const viewerBodyRef = useRef<HTMLDivElement | null>(null);
  const pdfViewportRef = useRef<HTMLDivElement | null>(null);
  const workspaceShellRef = useRef<HTMLDivElement | null>(null);
  const modelPickerRef = useRef<HTMLDivElement | null>(null);
  const settingsDropdownRef = useRef<HTMLDivElement | null>(null);
  const workspaceImportRef = useRef<HTMLDivElement | null>(null);
  const chatAttachmentInputRef = useRef<HTMLInputElement | null>(null);
  const searchRequestIdRef = useRef(0);
  const modelDiscoveryRequestIdRef = useRef(0);
  const chatAbortControllerRef = useRef<AbortController | null>(null);
  const activeFileNameRef = useRef("");
  const historySyncRef = useRef<Promise<void>>(Promise.resolve());
  const suppressAbortCleanupRef = useRef(false);
  const agentStepRefs = useRef(new Map<number, HTMLElement>());
  const agentStepPillRefs = useRef(new Map<number, HTMLButtonElement>());
  const agentStepNavigatorRef = useRef<HTMLDivElement | null>(null);
  const agentStepScrollbarRef = useRef<HTMLDivElement | null>(null);
  const agentStepScrollbarThumbRef = useRef<HTMLSpanElement | null>(null);
  const viewportLockTargetRef = useRef<{ x: number; y: number } | null>(null);
  const viewportLockIntervalRef = useRef<number | null>(null);
  const viewportLockTimeoutRef = useRef<number | null>(null);
  const copyFeedbackTimeoutRef = useRef<number | null>(null);

  function clearViewportLock() {
    if (viewportLockIntervalRef.current !== null) {
      window.clearInterval(viewportLockIntervalRef.current);
      viewportLockIntervalRef.current = null;
    }
    if (viewportLockTimeoutRef.current !== null) {
      window.clearTimeout(viewportLockTimeoutRef.current);
      viewportLockTimeoutRef.current = null;
    }
    viewportLockTargetRef.current = null;
  }

  function lockViewportFor(durationMs = 1200) {
    const target = { x: window.scrollX, y: window.scrollY };
    viewportLockTargetRef.current = target;
    restoreWindowScroll(target.x, target.y);

    if (viewportLockIntervalRef.current !== null) {
      window.clearInterval(viewportLockIntervalRef.current);
    }
    viewportLockIntervalRef.current = window.setInterval(() => {
      const currentTarget = viewportLockTargetRef.current;
      if (currentTarget) {
        restoreWindowScroll(currentTarget.x, currentTarget.y);
      }
    }, 16);

    if (viewportLockTimeoutRef.current !== null) {
      window.clearTimeout(viewportLockTimeoutRef.current);
    }
    viewportLockTimeoutRef.current = window.setTimeout(() => {
      clearViewportLock();
    }, durationMs);

    requestAnimationFrame(() => {
      const currentTarget = viewportLockTargetRef.current;
      if (!currentTarget) {
        return;
      }
      restoreWindowScroll(currentTarget.x, currentTarget.y);
      requestAnimationFrame(() => {
        const latestTarget = viewportLockTargetRef.current;
        if (latestTarget) {
          restoreWindowScroll(latestTarget.x, latestTarget.y);
        }
      });
    });
  }

  useEffect(() => {
    localStorage.removeItem(LEGACY_CONFIG_KEY);
  }, []);

  useEffect(() => {
    let cancelled = false;
    const legacyHistory = parseConversationHistory(localStorage.getItem(LEGACY_HISTORY_KEY));
    const legacyActiveId = localStorage.getItem(LEGACY_ACTIVE_HISTORY_KEY);

    function hydrateHistory(history: ConversationHistoryItem[], requestedActiveId: string | null) {
      const active = history.find((item) => item.id === requestedActiveId) ?? null;
      setConversationHistory(history);
      setActiveHistoryId(active?.id ?? null);
      if (!active) {
        return;
      }
      const process = latestAssistantProcess(active.messages);
      const hasMessageProcess = Boolean(
        process.agentTrace?.length || process.agentState || process.responseMode === "docpilot",
      );
      setMessages(active.messages.map((message) => ({ ...message })));
      setSelectedFiles([...active.selectedFiles]);
      setLastAgentTrace(hasMessageProcess ? (process.agentTrace ?? []) : (active.agentTrace ?? []));
      setLastAgentState(hasMessageProcess ? (process.agentState ?? null) : (active.agentState ?? null));
      setLastResponseMode(hasMessageProcess ? (process.responseMode ?? "chat") : (active.responseMode ?? "chat"));
      setCurrentAgent(active.agent ?? "default");
    }

    async function loadHistory() {
      try {
        const stored = await fetchConversationHistory();
        let history = parseConversationHistory(JSON.stringify(stored.conversations));
        let activeId = stored.active_id;
        if (history.length === 0 && legacyHistory.length > 0) {
          history = legacyHistory;
          activeId = legacyHistory.some((item) => item.id === legacyActiveId)
            ? legacyActiveId
            : null;
          await saveConversationHistory({ active_id: activeId, conversations: history });
        }
        if (cancelled) {
          return;
        }
        hydrateHistory(history, activeId);
        localStorage.removeItem(LEGACY_HISTORY_KEY);
        localStorage.removeItem(LEGACY_ACTIVE_HISTORY_KEY);
        setIsHistoryLoaded(true);
      } catch (error) {
        if (!cancelled) {
          hydrateHistory(legacyHistory, legacyActiveId);
          setErrorText(`Unable to load local conversation history: ${String(error)}`);
        }
      }
    }

    void loadHistory();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    return () => {
      clearViewportLock();
      if (copyFeedbackTimeoutRef.current !== null) {
        window.clearTimeout(copyFeedbackTimeoutRef.current);
      }
    };
  }, []);

  useEffect(() => {
    localStorage.setItem(CONFIG_KEY, JSON.stringify(config));
  }, [config]);

  useEffect(() => {
    if (currentAgent === "chat" && config.provider !== "custom") {
      setConfig((current) => ({ ...current, provider: "custom" }));
    }
  }, [currentAgent, config.provider]);

  useEffect(() => {
    localStorage.setItem(AGENT_KEY, currentAgent);
  }, [currentAgent]);

  useEffect(() => {
    if (!isHistoryLoaded) {
      return;
    }
    const snapshot = { active_id: activeHistoryId, conversations: conversationHistory };
    historySyncRef.current = historySyncRef.current
      .catch(() => undefined)
      .then(async () => {
        await saveConversationHistory(snapshot);
      })
      .catch((error) => {
        setErrorText(`Unable to save local conversation history: ${String(error)}`);
      });
  }, [activeHistoryId, conversationHistory, isHistoryLoaded]);

  useEffect(() => {
    if (!isSending) {
      archiveCurrentConversation();
    }
  }, [isSending]);

  useEffect(() => {
    const list = messageListRef.current;
    if (!list) {
      return;
    }
    list.scrollTo({ top: list.scrollHeight, behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (lastResponseMode !== "docpilot" || lastAgentTrace.length === 0) {
      setActiveAgentStepIndex(0);
      setExpandedAgentStepIndexes(new Set());
      return;
    }
    let nextIndex = lastAgentTrace.length - 1;
    while (nextIndex >= 0 && isAgentThinkingPlaceholder(lastAgentTrace[nextIndex])) {
      nextIndex -= 1;
    }
    if (nextIndex < 0) {
      return;
    }
    setActiveAgentStepIndex(nextIndex);
    setExpandedAgentStepIndexes(new Set([nextIndex]));
  }, [lastAgentTrace, lastResponseMode]);

  useEffect(() => {
    const container = agentStepNavigatorRef.current;
    const target = agentStepPillRefs.current.get(activeAgentStepIndex);
    if (!container || !target) {
      return;
    }
    container.scrollTo({
      left: target.offsetLeft - container.clientWidth / 2 + target.offsetWidth / 2,
      behavior: "smooth",
    });
  }, [activeAgentStepIndex, lastAgentTrace.length]);

  useLayoutEffect(() => {
    syncAgentStepScrollbar();
    window.addEventListener("resize", syncAgentStepScrollbar);
    return () => window.removeEventListener("resize", syncAgentStepScrollbar);
  }, [lastAgentTrace.length, lastResponseMode]);

  useEffect(() => {
    if (!isSending || lastResponseMode !== "docpilot") {
      return;
    }
    const target = agentStepRefs.current.get(activeAgentStepIndex);
    const container = messageListRef.current;
    if (target && container) {
      scrollElementWithinContainer(container, target);
    }
  }, [activeAgentStepIndex, lastAgentTrace.length, isSending, lastResponseMode]);

  useEffect(() => {
    function handlePointerDown(event: MouseEvent) {
      const target = event.target;
      if (!(target instanceof Node)) {
        return;
      }
      if (!modelPickerRef.current?.contains(target)) {
        setIsModelMenuOpen(false);
      }
      if (!settingsDropdownRef.current?.contains(target)) {
        setIsSettingsOpen(false);
        setIsAgentMenuOpen(false);
      }
      if (!workspaceImportRef.current?.contains(target)) {
        setIsImportOpen(false);
      }
    }

    document.addEventListener("mousedown", handlePointerDown);
    return () => document.removeEventListener("mousedown", handlePointerDown);
  }, []);

  useEffect(() => {
    const container = viewerBodyRef.current;
    if (!container || !activeViewerQuery.trim()) {
      return;
    }

    const marks = Array.from(container.querySelectorAll("mark"));
    marks.forEach((mark) => mark.classList.remove("active-search-hit"));

    if (viewer.mode === "pdf") {
      const activeMatch = pdfSearchMatches[activeSearchIndex];
      if (!activeMatch || activeMatch.pageNumber !== pdfDisplayedPageNumber) {
        return;
      }
      const target = container.querySelector(
        `.pdf-search-box[data-search-index="${activeSearchIndex}"]`,
      );
      if (target instanceof HTMLElement) {
        const pdfScroller = container.querySelector(".pdf-viewer-shell");
        scrollElementWithinContainer(
          pdfScroller instanceof HTMLElement ? pdfScroller : container,
          target,
        );
      }
      return;
    }

    if (marks.length === 0) {
      return;
    }
    const safeIndex = Math.min(Math.max(activeSearchIndex, 0), marks.length - 1);
    const target = marks[safeIndex];
    if (target instanceof HTMLElement) {
      target.classList.add("active-search-hit");
      scrollElementWithinContainer(container, target);
    }
  }, [
    activeSearchIndex,
    activeViewerQuery,
    pdfDisplayedPageNumber,
    pdfSearchMatches,
    viewer.mode,
    viewer.sourceUrl,
  ]);

  useEffect(() => {
    if (!isResizing) {
      return;
    }

    function handleMove(event: MouseEvent) {
      const shell = workspaceShellRef.current;
      if (!shell) {
        return;
      }
      const bounds = shell.getBoundingClientRect();
      const nextWidth = event.clientX - bounds.left;
      const minWidth = 420;
      const maxWidth = Math.max(minWidth, bounds.width - 440);
      setHasManualResize(true);
      setChatPanelWidth(Math.min(Math.max(nextWidth, minWidth), maxWidth));
    }

    function handleUp() {
      setIsResizing(false);
    }

    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleUp);
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";

    return () => {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", handleUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, [isResizing]);

  useEffect(() => {
    if (hasManualResize) {
      return;
    }

    function syncDefaultSplit() {
      const shell = workspaceShellRef.current;
      if (!shell) {
        return;
      }
      const bounds = shell.getBoundingClientRect();
      const minWidth = 420;
      const maxWidth = Math.max(minWidth, bounds.width - 440);
      const targetWidth = Math.round(bounds.width * 0.6);
      setChatPanelWidth(Math.min(Math.max(targetWidth, minWidth), maxWidth));
    }

    syncDefaultSplit();
    window.addEventListener("resize", syncDefaultSplit);
    return () => window.removeEventListener("resize", syncDefaultSplit);
  }, [hasManualResize]);

  useEffect(() => {
    const shell = workspaceShellRef.current;
    if (!shell) {
      return;
    }

    const observer = new ResizeObserver(() => {
      setChatPanelWidth((current) => {
        if (current === null) {
          return current;
        }
        const bounds = shell.getBoundingClientRect();
        const minWidth = 420;
        const maxWidth = Math.max(minWidth, bounds.width - 440);
        return Math.min(Math.max(current, minWidth), maxWidth);
      });
    });
    observer.observe(shell);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    void refreshFiles();
  }, [workspaceSessionId]);

  useEffect(() => {
    const hasPending = workspaceFiles.some((file) =>
      file.processing_status === "queued" || file.processing_status === "processing",
    );
    if (!hasPending) {
      return;
    }
    const timer = window.setInterval(() => {
      void refreshFiles(false);
    }, 4000);
    return () => window.clearInterval(timer);
  }, [workspaceFiles, workspaceSessionId]);

  async function refreshFiles(showBusy = true) {
    if (showBusy) {
      setIsLoadingFiles(true);
    }
    try {
      const files = await fetchWorkspaceFiles(workspaceSessionId);
      setWorkspaceFiles(files);
      setSelectedFiles((current) =>
        current.filter((name) => files.some((file) => file.name === name)),
      );
      setWorkspaceSelection((current) => {
        if (current && files.some((file) => file.name === current)) {
          return current;
        }
        if (
          activeFileNameRef.current &&
          files.some((file) => file.name === activeFileNameRef.current)
        ) {
          return activeFileNameRef.current;
        }
        return files[0]?.name ?? "";
      });
      const currentActiveName = activeFileNameRef.current;
      const nextActiveFile = currentActiveName
        ? files.find((file) => file.name === currentActiveName) ?? null
        : null;

      if (nextActiveFile) {
        setActiveFile(nextActiveFile);
      } else if (files.length > 0) {
        await openViewer(files[0]);
      } else {
        activeFileNameRef.current = "";
        setActiveFile(null);
        setViewer(EMPTY_VIEWER);
      }
    } catch (error) {
      setErrorText(String(error));
    } finally {
      if (showBusy) {
        setIsLoadingFiles(false);
      }
    }
  }

  function invalidateDiscoveredModels() {
    modelDiscoveryRequestIdRef.current += 1;
    setAvailableModels([]);
    setIsModelMenuOpen(false);
  }

  function loadPreset(provider: "vllm" | "custom") {
    setErrorText("");
    invalidateDiscoveredModels();
    setConfig({ ...PRESET_CONFIGS[provider] });
  }

  async function handleDiscoverModels() {
    if (availableModels.length > 0) {
      setIsModelMenuOpen((current) => !current);
      return;
    }

    setErrorText("");
    const requestId = ++modelDiscoveryRequestIdRef.current;
    try {
      const models = await discoverModels(config);
      if (requestId !== modelDiscoveryRequestIdRef.current) return;
      setAvailableModels(models);
      setIsModelMenuOpen(models.length > 0);
    } catch (error) {
      if (requestId !== modelDiscoveryRequestIdRef.current) return;
      setErrorText(String(error));
    }
  }

  async function storeWorkspaceFiles(files: File[]): Promise<WorkspaceFile[]> {
    const nextFiles = await uploadWorkspaceFiles(workspaceSessionId, files);
    setWorkspaceFiles(nextFiles);
    const uploadedNames = new Set(files.map((file) => file.name));
    const uploadedFiles = nextFiles.filter((file) => uploadedNames.has(file.name));
    if (uploadedFiles[0]) {
      await openViewer(uploadedFiles[0]);
    }
    return uploadedFiles;
  }

  async function handleUpload(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    if (!files.length) {
      return;
    }
    setErrorText("");
    try {
      await storeWorkspaceFiles(files);
    } catch (error) {
      setErrorText(String(error));
    } finally {
      event.target.value = "";
    }
  }

  async function attachChatFiles(files: File[]) {
    if (currentAgent !== "chat" || isUploadingAttachments || files.length === 0) {
      return;
    }
    const supported = files.filter((file) =>
      CHAT_ATTACHMENT_EXTENSIONS.has(getFileExtension(file.name)),
    );
    const unsupported = files.filter(
      (file) => !CHAT_ATTACHMENT_EXTENSIONS.has(getFileExtension(file.name)),
    );
    if (supported.length === 0) {
      setErrorText("Chat attachments support PDF, PNG, JPG, WebP, and GIF files.");
      return;
    }

    setErrorText("");
    setIsUploadingAttachments(true);
    try {
      const uploadedFiles = await storeWorkspaceFiles(supported);
      setSelectedFiles((current) => [
        ...current,
        ...uploadedFiles.map((file) => file.name).filter((name) => !current.includes(name)),
      ]);
      if (unsupported.length > 0) {
        setErrorText(
          `Unsupported attachment${unsupported.length > 1 ? "s" : ""}: ${unsupported.map((file) => file.name).join(", ")}`,
        );
      }
    } catch (error) {
      setErrorText(String(error));
    } finally {
      setIsUploadingAttachments(false);
      if (chatAttachmentInputRef.current) {
        chatAttachmentInputRef.current.value = "";
      }
    }
  }

  function handleComposerDrag(event: DragEvent<HTMLDivElement>) {
    if (currentAgent !== "chat" || !event.dataTransfer.types.includes("Files")) {
      return;
    }
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
    setIsDraggingAttachment(true);
  }

  function handleComposerDragLeave(event: DragEvent<HTMLDivElement>) {
    const nextTarget = event.relatedTarget;
    if (nextTarget instanceof Node && event.currentTarget.contains(nextTarget)) {
      return;
    }
    setIsDraggingAttachment(false);
  }

  function handleComposerDrop(event: DragEvent<HTMLDivElement>) {
    if (currentAgent !== "chat") {
      return;
    }
    event.preventDefault();
    setIsDraggingAttachment(false);
    void attachChatFiles(Array.from(event.dataTransfer.files));
  }

  async function handleServerPathImport() {
    const paths = serverPathInput
      .split("\n")
      .map((item) => item.trim())
      .filter(Boolean);
    if (!paths.length) {
      setErrorText("请至少输入一个服务器文件路径。");
      return;
    }

    setErrorText("");
    try {
      const result = await importWorkspaceFilesFromPaths(workspaceSessionId, paths);
      setWorkspaceFiles(result.files);
      if (result.errors.length > 0) {
        setErrorText(result.errors.join("\n"));
      }
      setServerPathInput("");
      if (result.files.length > 0) {
        const firstImported = result.files.find((file) => result.imported.includes(file.name));
        if (firstImported) {
          await openViewer(firstImported);
        }
      }
    } catch (error) {
      setErrorText(String(error));
    }
  }

  async function openViewer(file: WorkspaceFile, preserveEvidence = false) {
    const sourceUrl = toAbsoluteUrl(file.download_url);
    if (!preserveEvidence) {
      setActiveEvidenceIndex(-1);
    }
    activeFileNameRef.current = file.name;
    setActiveFile(file);
    setWorkspaceSelection(file.name);
    setViewerZoom(1);
    setViewerQuery("");
    setPdfPageNumber(1);
    if (viewer.sourceUrl !== sourceUrl) {
      setPdfPageCount(0);
    }
    setIsViewerLoading(true);
    setErrorText("");
    try {
      const extension = getFileExtension(file.name);

      if (extension === "pdf") {
        setViewer({
          mode: "pdf",
          title: file.name,
          sourceUrl,
          htmlContent: "",
          textContent: "",
        });
        return;
      }

      if (CHAT_ATTACHMENT_EXTENSIONS.has(extension)) {
        setViewer({
          mode: "image",
          title: file.name,
          sourceUrl,
          htmlContent: "",
          textContent: "",
        });
        return;
      }

      if (extension === "docx") {
        const response = await fetch(sourceUrl);
        const arrayBuffer = await response.arrayBuffer();
        const result = await mammoth.convertToHtml({ arrayBuffer });
        setViewer({
          mode: "docx",
          title: file.name,
          sourceUrl,
          htmlContent: result.value,
          textContent: result.value.replace(/<[^>]+>/g, " "),
        });
        return;
      }

      if (extension === "md" || extension === "markdown") {
        const response = await fetch(sourceUrl);
        const markdown = await response.text();
        setViewer({
          mode: "markdown",
          title: file.name,
          sourceUrl,
          htmlContent: await marked.parse(markdown),
          textContent: markdown,
        });
        return;
      }

      if (TEXT_EXTENSIONS.has(extension)) {
        const response = await fetch(sourceUrl);
        const text = await response.text();
        setViewer({
          mode: "text",
          title: file.name,
          sourceUrl,
          htmlContent: "",
          textContent: text,
        });
        return;
      }

      setViewer({
        mode: "unsupported",
        title: file.name,
        sourceUrl,
        htmlContent: "",
        textContent:
          "当前浏览器查看器已支持 pdf、docx、md 和常见纯文本格式。此文件可以下载或外部打开。",
      });
    } catch (error) {
      setViewer({
        mode: "unsupported",
        title: file.name,
        sourceUrl: toAbsoluteUrl(file.download_url),
        htmlContent: "",
        textContent: `加载失败: ${String(error)}`,
      });
    } finally {
      setIsViewerLoading(false);
    }
  }

  async function handleDelete(fileName: string) {
    setErrorText("");
    try {
      const nextFiles = await deleteWorkspaceFile(workspaceSessionId, fileName);
      setWorkspaceFiles(nextFiles);
      setSelectedFiles((current) => current.filter((item) => item !== fileName));
      setWorkspaceSelection((current) => {
        if (current !== fileName) {
          return current;
        }
        return nextFiles[0]?.name ?? "";
      });
      const nextActive = nextFiles.find((file) => file.name === activeFileNameRef.current) ?? null;
      if (nextActive) {
        setActiveFile(nextActive);
      } else if (nextFiles.length > 0) {
        await openViewer(nextFiles[0]);
      } else {
        activeFileNameRef.current = "";
        setActiveFile(null);
        setViewer(EMPTY_VIEWER);
      }
    } catch (error) {
      setErrorText(String(error));
    }
  }

  async function handleRetry(fileName: string) {
    setErrorText("");
    try {
      const nextFiles = await retryWorkspaceFile(workspaceSessionId, fileName);
      setWorkspaceFiles(nextFiles);
      const nextActive = nextFiles.find((file) => file.name === fileName) ?? null;
      if (nextActive) {
        activeFileNameRef.current = nextActive.name;
        setActiveFile(nextActive);
      }
    } catch (error) {
      setErrorText(String(error));
    }
  }

  function toggleSelectedFile(fileName: string) {
    setSelectedFiles((current) =>
      current.includes(fileName)
        ? current.filter((item) => item !== fileName)
        : [...current, fileName],
    );
  }

  async function handleSend() {
    const content = input.trim();
    if (!content || isSending || isUploadingAttachments) {
      return;
    }
    if (!config.model.trim()) {
      setErrorText("请先填写模型名。");
      return;
    }

    setIsHistoryOpen(false);
    setIsSettingsOpen(false);
    setIsAgentMenuOpen(false);
    setErrorText("");
    setInput("");

    const archivedMessages = attachProcessToLatestAssistant(
      messages,
      lastAgentTrace,
      lastAgentState,
      lastResponseMode,
    );
    const nextMessages: Message[] = [
      ...archivedMessages,
      { id: createId(), role: "user", content },
      { id: createId(), role: "assistant", content: "" },
    ];

    const conversationId = activeHistoryId ?? createId();
    if (!activeHistoryId) {
      setActiveHistoryId(conversationId);
    }
    setMessages(nextMessages);
    setIsSending(true);
    setLastAgentTrace([]);
    setLastAgentState(null);
    setLastResponseMode(currentAgent === "default" ? "docpilot" : "chat");
    setExpandedAgentStepIndexes(new Set());
    const controller = new AbortController();
    chatAbortControllerRef.current = controller;

    try {
      await streamChat(
        conversationSessionId,
        workspaceSessionId,
        conversationId,
        config,
        nextMessages.filter((item) => item.role === "user" || item.content),
        selectedFiles,
        currentAgent,
        (event) => {
          if (event.type === "meta") {
            const mode = event.mode ?? "chat";
            setLastResponseMode(mode);
            return;
          }
          if (event.type === "error") {
            setErrorText(event.message);
            setLastAgentTrace((current) => current.filter((item) => !isPendingAgentStep(item)));
            setMessages((current) => {
              const cloned = [...current];
              const last = cloned[cloned.length - 1];
              if (last?.role === "assistant" && !last.content.trim()) {
                cloned[cloned.length - 1] = {
                  ...last,
                  content: `Error: ${event.message}`,
                };
              }
              return cloned;
            });
            return;
          }
          if (event.type === "step") {
            if (event.history) {
              setLastAgentTrace(event.history);
            }
            if (event.state) {
              setLastAgentState(event.state);
            }
            return;
          }
          if (event.type === "delta") {
            setMessages((current) => {
              const cloned = [...current];
              const last = cloned[cloned.length - 1];
              if (last?.role === "assistant") {
                cloned[cloned.length - 1] = {
                  ...last,
                  content: last.content + event.content,
                };
              }
              return cloned;
            });
            if (event.history) {
              setLastAgentTrace(event.history);
            }
            if (event.state) {
              setLastAgentState(event.state);
            }
            return;
          }
          if (event.type === "done") {
          }
        },
        controller.signal,
      );
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        if (suppressAbortCleanupRef.current) {
          suppressAbortCleanupRef.current = false;
        } else {
          setLastAgentTrace((current) =>
            current.filter((item) => !isPendingAgentStep(item)),
          );
          setMessages((current) => {
            const cloned = [...current];
            const last = cloned[cloned.length - 1];
            if (last?.role === "assistant" && !last.content.trim()) {
              cloned[cloned.length - 1] = { ...last, content: "Stopped by user." };
            }
            return cloned;
          });
        }
      } else {
        setLastAgentTrace((current) => current.filter((item) => !isPendingAgentStep(item)));
        setErrorText(String(error));
      }
    } finally {
      chatAbortControllerRef.current = null;
      setIsSending(false);
    }
  }

  function handleStopSending() {
    chatAbortControllerRef.current?.abort();
  }

  function archiveCurrentConversation(preserveOrder = false) {
    const snapshot = attachProcessToLatestAssistant(
      messages.filter((message) => message.role === "user" || message.content.trim()),
      lastAgentTrace,
      lastAgentState,
      lastResponseMode,
    );
    if (!snapshot.some((message) => message.role === "user" && message.content.trim())) {
      return;
    }
    const item: ConversationHistoryItem = {
      id: activeHistoryId ?? createId(),
      title: getConversationTitle(snapshot),
      messages: snapshot,
      selectedFiles: [...selectedFiles],
      agentTrace: lastAgentTrace,
      agentState: lastAgentState,
      responseMode: lastResponseMode,
      agent: currentAgent,
      updatedAt: new Date().toISOString(),
    };
    setConversationHistory((current) => preserveOrder
      ? updateConversationHistoryInPlace(current, item)
      : upsertConversationHistory(current, item));
  }

  function resetConversation() {
    suppressAbortCleanupRef.current = true;
    chatAbortControllerRef.current?.abort();
    chatAbortControllerRef.current = null;
    setMessages([]);
    setLastAgentTrace([]);
    setLastAgentState(null);
    setLastResponseMode("chat");
    setExpandedAgentStepIndexes(new Set());
    setExpandedArchivedAgentSteps(new Set());
    setErrorText("");
    setIsSending(false);
    setIsHistoryOpen(false);
  }

  async function deletePersistedConversation(id: string) {
    const deletion = historySyncRef.current
      .catch(() => undefined)
      .then(() => deleteConversationHistory(id));
    historySyncRef.current = deletion.then(() => undefined);
    return deletion;
  }

  async function handleClearConversation() {
    const conversationId = activeHistoryId;
    resetConversation();
    if (!conversationId) {
      return;
    }
    try {
      const stored = await deletePersistedConversation(conversationId);
      setConversationHistory(stored.conversations);
    } catch (error) {
      setErrorText(`Unable to delete conversation: ${String(error)}`);
    }
  }

  async function handleDeleteHistoryItem(id: string) {
    try {
      const stored = await deletePersistedConversation(id);
      setConversationHistory(stored.conversations);
    } catch (error) {
      setErrorText(`Unable to delete conversation: ${String(error)}`);
      return;
    }
    if (activeHistoryId !== id) {
      return;
    }
    resetConversation();
    setSelectedFiles([]);
    setActiveHistoryId(null);
  }

  function handleNewConversation() {
    archiveCurrentConversation(true);
    resetConversation();
    setSelectedFiles([]);
    const id = createId();
    setConversationHistory((current) =>
      upsertConversationHistory(current, {
        id,
        title: "New conversation",
        messages: [],
        selectedFiles: [],
        agentTrace: [],
        agentState: null,
        responseMode: "chat",
        agent: currentAgent,
        updatedAt: new Date().toISOString(),
      }),
    );
    setActiveHistoryId(id);
  }

  function restoreConversation(item: ConversationHistoryItem) {
    if (isSending || activeHistoryId === item.id) {
      setIsHistoryOpen(false);
      return;
    }
    archiveCurrentConversation(true);
    const process = latestAssistantProcess(item.messages);
    const hasMessageProcess = Boolean(
      process.agentTrace?.length || process.agentState || process.responseMode === "docpilot",
    );
    setMessages(item.messages.map((message) => ({ ...message })));
    setSelectedFiles(item.selectedFiles);
    setInput("");
    setLastAgentTrace(hasMessageProcess ? (process.agentTrace ?? []) : (item.agentTrace ?? []));
    setLastAgentState(hasMessageProcess ? (process.agentState ?? null) : (item.agentState ?? null));
    setLastResponseMode(hasMessageProcess ? (process.responseMode ?? "chat") : (item.responseMode ?? "chat"));
    setCurrentAgent(item.agent ?? currentAgent);
    setExpandedAgentStepIndexes(new Set());
    setExpandedArchivedAgentSteps(new Set());
    setErrorText("");
    setActiveHistoryId(item.id);
    setIsHistoryOpen(false);
  }

  function jumpToAgentStep(index: number) {
    setActiveAgentStepIndex(index);
    setExpandedAgentStepIndexes((current) => {
      const next = new Set(current);
      next.add(index);
      return next;
    });
    const target = agentStepRefs.current.get(index);
    const container = messageListRef.current;
    if (target && container) {
      scrollElementWithinContainer(container, target);
    }
  }

  function syncAgentStepScrollbar() {
    const navigator = agentStepNavigatorRef.current;
    const scrollbar = agentStepScrollbarRef.current;
    const thumb = agentStepScrollbarThumbRef.current;
    if (!navigator || !scrollbar || !thumb) {
      return;
    }
    const maxScrollLeft = navigator.scrollWidth - navigator.clientWidth;
    if (maxScrollLeft <= 1) {
      scrollbar.hidden = true;
      return;
    }
    scrollbar.hidden = false;
    const trackWidth = scrollbar.clientWidth;
    const thumbWidth = Math.min(
      84,
      Math.max(32, trackWidth * (navigator.clientWidth / navigator.scrollWidth)),
    );
    const maxThumbLeft = Math.max(0, trackWidth - thumbWidth);
    const thumbLeft = (navigator.scrollLeft / maxScrollLeft) * maxThumbLeft;
    thumb.style.width = `${thumbWidth}px`;
    thumb.style.transform = `translateX(${thumbLeft}px)`;
  }

  function dragAgentStepScrollbar(event: ReactPointerEvent<HTMLDivElement>) {
    const navigator = agentStepNavigatorRef.current;
    const scrollbar = agentStepScrollbarRef.current;
    const thumb = agentStepScrollbarThumbRef.current;
    if (!navigator || !scrollbar || !thumb) {
      return;
    }
    const maxScrollLeft = navigator.scrollWidth - navigator.clientWidth;
    const maxThumbLeft = scrollbar.clientWidth - thumb.offsetWidth;
    if (maxScrollLeft <= 0 || maxThumbLeft <= 0) {
      return;
    }

    event.preventDefault();
    const track = event.currentTarget;
    let startScrollLeft = navigator.scrollLeft;
    if (event.target === track) {
      const left = Math.min(
        maxThumbLeft,
        Math.max(0, event.clientX - track.getBoundingClientRect().left - thumb.offsetWidth / 2),
      );
      startScrollLeft = (left / maxThumbLeft) * maxScrollLeft;
      navigator.scrollLeft = startScrollLeft;
    }
    const startX = event.clientX;
    track.setPointerCapture(event.pointerId);

    const handleMove = (moveEvent: PointerEvent) => {
      navigator.scrollLeft =
        startScrollLeft + ((moveEvent.clientX - startX) / maxThumbLeft) * maxScrollLeft;
    };
    const handleUp = () => {
      track.removeEventListener("pointermove", handleMove);
      track.removeEventListener("pointerup", handleUp);
      track.removeEventListener("pointercancel", handleUp);
    };
    track.addEventListener("pointermove", handleMove);
    track.addEventListener("pointerup", handleUp);
    track.addEventListener("pointercancel", handleUp);
  }

  function toggleAgentStep(index: number, messageId: string) {
    if (messageId !== latestAssistantMessageId) {
      const key = `${messageId}:${index}`;
      setExpandedArchivedAgentSteps((current) => {
        const next = new Set(current);
        if (next.has(key)) {
          next.delete(key);
        } else {
          next.add(key);
        }
        return next;
      });
      return;
    }
    setExpandedAgentStepIndexes((current) => {
      const next = new Set(current);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  }

  async function copyAgentStep(step: AgentStepView, messageId: string) {
    const content = [
      step.title,
      step.thought ? `Thinking\n${step.thought}` : "",
      step.reasoning ? `Action thinking\n${step.reasoning}` : "",
      step.result ? `Result\n${step.result}` : "",
    ].filter(Boolean).join("\n\n");

    try {
      await navigator.clipboard.writeText(content);
      if (copyFeedbackTimeoutRef.current !== null) {
        window.clearTimeout(copyFeedbackTimeoutRef.current);
      }
      setCopiedAgentStepKey(`${messageId}:${step.index}`);
      copyFeedbackTimeoutRef.current = window.setTimeout(() => {
        setCopiedAgentStepKey(null);
        copyFeedbackTimeoutRef.current = null;
      }, 1600);
    } catch (error) {
      setErrorText(`Unable to copy action: ${String(error)}`);
    }
  }

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.nativeEvent.isComposing) {
      return;
    }

    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void handleSend();
    }
  }

  const conversationTitle = useMemo(() => getConversationTitle(messages), [messages]);

  const selectedWorkspaceFile = useMemo(
    () => workspaceFiles.find((file) => file.name === workspaceSelection) ?? workspaceFiles[0] ?? null,
    [workspaceFiles, workspaceSelection],
  );

  const latestAssistantMessageId = useMemo(() => {
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      if (messages[index]?.role === "assistant") {
        return messages[index].id;
      }
    }
    return null;
  }, [messages]);

  const agentSteps = useMemo(
    () => buildAgentStepViews(lastAgentTrace, Boolean(lastAgentState?.done)),
    [lastAgentTrace, lastAgentState],
  );
  const answerEvidence = useMemo(
    () => lastResponseMode === "docpilot"
      ? collectAnswerEvidenceLocators(lastAgentState)
      : [],
    [lastAgentState, lastResponseMode],
  );
  const activeEvidence = activeEvidenceIndex >= 0
    ? answerEvidence[activeEvidenceIndex] ?? null
    : null;
  const isPlanningNextAgentStep = Boolean(
    isSending &&
    lastAgentTrace.length > 0 &&
    isAgentThinkingPlaceholder(lastAgentTrace[lastAgentTrace.length - 1]),
  );

  useEffect(() => {
    if (activeEvidenceIndex >= answerEvidence.length) {
      setActiveEvidenceIndex(-1);
    }
  }, [activeEvidenceIndex, answerEvidence.length]);

  useEffect(() => {
    setActiveEvidenceIndex(-1);
  }, [activeHistoryId]);

  useEffect(() => {
    if (!activeEvidence || viewer.mode !== "pdf") {
      return;
    }
    if (activeEvidence.page !== pdfDisplayedPageNumber) {
      return;
    }
    const target = viewerBodyRef.current?.querySelector(
      `.pdf-evidence-box[data-evidence-index="${activeEvidenceIndex}"]`,
    );
    if (target instanceof HTMLElement && pdfViewportRef.current) {
      scrollElementWithinContainer(pdfViewportRef.current, target);
    }
  }, [activeEvidence, activeEvidenceIndex, pdfDisplayedPageNumber, viewer.mode, viewer.sourceUrl]);

  async function showEvidence(index: number) {
    if (answerEvidence.length === 0) {
      return;
    }
    const wrappedIndex = (index + answerEvidence.length) % answerEvidence.length;
    const evidence = answerEvidence[wrappedIndex];
    const file = workspaceFiles.find((item) => item.doc_id === evidence.docId);
    if (!file) {
      setErrorText(`Evidence document is no longer available in Workspace: ${evidence.docId}`);
      return;
    }
    setErrorText("");
    if (activeFileNameRef.current !== file.name) {
      await openViewer(file, true);
    }
    setPdfPageNumber(evidence.page);
    setActiveEvidenceIndex(wrappedIndex);
  }

  function toggleEvidence() {
    if (activeEvidenceIndex >= 0) {
      setActiveEvidenceIndex(-1);
      return;
    }
    void withViewportLock(() => showEvidence(0));
  }

  function shouldRenderEvidence(index: number, page: number) {
    const evidence = answerEvidence[index];
    return evidence?.docId === activeFile?.doc_id &&
      evidence.page === page &&
      (activeEvidence?.box ? Boolean(evidence.box) : index === activeEvidenceIndex);
  }

  const highlightedHtml = useMemo(() => {
    if (viewer.mode === "markdown" || viewer.mode === "docx") {
      return highlightHtml(viewer.htmlContent, activeViewerQuery);
    }
    if (viewer.mode === "text") {
      return highlightHtml(escapeHtml(viewer.textContent), activeViewerQuery).replaceAll(
        "\n",
        "<br/>",
      );
    }
    return "";
  }, [viewer, activeViewerQuery]);

  useLayoutEffect(() => {
    if (viewer.mode !== "pdf") {
      setPdfPageBaseSize(null);
      setPdfViewportWidth(0);
      return;
    }

    const container = viewerBodyRef.current;
    const pdfScroller = pdfViewportRef.current;
    if (!container || !pdfScroller) {
      return;
    }

    const updateSize = () => {
      const styles = window.getComputedStyle(pdfScroller);
      const width = Math.max(
        0,
        container.clientWidth -
          Number.parseFloat(styles.paddingLeft) -
          Number.parseFloat(styles.paddingRight),
      );
      setPdfViewportWidth((current) => (current === width ? current : width));
    };

    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(container);
    return () => observer.disconnect();
  }, [viewer.mode, viewer.sourceUrl]);

  useEffect(() => {
    if (viewer.mode !== "pdf") {
      return;
    }
    const node = pdfViewportRef.current;
    if (!node) {
      return;
    }

    function handleCtrlWheel(event: WheelEvent) {
      if (!event.ctrlKey || event.deltaY === 0) {
        return;
      }
      event.preventDefault();
      const direction = event.deltaY < 0 ? 0.1 : -0.1;
      setViewerZoom((current) =>
        Math.min(2.4, Math.max(0.6, Number((current + direction).toFixed(2)))),
      );
    }

    node.addEventListener("wheel", handleCtrlWheel, { passive: false });
    return () => node.removeEventListener("wheel", handleCtrlWheel);
  }, [viewer.mode, viewer.sourceUrl]);

  const pdfRenderWidth = useMemo(() => {
    if (!pdfPageBaseSize || pdfViewportWidth <= 0) {
      return undefined;
    }

    // Keep PDF sizing stable by fitting to viewport width only.
    // Height-constrained fitting can oscillate when scrollbar/viewport height changes during render.
    const widthScale = pdfViewportWidth / pdfPageBaseSize.width;
    const fitScale = widthScale;
    return Math.max(180, Math.floor(pdfPageBaseSize.width * fitScale * viewerZoom));
  }, [pdfPageBaseSize, pdfViewportWidth, viewerZoom]);

  useEffect(() => {
    if (viewer.mode !== "pdf") {
      setPdfDisplayedPageNumber(1);
      setPdfPendingPageNumber(null);
      return;
    }

    if (pdfPageNumber !== pdfDisplayedPageNumber) {
      setPdfPendingPageNumber(pdfPageNumber);
    }
  }, [viewer.mode, pdfPageNumber, viewer.sourceUrl]);

  useEffect(() => {
    setActiveViewerQuery("");
    setViewerQuery("");
    setSearchResultCount(0);
    setActiveSearchIndex(-1);
    setPdfSearchMatches([]);
    setIsSearchingDocument(false);
  }, [viewer.sourceUrl]);

  useEffect(() => {
    if (!activeViewerQuery.trim() || viewer.mode === "pdf") {
      return;
    }

    const container = viewerBodyRef.current;
    if (!container) {
      return;
    }

    const marks = container.querySelectorAll("mark");
    setSearchResultCount(marks.length);
    setActiveSearchIndex(marks.length > 0 ? 0 : -1);
  }, [activeViewerQuery, highlightedHtml, viewer.mode]);

  async function commitViewerSearch() {
    const query = viewerQuery.trim();
    const currentPageNumber = pdfDisplayedPageNumber;
    setActiveViewerQuery(query);

    if (!query) {
      setSearchResultCount(0);
      setActiveSearchIndex(-1);
      setPdfSearchMatches([]);
      return;
    }

    if (viewer.mode !== "pdf") {
      return;
    }

    const requestId = ++searchRequestIdRef.current;
    setIsSearchingDocument(true);

    try {
      const loadingTask = pdfjs.getDocument(viewer.sourceUrl);
      const pdfDocument = await loadingTask.promise;
      const matches: PdfSearchMatch[] = [];
      const normalizedQuery = query.toLowerCase();

      for (let pageNumber = 1; pageNumber <= pdfDocument.numPages; pageNumber += 1) {
        const page = await pdfDocument.getPage(pageNumber);
        const viewport = page.getViewport({ scale: 1 });
        const textContent = await page.getTextContent();
        let occurrenceInPage = 0;
        textContent.items.forEach((item) => {
          if (!("str" in item)) {
            return;
          }
          const itemMatches = Array.from(
            item.str.toLowerCase().matchAll(buildHighlightPattern(normalizedQuery)),
          );
          if (itemMatches.length === 0 || item.str.length === 0) {
            return;
          }
          const transform = pdfjs.Util.transform(viewport.transform, item.transform);
          const angle = Math.atan2(transform[1], transform[0]);
          const fontHeight = Math.hypot(transform[2], transform[3]);
          const cos = Math.cos(angle);
          const sin = Math.sin(angle);

          itemMatches.forEach((match) => {
            const start = match.index ?? 0;
            const startOffset = item.width * (start / item.str.length);
            const matchWidth = item.width * (normalizedQuery.length / item.str.length);
            const baselineX = transform[4] + cos * startOffset;
            const baselineY = transform[5] + sin * startOffset;
            matches.push({
              pageNumber,
              occurrenceInPage,
              left: ((baselineX + sin * fontHeight) / viewport.width) * 100,
              top: ((baselineY - cos * fontHeight) / viewport.height) * 100,
              width: (matchWidth / viewport.width) * 100,
              height: (fontHeight / viewport.height) * 100,
              rotation: angle,
            });
            occurrenceInPage += 1;
          });
        });
      }

      if (searchRequestIdRef.current !== requestId) {
        return;
      }

      setPdfSearchMatches(matches);
      setSearchResultCount(matches.length);
      const currentPageMatchIndex = matches.findIndex(
        (match) => match.pageNumber === currentPageNumber,
      );
      const nextActiveIndex =
        currentPageMatchIndex >= 0 ? currentPageMatchIndex : matches.length > 0 ? 0 : -1;
      setActiveSearchIndex(nextActiveIndex);
      if (nextActiveIndex >= 0) {
        setPdfPageNumber(matches[nextActiveIndex].pageNumber);
      }
    } catch (error) {
      if (searchRequestIdRef.current === requestId) {
        setErrorText(`搜索失败: ${String(error)}`);
        setPdfSearchMatches([]);
        setSearchResultCount(0);
        setActiveSearchIndex(-1);
      }
    } finally {
      if (searchRequestIdRef.current === requestId) {
        setIsSearchingDocument(false);
      }
    }
  }

  function handleViewerSearchKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "Enter") {
      event.preventDefault();
      void withViewportLock(commitViewerSearch);
    }
  }

  useEffect(() => {
    if (!isPdfPageInputFocused) {
      setPdfPageInput(String(pdfPageNumber));
    }
  }, [pdfPageNumber, isPdfPageInputFocused]);

  function commitPdfPageInput() {
    const maxPage = viewer.mode === "pdf" ? pdfPageCount || 1 : 1;
    const parsed = Number.parseInt(pdfPageInput, 10);
    const clamped = Number.isFinite(parsed)
      ? Math.min(Math.max(parsed, 1), maxPage)
      : pdfPageNumber;
    setPdfPageNumber(clamped);
    setPdfPageInput(String(clamped));
  }

  function handlePdfPageInputKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "Enter") {
      event.preventDefault();
      event.currentTarget.blur();
    }
  }

  function jumpToSearchResult(nextIndex: number) {
    if (searchResultCount === 0) {
      return;
    }
    const wrappedIndex = (nextIndex + searchResultCount) % searchResultCount;
    setActiveSearchIndex(wrappedIndex);
    if (viewer.mode === "pdf") {
      const match = pdfSearchMatches[wrappedIndex];
      if (match) {
        setPdfPageNumber(match.pageNumber);
      }
    }
  }

  function goToNextSearchResult() {
    jumpToSearchResult(activeSearchIndex + 1);
  }

  function goToPreviousSearchResult() {
    jumpToSearchResult(activeSearchIndex - 1);
  }

  function zoomIn() {
    setViewerZoom((current) => Math.min(Number((current + 0.1).toFixed(2)), 2.4));
  }

  function zoomOut() {
    setViewerZoom((current) => Math.max(Number((current - 0.1).toFixed(2)), 0.6));
  }

  function resetZoom() {
    setViewerZoom(1);
  }

  async function withViewportLock(task: () => void | Promise<void>) {
    lockViewportFor(1200);
    try {
      await Promise.resolve(task());
    } finally {
      const target = viewportLockTargetRef.current;
      if (target) {
        restoreWindowScroll(target.x, target.y);
      }
    }
  }

  return (
    <div className="app-shell">
      <div className="page-shell">
        <main
          ref={workspaceShellRef}
          className={`layout-grid workspace-split ${isResizing ? "resizing" : ""}`}
          style={{
            gridTemplateColumns:
              chatPanelWidth === null
                ? "minmax(0, 3fr) 8px minmax(360px, 2fr)"
                : `${chatPanelWidth}px 8px minmax(360px, 1fr)`,
          }}
        >
        <section className="panel chat-panel">
          <div className="panel-title-row">
            <div className="chat-brand">
              <img className="chat-brand-logo" src="/docpilot-logo.png" alt="" aria-hidden="true" />
              <p className="section-kicker">DocPilot</p>
            </div>
            <div className="conversation-actions">
              <div className="conversation-title" title={conversationTitle}>
                {conversationTitle}
              </div>
              <div className="conversation-icon-group">
                <div
                  className="history-control"
                  onMouseEnter={() => setIsHistoryOpen(true)}
                  onMouseLeave={() => setIsHistoryOpen(false)}
                  onFocus={() => setIsHistoryOpen(true)}
                  onBlur={(event) => {
                    if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
                      setIsHistoryOpen(false);
                    }
                  }}
                >
                  <button
                    className={`secondary mini-button icon-button ${isHistoryOpen ? "active" : ""}`}
                    type="button"
                    onClick={() => setIsHistoryOpen(true)}
                    aria-label="Conversation history"
                    aria-expanded={isHistoryOpen}
                    aria-controls="conversation-history"
                    title="Conversation history"
                  >
                    <svg viewBox="0 0 24 24" aria-hidden="true">
                      <circle cx="12" cy="12" r="9" />
                      <path d="M12 7v5l3 2" />
                    </svg>
                  </button>
                  {isHistoryOpen ? (
                    <div
                      id="conversation-history"
                      className="history-popover"
                      role="dialog"
                      aria-label="Conversation history"
                    >
                      <div className="history-popover-title">Conversation history</div>
                      {conversationHistory.length === 0 ? (
                        <div className="history-empty">No saved conversations yet.</div>
                      ) : (
                        <div className="history-list">
                          {conversationHistory.map((item) => (
                            <div
                              className={`history-item ${activeHistoryId === item.id ? "active" : ""}`}
                              key={item.id}
                            >
                              <button
                                className="history-item-main"
                                type="button"
                                onClick={() => restoreConversation(item)}
                              >
                                <span className="history-item-title">{item.title}</span>
                                <span className="history-item-meta">
                                  {new Date(item.updatedAt).toLocaleString()} · {item.messages.length} messages
                                </span>
                              </button>
                              <button
                                className="history-delete-button"
                                type="button"
                                onClick={() => void handleDeleteHistoryItem(item.id)}
                                aria-label={`Delete ${item.title}`}
                                title="Delete conversation"
                              >
                                <svg viewBox="0 0 24 24" aria-hidden="true">
                                  <path d="M4 7h16M9 7V4h6v3m3 0-1 13H7L6 7m4 4v5m4-5v5" />
                                </svg>
                              </button>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ) : null}
                </div>
                <button
                  className="secondary mini-button icon-button"
                  type="button"
                  onClick={handleNewConversation}
                  aria-label="New conversation"
                  title="New conversation"
                >
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M12 20H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h7" />
                    <path d="M17 3v8m-4-4h8" />
                  </svg>
                </button>
                <button
                  className="danger-link mini-button icon-button"
                  type="button"
                  onClick={() => void handleClearConversation()}
                  disabled={messages.length === 0 && !isSending}
                  aria-label="Clear conversation"
                  title="Clear conversation"
                >
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M4 7h16M9 7V4h6v3m3 0-1 13H7L6 7m4 4v5m4-5v5" />
                  </svg>
                </button>
              </div>
            </div>
          </div>

          <div
            className={`message-list ${messages.length === 0 ? "empty" : ""}`}
            ref={messageListRef}
          >
            {messages.length === 0 ? (
              <div className="empty-state empty-state-hero">
                <h3>Start with DocPilot agent</h3>
                <p>
                  Import documents in Workspace, wait for MinerU on PDF files, then add
                  the documents you want the agent to use.
                </p>
              </div>
            ) : null}
            {messages.map((message) => {
              const isLatestAssistant =
                message.role === "assistant" && message.id === latestAssistantMessageId;
              const messageTrace = isLatestAssistant ? lastAgentTrace : (message.agentTrace ?? []);
              const messageState = isLatestAssistant ? lastAgentState : (message.agentState ?? null);
              const messageMode = isLatestAssistant
                ? lastResponseMode
                : (message.responseMode ?? "chat");
              const messageAgentSteps = isLatestAssistant
                ? agentSteps
                : buildAgentStepViews(messageTrace, Boolean(messageState?.done));
              const isProcessMessage =
                message.role === "assistant" &&
                messageMode === "docpilot" &&
                messageAgentSteps.length > 0;
              const isWaitingAssistantMessage =
                isLatestAssistant &&
                isSending &&
                !message.content.trim() &&
                messageAgentSteps.length === 0;

              return (
                <div className="message-entry" key={message.id}>
                  {isProcessMessage && isLatestAssistant ? (
                    <div className="agent-step-navigator-shell">
                      <div
                        className="agent-step-navigator"
                        ref={agentStepNavigatorRef}
                        onScroll={syncAgentStepScrollbar}
                      >
                        {agentSteps.map((step) => {
                          const isActive = activeAgentStepIndex === step.index;
                          const isComplete = step.done && step.success;
                          const isFailed = step.done && !step.success;
                          const isRunning = isSending && isActive && !step.done;
                          return (
                            <button
                              key={step.key}
                              ref={(node) => {
                                if (node) {
                                  agentStepPillRefs.current.set(step.index, node);
                                } else {
                                  agentStepPillRefs.current.delete(step.index);
                                }
                              }}
                              type="button"
                              className={`agent-step-pill ${isActive ? "active" : ""} ${isComplete ? "done" : ""} ${isFailed ? "failed" : ""} ${isRunning ? "running" : ""}`}
                              onClick={() => jumpToAgentStep(step.index)}
                            >
                              <span className="agent-step-pill-index">
                                {isComplete ? "✓" : step.index + 1}
                              </span>
                              <span className="agent-step-pill-label">{step.title}</span>
                            </button>
                          );
                        })}
                      </div>
                      <div
                        className="agent-step-scrollbar"
                        ref={agentStepScrollbarRef}
                        aria-hidden="true"
                        onPointerDown={dragAgentStepScrollbar}
                        hidden
                      >
                        <span ref={agentStepScrollbarThumbRef} />
                      </div>
                    </div>
                  ) : null}
                  <article
                    className={`message-card ${message.role === "user" ? "user" : "assistant"} ${isProcessMessage || isWaitingAssistantMessage ? "process-message-card" : ""}`}
                  >
                  <div className="message-label">
                    <span>{message.role === "user" ? "User" : "Assistant"}</span>
                    {isProcessMessage ? (
                      <span className="message-step-count">Steps: {messageAgentSteps.length}</span>
                    ) : null}
                  </div>
                  {message.role === "assistant" ? (
                    <>
                      {isProcessMessage ? (
                        <div className="process-message-shell">
                          <div className="agent-step-list">
                            {messageAgentSteps.map((step) => {
                              const archivedStepKey = `${message.id}:${step.index}`;
                              const isActive = isLatestAssistant && activeAgentStepIndex === step.index;
                              const isComplete = step.done && step.success;
                              const isFailed = step.done && !step.success;
                              const isRunning = isLatestAssistant && isSending && isActive && !step.done;
                              const isExpanded = isLatestAssistant
                                ? expandedAgentStepIndexes.has(step.index)
                                : expandedArchivedAgentSteps.has(archivedStepKey);
                              const isCopied = copiedAgentStepKey === archivedStepKey;
                              const status = isComplete
                                ? "Completed"
                                : isFailed
                                  ? "Failed"
                                  : isRunning
                                    ? "Processing"
                                    : "Recorded";
                              return (
                                <div
                                  key={step.key}
                                  ref={(node) => {
                                    if (!isLatestAssistant) {
                                      return;
                                    }
                                    if (node) {
                                      agentStepRefs.current.set(step.index, node);
                                    } else {
                                      agentStepRefs.current.delete(step.index);
                                    }
                                  }}
                                  className="agent-step-entry"
                                >
                                  {step.thought ? (
                                    <details className="agent-thinking-panel">
                                      <summary>
                                        <svg className="agent-thinking-icon" viewBox="0 0 24 24" aria-hidden="true">
                                          <path d="M14.7 6.3a4 4 0 0 1-5 5L4 17l3 3 5.7-5.7a4 4 0 0 1 5-5l-3 3-3-3 3-3Z" />
                                        </svg>
                                        <span className="agent-thinking-title">Thinking</span>
                                        <div className="agent-thinking-preview">
                                          {step.thought.replace(/\s+/g, " ").trim()}
                                        </div>
                                        <svg className="agent-thinking-chevron" viewBox="0 0 24 24" aria-hidden="true">
                                          <path d="m7 9 5 5 5-5" />
                                        </svg>
                                      </summary>
                                      <div className="agent-thinking-content markdown-body">
                                        <ReactMarkdown>{step.thought}</ReactMarkdown>
                                      </div>
                                    </details>
                                  ) : null}

                                  <section
                                    className={`agent-step-card ${isActive ? "active" : ""} ${isComplete ? "done" : ""} ${isFailed ? "failed" : ""} ${isRunning ? "running" : ""} ${isExpanded ? "expanded" : "collapsed"}`}
                                  >
                                    <div className="agent-step-card-header">
                                    <button
                                      type="button"
                                      className="agent-step-card-title"
                                      onClick={() => toggleAgentStep(step.index, message.id)}
                                    >
                                      <span className="agent-step-card-index">{step.index + 1}</span>
                                      <span className="agent-step-card-name">{step.title}</span>
                                    </button>
                                    <div className="agent-step-card-meta">
                                      <span
                                        className={`agent-step-card-status ${isComplete ? "completed" : isFailed ? "failed" : isRunning ? "running" : "recorded"}`}
                                        role="img"
                                        aria-label={status}
                                        title={status}
                                      >
                                        {isRunning ? (
                                          <svg viewBox="0 0 24 24" aria-hidden="true">
                                            <path d="M20 12a8 8 0 1 1-2.34-5.66" />
                                          </svg>
                                        ) : isFailed ? (
                                          <svg viewBox="0 0 24 24" aria-hidden="true">
                                            <circle cx="12" cy="12" r="9" />
                                            <path d="m9 9 6 6m0-6-6 6" />
                                          </svg>
                                        ) : isComplete ? (
                                          <svg viewBox="0 0 24 24" aria-hidden="true">
                                            <circle cx="12" cy="12" r="9" />
                                            <path d="m8 12 2.5 2.5L16 9" />
                                          </svg>
                                        ) : (
                                          <svg viewBox="0 0 24 24" aria-hidden="true">
                                            <circle cx="12" cy="12" r="9" />
                                            <path d="M12 7v5l3 2" />
                                          </svg>
                                        )}
                                      </span>
                                      <button
                                        type="button"
                                        className={`agent-step-card-icon-button ${isCopied ? "copied" : ""}`}
                                        onClick={() => void copyAgentStep(step, message.id)}
                                        aria-label={isCopied ? `${step.title} copied` : `Copy ${step.title}`}
                                        title={isCopied ? "Copied" : "Copy"}
                                      >
                                        {isCopied ? (
                                          <>
                                            <svg viewBox="0 0 24 24" aria-hidden="true">
                                              <path d="m6 12 4 4 8-8" />
                                            </svg>
                                            <span className="agent-step-copy-feedback" role="status">
                                              Copied
                                            </span>
                                          </>
                                        ) : (
                                          <svg viewBox="0 0 24 24" aria-hidden="true">
                                            <rect x="9" y="9" width="10" height="10" rx="2" />
                                            <path d="M15 9V7a2 2 0 0 0-2-2H7a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2h2" />
                                          </svg>
                                        )}
                                      </button>
                                      <button
                                        type="button"
                                        className="agent-step-card-icon-button"
                                        onClick={() => toggleAgentStep(step.index, message.id)}
                                        aria-label={`${isExpanded ? "Collapse" : "Expand"} ${step.title}`}
                                        title={isExpanded ? "Collapse" : "Expand"}
                                      >
                                        <svg
                                          className={`agent-step-card-toggle ${isExpanded ? "expanded" : ""}`}
                                          viewBox="0 0 24 24"
                                          aria-hidden="true"
                                        >
                                          <path d="m7 9 5 5 5-5" />
                                        </svg>
                                      </button>
                                    </div>
                                    </div>

                                    <div className="agent-step-card-body">
                                    {step.reasoning ? (
                                      <div className="agent-step-block">
                                        <div className="agent-step-block-label">Thinking</div>
                                        <div className="markdown-body">
                                          <ReactMarkdown>{step.reasoning}</ReactMarkdown>
                                        </div>
                                      </div>
                                    ) : null}

                                    {step.result ? (
                                      <div className="agent-step-block">
                                        <div className="agent-step-block-label">Result</div>
                                        <div className="agent-step-result markdown-body">
                                          <ReactMarkdown>{step.result}</ReactMarkdown>
                                        </div>
                                      </div>
                                    ) : null}

                                    {step.images.length ? (
                                      <div className="agent-step-images">
                                        {step.images.map((item) => {
                                          const imageUrl = item.url.startsWith("/")
                                            ? toAbsoluteUrl(item.url)
                                            : item.url;
                                          return (
                                            <figure className="agent-step-image" key={item.url}>
                                              <a href={imageUrl} target="_blank" rel="noreferrer">
                                                <img src={imageUrl} alt={item.label} loading="lazy" />
                                              </a>
                                              <figcaption>{item.label}</figcaption>
                                            </figure>
                                          );
                                        })}
                                      </div>
                                    ) : null}
                                    </div>
                                  </section>
                                </div>
                              );
                            })}
                            {isLatestAssistant && isPlanningNextAgentStep ? (
                              <div className="agent-thinking-panel pending" role="status" aria-live="polite">
                                <div className="agent-thinking-summary">
                                  <svg className="agent-thinking-icon" viewBox="0 0 24 24" aria-hidden="true">
                                    <path d="M14.7 6.3a4 4 0 0 1-5 5L4 17l3 3 5.7-5.7a4 4 0 0 1 5-5l-3 3-3-3 3-3Z" />
                                  </svg>
                                  <span className="agent-thinking-title">Thinking</span>
                                  <span className="agent-thinking-preview">Planning the next action...</span>
                                  <span className="agent-thinking-dots" aria-hidden="true"><i /><i /><i /></span>
                                </div>
                              </div>
                            ) : null}
                          </div>

                          <div className={`process-answer-block ${message.content.trim() ? "visible" : ""}`}>
                            <div className="agent-step-block-label">Answer</div>
                            <div className="markdown-body">
                              <ReactMarkdown>{message.content}</ReactMarkdown>
                            </div>
                          </div>
                        </div>
                      ) : isWaitingAssistantMessage ? (
                        <div className="agent-thinking-panel pending" role="status" aria-live="polite">
                          <div className="agent-thinking-summary">
                            <svg className="agent-thinking-icon" viewBox="0 0 24 24" aria-hidden="true">
                              <path d="M14.7 6.3a4 4 0 0 1-5 5L4 17l3 3 5.7-5.7a4 4 0 0 1 5-5l-3 3-3-3 3-3Z" />
                            </svg>
                            <span className="agent-thinking-title">Thinking</span>
                            <span className="agent-thinking-preview">
                              {lastResponseMode === "docpilot" ? "Planning the next action..." : "Preparing response..."}
                            </span>
                            <span className="agent-thinking-dots" aria-hidden="true"><i /><i /><i /></span>
                          </div>
                        </div>
                      ) : (
                        <div className="markdown-body">
                          <ReactMarkdown>{message.content}</ReactMarkdown>
                        </div>
                      )}
                    </>
                  ) : (
                    <p className="plain-message">{message.content}</p>
                  )}
                  </article>
                </div>
              );
            })}
            <div ref={messagesEndRef} />
          </div>

          {selectedFiles.length > 0 ? (
            <div className="context-strip">
              {selectedFiles.map((fileName) => (
                <button
                  key={fileName}
                  className="context-chip"
                  onClick={() => toggleSelectedFile(fileName)}
                >
                  {fileName} ×
                </button>
              ))}
            </div>
          ) : null}

          <div
            className={`composer ${isDraggingAttachment ? "attachment-dragging" : ""}`}
            onDragEnter={handleComposerDrag}
            onDragOver={handleComposerDrag}
            onDragLeave={handleComposerDragLeave}
            onDrop={handleComposerDrop}
          >
            {isDraggingAttachment ? (
              <div className="attachment-drop-hint">Drop images or PDFs to attach</div>
            ) : null}
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleComposerKeyDown}
              placeholder="Ask DocPilot a question. Add documents only if you want context."
              rows={4}
            />
            <div className="composer-footer">
              {currentAgent === "chat" ? (
                <>
                  <input
                    ref={chatAttachmentInputRef}
                    type="file"
                    accept=".pdf,image/png,image/jpeg,image/webp,image/gif"
                    multiple
                    hidden
                    onChange={(event) =>
                      void attachChatFiles(Array.from(event.target.files ?? []))
                    }
                  />
                  <button
                    className={`secondary icon-button composer-attachment-button ${isUploadingAttachments ? "active" : ""}`}
                    type="button"
                    onClick={() => chatAttachmentInputRef.current?.click()}
                    disabled={isSending || isUploadingAttachments}
                    aria-label={isUploadingAttachments ? "Uploading attachments" : "Attach images or PDFs"}
                    aria-busy={isUploadingAttachments}
                    title={isUploadingAttachments ? "Uploading..." : "Attach images or PDFs"}
                  >
                    <svg viewBox="0 0 24 24" aria-hidden="true">
                      <path d="m21.4 11.1-9.2 9.2a6 6 0 0 1-8.5-8.5l9.2-9.2a4 4 0 0 1 5.7 5.7l-9.2 9.2a2 2 0 0 1-2.8-2.8l8.5-8.5" />
                    </svg>
                  </button>
                </>
              ) : null}
              <div className="composer-actions">
                <div className="settings-dropdown" ref={settingsDropdownRef}>
                  <button
                    className={`secondary settings-trigger ${isSettingsOpen ? "active" : ""}`}
                    type="button"
                    onClick={() => setIsSettingsOpen((current) => !current)}
                    aria-label="Model settings"
                    aria-expanded={isSettingsOpen}
                    aria-controls="model-settings"
                  >
                    <svg viewBox="0 0 24 24" aria-hidden="true">
                      <path d="M4 7h10M18 7h2M4 17h2M10 17h10" />
                      <circle cx="16" cy="7" r="2" />
                      <circle cx="8" cy="17" r="2" />
                    </svg>
                    <span>{config.model.trim() || "Model settings"}</span>
                    <svg className="settings-chevron" viewBox="0 0 24 24" aria-hidden="true">
                      <path d="m8 10 4 4 4-4" />
                    </svg>
                  </button>

                  {isSettingsOpen ? (
                    <div
                      id="model-settings"
                      className="settings-popover"
                      role="dialog"
                      aria-label="Model settings"
                    >
                      <div className="agent-selector">
                        <div className="agent-selector-label">
                          Current Agent <span>(2)</span>
                        </div>
                        <button
                          className="agent-selector-trigger"
                          type="button"
                          onClick={() => setIsAgentMenuOpen((current) => !current)}
                          aria-expanded={isAgentMenuOpen}
                        >
                          <span className="agent-selector-icon" aria-hidden="true">
                            {currentAgent === "default" ? (
                              <svg viewBox="0 0 24 24">
                                <rect x="5" y="7" width="14" height="11" rx="3" />
                                <path d="M12 3v4M9 12h.01M15 12h.01M9 15h6" />
                              </svg>
                            ) : (
                              <svg viewBox="0 0 24 24">
                                <path d="M5 5h14a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H9l-5 4v-4H5a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2Z" />
                              </svg>
                            )}
                          </span>
                          <span className="agent-selector-copy">
                            <strong>{currentAgent === "default" ? "Default" : "Chat"}</strong>
                            <small>
                              {currentAgent === "default" ? "DocPilot pipeline" : "Direct chat run"}
                            </small>
                          </span>
                          <svg className="agent-selector-chevron" viewBox="0 0 24 24" aria-hidden="true">
                            <path d="m8 10 4 4 4-4" />
                          </svg>
                        </button>

                        {isAgentMenuOpen ? (
                          <div className="agent-menu">
                            <button
                              className={`agent-option ${currentAgent === "default" ? "active" : ""}`}
                              type="button"
                              onClick={() => {
                                setCurrentAgent("default");
                                setIsAgentMenuOpen(false);
                              }}
                            >
                              <span className="agent-option-icon" aria-hidden="true">D</span>
                              <span>
                                <strong>Default</strong>
                                <small>Run the DocPilot document pipeline</small>
                              </span>
                            </button>
                            <button
                              className={`agent-option ${currentAgent === "chat" ? "active" : ""}`}
                              type="button"
                              onClick={() => {
                                setCurrentAgent("chat");
                                setIsAgentMenuOpen(false);
                              }}
                            >
                              <span className="agent-option-icon" aria-hidden="true">C</span>
                              <span>
                                <strong>Chat</strong>
                                <small>Call the model directly without the pipeline</small>
                              </span>
                            </button>
                          </div>
                        ) : null}
                      </div>

                      <div className="stack">
                        <div className="settings-field-heading">
                          <span>Base URL</span>
                          {currentAgent === "default" ? (
                            <div className="provider-selector" role="group" aria-label="Model provider">
                              <button
                                type="button"
                                className={config.provider === "vllm" ? "active" : ""}
                                onClick={() => loadPreset("vllm")}
                                title="DocPilot models trained for the vLLM Tags protocol"
                              >
                                vLLM Tags
                              </button>
                              <button
                                type="button"
                                className={config.provider === "custom" ? "active" : ""}
                                onClick={() => loadPreset("custom")}
                                title="Standard OpenAI-compatible APIs, including untrained vLLM models"
                              >
                                Custom
                              </button>
                            </div>
                          ) : null}
                        </div>
                        <input
                          aria-label="Base URL"
                          value={config.base_url}
                          onChange={(event) => {
                            invalidateDiscoveredModels();
                            setConfig((current) => ({ ...current, base_url: event.target.value }));
                          }}
                          placeholder="http://127.0.0.1:8000/v1"
                        />
                      </div>

                      <label className="stack">
                        <span>API Key</span>
                        <input
                          value={config.api_key}
                          onChange={(event) => {
                            invalidateDiscoveredModels();
                            setConfig((current) => ({ ...current, api_key: event.target.value }));
                          }}
                          placeholder="docpilot-local"
                          type="password"
                        />
                      </label>

                      <div className="model-picker" ref={modelPickerRef}>
                        <div className="settings-field-heading">
                          <span>Model</span>
                          <div
                            className={`reasoning-setting ${usesVllmTags ? "disabled" : ""}`}
                            title={usesVllmTags ? "vLLM Tags uses <Analyze> reasoning" : undefined}
                          >
                            <span>Reasoning</span>
                            <button
                              className={`reasoning-toggle ${config.reasoning && !usesVllmTags ? "active" : ""}`}
                              type="button"
                              role="switch"
                              aria-label="Reasoning"
                              aria-checked={usesVllmTags ? false : config.reasoning}
                              disabled={usesVllmTags}
                              onClick={() =>
                                setConfig((current) => ({
                                  ...current,
                                  reasoning: !current.reasoning,
                                }))
                              }
                            >
                              <span className="reasoning-toggle-copy">
                                {usesVllmTags ? "Analyze" : config.reasoning ? "On" : "Off"}
                              </span>
                              <span className="reasoning-toggle-track" aria-hidden="true">
                                <span />
                              </span>
                            </button>
                          </div>
                        </div>
                        <div className="inline-row">
                          <input
                            className="grow"
                            aria-label="Model"
                            value={config.model}
                            onChange={(event) =>
                              setConfig((current) => ({ ...current, model: event.target.value }))
                            }
                            placeholder="Qwen/Qwen2.5-7B-Instruct"
                          />
                          <button
                            className="secondary"
                            type="button"
                            onClick={() => void handleDiscoverModels()}
                          >
                            {isModelMenuOpen ? "Hide" : "Discover"}
                          </button>
                        </div>

                        {isModelMenuOpen && availableModels.length > 0 ? (
                          <div className="model-dropdown">
                            {availableModels.map((modelName) => (
                              <button
                                type="button"
                                key={modelName}
                                className={`model-option ${config.model === modelName ? "active" : ""}`}
                                onClick={() => {
                                  setConfig((current) => ({ ...current, model: modelName }));
                                  setIsModelMenuOpen(false);
                                }}
                              >
                                {modelName}
                              </button>
                            ))}
                          </div>
                        ) : null}
                      </div>

                      {errorText ? <div className="error-box">{errorText}</div> : null}
                    </div>
                  ) : null}
                </div>
                <button
                  className="send-button"
                  type="button"
                  onClick={isSending ? handleStopSending : () => void handleSend()}
                  disabled={isUploadingAttachments}
                >
                  {isSending ? "Stop" : "Send"}
                </button>
              </div>
            </div>
          </div>
        </section>

        <div
          className="panel-divider"
          onMouseDown={() => setIsResizing(true)}
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize conversation and workspace panels"
        />

        <section className="panel workspace-panel">
          <div className="panel-title-row">
            <div>
              <p className="section-kicker">Workspace</p>
            </div>
            <div className="workspace-header-actions">
              <button
                className={`secondary mini-button icon-button workspace-evidence-button ${activeEvidence ? "active" : ""}`}
                type="button"
                onClick={toggleEvidence}
                disabled={answerEvidence.length === 0}
                aria-label={activeEvidence ? "Hide answer evidence" : "Show answer evidence"}
                aria-pressed={Boolean(activeEvidence)}
                title={
                  answerEvidence.length > 0
                    ? activeEvidence
                      ? "Hide answer evidence"
                      : `Show ${answerEvidence.length} answer evidence item(s)`
                    : "No evidence for the latest answer"
                }
              >
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <circle cx="12" cy="12" r="7" />
                  <circle cx="12" cy="12" r="2" />
                  <path d="M12 2v3M12 19v3M2 12h3M19 12h3" />
                </svg>
                {answerEvidence.length > 0 ? (
                  <span className="workspace-evidence-count">{answerEvidence.length}</span>
                ) : null}
              </button>
              <div className="workspace-import-control" ref={workspaceImportRef}>
                <button
                  className={`secondary mini-button ${isImportOpen ? "active" : ""}`}
                  type="button"
                  onClick={() => setIsImportOpen((current) => !current)}
                  aria-expanded={isImportOpen}
                >
                  Import
                </button>
                {isImportOpen ? <div className="workspace-import-popover">
                  <div className="upload-mode-tabs">
                    <button
                      className={uploadMode === "local" ? "active" : ""}
                      onClick={() => setUploadMode("local")}
                      type="button"
                    >
                      Local
                    </button>
                    <button
                      className={uploadMode === "server" ? "active" : ""}
                      onClick={() => setUploadMode("server")}
                      type="button"
                    >
                      Server
                    </button>
                  </div>
                  {uploadMode === "local" ? (
                    <input type="file" multiple onChange={handleUpload} />
                  ) : (
                    <div className="server-path-inline">
                      <textarea
                        rows={3}
                        value={serverPathInput}
                        onChange={(event) => setServerPathInput(event.target.value)}
                        placeholder={"/data/path/sample.pdf\n/data/path/example.md"}
                      />
                      <button type="button" className="secondary" onClick={() => void handleServerPathImport()}>
                        Import
                      </button>
                    </div>
                  )}
                </div> : null}
              </div>
              <button
                className="secondary mini-button icon-button"
                onClick={() => void refreshFiles()}
                disabled={isLoadingFiles}
                aria-label="Refresh workspace"
                aria-busy={isLoadingFiles}
                title="Refresh workspace"
              >
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M20 11a8 8 0 1 0-2.3 5.7" />
                  <path d="M20 5v6h-6" />
                </svg>
              </button>
            </div>
          </div>

          <div className="workspace-doc-controls">
            <div className="workspace-file-toolbar">
              <select
                className="workspace-file-select"
                value={selectedWorkspaceFile?.name ?? ""}
                onChange={(event) => {
                  const nextFile = workspaceFiles.find((file) => file.name === event.target.value);
                  if (nextFile) {
                    void withViewportLock(() => openViewer(nextFile));
                  }
                }}
              >
                {workspaceFiles.length === 0 ? <option value="">No files</option> : null}
                {workspaceFiles.map((file) => (
                  <option key={file.name} value={file.name}>
                    {file.name}
                  </option>
                ))}
              </select>
              <span
                className={`workspace-file-meta status-${selectedWorkspaceFile?.processing_status ?? "none"} ${
                  selectedWorkspaceFile?.processing_status === "failed" && currentAgent === "default"
                    ? "retryable"
                    : ""
                }`}
                role={selectedWorkspaceFile?.processing_status === "failed" && currentAgent === "default" ? "button" : undefined}
                tabIndex={selectedWorkspaceFile?.processing_status === "failed" && currentAgent === "default" ? 0 : undefined}
                title={selectedWorkspaceFile?.processing_status === "failed" && currentAgent === "default" ? "Click to retry MinerU" : undefined}
                aria-label={selectedWorkspaceFile?.processing_status === "failed" && currentAgent === "default" ? `Retry MinerU for ${selectedWorkspaceFile.name}` : undefined}
                onClick={() => {
                  if (selectedWorkspaceFile?.processing_status === "failed" && currentAgent === "default") {
                    void handleRetry(selectedWorkspaceFile.name);
                  }
                }}
                onKeyDown={(event) => {
                  if (
                    selectedWorkspaceFile?.processing_status === "failed" &&
                    currentAgent === "default" &&
                    (event.key === "Enter" || event.key === " ")
                  ) {
                    event.preventDefault();
                    void handleRetry(selectedWorkspaceFile.name);
                  }
                }}
              >
                {selectedWorkspaceFile
                  ? `${formatBytes(selectedWorkspaceFile.size)} · ${statusLabel(selectedWorkspaceFile)}`
                  : "No file"}
              </span>
              <button
                className={`mini-button ${
                  selectedWorkspaceFile && selectedFiles.includes(selectedWorkspaceFile.name)
                    ? "danger-link"
                    : "secondary"
                }`}
                onClick={() => {
                  if (selectedWorkspaceFile) {
                    toggleSelectedFile(selectedWorkspaceFile.name);
                  }
                }}
                disabled={!selectedWorkspaceFile}
              >
                {selectedWorkspaceFile && selectedFiles.includes(selectedWorkspaceFile.name)
                  ? "Remove"
                  : "Add"}
              </button>
              <button
                className="danger-link mini-button"
                onClick={() => {
                  if (selectedWorkspaceFile) {
                    void handleDelete(selectedWorkspaceFile.name);
                  }
                }}
                disabled={!selectedWorkspaceFile}
              >
                Delete
              </button>
            </div>
          </div>

          <div className="viewer-toolbar">
            <div className="viewer-search-row">
              <input
                value={viewerQuery}
                onChange={(event) => setViewerQuery(event.target.value)}
                onKeyDown={handleViewerSearchKeyDown}
                placeholder="Search in current document"
              />
              <button
                className="secondary viewer-icon-button"
                onClick={() => void withViewportLock(commitViewerSearch)}
                aria-label="Search document"
                title="Search document"
              >
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <circle cx="11" cy="11" r="6" />
                  <path d="m16 16 4 4" />
                </svg>
              </button>
              <div className="viewer-search-status">
                {isSearchingDocument
                  ? "Searching..."
                  : searchResultCount > 0
                    ? `${activeSearchIndex + 1} / ${searchResultCount}`
                    : activeViewerQuery
                      ? "0 results"
                      : ""}
              </div>
              <button
                className="secondary viewer-icon-button"
                onClick={goToPreviousSearchResult}
                disabled={searchResultCount <= 0}
                aria-label="Previous search result"
                title="Previous search result"
              >
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <path d="m7 14 5-5 5 5" />
                </svg>
              </button>
              <button
                className="secondary viewer-icon-button"
                onClick={goToNextSearchResult}
                disabled={searchResultCount <= 0}
                aria-label="Next search result"
                title="Next search result"
              >
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <path d="m7 10 5 5 5-5" />
                </svg>
              </button>
            </div>
            <div className="viewer-control-row">
              <div className="viewer-pagination" aria-label="PDF pagination controls">
                <button
                  className="secondary viewer-icon-button"
                  onClick={() =>
                    void withViewportLock(() =>
                      setPdfPageNumber((current) => Math.max(current - 1, 1)),
                    )
                  }
                  disabled={viewer.mode !== "pdf" || pdfPageNumber <= 1}
                  aria-label="Previous page"
                  title="Previous page"
                >
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="m14 7-5 5 5 5" />
                  </svg>
                </button>
                <span className="viewer-page-jump">
                  <input
                    type="number"
                    className="viewer-page-input"
                    min={1}
                    max={viewer.mode === "pdf" ? pdfPageCount || 1 : 1}
                    value={pdfPageInput}
                    disabled={viewer.mode !== "pdf"}
                    onFocus={() => setIsPdfPageInputFocused(true)}
                    onBlur={() => {
                      setIsPdfPageInputFocused(false);
                      commitPdfPageInput();
                    }}
                    onChange={(event) => setPdfPageInput(event.target.value)}
                    onKeyDown={handlePdfPageInputKeyDown}
                    aria-label="Current page"
                  />
                  {" / "}
                  {viewer.mode === "pdf" ? pdfPageCount || 1 : 1}
                </span>
                <button
                  className="secondary viewer-icon-button"
                  onClick={() =>
                    void withViewportLock(() =>
                      setPdfPageNumber((current) =>
                        Math.min(current + 1, pdfPageCount || current + 1),
                      ),
                    )
                  }
                  disabled={
                    viewer.mode !== "pdf" || pdfPageCount === 0 || pdfPageNumber >= pdfPageCount
                  }
                  aria-label="Next page"
                  title="Next page"
                >
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="m10 7 5 5-5 5" />
                  </svg>
                </button>
              </div>
              <div className="viewer-zoom-controls" aria-label="Document zoom controls">
                <button
                  className="secondary viewer-icon-button"
                  onClick={zoomOut}
                  aria-label="Zoom out"
                  title="Zoom out"
                >
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M6 12h12" />
                  </svg>
                </button>
                <span className="zoom-readout">{Math.round(viewerZoom * 100)}%</span>
                <button
                  className="secondary viewer-icon-button"
                  onClick={zoomIn}
                  aria-label="Zoom in"
                  title="Zoom in"
                >
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M12 6v12M6 12h12" />
                  </svg>
                </button>
                <button
                  className="secondary viewer-icon-button"
                  onClick={resetZoom}
                  aria-label="Reset zoom"
                  title="Reset zoom"
                >
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M20 11a8 8 0 1 0-2.3 5.7" />
                    <path d="M20 5v6h-6" />
                  </svg>
                </button>
              </div>
            </div>
          </div>

          <div
            className={`viewer-surface ${viewer.mode === "pdf" ? "pdf-active" : ""} ${viewer.mode === "image" ? "image-active" : ""}`}
            ref={viewerBodyRef}
          >
            {activeEvidence ? (
              <div className="workspace-evidence-nav" role="status" aria-live="polite">
                <button
                  type="button"
                  onClick={() => void withViewportLock(() => showEvidence(activeEvidenceIndex - 1))}
                  aria-label="Previous evidence"
                  title="Previous evidence"
                >
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="m14 7-5 5 5 5" />
                  </svg>
                </button>
                <span>
                  Evidence {activeEvidenceIndex + 1}/{answerEvidence.length}
                </span>
                <button
                  type="button"
                  onClick={() => void withViewportLock(() => showEvidence(activeEvidenceIndex + 1))}
                  aria-label="Next evidence"
                  title="Next evidence"
                >
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="m10 7 5 5-5 5" />
                  </svg>
                </button>
                <button
                  type="button"
                  onClick={() => setActiveEvidenceIndex(-1)}
                  aria-label="Hide evidence"
                  title="Hide evidence"
                >
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="m7 7 10 10M17 7 7 17" />
                  </svg>
                </button>
              </div>
            ) : null}
            {isViewerLoading ? <div className="empty-state">Loading document...</div> : null}

            {!isViewerLoading && viewer.mode === "empty" ? (
              <div className="empty-state">
                <h3>Workspace is ready</h3>
                <p>Import a document, then select it above to inspect it here.</p>
              </div>
            ) : null}

            {!isViewerLoading && viewer.mode === "pdf" ? (
              <div className="pdf-viewer-shell" ref={pdfViewportRef}>
                <Document
                  file={viewer.sourceUrl}
                  onLoadSuccess={({ numPages }) => {
                    const targetPage = Math.min(pdfPageNumber, numPages);
                    setPdfPageCount(numPages);
                    setPdfPageNumber(targetPage);
                    setPdfDisplayedPageNumber(targetPage);
                    setPdfPendingPageNumber(null);
                  }}
                >
                  <div className={`pdf-page-stage ${pdfRenderWidth === undefined ? "sizing" : ""}`}>
                    <Page
                      key={`pdf-page-${pdfDisplayedPageNumber}`}
                      pageNumber={pdfDisplayedPageNumber}
                      width={pdfRenderWidth}
                      renderAnnotationLayer={false}
                      renderTextLayer={false}
                      loading=""
                      onLoadSuccess={(page) => {
                        const viewport = page.getViewport({ scale: 1 });
                        setPdfPageBaseSize({
                          width: viewport.width,
                          height: viewport.height,
                        });
                      }}
                    >
                      <div className="pdf-search-overlay" aria-hidden="true">
                        {pdfSearchMatches.map((match, index) =>
                          match.pageNumber === pdfDisplayedPageNumber ? (
                            <span
                              className={`pdf-search-box ${index === activeSearchIndex ? "active" : ""}`}
                              data-search-index={index}
                              key={`${match.pageNumber}-${match.occurrenceInPage}`}
                              style={{
                                left: `${match.left}%`,
                                top: `${match.top}%`,
                                width: `${match.width}%`,
                                height: `${match.height}%`,
                                transform: `rotate(${match.rotation}rad)`,
                              }}
                            />
                          ) : null,
                        )}
                      </div>
                      {activeEvidence ? (
                        <div className="pdf-evidence-overlay" aria-hidden="true">
                          {answerEvidence.map((evidence, index) =>
                            shouldRenderEvidence(index, pdfDisplayedPageNumber) ? (
                              <span
                                className={`pdf-evidence-box ${!evidence.box ? "page-level" : ""} ${index === activeEvidenceIndex ? "active" : ""}`}
                                data-evidence-index={index}
                                key={evidence.id}
                                style={evidence.box ? {
                                  left: `${evidence.box.left}%`,
                                  top: `${evidence.box.top}%`,
                                  width: `${evidence.box.width}%`,
                                  height: `${evidence.box.height}%`,
                                } : { inset: 0 }}
                              >
                                <span>E{index + 1}</span>
                              </span>
                            ) : null,
                          )}
                        </div>
                      ) : null}
                    </Page>
                    {pdfPendingPageNumber && pdfPendingPageNumber !== pdfDisplayedPageNumber ? (
                      <div className="pdf-page-buffer" aria-hidden="true">
                        <Page
                          key={`pdf-buffer-${pdfPendingPageNumber}`}
                          pageNumber={pdfPendingPageNumber}
                          width={pdfRenderWidth}
                          renderAnnotationLayer={false}
                          renderTextLayer={false}
                          loading=""
                          onLoadSuccess={(page) => {
                            const viewport = page.getViewport({ scale: 1 });
                            setPdfPageBaseSize({
                              width: viewport.width,
                              height: viewport.height,
                            });
                            setPdfDisplayedPageNumber(pdfPendingPageNumber);
                            setPdfPendingPageNumber(null);
                          }}
                        >
                          <div className="pdf-search-overlay">
                            {pdfSearchMatches.map((match, index) =>
                              match.pageNumber === pdfPendingPageNumber ? (
                                <span
                                  className={`pdf-search-box ${index === activeSearchIndex ? "active" : ""}`}
                                  data-search-index={index}
                                  key={`${match.pageNumber}-${match.occurrenceInPage}`}
                                  style={{
                                    left: `${match.left}%`,
                                    top: `${match.top}%`,
                                    width: `${match.width}%`,
                                    height: `${match.height}%`,
                                    transform: `rotate(${match.rotation}rad)`,
                                  }}
                                />
                              ) : null,
                            )}
                          </div>
                          {activeEvidence ? (
                            <div className="pdf-evidence-overlay" aria-hidden="true">
                              {answerEvidence.map((evidence, index) =>
                                shouldRenderEvidence(index, pdfPendingPageNumber) ? (
                                  <span
                                    className={`pdf-evidence-box ${!evidence.box ? "page-level" : ""} ${index === activeEvidenceIndex ? "active" : ""}`}
                                    data-evidence-index={index}
                                    key={evidence.id}
                                    style={evidence.box ? {
                                      left: `${evidence.box.left}%`,
                                      top: `${evidence.box.top}%`,
                                      width: `${evidence.box.width}%`,
                                      height: `${evidence.box.height}%`,
                                    } : { inset: 0 }}
                                  >
                                    <span>E{index + 1}</span>
                                  </span>
                                ) : null,
                              )}
                            </div>
                          ) : null}
                        </Page>
                      </div>
                    ) : null}
                  </div>
                </Document>
              </div>
            ) : null}

            {!isViewerLoading && viewer.mode === "image" ? (
              <div className="image-viewer-shell">
                <img
                  src={viewer.sourceUrl}
                  alt={viewer.title}
                  style={{ width: `${viewerZoom * 100}%` }}
                />
              </div>
            ) : null}

            {!isViewerLoading &&
            (viewer.mode === "docx" || viewer.mode === "markdown" || viewer.mode === "text") ? (
              <div
                className="rich-viewer"
                style={{ fontSize: `${viewerZoom}em` }}
                dangerouslySetInnerHTML={{ __html: highlightedHtml }}
              />
            ) : null}

            {!isViewerLoading && viewer.mode === "unsupported" ? (
              <div className="empty-state">
                <h3>Preview not available</h3>
                <p>{viewer.textContent}</p>
                {viewer.sourceUrl ? (
                  <a href={viewer.sourceUrl} target="_blank" rel="noreferrer">
                    Open original file
                  </a>
                ) : null}
              </div>
            ) : null}
          </div>
        </section>
        </main>
      </div>
    </div>
  );
}
