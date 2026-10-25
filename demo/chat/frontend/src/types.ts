export type ProviderType = "vllm" | "custom";
export type AgentType = "default" | "chat";

export type ModelConfig = {
  provider: ProviderType;
  base_url: string;
  api_key: string;
  model: string;
  reasoning: boolean;
  timeout: number;
};

export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  agentTrace?: Array<Record<string, unknown>>;
  agentState?: Record<string, unknown> | null;
  responseMode?: "chat" | "docpilot";
};

export type WorkspaceFile = {
  name: string;
  size: number;
  modified_at: string;
  content_type: string;
  download_url: string;
  extension: string;
  processing_status:
    | "queued"
    | "processing"
    | "ready"
    | "ready_image"
    | "ready_text"
    | "skipped"
    | "failed";
  processing_error?: string | null;
  doc_id: string;
  mineru_output_dir?: string | null;
  artifacts?: Record<string, unknown>;
  viewer_url?: string | null;
  viewer_type?: string | null;
};

export type OcrLayout = {
  file_name: string;
  page: number;
  blocks: Array<{ type: string; bbox: number[] }>;
};

export type ParsedOutputFormat = "markdown" | "json";

export type ParsedOutput = {
  content: string;
  image_base_url: string;
};

export type ChatEvent =
  | { type: "meta"; model: string; mode?: "chat" | "docpilot" }
  | {
      type: "step";
      step_index?: number;
      action?: Record<string, unknown>;
      observation?: Record<string, unknown>;
      state?: Record<string, unknown>;
      history?: Array<Record<string, unknown>>;
    }
  | { type: "delta"; content: string; state?: Record<string, unknown>; history?: Array<Record<string, unknown>> }
  | { type: "done" }
  | { type: "error"; message: string };
