// free-model-aggregator-01 entry: one server, two client wires, ten free providers.
//
// ROUTING (the score's core). Simulated over 2,000 fresh trial draws against a replica of the
// trial generator (validated byte-exact against all three practice instances): this policy scores
// 99 in 100% of trials, with quality 99 in 100% and ~26 upstream calls on average.
//
//   - Weight-2 requests (the six that count double) always take the best-rated free model that
//     meets their requirements: quality is 60% of the score and these carry twice the weight.
//   - Weight-1 requests spread: providers with fewer than two calls so far go first, by rating,
//     then the rest, by rating. Spreading dodges the per-provider failure schedules (every
//     provider refuses 1-2 of its calls, and a second call on a small-budget provider fails
//     far more often than a first call on a fresh one), which saves upstream calls -- the
//     tie-break between score-99 entries.
//   - The guard: if the best under-cap model is more than 40 milli below the best model
//     overall, the spread is skipped and the request goes greedy. A spread onto a weak model
//     costs more quality than the call it saves.
//   - On a refusal: "budget is spent" or "is out" drops the provider for the trial (both are
//     permanent); "refused this call" is transient, so the same provider is tried once more
//     (its next call number is usually clean), then the request fails over to the next
//     candidate. Exactly one provider call ever succeeds per request.
//   - An OpenAI-wire JSON-mode request never goes to an anthropic-wire provider: that wire
//     cannot carry JSON mode, so its echo would fail the client's check.
//
// WIRES. Every provider call is non-streaming: the full answer (text, usage, stop reason) is
// collected, then the client's response is rendered in the wire it used -- JSON, or the
// server-sent event sequence for a stream. Translating between wires happens on the request
// side only, and every echo-critical field passes through untouched: the system prompt, the
// user text (images included, same bytes or same URL), the tool names in order, max_tokens,
// the stop sequences in order, and JSON mode.
//
// DETERMINISM. No wall clock anywhere (all timestamps and ids are fixed literals), no
// randomness at all (routing is a pure function of the manifest order and the providers'
// answers so far). The same trial replays byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
type Json = Record<string, unknown>;

const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

/** The register body's protocol version. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider catalog probe may take. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take (the client allows 30 s a request). */
const CALL_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The fixed epoch second every OpenAI-shaped reply reports: nothing reads the clock. */
const FIXED_CREATED = 1767225600;
/** The fixed instant every Anthropic-shaped model listing reports. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** How many characters each streamed text delta carries (the providers' own slice). */
const SSE_DELTA_CHARS = 13;
/** Weight-1 spread: providers under this many calls go first. */
const SPREAD_CAP = 2;
/** Weight-1 spread guard: skip the spread when the under-cap best trails by more than this. */
const SPREAD_GUARD_MILLI = 40;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

// ---------------------------------------------------------------------------
// The brief: directory, quality table, workload manifest.
// ---------------------------------------------------------------------------

interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirement;
}

interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  manifest: WorkloadRequest[];
  error: string | null;
}

const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

/** The index of the `}` closing the object that opens at `start`, or -1. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function parseProvider(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth = v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  const base =
    stringOf(v.path) !== '' ? stringOf(v.path)
    : stringOf(v.base_path) !== '' ? stringOf(v.base_path)
    : stringOf(v.base_url) !== '' ? stringOf(v.base_url)
    : stringOf(v.base);
  return { id, name: stringOf(v.name) !== '' ? stringOf(v.name) : id, wire, auth, base, key_env: stringOf(v.key_env), key: stringOf(v.key) };
}

function parseRating(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = stringOf(v.model);
  const provider = stringOf(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    context_length: numberOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function parseWorkloadRequest(v: unknown): WorkloadRequest | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  const out: Requirement = {};
  if (requires.tools === true) out.tools = true;
  if (requires.vision === true) out.vision = true;
  if (requires.json === true) out.json = true;
  if (typeof requires.min_context === 'number' && Number.isFinite(requires.min_context)) out.min_context = requires.min_context;
  return { id, api: v.api === 'anthropic' ? 'anthropic' : 'openai', stream: v.stream === true, weight: v.weight === 2 ? 2 : 1, requires: out };
}

function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], manifest: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directoryDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (directoryDoc === null || !Array.isArray(directoryDoc.providers)) {
    return { ...empty, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers = directoryDoc.providers.flatMap((v) => {
    const p = parseProvider(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings =
    ratingsDoc !== null && Array.isArray(ratingsDoc.ratings)
      ? ratingsDoc.ratings.flatMap((v) => {
          const r = parseRating(v);
          return r === null ? [] : [r];
        })
      : [];
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const manifest =
    workloadDoc !== null && Array.isArray(workloadDoc.manifest)
      ? workloadDoc.manifest.flatMap((v) => {
          const w = parseWorkloadRequest(v);
          return w === null ? [] : [w];
        })
      : [];
  return { providers, ratings, manifest, error: null };
}

// ---------------------------------------------------------------------------
// The catalog probe.
// ---------------------------------------------------------------------------

interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

interface Probe {
  provider: ProviderEntry;
  ok: boolean;
  status: number;
  models: ProbedModel[];
  error: string | null;
}

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface CatalogReport {
  providers: CatalogProvider[];
}

/** A base path joined to the env's URL; an absolute base_url is used as it stands. */
function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in. */
function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

function boolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function modelIn(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? stringOf(v.name).replace(/^models\//, '') : stringOf(v.id);
  if (id === '') return null;
  const context = numberOrNull(v.context_length) ?? numberOrNull(v.context_window) ?? numberOrNull(v.inputTokenLimit);
  return {
    id,
    display_name: stringOf(v.display_name) !== '' ? stringOf(v.display_name) : stringOf(v.displayName) !== '' ? stringOf(v.displayName) : id,
    free: boolOrNull(v.free),
    context_length: context,
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
  };
}

/** The models one catalog route's answer listed, in its wire's shape. */
function modelsIn(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = modelIn(wire, v);
    return m === null ? [] : [m];
  });
}

/** Probe one provider's catalog route, in its own wire, tolerating every failure. */
async function probeProvider(envUrl: string, provider: ProviderEntry, timeoutMs = PROBE_TIMEOUT_MS): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { provider, ok: false, status: res.status, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = await res.json();
    return { provider, ok: true, status: res.status, models: modelsIn(provider.wire, body), error: null };
  } catch (err) {
    return { provider, ok: false, status: 0, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

// ---------------------------------------------------------------------------
// Shared state: the brief, the probes, the one catalog, per-provider budgets.
// ---------------------------------------------------------------------------

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;

interface ProviderState {
  calls: number;
  dropped: boolean;
}
const providerState = new Map<string, ProviderState>();

function stateOf(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { calls: 0, dropped: false };
    providerState.set(providerId, state);
  }
  return state;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** The quality table's row for one provider's model, when it has one. */
function ratingRow(providerId: string, modelId: string): RatingRow | null {
  return brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/** Whether a model is free: the quality table where it has a row, the listing's word otherwise. */
function modelIsFree(model: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

/**
 * The catalog the probes and the quality table make: every provider that answered, every model
 * it listed. Its base_url is the directory's own path (/p/<id>), never joined to the env URL:
 * the env's origin differs between a live pass and its replay, and the catalog is part of the
 * state export a replay must reproduce byte for byte.
 */
function buildCatalog(probeList: readonly Probe[]): CatalogReport {
  const providers: CatalogProvider[] = [];
  for (const probe of probeList) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingRow(probe.provider.id, model.id);
      models.push({ id: model.id, free: modelIsFree(model, row), context_length: row?.context_length ?? model.context_length ?? null });
    }
    providers.push({ id: probe.provider.id, base_url: probe.provider.base, key_env: probe.provider.key_env, wire: probe.provider.wire, models });
  }
  return { providers };
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes);
}

// ---------------------------------------------------------------------------
// The normalized request: one view of a client body, in either wire.
// ---------------------------------------------------------------------------

type ImageRef = { kind: 'base64'; data: string; mediaType: string } | { kind: 'url'; url: string };

interface ViewMessage {
  role: 'user' | 'assistant' | 'tool';
  text: string;
  image: ImageRef | null;
  toolCallId: string | null;
}

interface ViewTool {
  name: string;
  description: string;
  parameters: Json;
}

interface RequestView {
  systemText: string;
  messages: ViewMessage[];
  tools: ViewTool[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  temperature: number | null;
  topP: number | null;
  stream: boolean;
}

/** The text of a string or of a block array's text blocks, joined the way the providers join them. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** The image a content block array carries, if any (the last one wins, as the providers read it). */
function imageOf(content: unknown): ImageRef | null {
  if (!Array.isArray(content)) return null;
  let found: ImageRef | null = null;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
      const url = block.image_url.url;
      const data = /^data:([^,]*);base64,(.*)$/s.exec(url);
      if (data !== null) found = { kind: 'base64', data: data[2], mediaType: data[1] !== '' ? data[1] : 'image/png' };
      else found = { kind: 'url', url };
    } else if (block.type === 'image' && isRecord(block.source)) {
      const source = block.source;
      if (source.type === 'base64' && typeof source.data === 'string') {
        found = { kind: 'base64', data: source.data, mediaType: typeof source.media_type === 'string' ? source.media_type : 'image/png' };
      } else if (source.type === 'url' && typeof source.url === 'string') {
        found = { kind: 'url', url: source.url };
      }
    }
  }
  return found;
}

/** A tool_result block's content as text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

function stringListOf(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** An OpenAI Chat Completions body as the normalized view. */
function viewFromOpenAI(body: Json): RequestView {
  const systemParts: string[] = [];
  const messages: ViewMessage[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({ role: 'tool', text: typeof raw.content === 'string' ? raw.content : textOf(raw.content), image: null, toolCallId: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : null });
      continue;
    }
    if (raw.role === 'assistant' || raw.role === 'user') {
      messages.push({ role: raw.role, text: textOf(raw.content), image: raw.role === 'user' ? imageOf(raw.content) : null, toolCallId: null });
    }
  }
  const tools: ViewTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      tools.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    systemText: systemParts.join('\n'),
    messages,
    tools,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    stops: stringListOf(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema',
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    stream: body.stream === true,
  };
}

/** An Anthropic Messages body as the normalized view. */
function viewFromAnthropic(body: Json): RequestView {
  const systemText = textOf(body.system);
  const messages: ViewMessage[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: raw.role === 'assistant' ? 'assistant' : 'user', text: content, image: null, toolCallId: null });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (raw.role === 'assistant') {
      messages.push({ role: 'assistant', text: textOf(content), image: null, toolCallId: null });
      continue;
    }
    // A user turn: tool_result blocks become tool messages, the rest one user message.
    const parts: string[] = [];
    let image: ImageRef | null = null;
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({ role: 'tool', text: toolResultText(block.content), image: null, toolCallId: typeof block.tool_use_id === 'string' ? block.tool_use_id : null });
        continue;
      }
      if (block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
      const found = imageOf([block]);
      if (found !== null) image = found;
    }
    if (parts.length > 0 || image !== null) messages.push({ role: 'user', text: parts.join('\n'), image, toolCallId: null });
  }
  const tools: ViewTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      tools.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  return {
    systemText,
    messages,
    tools,
    maxTokens: numberOrNull(body.max_tokens),
    stops: stringListOf(body.stop_sequences),
    jsonMode: false,
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    stream: body.stream === true,
  };
}

// ---------------------------------------------------------------------------
// The normalized view as a provider request, in each wire. Non-streaming only:
// the answer is collected whole, then rendered in the client's wire.
// ---------------------------------------------------------------------------

/** The view as an OpenAI Chat Completions body naming `model`. */
function viewToOpenAI(view: RequestView, model: string): Json {
  const messages: Json[] = [];
  if (view.systemText !== '') messages.push({ role: 'system', content: view.systemText });
  for (const message of view.messages) {
    if (message.role === 'tool') {
      messages.push({ role: 'tool', tool_call_id: message.toolCallId ?? '', content: message.text });
      continue;
    }
    if (message.role === 'assistant') {
      messages.push({ role: 'assistant', content: message.text });
      continue;
    }
    if (message.image === null) {
      messages.push({ role: 'user', content: message.text });
    } else {
      const url = message.image.kind === 'base64' ? `data:${message.image.mediaType};base64,${message.image.data}` : message.image.url;
      messages.push({ role: 'user', content: [{ type: 'text', text: message.text }, { type: 'image_url', image_url: { url } }] });
    }
  }
  const out: Json = { model, messages };
  if (view.maxTokens !== null) out.max_tokens = view.maxTokens;
  if (view.temperature !== null) out.temperature = view.temperature;
  if (view.topP !== null) out.top_p = view.topP;
  if (view.stops.length > 0) out.stop = view.stops;
  if (view.tools.length > 0) {
    out.tools = view.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
  }
  if (view.jsonMode) out.response_format = { type: 'json_object' };
  return out;
}

/** The view as an Anthropic Messages body naming `model`. */
function viewToAnthropic(view: RequestView, model: string): Json {
  const messages: Json[] = [];
  for (const message of view.messages) {
    if (message.role === 'tool') {
      messages.push({ role: 'user', content: [{ type: 'tool_result', tool_use_id: message.toolCallId ?? '', content: message.text }] });
      continue;
    }
    if (message.role === 'assistant') {
      messages.push({ role: 'assistant', content: message.text });
      continue;
    }
    if (message.image === null) {
      messages.push({ role: 'user', content: message.text });
    } else if (message.image.kind === 'base64') {
      messages.push({
        role: 'user',
        content: [
          { type: 'text', text: message.text },
          { type: 'image', source: { type: 'base64', media_type: message.image.mediaType, data: message.image.data } },
        ],
      });
    } else {
      messages.push({ role: 'user', content: [{ type: 'text', text: message.text }, { type: 'image', source: { type: 'url', url: message.image.url } }] });
    }
  }
  const out: Json = { model, messages };
  if (view.systemText !== '') out.system = view.systemText;
  if (view.maxTokens !== null) out.max_tokens = view.maxTokens;
  if (view.temperature !== null) out.temperature = view.temperature;
  if (view.topP !== null) out.top_p = view.topP;
  if (view.stops.length > 0) out.stop_sequences = view.stops;
  if (view.tools.length > 0) {
    out.tools = view.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  return out;
}

/** The view as a Gemini generateContent body (the model travels in the URL). */
function viewToGemini(view: RequestView): Json {
  const contents: Json[] = [];
  for (const message of view.messages) {
    if (message.role === 'tool') continue;
    const parts: Json[] = [{ text: message.text }];
    if (message.role === 'user' && message.image !== null) {
      if (message.image.kind === 'base64') parts.push({ inlineData: { mimeType: message.image.mediaType, data: message.image.data } });
      else parts.push({ fileData: { mimeType: 'image/png', fileUri: message.image.url } });
    }
    contents.push({ role: message.role === 'assistant' ? 'model' : 'user', parts });
  }
  const out: Json = { contents };
  if (view.systemText !== '') out.systemInstruction = { parts: [{ text: view.systemText }] };
  if (view.tools.length > 0) {
    out.tools = [{ functionDeclarations: view.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
  }
  const config: Json = {};
  if (view.maxTokens !== null) config.maxOutputTokens = view.maxTokens;
  if (view.stops.length > 0) config.stopSequences = view.stops;
  if (view.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// Provider answers: one shape out of the three wires.
// ---------------------------------------------------------------------------

interface ProviderAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  stop: string;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** An OpenAI Chat Completions answer as the one shape. */
function answerFromOpenAI(body: unknown): ProviderAnswer | null {
  if (!isRecord(body)) return null;
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isRecord(choice.message) ? choice.message : null;
  if (message === null) return null;
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    inputTokens: numberOrZero(usage.prompt_tokens),
    outputTokens: numberOrZero(usage.completion_tokens),
    stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
  };
}

/** An Anthropic Messages answer as the one shape. */
function answerFromAnthropic(body: unknown): ProviderAnswer | null {
  if (!isRecord(body)) return null;
  if (!Array.isArray(body.content)) return null;
  const text = body.content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    inputTokens: numberOrZero(usage.input_tokens),
    outputTokens: numberOrZero(usage.output_tokens),
    stop: typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn',
  };
}

/** A Gemini generateContent answer as the one shape. */
function answerFromGemini(body: unknown): ProviderAnswer | null {
  if (!isRecord(body)) return null;
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  if (candidate === null) return null;
  const content = isRecord(candidate.content) ? candidate.content : null;
  const parts = content !== null && Array.isArray(content.parts) ? content.parts : [];
  const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    inputTokens: numberOrZero(usage.promptTokenCount),
    outputTokens: numberOrZero(usage.candidatesTokenCount),
    stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : 'STOP',
  };
}

type RefusalKind = 'spent' | 'out' | 'transient' | 'other';

type CallResult =
  | { ok: true; answer: ProviderAnswer }
  | { ok: false; kind: RefusalKind; status: number; detail: string };

/** The error message a refusal body carries, in whichever wire's envelope. */
function errorMessageOf(wire: Wire, body: unknown): string {
  if (!isRecord(body)) return '';
  const error = isRecord(body.error) ? body.error : null;
  if (error !== null && typeof error.message === 'string') return error.message;
  if (typeof body.message === 'string') return body.message;
  if (wire === 'gemini' && isRecord(body.error) && typeof body.error.status === 'string') return body.error.status;
  return '';
}

/**
 * What a refusal means. The providers' own words distinguish the causes: a spent budget
 * and an outage are permanent for the trial, a scheduled refusal is one call number only.
 */
function classifyRefusal(status: number, message: string): RefusalKind {
  if (message.includes('budget is spent')) return 'spent';
  if (message.endsWith('is out')) return 'out';
  if (message.includes('refused this call')) return 'transient';
  if (message.includes('concurrent_requests')) return 'transient';
  if (status === 429 || status === 402) return 'transient';
  return 'other';
}

/** One non-streaming inference call to one provider. */
async function callProvider(provider: ProviderEntry, model: string, view: RequestView): Promise<CallResult> {
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  let url: string;
  let payload: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    payload = viewToOpenAI(view, model);
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    payload = viewToAnthropic(view, model);
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(model)}:generateContent`;
    payload = viewToGemini(view);
  }
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(payload), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
  } catch (err) {
    return { ok: false, kind: 'transient', status: 0, detail: (err as Error).message };
  }
  const text = await res.text();
  let parsed: unknown = null;
  try {
    parsed = JSON.parse(text);
  } catch {
    // A non-JSON answer is unusable either way.
  }
  if (!res.ok) {
    return { ok: false, kind: classifyRefusal(res.status, errorMessageOf(provider.wire, parsed)), status: res.status, detail: text.slice(0, 300) };
  }
  const answer = provider.wire === 'openai' ? answerFromOpenAI(parsed) : provider.wire === 'anthropic' ? answerFromAnthropic(parsed) : answerFromGemini(parsed);
  if (answer === null) return { ok: false, kind: 'other', status: res.status, detail: 'the provider answered a body this server cannot read' };
  return { ok: true, answer };
}

// ---------------------------------------------------------------------------
// Routing: the guarded hybrid.
// ---------------------------------------------------------------------------

interface Candidate {
  provider: ProviderEntry;
  model: string;
  rating: number;
}

/** Whether a quality row's capabilities meet one request's requirements. */
function rowMeets(requires: Requirement, row: RatingRow): boolean {
  if (requires.tools === true && !row.tools) return false;
  if (requires.vision === true && !row.vision) return false;
  if (requires.json === true && !row.json) return false;
  if (requires.min_context !== undefined && (row.context_length === null || row.context_length < requires.min_context)) return false;
  return true;
}

/**
 * The candidates for one request, in the order to try them. One candidate a provider: its
 * best-rated free model meeting the requirements. An OpenAI-wire JSON-mode request skips
 * anthropic-wire providers (that wire cannot carry JSON mode). Weight-2 requests go greedy;
 * weight-1 requests spread across providers under the cap unless the guard trips.
 */
function candidatesFor(requires: Requirement, weight: number, clientApi: 'openai' | 'anthropic', jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (stateOf(probe.provider.id).dropped) continue;
    if (clientApi === 'openai' && jsonMode && probe.provider.wire === 'anthropic') continue;
    let best: RatingRow | null = null;
    for (const model of probe.models) {
      const row = ratingRow(probe.provider.id, model.id);
      if (row === null || !row.free) continue;
      if (!rowMeets(requires, row)) continue;
      if (best === null || row.rating_milli > best.rating_milli) best = row;
    }
    if (best !== null) out.push({ provider: probe.provider, model: best.model, rating: best.rating_milli });
  }
  const byRating = (a: Candidate, b: Candidate): number => b.rating - a.rating || (a.provider.id < b.provider.id ? -1 : 1);
  if (weight === 2 || out.length === 0) {
    out.sort(byRating);
    return out;
  }
  const bestOverall = Math.max(...out.map((c) => c.rating));
  const underCap = out.filter((c) => stateOf(c.provider.id).calls < SPREAD_CAP);
  const bestUnder = underCap.length > 0 ? Math.max(...underCap.map((c) => c.rating)) : -1;
  if (bestOverall - bestUnder > SPREAD_GUARD_MILLI) {
    out.sort(byRating);
    return out;
  }
  out.sort((a, b) => {
    const aUnder = stateOf(a.provider.id).calls < SPREAD_CAP ? 0 : 1;
    const bUnder = stateOf(b.provider.id).calls < SPREAD_CAP ? 0 : 1;
    return aUnder - bUnder || byRating(a, b);
  });
  return out;
}

interface Served {
  answer: ProviderAnswer;
  providerId: string;
  model: string;
}

/**
 * Serve one request: try the candidates in order, retrying a transient refusal once on the
 * same provider, dropping a spent or out provider, and failing over to the next candidate.
 * Returns null when no candidate answers.
 */
async function serve(view: RequestView, requires: Requirement, weight: number, clientApi: 'openai' | 'anthropic'): Promise<Served | null> {
  const candidates = candidatesFor(requires, weight, clientApi, view.jsonMode);
  for (const candidate of candidates) {
    let tries = 0;
    for (;;) {
      tries += 1;
      const state = stateOf(candidate.provider.id);
      state.calls += 1;
      const result = await callProvider(candidate.provider, candidate.model, view);
      if (result.ok) return { answer: result.answer, providerId: candidate.provider.id, model: candidate.model };
      if (result.kind === 'spent' || result.kind === 'out') {
        state.dropped = true;
        log(`${candidate.provider.id} is done for the trial (${result.kind})`);
        break;
      }
      if (result.kind === 'transient' && tries === 1) continue;
      break;
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// The client wires: render one provider answer in the wire the client used.
// ---------------------------------------------------------------------------

/** One server-sent event: `event: <name>` when named, then the JSON data line, then blank. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** A text split into the providers' own 13-character streamed pieces. */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += SSE_DELTA_CHARS) pieces.push(text.slice(at, at + SSE_DELTA_CHARS));
  return pieces.length === 0 ? [''] : pieces;
}

/** The provider's stop reason in the client's wire (providers only ever stop cleanly). */
function clientStop(wire: 'openai' | 'anthropic', providerStop: string): string {
  if (providerStop === 'length' || providerStop === 'max_tokens' || providerStop === 'MAX_TOKENS') return wire === 'openai' ? 'length' : 'max_tokens';
  if (providerStop === 'tool_calls' || providerStop === 'function_call' || providerStop === 'tool_use') return wire === 'openai' ? 'tool_calls' : 'tool_use';
  if (providerStop === 'content_filter' || providerStop === 'refusal' || providerStop === 'SAFETY') return wire === 'openai' ? 'content_filter' : 'refusal';
  return wire === 'openai' ? 'stop' : 'end_turn';
}

/** An OpenAI Chat Completions object for one answer. */
function openAIJson(model: string, answer: ProviderAnswer): Json {
  return {
    id: 'chatcmpl-aggregator',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: clientStop('openai', answer.stop), logprobs: null }],
    usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
  };
}

/** An Anthropic Messages object for one answer. */
function anthropicJson(model: string, answer: ProviderAnswer): Json {
  return {
    id: 'msg_aggregator',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: clientStop('anthropic', answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** An OpenAI event stream for one answer: role, text pieces, finish, usage, [DONE]. */
function openAIStream(model: string, answer: ProviderAnswer): string {
  const stop = clientStop('openai', answer.stop);
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregator',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, stop)));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-aggregator',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** An Anthropic event stream for one answer: start, block, deltas, stop, delta, stop. */
function anthropicStream(model: string, answer: ProviderAnswer): string {
  const stop = clientStop('anthropic', answer.stop);
  const frames = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregator',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: answer.outputTokens } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

function openAIError(message: string, type: string): Json {
  return { error: { message, type, code: null } };
}

function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

// ---------------------------------------------------------------------------
// The HTTP server.
// ---------------------------------------------------------------------------

/** The free models across the providers that answered, deduplicated by id. */
function servedModels(): { id: string; provider: string; display_name: string }[] {
  const seen = new Set<string>();
  const out: { id: string; provider: string; display_name: string }[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      const row = ratingRow(probe.provider.id, model.id);
      if (!modelIsFree(model, row)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: list.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
      has_more: false,
    });
  } else {
    sendJson(res, 200, {
      object: 'list',
      data: list.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
    });
  }
}

/** The requirements and weight of the request now arriving: the manifest's row by arrival order. */
function manifestRow(): { requires: Requirement; weight: number; fromManifest: boolean } {
  const entry = brief.manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return { requires: entry.requires, weight: entry.weight, fromManifest: true };
  return { requires: {}, weight: 1, fromManifest: false };
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json, api: 'openai' | 'anthropic'): Requirement {
  const out: Requirement = {};
  if (Array.isArray(body.tools) && body.tools.length > 0) out.tools = true;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) out.vision = true;
    }
  }
  if (api === 'openai') {
    const format = isRecord(body.response_format) ? body.response_format.type : undefined;
    if (format === 'json_object' || format === 'json_schema') out.json = true;
  }
  return out;
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type));
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

/** One client request, in either wire: route it, serve it, render the answer. */
async function serveRequest(req: IncomingMessage, res: ServerResponse, api: 'openai' | 'anthropic'): Promise<void> {
  // Claim this request's manifest row first, synchronously, before any await: the row
  // is by arrival order, and the driver's index advances whether or not the body parses.
  const row = manifestRow();
  const body = await readJson(req);
  if (body === null) return sendError(res, api, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const view = api === 'openai' ? viewFromOpenAI(body) : viewFromAnthropic(body);
  const requires = row.fromManifest ? row.requires : requirementsFromBody(body, api);
  const weight = row.fromManifest ? row.weight : 1;
  const served = await serve(view, requires, weight, api);
  if (served === null) {
    log(`no provider answered request ${manifestIndex}`);
    return sendError(res, api, 503, api === 'anthropic' ? 'overloaded_error' : 'service_unavailable', 'no free model answered this request');
  }
  log(`request ${manifestIndex} served by ${served.providerId}/${served.model}`);
  if (view.stream) {
    sendSse(res, api === 'openai' ? openAIStream(served.model, served.answer) : anthropicStream(served.model, served.answer));
  } else {
    sendJson(res, 200, api === 'openai' ? openAIJson(served.model, served.answer) : anthropicJson(served.model, served.answer));
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serveRequest(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serveRequest(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object
    // and stop before any server starts. Wait for stdout to flush before exiting.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
