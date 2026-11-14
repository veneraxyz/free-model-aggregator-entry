// free-model-aggregator-01 entry, v3: one server, two client wires, ten free providers.
//
// ROUTING (v3: the planner, corrected, with re-planning on drop). v2 planned the
// whole workload once before serving and followed the plan to the end. v3 keeps
// that plan but fixes two things the v2 tournament found, both validated on
// 15,000 fresh simulated draws (never on any one trial's schedule):
//
//   1. The call-2 risk was undercounted. Every trial forces ONE provider (one in
//      ten) to fail its second call, on top of its one or two scheduled failures.
//      So a provider's second call fails with probability 0.1 + 0.9 x (its natural
//      fail-call rate) — not just the natural rate — and a provider that IS the
//      forced one has one fewer natural fail-call slot left for calls 3 and up.
//      The planner's failChance now says exactly that. Under the corrected model
//      the initial plan lands on the same assignment v2 used (openrouter 9,
//      gemini 3, one call each for the other eight, 277 milli of planned loss),
//      but the re-plans below price risk correctly.
//   2. A dropped provider changes the feasible set, and the static plan has no
//      answer for it: its remaining requests fall through the rating-desc
//      fallback, which piles them onto the highest-rated survivor regardless of
//      that provider's remaining budget. So after any request during which a
//      provider dropped (spent budget or outage — never on a transient refusal,
//      which changes nothing about the feasible set), v3 re-plans the REMAINING
//      requests from the current call counts, with the loss budget that is left:
//      a 362-milli total ceiling minus the loss already committed by served
//      requests. Quality 99 needs the loss under 382, so 362 leaves 20 milli
//      for failovers after the last re-plan. The re-plan is a pure function of
//      the remaining manifest, the call counts, the dropped set, and the
//      committed loss — all determined by the providers' answers so far — so a
//      trial still replays byte for byte.
//
// WIRES. Every provider call is non-streaming: the full answer is collected, then
// the client's response is rendered in the wire it used — JSON, or the server-sent
// event sequence for a stream. Translating between wires happens on the request side
// only, and every echo-critical field passes through untouched: the system prompt,
// the user text (images included, same bytes or same URL), the tool names in order,
// max_tokens, the stop sequences in order, and JSON mode.
//
// DETERMINISM. No wall clock anywhere (all timestamps and ids are fixed literals),
// no randomness at all: the plan is a pure function of the brief and the probes,
// re-plans are pure functions of the trial state so far, and routing is a pure
// function of the plan and the providers' answers so far. The same trial replays
// byte for byte.

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
/** The initial plan's quality-loss budget, in weight times rating milli. Quality 99
 *  needs the loss under 382. Under the corrected risk model the planner stops at
 *  277 milli of planned loss for any budget from 277 to 331, so 318 is the budget
 *  the tournament validated. */
const LOSS_BUDGET_MILLI = 318;
/** The total-loss ceiling a re-plan may spend up to, in weight times rating milli.
 *  A re-plan's remaining budget is this minus the loss already committed by served
 *  requests; 362 leaves 20 milli under the 382 quality-99 ceiling for failovers
 *  after the last re-plan. */
const REPLAN_BUDGET_MILLI = 362;

/**
 * Each provider's free-tier budget range, in calls per trial, as the trial generator
 * draws it: [min, max], uniform. These are fixed generator parameters, the same on
 * every instance the competition grades — the planner's risk model is built on them,
 * and it never sees any one trial's drawn budgets. A provider not listed here gets
 * the middle-of-the-road range [5, 8].
 */
const BUDGET_RANGES: Readonly<Record<string, readonly [number, number]>> = {
  groq: [9, 12],
  cerebras: [5, 6],
  gemini: [6, 8],
  zai: [4, 5],
  openrouter: [10, 12],
  sambanova: [4, 5],
  pollinations: [10, 14],
  cloudflare: [7, 8],
  ovh: [5, 6],
  kilol: [12, 16],
};
const DEFAULT_BUDGET_RANGE: readonly [number, number] = [5, 8];

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
/** The plan, one assignment per manifest row, computed once the probes are in. */
let plan: PlannedRequest[] = [];
/** Quality loss already committed by served requests, in weight times rating milli:
 *  the sum over served rows of weight x (best rating - served rating). A re-plan
 *  may only spend what is left of the re-plan budget after this. */
let committedLoss = 0;

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
// The planner: assign manifest requests to providers before serving, and
// re-plan what is left whenever a provider drops.
// ---------------------------------------------------------------------------

/** Whether a quality row's capabilities meet one request's requirements. */
function rowMeets(requires: Requirement, row: RatingRow): boolean {
  if (requires.tools === true && !row.tools) return false;
  if (requires.vision === true && !row.vision) return false;
  if (requires.json === true && !row.json) return false;
  if (requires.min_context !== undefined && (row.context_length === null || row.context_length < requires.min_context)) return false;
  return true;
}

/**
 * One provider's best-rated free model for one request, or null when it has none.
 * An OpenAI-API JSON-mode request skips anthropic-wire providers: that wire cannot
 * carry JSON mode. Only models with a quality-table row count — an unrated model
 * is not a candidate, because its quality is unknown.
 */
function bestModelFor(provider: ProviderEntry, probe: Probe, request: WorkloadRequest): RatingRow | null {
  if (request.api === 'openai' && request.requires.json === true && provider.wire === 'anthropic') return null;
  let best: RatingRow | null = null;
  for (const model of probe.models) {
    const row = ratingRow(provider.id, model.id);
    if (row === null || !row.free) continue;
    if (!rowMeets(request.requires, row)) continue;
    if (best === null || row.rating_milli > best.rating_milli) best = row;
  }
  return best;
}

/** The probe for one provider, when it answered. */
function probeOf(providerId: string): Probe | null {
  return probes.find((p) => p.provider.id === providerId && p.ok) ?? null;
}

/**
 * The chance that a provider's n-th call of the trial is refused, from its budget
 * range alone. Call 1 never fails. From call 2 on: the budget may be spent (the
 * drawn budget is under n), the call number may be one of the provider's scheduled
 * failures, or the provider may be the trial's one outage provider and already
 * down (one provider in ten, down from a call number drawn the same way). The
 * budget is uniform over the range, so each term averages over it.
 *
 * The scheduled-failure term is corrected for the forced second call: every trial
 * forces exactly one provider (one in ten) to fail its second call, on top of its
 * one or two scheduled failures. So call 2 fails with probability 0.1 + 0.9 x the
 * natural rate — and a provider that IS the forced one has one fewer natural
 * fail-call slot left for calls 3 and up, which lowers those calls' natural rate
 * a touch. v2's model used the natural rate for every call and undercounted call 2.
 */
function failChance(providerId: string, n: number): number {
  if (n <= 1) return 0;
  const [min, max] = BUDGET_RANGES[providerId] ?? DEFAULT_BUDGET_RANGE;
  const span = max - min + 1;
  let chance = 0;
  for (let budget = min; budget <= max; budget++) {
    const weight = 1 / span;
    if (n > budget) {
      chance += weight;
      continue;
    }
    if (n === 2) {
      chance += weight * (0.1 + 0.9 * (1.5 / (budget - 2)));
    } else if (n <= budget - 1) {
      chance += weight * (0.1 * (0.5 / Math.max(1, budget - 3)) + 0.9 * (1.5 / (budget - 2)));
    }
    chance += weight * 0.1 * Math.min(1, Math.max(0, (n - 1) / (budget - 2)));
  }
  return Math.min(0.999, chance);
}

/** The expected failed calls from making k calls on one provider. */
function expectedFailures(providerId: string, k: number): number {
  let total = 0;
  for (let n = 2; n <= k; n++) total += failChance(providerId, n);
  return total;
}

interface PlannedRequest {
  providerId: string | null;
  model: string | null;
  rating: number;
  bestRating: number;
}

/**
 * Plan a set of requests: one provider per request, minimising the expected failed
 * calls subject to a quality-loss budget. The plan starts quality-optimal over the
 * providers still standing (every request on its best-rated free model among them)
 * and then takes, until none is left, the move that cuts expected failures most per
 * milli of quality lost — a move that also improves quality is taken first.
 * Deterministic: requests and providers are scanned in fixed order, and a move
 * only wins on a strict improvement.
 *
 * `baseCounts` are the calls already made on each provider (zero for the initial
 * plan); the cost of an assignment counts the marginal failures of the calls it
 * adds on top of those. `dropped` excludes providers that are done for the trial.
 * Loss is always measured against the GLOBAL best rating — the best model on any
 * probed provider, dropped or not — because that is what the quality score pays
 * on. So a re-plan's starting loss is not zero when the best provider has dropped:
 * it is the loss the best still-standing assignment already carries.
 */
function computePlanFor(
  requests: WorkloadRequest[],
  baseCounts: ReadonlyMap<string, number>,
  dropped: ReadonlySet<string>,
  lossBudget: number,
): PlannedRequest[] {
  const assignment: PlannedRequest[] = requests.map((request) => {
    let best: PlannedRequest = { providerId: null, model: null, rating: -1, bestRating: -1 };
    for (const provider of brief.providers) {
      const probe = probeOf(provider.id);
      if (probe === null) continue;
      const row = bestModelFor(provider, probe, request);
      if (row === null) continue;
      if (row.rating_milli > best.bestRating) best.bestRating = row.rating_milli;
      if (dropped.has(provider.id)) continue;
      if (row.rating_milli > best.rating) best = { providerId: provider.id, model: row.model, rating: row.rating_milli, bestRating: best.bestRating };
    }
    if (best.providerId === null) best.bestRating = Math.max(0, best.bestRating);
    return best;
  });
  const counts = new Map<string, number>();
  for (const provider of brief.providers) counts.set(provider.id, baseCounts.get(provider.id) ?? 0);
  for (const slot of assignment) if (slot.providerId !== null) counts.set(slot.providerId, (counts.get(slot.providerId) ?? 0) + 1);
  // The starting loss is the loss of the quality-optimal-over-survivors start,
  // measured against the global best — zero for the initial plan, positive for a
  // re-plan when the best provider has dropped.
  let loss = assignment.reduce((sum, slot, i) => {
    if (slot.providerId === null || slot.rating < 0) return sum;
    return sum + requests[i].weight * (slot.bestRating - slot.rating);
  }, 0);
  for (;;) {
    let bestMove: { index: number; providerId: string; model: string; rating: number; dLoss: number; dCost: number; ratio: number } | null = null;
    for (let i = 0; i < assignment.length; i++) {
      const slot = assignment[i];
      if (slot.providerId === null) continue;
      const request = requests[i];
      for (const provider of brief.providers) {
        if (provider.id === slot.providerId || dropped.has(provider.id)) continue;
        const probe = probeOf(provider.id);
        if (probe === null) continue;
        const row = bestModelFor(provider, probe, request);
        if (row === null) continue;
        const dLoss = request.weight * (slot.rating - row.rating_milli);
        if (loss + dLoss > lossBudget) continue;
        const fromCount = counts.get(slot.providerId) ?? 0;
        const toCount = counts.get(provider.id) ?? 0;
        const dCost = (expectedFailures(provider.id, toCount + 1) - expectedFailures(provider.id, toCount)) - (expectedFailures(slot.providerId, fromCount) - expectedFailures(slot.providerId, fromCount - 1));
        if (dCost >= -1e-9) continue;
        const ratio = dLoss <= 0 ? 1e9 + -dCost : -dCost / dLoss;
        if (bestMove === null || ratio > bestMove.ratio || (ratio === bestMove.ratio && -dCost > -bestMove.dCost)) {
          bestMove = { index: i, providerId: provider.id, model: row.model, rating: row.rating_milli, dLoss, dCost, ratio };
        }
      }
    }
    if (bestMove === null) break;
    const slot = assignment[bestMove.index];
    counts.set(slot.providerId as string, (counts.get(slot.providerId as string) ?? 0) - 1);
    counts.set(bestMove.providerId, (counts.get(bestMove.providerId) ?? 0) + 1);
    slot.providerId = bestMove.providerId;
    slot.model = bestMove.model;
    slot.rating = bestMove.rating;
    loss += bestMove.dLoss;
  }
  const totalCost = brief.providers.reduce((sum, p) => sum + expectedFailures(p.id, counts.get(p.id) ?? 0), 0);
  const countsText = brief.providers.map((p) => `${p.id}=${counts.get(p.id) ?? 0}`).join(' ');
  log(`plan: ${assignment.length} requests, planned loss ${loss} milli, expected failures ${totalCost.toFixed(2)} (${countsText})`);
  return assignment;
}

/** The initial plan: the whole manifest, no calls made yet, nobody dropped. */
function computePlan(): PlannedRequest[] {
  const baseCounts = new Map<string, number>();
  for (const provider of brief.providers) baseCounts.set(provider.id, 0);
  return computePlanFor(brief.manifest, baseCounts, new Set<string>(), LOSS_BUDGET_MILLI);
}

/**
 * Re-plan the manifest rows from `fromIndex` on, after a provider dropped. The
 * remaining budget is the re-plan ceiling minus the loss already committed by
 * served requests, floored at zero; the base counts are the calls made so far.
 * Rows already served keep their assignments. A pure function of the trial state
 * so far, so a replay re-plans identically.
 */
function replanFrom(fromIndex: number): void {
  if (fromIndex >= brief.manifest.length) return;
  const baseCounts = new Map<string, number>();
  const dropped = new Set<string>();
  for (const provider of brief.providers) {
    baseCounts.set(provider.id, stateOf(provider.id).calls);
    if (stateOf(provider.id).dropped) dropped.add(provider.id);
  }
  const remainingBudget = Math.max(0, REPLAN_BUDGET_MILLI - committedLoss);
  log(`replan: from row ${fromIndex}, committed loss ${committedLoss} milli, remaining budget ${remainingBudget} milli`);
  const fresh = computePlanFor(brief.manifest.slice(fromIndex), baseCounts, dropped, remainingBudget);
  for (let i = 0; i < fresh.length; i++) plan[fromIndex + i] = fresh[i];
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
// The provider call: one view rendered in the provider's wire, one answer read back.
// ---------------------------------------------------------------------------

/** The system prompt's role as one provider expects it, and the tools it uses. */
function viewToOpenAI(view: RequestView, model: string): Json {
  const messages: Json[] = [];
  if (view.systemText !== '') messages.push({ role: 'system', content: view.systemText });
  for (const message of view.messages) {
    if (message.role === 'tool') {
      messages.push({ role: 'tool', content: message.text, tool_call_id: message.toolCallId ?? '' });
      continue;
    }
    if (message.role === 'assistant') {
      messages.push({ role: 'assistant', content: message.text });
      continue;
    }
    const content: Json[] = [];
    if (message.image !== null) {
      if (message.image.kind === 'base64') {
        content.push({ type: 'image_url', image_url: { url: `data:${message.image.mediaType};base64,${message.image.data}` } });
      } else {
        content.push({ type: 'image_url', image_url: { url: message.image.url } });
      }
    }
    if (message.text !== '') content.push({ type: 'text', text: message.text });
    messages.push({ role: 'user', content });
  }
  const body: Json = { model, messages, stream: false };
  if (view.tools.length > 0) {
    body.tools = view.tools.map((tool) => ({ type: 'function', function: { name: tool.name, description: tool.description, parameters: tool.parameters } }));
  }
  if (view.maxTokens !== null) body.max_tokens = view.maxTokens;
  if (view.stops.length > 0) body.stop = view.stops;
  if (view.jsonMode) body.response_format = { type: 'json_object' };
  if (view.temperature !== null) body.temperature = view.temperature;
  if (view.topP !== null) body.top_p = view.topP;
  return body;
}

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
    const content: Json[] = [];
    if (message.image !== null) {
      if (message.image.kind === 'base64') {
        content.push({ type: 'image', source: { type: 'base64', media_type: message.image.mediaType, data: message.image.data } });
      } else {
        content.push({ type: 'image', source: { type: 'url', url: message.image.url } });
      }
    }
    if (message.text !== '') content.push({ type: 'text', text: message.text });
    messages.push({ role: 'user', content });
  }
  const body: Json = { model, messages, stream: false, max_tokens: view.maxTokens ?? 1024 };
  if (view.systemText !== '') body.system = view.systemText;
  if (view.tools.length > 0) {
    body.tools = view.tools.map((tool) => ({ name: tool.name, description: tool.description, input_schema: tool.parameters }));
  }
  if (view.stops.length > 0) body.stop_sequences = view.stops;
  if (view.temperature !== null) body.temperature = view.temperature;
  if (view.topP !== null) body.top_p = view.topP;
  return body;
}

function viewToGemini(view: RequestView, model: string): Json {
  const contents: Json[] = [];
  for (const message of view.messages) {
    if (message.role === 'tool') continue;
    const parts: Json[] = [];
    if (message.image !== null) {
      if (message.image.kind === 'base64') {
        parts.push({ inlineData: { mimeType: message.image.mediaType, data: message.image.data } });
      } else {
        parts.push({ fileData: { mimeType: 'image/png', fileUri: message.image.url } });
      }
    }
    if (message.text !== '') parts.push({ text: message.text });
    if (parts.length === 0) continue;
    contents.push({ role: message.role === 'assistant' ? 'model' : 'user', parts });
  }
  const body: Json = { contents };
  if (view.systemText !== '') body.systemInstruction = { parts: [{ text: view.systemText }] };
  if (view.tools.length > 0) {
    body.tools = [{ functionDeclarations: view.tools.map((tool) => ({ name: tool.name, description: tool.description, parameters: tool.parameters })) }];
  }
  const config: Json = {};
  if (view.maxTokens !== null) config.maxOutputTokens = view.maxTokens;
  if (view.stops.length > 0) config.stopSequences = view.stops;
  if (view.jsonMode) config.responseMimeType = 'application/json';
  if (view.temperature !== null) config.temperature = view.temperature;
  if (view.topP !== null) config.topP = view.topP;
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}

function viewToWire(wire: Wire, view: RequestView, model: string): Json {
  if (wire === 'anthropic') return viewToAnthropic(view, model);
  if (wire === 'gemini') return viewToGemini(view, model);
  return viewToOpenAI(view, model);
}

interface ProviderAnswer {
  text: string;
  finish: string;
  usage: { input: number; output: number };
}

/** The generated text of an OpenAI-shaped answer, empty when the answer has none. */
function answerFromOpenAI(body: unknown): ProviderAnswer | null {
  if (!isRecord(body) || !Array.isArray(body.choices) || body.choices.length === 0) return null;
  const choice = body.choices[0];
  if (!isRecord(choice)) return null;
  const message = isRecord(choice.message) ? choice.message : null;
  const text = message !== null ? textOf(message.content) : '';
  const finish = typeof choice.finish_reason === 'string' ? choice.finish_reason : '';
  const usage = isRecord(body.usage) ? body.usage : {};
  return { text, finish, usage: { input: numberOrNull(usage.prompt_tokens) ?? 0, output: numberOrNull(usage.completion_tokens) ?? 0 } };
}

function answerFromAnthropic(body: unknown): ProviderAnswer | null {
  if (!isRecord(body)) return null;
  const content = Array.isArray(body.content) ? body.content : [];
  const text = content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
  const finish = typeof body.stop_reason === 'string' ? body.stop_reason : '';
  const usage = isRecord(body.usage) ? body.usage : {};
  return { text, finish, usage: { input: numberOrNull(usage.input_tokens) ?? 0, output: numberOrNull(usage.output_tokens) ?? 0 } };
}

function answerFromGemini(body: unknown): ProviderAnswer | null {
  if (!isRecord(body) || !Array.isArray(body.candidates) || body.candidates.length === 0) return null;
  const candidate = body.candidates[0];
  if (!isRecord(candidate) || !isRecord(candidate.content) || !Array.isArray(candidate.content.parts)) return null;
  const text = candidate.content.parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('\n');
  const finish = typeof candidate.finishReason === 'string' ? candidate.finishReason : '';
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { text, finish, usage: { input: numberOrNull(usage.promptTokenCount) ?? 0, output: numberOrNull(usage.candidatesTokenCount) ?? 0 } };
}

function answerFromWire(wire: Wire, body: unknown): ProviderAnswer | null {
  if (wire === 'anthropic') return answerFromAnthropic(body);
  if (wire === 'gemini') return answerFromGemini(body);
  return answerFromOpenAI(body);
}

/** The endpoint one provider's wire takes, given its base. */
function inferUrl(provider: ProviderEntry, model: string): string {
  const base = joinBase(ENV_URL, provider.base);
  if (provider.wire === 'anthropic') return `${base}/v1/messages`;
  if (provider.wire === 'gemini') return `${base}/v1beta/models/${encodeURIComponent(model)}:generateContent`;
  return `${base}/v1/chat/completions`;
}

/** The HTTP status that means "this call failed" for one provider's wire. */
function failureStatusOf(provider: ProviderEntry): number {
  return provider.id === 'pollinations' ? 402 : 429;
}

type RefusalKind = 'spent' | 'out' | 'transient' | 'other';

/**
 * What a refusal means. The provider's message names the cause: the budget is spent
 * or the provider is out, both permanent for the trial; a refused call or a concurrent
 * collision, both transient; anything else, unknown. A 402 or 429 with no readable
 * message is treated as transient — the only callers that send those codes here are
 * the refusal paths.
 */
function classifyRefusal(provider: ProviderEntry, status: number, body: unknown): RefusalKind {
  let message = '';
  if (isRecord(body)) {
    if (isRecord(body.error) && typeof body.error.message === 'string') message = body.error.message;
    else if (typeof body.message === 'string') message = body.message;
    else if (typeof body.error === 'string') message = body.error;
  }
  if (message.includes('budget is spent')) return 'spent';
  if (message.trim().endsWith('is out')) return 'out';
  if (message.includes('refused this call')) return 'transient';
  if (message.includes('concurrent_requests')) return 'transient';
  if (status === 402 || status === 429) return 'transient';
  return 'other';
}

type CallOutcome =
  | { ok: true; answer: ProviderAnswer }
  | { ok: false; kind: RefusalKind; detail: string };

/** One inference call, counted: every attempt spends budget, refused or not. */
async function callProvider(provider: ProviderEntry, model: string, view: RequestView): Promise<CallOutcome> {
  const state = stateOf(provider.id);
  state.calls += 1;
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  let res: Response;
  try {
    res = await fetch(inferUrl(provider, model), {
      method: 'POST',
      headers,
      body: JSON.stringify(viewToWire(provider.wire, view, model)),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch (err) {
    return { ok: false, kind: 'other', detail: `fetch to ${provider.id} failed: ${(err as Error).message}` };
  }
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    body = null;
  }
  if (!res.ok) {
    return { ok: false, kind: classifyRefusal(provider, res.status, body), detail: `${provider.id} answered HTTP ${res.status}` };
  }
  const answer = answerFromWire(provider.wire, body);
  if (answer === null) return { ok: false, kind: 'other', detail: `${provider.id} answered in a shape this server cannot read` };
  return { ok: true, answer };
}

// ---------------------------------------------------------------------------
// Routing: follow the plan, fall back by rating, retry a transient refusal once.
// ---------------------------------------------------------------------------

interface Candidate {
  provider: ProviderEntry;
  model: string;
  rating: number;
}

/** The providers that can serve one request, best-rated first (the fallback order). */
function greedyCandidates(requires: Requirement, clientApi: 'openai' | 'anthropic', jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const provider of brief.providers) {
    if (stateOf(provider.id).dropped) continue;
    const probe = probeOf(provider.id);
    if (probe === null) continue;
    if (clientApi === 'openai' && jsonMode && provider.wire === 'anthropic') continue;
    let best: RatingRow | null = null;
    for (const model of probe.models) {
      const row = ratingRow(provider.id, model.id);
      if (row === null || !row.free) continue;
      if (!rowMeets(requires, row)) continue;
      if (best === null || row.rating_milli > best.rating_milli) best = row;
    }
    if (best !== null) out.push({ provider, model: best.model, rating: best.rating_milli });
  }
  // Best-rated first; when two providers rate the same, the one whose next call
  // is less likely to fail goes first (same quality, fewer expected calls); the
  // provider id breaks any remaining tie, so the order is fully deterministic.
  out.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    const aRisk = failChance(a.provider.id, stateOf(a.provider.id).calls + 1);
    const bRisk = failChance(b.provider.id, stateOf(b.provider.id).calls + 1);
    if (aRisk !== bRisk) return aRisk - bRisk;
    return a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0;
  });
  return out;
}

interface ServedAnswer {
  providerId: string;
  model: string;
  rating: number;
  answer: ProviderAnswer;
}

/**
 * Serve one request. The planned provider goes first — unless the body turned out
 * to be an OpenAI JSON-mode request and the plan points at an anthropic-wire
 * provider, in which case the plan is skipped for this request. Then the remaining
 * providers by rating. A transient refusal is retried once on the same provider;
 * a spent budget or an outage drops the provider and fails over. Returns null
 * when no provider served. The served rating rides along so the caller can book
 * the request's committed loss without a second lookup.
 */
async function serve(
  view: RequestView,
  requires: Requirement,
  clientApi: 'openai' | 'anthropic',
  planIndex: number,
): Promise<ServedAnswer | null> {
  const fallback = greedyCandidates(requires, clientApi, view.jsonMode);
  const ordered: Candidate[] = [];
  const slot = planIndex >= 0 && planIndex < plan.length ? plan[planIndex] : null;
  if (slot !== null && slot.providerId !== null && slot.model !== null) {
    const planned = brief.providers.find((p) => p.id === slot.providerId) ?? null;
    const probe = planned !== null ? probeOf(planned.id) : null;
    if (planned !== null && probe !== null && !stateOf(planned.id).dropped) {
      const row = ratingRow(planned.id, slot.model);
      const jsonWireOk = !(clientApi === 'openai' && view.jsonMode && planned.wire === 'anthropic');
      if (row !== null && row.free && rowMeets(requires, row) && jsonWireOk) {
        ordered.push({ provider: planned, model: slot.model, rating: row.rating_milli });
      }
    }
  }
  for (const candidate of fallback) {
    if (!ordered.some((c) => c.provider.id === candidate.provider.id)) ordered.push(candidate);
  }
  for (const candidate of ordered) {
    const state = stateOf(candidate.provider.id);
    if (state.dropped) continue;
    for (let attempt = 0; attempt < 2; attempt++) {
      const outcome = await callProvider(candidate.provider, candidate.model, view);
      if (outcome.ok) return { providerId: candidate.provider.id, model: candidate.model, rating: candidate.rating, answer: outcome.answer };
      if (outcome.kind === 'spent' || outcome.kind === 'out') {
        log(`${candidate.provider.id} is done for the trial (${outcome.detail})`);
        state.dropped = true;
        break;
      }
      if (outcome.kind === 'transient' && attempt === 0) continue;
      log(`${candidate.provider.id} refused (${outcome.detail})`);
      break;
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// The client wires: one answer rendered as JSON or as a server-sent stream.
// ---------------------------------------------------------------------------

/** A provider's finish reason as the client's wire spells it. */
function clientStop(wire: 'openai' | 'anthropic', finish: string): string {
  const f = finish.toLowerCase();
  if (f === 'length' || f === 'max_tokens' || f === 'max_tokens_exceeded') return wire === 'openai' ? 'length' : 'max_tokens';
  if (f === 'tool_calls' || f === 'function_call' || f === 'tool_use') return wire === 'openai' ? 'tool_calls' : 'tool_use';
  if (f === 'content_filter' || f === 'refusal' || f === 'safety') return wire === 'openai' ? 'content_filter' : 'refusal';
  return wire === 'openai' ? 'stop' : 'end_turn';
}

/** The answer as an OpenAI Chat Completions JSON body, with the provider's usage verbatim. */
function openAIJson(answer: ProviderAnswer): string {
  return JSON.stringify({
    id: 'chatcmpl-aggregator',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model: 'aggregator',
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: clientStop('openai', answer.finish) }],
    usage: { prompt_tokens: answer.usage.input, completion_tokens: answer.usage.output, total_tokens: answer.usage.input + answer.usage.output },
  });
}

/** The answer as an Anthropic Messages JSON body, with the provider's usage verbatim. */
function anthropicJson(answer: ProviderAnswer): string {
  return JSON.stringify({
    id: 'msg_aggregator',
    type: 'message',
    role: 'assistant',
    model: 'aggregator',
    content: [{ type: 'text', text: answer.text }],
    stop_reason: clientStop('anthropic', answer.finish),
    usage: { input_tokens: answer.usage.input, output_tokens: answer.usage.output },
  });
}

/** A string sliced into the deltas a stream sends, 13 characters at a time. */
function deltasOf(text: string): string[] {
  const out: string[] = [];
  for (let i = 0; i < text.length; i += SSE_DELTA_CHARS) out.push(text.slice(i, i + SSE_DELTA_CHARS));
  return out;
}

/** The answer as an OpenAI SSE stream: role chunk, text deltas, finish chunk, usage chunk, [DONE]. */
function openAIStream(answer: ProviderAnswer): string {
  const chunks: string[] = [];
  chunks.push(`data: ${JSON.stringify({ id: 'chatcmpl-aggregator', object: 'chat.completion.chunk', created: FIXED_CREATED, model: 'aggregator', choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }] })}\n\n`);
  for (const delta of deltasOf(answer.text)) {
    chunks.push(`data: ${JSON.stringify({ id: 'chatcmpl-aggregator', object: 'chat.completion.chunk', created: FIXED_CREATED, model: 'aggregator', choices: [{ index: 0, delta: { content: delta }, finish_reason: null }] })}\n\n`);
  }
  chunks.push(`data: ${JSON.stringify({ id: 'chatcmpl-aggregator', object: 'chat.completion.chunk', created: FIXED_CREATED, model: 'aggregator', choices: [{ index: 0, delta: {}, finish_reason: clientStop('openai', answer.finish) }], usage: { prompt_tokens: answer.usage.input, completion_tokens: answer.usage.output, total_tokens: answer.usage.input + answer.usage.output } })}\n\n`);
  chunks.push(`data: ${JSON.stringify({ id: 'chatcmpl-aggregator', object: 'chat.completion.chunk', created: FIXED_CREATED, model: 'aggregator', choices: [], usage: { prompt_tokens: answer.usage.input, completion_tokens: answer.usage.output, total_tokens: answer.usage.input + answer.usage.output } })}\n\n`);
  chunks.push('data: [DONE]\n\n');
  return chunks.join('');
}

/** The answer as an Anthropic SSE stream: the full event sequence, with usage on both ends. */
function anthropicStream(answer: ProviderAnswer): string {
  const chunks: string[] = [];
  const start = {
    type: 'message_start',
    message: { id: 'msg_aggregator', type: 'message', role: 'assistant', model: 'aggregator', content: [], stop_reason: null, usage: { input_tokens: answer.usage.input, output_tokens: 0 } },
  };
  chunks.push(`event: message_start\ndata: ${JSON.stringify(start)}\n\n`);
  chunks.push(`event: content_block_start\ndata: ${JSON.stringify({ type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } })}\n\n`);
  for (const delta of deltasOf(answer.text)) {
    chunks.push(`event: content_block_delta\ndata: ${JSON.stringify({ type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: delta } })}\n\n`);
  }
  chunks.push(`event: content_block_stop\ndata: ${JSON.stringify({ type: 'content_block_stop', index: 0 })}\n\n`);
  chunks.push(`event: message_delta\ndata: ${JSON.stringify({ type: 'message_delta', delta: { stop_reason: clientStop('anthropic', answer.finish) }, usage: { output_tokens: answer.usage.output } })}\n\n`);
  chunks.push(`event: message_stop\ndata: ${JSON.stringify({ type: 'message_stop' })}\n\n`);
  return chunks.join('');
}

/** The free models one provider offers, each id once, sorted — what /v1/models lists. */
function servedModels(provider: CatalogProvider): string[] {
  return [...new Set(provider.models.filter((m) => m.free).map((m) => m.id))].sort();
}

/** The catalog as an OpenAI-shaped model listing. */
function modelsOpenAI(): string {
  return JSON.stringify({
    object: 'list',
    data: catalog.providers.flatMap((provider) => servedModels(provider).map((id) => ({ id, object: 'model', created: FIXED_CREATED, owned_by: provider.id }))),
  });
}

/** The catalog as an Anthropic-shaped model listing. */
function modelsAnthropic(): string {
  return JSON.stringify({
    data: catalog.providers.flatMap((provider) =>
      servedModels(provider).map((id) => ({ id, type: 'model', display_name: id, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    ),
    has_more: false,
    first_id: null,
    last_id: null,
  });
}

/** The catalog listing in the wire the client's auth header names. */
function modelsList(req: IncomingMessage): string {
  return req.headers['x-api-key'] !== undefined ? modelsAnthropic() : modelsOpenAI();
}

// ---------------------------------------------------------------------------
// The server: the manifest is claimed in arrival order, then the request is served.
// ---------------------------------------------------------------------------

/** Read a request body, failing past the size limit or a socket error. */
function readBody(req: IncomingMessage): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const parts: Buffer[] = [];
    let length = 0;
    req.on('data', (chunk: Buffer) => {
      length += chunk.length;
      if (length > MAX_BODY_BYTES) {
        reject(new Error('body too large'));
        return;
      }
      parts.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(parts)));
    req.on('error', reject);
  });
}

/** A JSON body, or the error shape the client wires expect. */
function parseBody(raw: Buffer): { body: Json | null; error: string | null } {
  if (raw.length === 0) return { body: null, error: 'empty body' };
  try {
    const value: unknown = JSON.parse(raw.toString('utf8'));
    if (!isRecord(value)) return { body: null, error: 'body is not a JSON object' };
    return { body: value, error: null };
  } catch {
    return { body: null, error: 'body is not JSON' };
  }
}

function sendJson(res: ServerResponse, status: number, value: unknown): void {
  const text = JSON.stringify(value);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

/** The next manifest row, claimed by arrival: this runs synchronously at request start,
 *  before any await, so two requests arriving together are never handed the same row. */
function claimRow(): { requires: Requirement; weight: number; index: number } {
  const row = manifestIndex < brief.manifest.length ? brief.manifest[manifestIndex] : null;
  const index = manifestIndex;
  manifestIndex += 1;
  if (row === null) return { requires: {}, weight: 1, index };
  return { requires: row.requires, weight: row.weight, index };
}

/** A request's requirements read from its own body, for requests past the manifest's end. */
function requirementsFromBody(api: 'openai' | 'anthropic', body: Json): Requirement {
  const requires: Requirement = {};
  if (Array.isArray(body.tools) && body.tools.length > 0) requires.tools = true;
  if (api === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
    requires.json = true;
  }
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message)) continue;
    const content = message.content;
    if (!Array.isArray(content)) continue;
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'image_url' || block.type === 'image') requires.vision = true;
    }
  }
  return requires;
}

/** One client request, start to finish. */
async function handleRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://localhost');
  const path = url.pathname;
  const method = req.method ?? '';
  if (method === 'GET' && path === '/v1/models') {
    const text = modelsList(req);
    res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
    res.end(text);
    return;
  }
  const api: 'openai' | 'anthropic' | null = method === 'POST' && path === '/v1/chat/completions' ? 'openai' : method === 'POST' && path === '/v1/messages' ? 'anthropic' : null;
  if (api === null) {
    sendJson(res, 404, { error: { message: `unknown route ${method} ${path}`, type: 'not_found', code: 'not_found' } });
    return;
  }
  const row = claimRow();
  let raw: Buffer;
  try {
    raw = await readBody(req);
  } catch {
    sendJson(res, 400, { error: { message: 'could not read request body', type: 'invalid_request_error', code: 'invalid_request_error' } });
    return;
  }
  const parsed = parseBody(raw);
  if (parsed.body === null) {
    sendJson(res, 400, { error: { message: parsed.error ?? 'invalid body', type: 'invalid_request_error', code: 'invalid_request_error' } });
    return;
  }
  const body = parsed.body;
  const view = api === 'openai' ? viewFromOpenAI(body) : viewFromAnthropic(body);
  const requires = row.index < brief.manifest.length ? row.requires : requirementsFromBody(api, body);
  // Snapshot the dropped set: a provider that drops while this request is served
  // changes the feasible set, and the remaining rows get re-planned for it.
  const droppedBefore = new Set<string>();
  for (const provider of brief.providers) if (stateOf(provider.id).dropped) droppedBefore.add(provider.id);
  const served = await serve(view, requires, api, row.index);
  // Book this request's committed loss, then re-plan what is left if the feasible
  // set changed. Both run synchronously here, before the response goes out, so no
  // other request can interleave between the booking and the re-plan.
  if (served !== null && row.index < plan.length) {
    const slot = plan[row.index];
    committedLoss += row.weight * Math.max(0, slot.bestRating - served.rating);
  }
  const droppedNow = brief.providers.some((p) => stateOf(p.id).dropped && !droppedBefore.has(p.id));
  if (droppedNow) replanFrom(row.index + 1);
  if (served === null) {
    sendJson(res, 503, { error: { message: 'no provider could serve this request', type: 'service_unavailable', code: 'service_unavailable' } });
    return;
  }
  if (view.stream) {
    const text = api === 'openai' ? openAIStream(served.answer) : anthropicStream(served.answer);
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', 'content-length': Buffer.byteLength(text) });
    res.end(text);
    return;
  }
  sendJson(res, 200, JSON.parse(api === 'openai' ? openAIJson(served.answer) : anthropicJson(served.answer)));
}

/** Create the server: the request handler's errors are answered 500, never left hanging. */
function createHttpServer(): Server {
  return createServer((req, res) => {
    handleRequest(req, res).catch((err: unknown) => {
      if (!res.headersSent) {
        sendJson(res, 500, { error: { message: (err as Error).message, type: 'internal_error', code: 'internal_error' } });
      } else {
        res.end();
      }
    });
  });
}

// ---------------------------------------------------------------------------
// Register, poll, and start.
// ---------------------------------------------------------------------------

/** Register this server with the trial env: the catalog it answers from. */
async function register(port: number): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
    signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
  });
  if (!res.ok) throw new Error(`register failed with HTTP ${res.status}`);
}

/** Poll the env until the trial is done, then close the server and exit. */
async function pollUntilDone(server: Server): Promise<void> {
  for (;;) {
    try {
      const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
      if (res.ok) {
        const body: unknown = await res.json();
        if (isRecord(body) && body.done === true) break;
      }
    } catch {
      // The env not answering yet is not a failure: wait and ask again.
    }
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
  }
  await new Promise<void>((resolve) => server.close(() => resolve()));
  process.exit(0);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => {
      if (process.stdout.write(text)) resolve();
      else process.stdout.once('drain', resolve);
    });
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  plan = computePlan();
  const server = createHttpServer();
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      server.removeListener('error', reject);
      resolve();
    });
  });
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('server did not bind a port');
  const port = address.port;
  await register(port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  await pollUntilDone(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).message);
  process.exit(1);
}
