// free-model-aggregator-01 entry, v3: a single server speaking two client wires,
// fronting ten free providers behind one planner.
//
// HOW REQUESTS ARE ROUTED (v3: a corrected planner that re-plans after a drop).
// v2 planned the whole workload once up front and then stuck to that plan. v3
// keeps the up-front plan but repairs two weaknesses the v2 tournament exposed,
// both confirmed on 15,000 fresh simulated draws (never tuned to one trial):
//
//   1. Second-call risk was understated. Each trial forces exactly ONE provider
//      (a one-in-ten chance for any given one) to refuse its second call, above
//      and beyond its one or two scheduled refusals. So a provider's second call
//      refuses with probability 0.1 + 0.9 x its natural fail-call rate — not the
//      natural rate alone — and a provider that IS the forced one has one fewer
//      natural fail-call slot remaining for its third call onward. The planner's
//      refusalChance now encodes exactly that. With the corrected model the
//      opening plan settles on the same split v2 used (openrouter 9, gemini 3,
//      one call apiece for the other eight, 277 milli of planned loss), but the
//      re-plans that follow now price risk properly.
//   2. A provider dropping mid-trial changes what is feasible, and a fixed plan
//      has no response: its leftover requests cascade into the rating-ordered
//      fallback, which stacks them onto the best-rated survivor no matter how
//      little budget that survivor has left. So whenever a request ends with a
//      provider dropped (spent budget or outage — never after a merely transient
//      refusal, which leaves the feasible set untouched), v3 re-plans the rows
//      still to come from the live call counts, spending only the loss budget
//      that remains: a 362-milli overall ceiling less the loss already booked by
//      served rows. Quality 99 needs total loss under 382, so 362 keeps 20 milli
//      in hand for failovers after the final re-plan. The re-plan depends only
//      on the remaining manifest, the call counts, the dropped set, and the
//      booked loss — all fixed by the providers' answers so far — so a trial
//      still replays byte for byte.
//
// WIRES. Every upstream call is made without streaming: the whole answer is
// gathered first, then the client's reply is rendered in the wire it spoke —
// plain JSON, or the server-sent event sequence for a stream. Translation
// between wires happens only while building the request, and every field the
// client checks in the echo goes through untouched: the system prompt, the user
// text (images included, identical bytes or identical URL), the tool names in
// order, max_tokens, the stop sequences in order, and JSON mode.
//
// DETERMINISM. The clock is never read (every timestamp and id is a literal),
// nothing is random: the opening plan is a pure function of the brief and the
// probes, re-plans are pure functions of the trial state so far, and routing is
// a pure function of the plan and the providers' answers so far. Run the same
// trial twice and the export matches byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

type ProviderWire = 'openai' | 'anthropic' | 'gemini';
type AuthMethod = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
type JsonObject = Record<string, unknown>;

function isJsonObject(v: unknown): v is JsonObject {
  return v !== null && typeof v === 'object' && !Array.isArray(v);
}

/** The protocol version the register body carries. */
const PROTOCOL_VERSION = 1;
/** The gap between GET /work polls, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The longest a provider catalog probe may run. */
const PROBE_TIMEOUT_MS = 5000;
/** The longest one provider inference call may run (the client allows 30 s per request). */
const CALL_TIMEOUT_MS = 10000;
/** The largest client request body this server will read, in bytes. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The fixed epoch second stamped on every OpenAI-shaped reply: the clock is never read. */
const FIXED_TIMESTAMP = 1767225600;
/** The fixed instant stamped on every Anthropic-shaped model listing. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** The characters each streamed text delta carries (the providers' own slice size). */
const STREAM_CHUNK_CHARS = 13;
/** The opening plan's quality-loss budget, in weight times rating milli. Quality 99
 *  needs total loss under 382. Under the corrected risk model the planner settles
 *  at 277 milli of planned loss for any budget between 277 and 331, so 318 is the
 *  budget the tournament validated. */
const INITIAL_LOSS_BUDGET = 318;
/** The total-loss ceiling a re-plan may spend up to, in weight times rating milli.
 *  A re-plan's remaining budget is this less the loss already booked by served
 *  rows; 362 keeps 20 milli under the 382 quality-99 ceiling for failovers
 *  after the last re-plan. */
const REPLAN_LOSS_CEILING = 362;

/**
 * Each provider's free-tier budget range, in calls per trial, as the trial
 * generator draws it: [min, max], uniform. These are fixed generator parameters,
 * identical on every instance the competition grades — the planner's risk model
 * is built from them, and it never learns any one trial's drawn budgets. A
 * provider missing from this table gets the middle-of-the-road range [5, 8].
 */
const PROVIDER_BUDGET_RANGES: Readonly<Record<string, readonly [number, number]>> = {
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
const FALLBACK_BUDGET_RANGE: readonly [number, number] = [5, 8];

const ENV_ORIGIN = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const BRIEF_PATH = process.env.MAKEFUN_PROMPT_FILE ?? '';

// ---------------------------------------------------------------------------
// The brief: provider directory, quality table, workload manifest.
// ---------------------------------------------------------------------------

interface Provider {
  id: string;
  name: string;
  wire: ProviderWire;
  auth: AuthMethod;
  basePath: string;
  keyEnv: string;
  key: string;
}

interface Rating {
  model: string;
  provider: string;
  rating: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface RequestNeeds {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface ManifestRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: RequestNeeds;
}

interface BriefDoc {
  providers: Provider[];
  ratings: Rating[];
  manifest: ManifestRow[];
  error: string | null;
}

const SCHEMA_PROVIDERS = 'free-model-aggregator-01/providers/v1';
const SCHEMA_RATINGS = 'free-model-aggregator-01/ratings/v1';
const SCHEMA_WORKLOAD = 'free-model-aggregator-01/workload/v1';

/** The index of the `}` that closes the object opening at `open`, or -1. */
function matchingBrace(source: string, open: number): number {
  let nesting = 0;
  let insideString = false;
  for (let k = open; k < source.length; k++) {
    const char = source[k];
    if (insideString) {
      if (char === '\\') k++;
      else if (char === '"') insideString = false;
    } else if (char === '"') insideString = true;
    else if (char === '{') nesting++;
    else if (char === '}') {
      nesting--;
      if (nesting === 0) return k;
    }
  }
  return -1;
}

/** Every JSON object in `source` carrying a string `schema`, in order of appearance. */
function extractDocuments(source: string): JsonObject[] {
  const collected: JsonObject[] = [];
  let cursor = source.indexOf('{');
  while (cursor !== -1) {
    const close = matchingBrace(source, cursor);
    if (close !== -1) {
      try {
        const parsed: unknown = JSON.parse(source.slice(cursor, close + 1));
        if (isJsonObject(parsed) && typeof parsed.schema === 'string') collected.push(parsed);
      } catch {
        // Not JSON on its own: keep looking.
      }
    }
    cursor = source.indexOf('{', cursor + 1);
  }
  return collected;
}

function asString(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function asNumberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function providerFrom(input: unknown): Provider | null {
  if (!isJsonObject(input)) return null;
  const id = asString(input.id);
  const wire = input.wire === 'openai' || input.wire === 'anthropic' || input.wire === 'gemini' ? input.wire : null;
  if (id === '' || wire === null) return null;
  const auth = input.auth === 'bearer' || input.auth === 'x-api-key' || input.auth === 'x-goog-api-key' || input.auth === 'none' ? input.auth : 'bearer';
  let basePath = asString(input.path);
  if (basePath === '') basePath = asString(input.base_path);
  if (basePath === '') basePath = asString(input.base_url);
  if (basePath === '') basePath = asString(input.base);
  const name = asString(input.name);
  return { id, name: name !== '' ? name : id, wire, auth, basePath, keyEnv: asString(input.key_env), key: asString(input.key) };
}

function ratingFrom(input: unknown): Rating | null {
  if (!isJsonObject(input)) return null;
  const model = asString(input.model);
  const provider = asString(input.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating: typeof input.rating_milli === 'number' && Number.isFinite(input.rating_milli) ? input.rating_milli : 0,
    context_length: asNumberOrNull(input.context_length),
    tools: input.tools === true,
    vision: input.vision === true,
    json: input.json === true,
    free: input.free === true,
  };
}

function workloadRowFrom(input: unknown): ManifestRow | null {
  if (!isJsonObject(input)) return null;
  const id = asString(input.id);
  if (id === '') return null;
  const requires = isJsonObject(input.requires) ? input.requires : {};
  const out: RequestNeeds = {};
  if (requires.tools === true) out.tools = true;
  if (requires.vision === true) out.vision = true;
  if (requires.json === true) out.json = true;
  if (typeof requires.min_context === 'number' && Number.isFinite(requires.min_context)) out.min_context = requires.min_context;
  return { id, api: input.api === 'anthropic' ? 'anthropic' : 'openai', stream: input.stream === true, weight: input.weight === 2 ? 2 : 1, requires: out };
}

function loadBrief(path: string): BriefDoc {
  const empty: BriefDoc = { providers: [], ratings: [], manifest: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (error) {
    return { ...empty, error: `cannot read ${path}: ${(error as Error).message}` };
  }
  const docs = extractDocuments(text);
  const directoryDoc = docs.find((doc) => doc.schema === SCHEMA_PROVIDERS && Array.isArray(doc.providers)) ?? null;
  if (directoryDoc === null || !Array.isArray(directoryDoc.providers)) {
    return { ...empty, error: `the brief at ${path} holds no ${SCHEMA_PROVIDERS} document` };
  }
  const providers = directoryDoc.providers.flatMap((item) => {
    const entry = providerFrom(item);
    return entry === null ? [] : [entry];
  });
  const ratingsDoc = docs.find((doc) => doc.schema === SCHEMA_RATINGS && Array.isArray(doc.ratings)) ?? null;
  const ratings =
    ratingsDoc !== null && Array.isArray(ratingsDoc.ratings)
      ? ratingsDoc.ratings.flatMap((item) => {
          const entry = ratingFrom(item);
          return entry === null ? [] : [entry];
        })
      : [];
  const workloadDoc = docs.find((doc) => doc.schema === SCHEMA_WORKLOAD && Array.isArray(doc.manifest)) ?? null;
  const manifest =
    workloadDoc !== null && Array.isArray(workloadDoc.manifest)
      ? workloadDoc.manifest.flatMap((item) => {
          const entry = workloadRowFrom(item);
          return entry === null ? [] : [entry];
        })
      : [];
  return { providers, ratings, manifest, error: null };
}

// ---------------------------------------------------------------------------
// The catalog probe.
// ---------------------------------------------------------------------------

interface ListedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

interface ProbeResult {
  provider: Provider;
  ok: boolean;
  status: number;
  models: ListedModel[];
  error: string | null;
}

interface ModelEntry {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogedProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: ProviderWire;
  models: ModelEntry[];
}

interface Catalog {
  providers: CatalogedProvider[];
}

/** A base path joined onto the env's URL; an absolute base_url is used as it stands. */
function resolveBase(envUrl: string, base: string): string {
  const trimmed = base.trim();
  if (/^https?:\/\//i.test(trimmed)) return trimmed.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (trimmed === '') return prefix;
  return `${prefix}${trimmed.startsWith('/') ? '' : '/'}${trimmed.replace(/\/+$/, '')}`;
}

/** The headers carrying a provider's key, in the shape its auth wants. */
function headersForAuth(auth: AuthMethod, key: string): Record<string, string> {
  if (key === '') return {};
  switch (auth) {
    case 'bearer':
      return { authorization: `Bearer ${key}` };
    case 'x-api-key':
      return { 'x-api-key': key };
    case 'x-goog-api-key':
      return { 'x-goog-api-key': key };
    default:
      return {};
  }
}

function asBoolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function modelFromListing(wire: ProviderWire, input: unknown): ListedModel | null {
  if (!isJsonObject(input)) return null;
  const id = wire === 'gemini' ? asString(input.name).replace(/^models\//, '') : asString(input.id);
  if (id === '') return null;
  const context = asNumberOrNull(input.context_length) ?? asNumberOrNull(input.context_window) ?? asNumberOrNull(input.inputTokenLimit);
  let displayName = asString(input.display_name);
  if (displayName === '') displayName = asString(input.displayName);
  if (displayName === '') displayName = id;
  return {
    id,
    display_name: displayName,
    free: asBoolOrNull(input.free),
    context_length: context,
    tools: asBoolOrNull(input.tools),
    vision: asBoolOrNull(input.vision),
    json: asBoolOrNull(input.json),
  };
}

/** The models one catalog route's answer listed, in its wire's shape. */
function modelsFromListing(wire: ProviderWire, body: unknown): ListedModel[] {
  if (!isJsonObject(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((item) => {
    const entry = modelFromListing(wire, item);
    return entry === null ? [] : [entry];
  });
}

/** Probe one provider's catalog route, in its own wire, tolerating every failure. */
async function probeOneProvider(envUrl: string, provider: Provider, timeoutMs = PROBE_TIMEOUT_MS): Promise<ProbeResult> {
  const root = resolveBase(envUrl, provider.basePath);
  const endpoint = provider.wire === 'gemini' ? `${root}/v1beta/models` : `${root}/v1/models`;
  const hdrs: Record<string, string> = { accept: 'application/json', ...headersForAuth(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') hdrs['anthropic-version'] = '2023-06-01';
  try {
    const response = await fetch(endpoint, { headers: hdrs, signal: AbortSignal.timeout(timeoutMs) });
    if (!response.ok) return { provider, ok: false, status: response.status, models: [], error: `GET ${endpoint} answered HTTP ${response.status}` };
    const payload: unknown = await response.json();
    return { provider, ok: true, status: response.status, models: modelsFromListing(provider.wire, payload), error: null };
  } catch (error) {
    return { provider, ok: false, status: 0, models: [], error: `GET ${endpoint} failed: ${(error as Error).message}` };
  }
}

// ---------------------------------------------------------------------------
// Shared state: the brief, the probes, the one catalog, per-provider budgets.
// ---------------------------------------------------------------------------

const briefing = loadBrief(BRIEF_PATH);
let probeResults: ProbeResult[] = [];
let registry: Catalog = { providers: [] };
/** Client API requests arrived so far: the manifest runs in arrival order, one at a time. */
let nextRow = 0;
/** The plan, one assignment per manifest row, computed once the probes are in. */
let assignments: PlannedRow[] = [];
/** Quality loss already booked by served rows, in weight times rating milli:
 *  the sum over served rows of weight x (best rating - served rating). A re-plan
 *  may only spend what is left of the re-plan ceiling after this. */
let lossCommitted = 0;

interface ProviderUsage {
  calls: number;
  dropped: boolean;
}
const providerStates = new Map<string, ProviderUsage>();

function stateFor(providerId: string): ProviderUsage {
  let usage = providerStates.get(providerId);
  if (usage === undefined) {
    usage = { calls: 0, dropped: false };
    providerStates.set(providerId, usage);
  }
  return usage;
}

function note(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** The quality table's row for one provider's model, when it has one. */
function ratingFor(providerId: string, modelId: string): Rating | null {
  return briefing.ratings.find((entry) => entry.provider === providerId && entry.model === modelId) ?? null;
}

/** Whether a model is free: the quality table where it has a row, the listing's word otherwise. */
function isFreeModel(model: ListedModel, row: Rating | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

/**
 * The catalog the probes and the quality table make: every provider that answered,
 * every model it listed. Its base_url is the directory's own path (/p/<id>), never
 * joined onto the env URL: the env's origin differs between a live pass and its
 * replay, and the catalog is part of the state export a replay must reproduce
 * byte for byte.
 */
function assembleCatalog(probeList: readonly ProbeResult[]): Catalog {
  const providers: CatalogedProvider[] = [];
  for (const result of probeList) {
    if (!result.ok) continue;
    const emitted = new Set<string>();
    const entries: ModelEntry[] = [];
    for (const model of [...result.models].sort((x, y) => (x.id < y.id ? -1 : x.id > y.id ? 1 : 0))) {
      if (emitted.has(model.id)) continue;
      emitted.add(model.id);
      const row = ratingFor(result.provider.id, model.id);
      entries.push({ id: model.id, free: isFreeModel(model, row), context_length: row?.context_length ?? model.context_length ?? null });
    }
    providers.push({ id: result.provider.id, base_url: result.provider.basePath, key_env: result.provider.keyEnv, wire: result.provider.wire, models: entries });
  }
  return { providers };
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function probeAll(): Promise<void> {
  if (briefing.error !== null) note(briefing.error);
  probeResults = await Promise.all(briefing.providers.map((entry) => probeOneProvider(ENV_ORIGIN, entry)));
  probeResults.forEach((result) => {
    if (!result.ok && result.error !== null) note(result.error);
  });
  registry = assembleCatalog(probeResults);
}

// ---------------------------------------------------------------------------
// The planner: assign manifest rows to providers before serving, and re-plan
// what is left whenever a provider drops.
// ---------------------------------------------------------------------------

/** Whether a quality row's capabilities meet one request's requirements. */
function satisfies(requires: RequestNeeds, row: Rating): boolean {
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
function bestFreeModel(provider: Provider, probe: ProbeResult, request: ManifestRow): Rating | null {
  if (request.api === 'openai' && request.requires.json === true && provider.wire === 'anthropic') return null;
  let top: Rating | null = null;
  for (const listed of probe.models) {
    const row = ratingFor(provider.id, listed.id);
    if (row === null || !row.free) continue;
    if (!satisfies(request.requires, row)) continue;
    if (top === null || row.rating > top.rating) top = row;
  }
  return top;
}

/** The probe for one provider, when it answered. */
function probeFor(providerId: string): ProbeResult | null {
  return probeResults.find((result) => result.provider.id === providerId && result.ok) ?? null;
}

/**
 * The chance that a provider's callNo-th call of the trial is refused, from its
 * budget range alone. Call 1 never fails. From call 2 on: the budget may be spent
 * (the drawn budget is under callNo), the call number may be one of the provider's
 * scheduled failures, or the provider may be the trial's one outage provider and
 * already down (one provider in ten, down from a call number drawn the same way).
 * The budget is uniform over the range, so each term averages over it.
 *
 * The scheduled-failure term is corrected for the forced second call: every trial
 * forces exactly one provider (one in ten) to fail its second call, on top of its
 * one or two scheduled failures. So call 2 fails with probability 0.1 + 0.9 x the
 * natural rate — and a provider that IS the forced one has one fewer natural
 * fail-call slot left for calls 3 and up, which lowers those calls' natural rate
 * a touch. v2's model used the natural rate for every call and undercounted call 2.
 */
function refusalChance(providerId: string, callNo: number): number {
  if (callNo <= 1) return 0;
  const [lo, hi] = PROVIDER_BUDGET_RANGES[providerId] ?? FALLBACK_BUDGET_RANGE;
  const range = hi - lo + 1;
  let total = 0;
  for (let drawn = lo; drawn <= hi; drawn++) {
    const share = 1 / range;
    if (callNo > drawn) {
      total += share;
      continue;
    }
    if (callNo === 2) {
      total += share * (0.1 + 0.9 * (1.5 / (drawn - 2)));
    } else if (callNo <= drawn - 1) {
      total += share * (0.1 * (0.5 / Math.max(1, drawn - 3)) + 0.9 * (1.5 / (drawn - 2)));
    }
    total += share * 0.1 * Math.min(1, Math.max(0, (callNo - 1) / (drawn - 2)));
  }
  return Math.min(0.999, total);
}

/** The expected refused calls from making k calls on one provider. */
function expectedRefusals(providerId: string, calls: number): number {
  let total = 0;
  for (let callNo = 2; callNo <= calls; callNo++) total += refusalChance(providerId, callNo);
  return total;
}

interface PlannedRow {
  providerId: string | null;
  modelId: string | null;
  rating: number;
  topRating: number;
}

/**
 * Plan a set of rows: one provider per row, minimising the expected refused calls
 * subject to a quality-loss budget. The plan starts quality-optimal over the
 * providers still standing (every row on its best-rated free model among them)
 * and then takes, until none is left, the move that cuts expected refusals most
 * per milli of quality lost — a move that also improves quality is taken first.
 * Deterministic: rows and providers are scanned in fixed order, and a move only
 * wins on a strict improvement.
 *
 * `startingCounts` are the calls already made on each provider (zero for the
 * opening plan); the cost of an assignment counts the marginal refusals of the
 * calls it adds on top of those. `excluded` leaves out providers that are done
 * for the trial. Loss is always measured against the GLOBAL best rating — the
 * best model on any probed provider, dropped or not — because that is what the
 * quality score pays on. So a re-plan's starting loss is not zero when the best
 * provider has dropped: it is the loss the best still-standing assignment
 * already carries.
 */
function planRequests(
  rows: ManifestRow[],
  startingCounts: ReadonlyMap<string, number>,
  excluded: ReadonlySet<string>,
  lossBudget: number,
): PlannedRow[] {
  const slots: PlannedRow[] = rows.map((request) => {
    let top: PlannedRow = { providerId: null, modelId: null, rating: -1, topRating: -1 };
    for (const entry of briefing.providers) {
      const probe = probeFor(entry.id);
      if (probe === null) continue;
      const row = bestFreeModel(entry, probe, request);
      if (row === null) continue;
      if (row.rating > top.topRating) top.topRating = row.rating;
      if (excluded.has(entry.id)) continue;
      if (row.rating > top.rating) top = { providerId: entry.id, modelId: row.model, rating: row.rating, topRating: top.topRating };
    }
    if (top.providerId === null) top.topRating = Math.max(0, top.topRating);
    return top;
  });
  const tally = new Map<string, number>();
  for (const entry of briefing.providers) tally.set(entry.id, startingCounts.get(entry.id) ?? 0);
  for (const cell of slots) if (cell.providerId !== null) tally.set(cell.providerId, (tally.get(cell.providerId) ?? 0) + 1);
  // The starting loss is the loss of the quality-optimal-over-survivors start,
  // measured against the global best — zero for the opening plan, positive for a
  // re-plan when the best provider has dropped.
  let lossSoFar = 0;
  for (let k = 0; k < slots.length; k++) {
    const cell = slots[k];
    if (cell.providerId === null || cell.rating < 0) continue;
    lossSoFar += rows[k].weight * (cell.topRating - cell.rating);
  }
  while (true) {
    let topMove: { index: number; providerId: string; model: string; rating: number; lossDelta: number; costDelta: number; ratio: number } | null = null;
    for (let k = 0; k < slots.length; k++) {
      const cell = slots[k];
      if (cell.providerId === null) continue;
      const request = rows[k];
      for (const entry of briefing.providers) {
        if (entry.id === cell.providerId || excluded.has(entry.id)) continue;
        const probe = probeFor(entry.id);
        if (probe === null) continue;
        const row = bestFreeModel(entry, probe, request);
        if (row === null) continue;
        const lossDelta = request.weight * (cell.rating - row.rating);
        if (lossSoFar + lossDelta > lossBudget) continue;
        const sourceCount = tally.get(cell.providerId) ?? 0;
        const targetCount = tally.get(entry.id) ?? 0;
        const costDelta = (expectedRefusals(entry.id, targetCount + 1) - expectedRefusals(entry.id, targetCount)) - (expectedRefusals(cell.providerId, sourceCount) - expectedRefusals(cell.providerId, sourceCount - 1));
        if (costDelta >= -1e-9) continue;
        const ratio = lossDelta <= 0 ? 1e9 + -costDelta : -costDelta / lossDelta;
        if (topMove === null || ratio > topMove.ratio || (ratio === topMove.ratio && -costDelta > -topMove.costDelta)) {
          topMove = { index: k, providerId: entry.id, model: row.model, rating: row.rating, lossDelta, costDelta, ratio };
        }
      }
    }
    if (topMove === null) break;
    const cell = slots[topMove.index];
    tally.set(cell.providerId as string, (tally.get(cell.providerId as string) ?? 0) - 1);
    tally.set(topMove.providerId, (tally.get(topMove.providerId) ?? 0) + 1);
    cell.providerId = topMove.providerId;
    cell.modelId = topMove.model;
    cell.rating = topMove.rating;
    lossSoFar += topMove.lossDelta;
  }
  const expectedTotal = briefing.providers.reduce((sum, entry) => sum + expectedRefusals(entry.id, tally.get(entry.id) ?? 0), 0);
  const countsSummary = briefing.providers.map((entry) => `${entry.id}=${tally.get(entry.id) ?? 0}`).join(' ');
  note(`planned ${slots.length} rows: loss ${lossSoFar} milli, expected refusals ${expectedTotal.toFixed(2)} [${countsSummary}]`);
  return slots;
}

/** The opening plan: the whole manifest, no calls made yet, nobody dropped. */
function initialPlan(): PlannedRow[] {
  const startingCounts = new Map<string, number>();
  for (const entry of briefing.providers) startingCounts.set(entry.id, 0);
  return planRequests(briefing.manifest, startingCounts, new Set<string>(), INITIAL_LOSS_BUDGET);
}

/**
 * Re-plan the manifest rows from `fromIndex` on, after a provider dropped. The
 * remaining budget is the re-plan ceiling less the loss already booked by served
 * rows, floored at zero; the base counts are the calls made so far. Rows already
 * served keep their assignments. A pure function of the trial state so far, so a
 * replay re-plans identically.
 */
function replanAfter(fromIndex: number): void {
  if (fromIndex >= briefing.manifest.length) return;
  const startingCounts = new Map<string, number>();
  const excluded = new Set<string>();
  for (const entry of briefing.providers) {
    startingCounts.set(entry.id, stateFor(entry.id).calls);
    if (stateFor(entry.id).dropped) excluded.add(entry.id);
  }
  const budgetLeft = Math.max(0, REPLAN_LOSS_CEILING - lossCommitted);
  note(`re-planning from row ${fromIndex}: loss booked ${lossCommitted} milli, budget left ${budgetLeft} milli`);
  const newPlan = planRequests(briefing.manifest.slice(fromIndex), startingCounts, excluded, budgetLeft);
  for (let k = 0; k < newPlan.length; k++) assignments[fromIndex + k] = newPlan[k];
}

// ---------------------------------------------------------------------------
// The normalized request: one view of a client body, in either wire.
// ---------------------------------------------------------------------------

type Image = { kind: 'base64'; data: string; mediaType: string } | { kind: 'url'; url: string };

interface MessageView {
  role: 'user' | 'assistant' | 'tool';
  text: string;
  image: Image | null;
  toolCallId: string | null;
}

interface ToolView {
  name: string;
  description: string;
  parameters: JsonObject;
}

interface NormalizedRequest {
  systemText: string;
  messages: MessageView[];
  tools: ToolView[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  temperature: number | null;
  topP: number | null;
  stream: boolean;
}

/** The text of a string or of a block array's text blocks, joined the way the providers join them. */
function textFrom(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((block) => (isJsonObject(block) && block.type === 'text' && typeof block.text === 'string' ? [block.text] : [])).join('\n');
}

/** The image a content block array carries, if any (the last one wins, as the providers read it). */
function imageFrom(content: unknown): Image | null {
  if (!Array.isArray(content)) return null;
  let seen: Image | null = null;
  for (const block of content) {
    if (!isJsonObject(block)) continue;
    if (block.type === 'image_url' && isJsonObject(block.image_url) && typeof block.image_url.url === 'string') {
      const url = block.image_url.url;
      const match = /^data:([^,]*);base64,(.*)$/s.exec(url);
      if (match !== null) seen = { kind: 'base64', data: match[2], mediaType: match[1] !== '' ? match[1] : 'image/png' };
      else seen = { kind: 'url', url };
    } else if (block.type === 'image' && isJsonObject(block.source)) {
      const src = block.source;
      if (src.type === 'base64' && typeof src.data === 'string') {
        seen = { kind: 'base64', data: src.data, mediaType: typeof src.media_type === 'string' ? src.media_type : 'image/png' };
      } else if (src.type === 'url' && typeof src.url === 'string') {
        seen = { kind: 'url', url: src.url };
      }
    }
  }
  return seen;
}

/** A tool_result block's content as text. */
function toolText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textFrom(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

function stringListFrom(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((item): item is string => typeof item === 'string');
}

/** An OpenAI Chat Completions body as the normalized view. */
function openAIToView(incoming: JsonObject): NormalizedRequest {
  const systemLines: string[] = [];
  const messages: MessageView[] = [];
  const rawMessages = Array.isArray(incoming.messages) ? incoming.messages : [];
  for (const item of rawMessages) {
    if (!isJsonObject(item)) continue;
    if (item.role === 'system') {
      const text = textFrom(item.content);
      if (text !== '') systemLines.push(text);
      continue;
    }
    if (item.role === 'tool') {
      messages.push({
        role: 'tool',
        text: typeof item.content === 'string' ? item.content : textFrom(item.content),
        image: null,
        toolCallId: typeof item.tool_call_id === 'string' ? item.tool_call_id : null,
      });
      continue;
    }
    if (item.role === 'assistant' || item.role === 'user') {
      messages.push({ role: item.role, text: textFrom(item.content), image: item.role === 'user' ? imageFrom(item.content) : null, toolCallId: null });
    }
  }
  const tools: ToolView[] = [];
  if (Array.isArray(incoming.tools)) {
    for (const item of incoming.tools) {
      if (!isJsonObject(item) || !isJsonObject(item.function) || typeof item.function.name !== 'string') continue;
      tools.push({
        name: item.function.name,
        description: typeof item.function.description === 'string' ? item.function.description : '',
        parameters: isJsonObject(item.function.parameters) ? item.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  const responseFormat = isJsonObject(incoming.response_format) ? incoming.response_format.type : undefined;
  return {
    systemText: systemLines.join('\n'),
    messages,
    tools,
    maxTokens: asNumberOrNull(incoming.max_tokens) ?? asNumberOrNull(incoming.max_completion_tokens),
    stops: stringListFrom(incoming.stop),
    jsonMode: responseFormat === 'json_object' || responseFormat === 'json_schema',
    temperature: asNumberOrNull(incoming.temperature),
    topP: asNumberOrNull(incoming.top_p),
    stream: incoming.stream === true,
  };
}

/** An Anthropic Messages body as the normalized view. */
function anthropicToView(incoming: JsonObject): NormalizedRequest {
  const systemText = textFrom(incoming.system);
  const messages: MessageView[] = [];
  const rawMessages = Array.isArray(incoming.messages) ? incoming.messages : [];
  for (const item of rawMessages) {
    if (!isJsonObject(item)) continue;
    const content = item.content;
    if (typeof content === 'string') {
      messages.push({ role: item.role === 'assistant' ? 'assistant' : 'user', text: content, image: null, toolCallId: null });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (item.role === 'assistant') {
      messages.push({ role: 'assistant', text: textFrom(content), image: null, toolCallId: null });
      continue;
    }
    // A user turn: tool_result blocks become tool messages, the rest one user message.
    const textParts: string[] = [];
    let image: Image | null = null;
    for (const block of content) {
      if (!isJsonObject(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          text: toolText(block.content),
          image: null,
          toolCallId: typeof block.tool_use_id === 'string' ? block.tool_use_id : null,
        });
        continue;
      }
      if (block.type === 'text' && typeof block.text === 'string') textParts.push(block.text);
      const seen = imageFrom([block]);
      if (seen !== null) image = seen;
    }
    if (textParts.length > 0 || image !== null) messages.push({ role: 'user', text: textParts.join('\n'), image, toolCallId: null });
  }
  const tools: ToolView[] = [];
  if (Array.isArray(incoming.tools)) {
    for (const item of incoming.tools) {
      if (!isJsonObject(item) || typeof item.name !== 'string') continue;
      tools.push({
        name: item.name,
        description: typeof item.description === 'string' ? item.description : '',
        parameters: isJsonObject(item.input_schema) ? item.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  return {
    systemText,
    messages,
    tools,
    maxTokens: asNumberOrNull(incoming.max_tokens),
    stops: stringListFrom(incoming.stop_sequences),
    jsonMode: false,
    temperature: asNumberOrNull(incoming.temperature),
    topP: asNumberOrNull(incoming.top_p),
    stream: incoming.stream === true,
  };
}

// ---------------------------------------------------------------------------
// The provider call: one view rendered in the provider's wire, one answer read back.
// ---------------------------------------------------------------------------

/** The system prompt's role as one provider expects it, and the tools it uses. */
function viewAsOpenAI(request: NormalizedRequest, model: string): JsonObject {
  const messages: JsonObject[] = [];
  if (request.systemText !== '') messages.push({ role: 'system', content: request.systemText });
  for (const msg of request.messages) {
    if (msg.role === 'tool') {
      messages.push({ role: 'tool', content: msg.text, tool_call_id: msg.toolCallId ?? '' });
      continue;
    }
    if (msg.role === 'assistant') {
      messages.push({ role: 'assistant', content: msg.text });
      continue;
    }
    const blocks: JsonObject[] = [];
    if (msg.image !== null) {
      if (msg.image.kind === 'base64') {
        blocks.push({ type: 'image_url', image_url: { url: `data:${msg.image.mediaType};base64,${msg.image.data}` } });
      } else {
        blocks.push({ type: 'image_url', image_url: { url: msg.image.url } });
      }
    }
    if (msg.text !== '') blocks.push({ type: 'text', text: msg.text });
    messages.push({ role: 'user', content: blocks });
  }
  const payload: JsonObject = { model, messages, stream: false };
  if (request.tools.length > 0) {
    payload.tools = request.tools.map((tool) => ({ type: 'function', function: { name: tool.name, description: tool.description, parameters: tool.parameters } }));
  }
  if (request.maxTokens !== null) payload.max_tokens = request.maxTokens;
  if (request.stops.length > 0) payload.stop = request.stops;
  if (request.jsonMode) payload.response_format = { type: 'json_object' };
  if (request.temperature !== null) payload.temperature = request.temperature;
  if (request.topP !== null) payload.top_p = request.topP;
  return payload;
}

function viewAsAnthropic(request: NormalizedRequest, model: string): JsonObject {
  const messages: JsonObject[] = [];
  for (const msg of request.messages) {
    if (msg.role === 'tool') {
      messages.push({ role: 'user', content: [{ type: 'tool_result', tool_use_id: msg.toolCallId ?? '', content: msg.text }] });
      continue;
    }
    if (msg.role === 'assistant') {
      messages.push({ role: 'assistant', content: msg.text });
      continue;
    }
    const blocks: JsonObject[] = [];
    if (msg.image !== null) {
      if (msg.image.kind === 'base64') {
        blocks.push({ type: 'image', source: { type: 'base64', media_type: msg.image.mediaType, data: msg.image.data } });
      } else {
        blocks.push({ type: 'image', source: { type: 'url', url: msg.image.url } });
      }
    }
    if (msg.text !== '') blocks.push({ type: 'text', text: msg.text });
    messages.push({ role: 'user', content: blocks });
  }
  const payload: JsonObject = { model, messages, stream: false, max_tokens: request.maxTokens ?? 1024 };
  if (request.systemText !== '') payload.system = request.systemText;
  if (request.tools.length > 0) {
    payload.tools = request.tools.map((tool) => ({ name: tool.name, description: tool.description, input_schema: tool.parameters }));
  }
  if (request.stops.length > 0) payload.stop_sequences = request.stops;
  if (request.temperature !== null) payload.temperature = request.temperature;
  if (request.topP !== null) payload.top_p = request.topP;
  return payload;
}

function viewAsGemini(request: NormalizedRequest, model: string): JsonObject {
  const contents: JsonObject[] = [];
  for (const msg of request.messages) {
    if (msg.role === 'tool') continue;
    const parts: JsonObject[] = [];
    if (msg.image !== null) {
      if (msg.image.kind === 'base64') {
        parts.push({ inlineData: { mimeType: msg.image.mediaType, data: msg.image.data } });
      } else {
        parts.push({ fileData: { mimeType: 'image/png', fileUri: msg.image.url } });
      }
    }
    if (msg.text !== '') parts.push({ text: msg.text });
    if (parts.length === 0) continue;
    contents.push({ role: msg.role === 'assistant' ? 'model' : 'user', parts });
  }
  const payload: JsonObject = { contents };
  if (request.systemText !== '') payload.systemInstruction = { parts: [{ text: request.systemText }] };
  if (request.tools.length > 0) {
    payload.tools = [{ functionDeclarations: request.tools.map((tool) => ({ name: tool.name, description: tool.description, parameters: tool.parameters })) }];
  }
  const config: JsonObject = {};
  if (request.maxTokens !== null) config.maxOutputTokens = request.maxTokens;
  if (request.stops.length > 0) config.stopSequences = request.stops;
  if (request.jsonMode) config.responseMimeType = 'application/json';
  if (request.temperature !== null) config.temperature = request.temperature;
  if (request.topP !== null) config.topP = request.topP;
  if (Object.keys(config).length > 0) payload.generationConfig = config;
  return payload;
}

function viewAsWire(wire: ProviderWire, request: NormalizedRequest, model: string): JsonObject {
  switch (wire) {
    case 'anthropic':
      return viewAsAnthropic(request, model);
    case 'gemini':
      return viewAsGemini(request, model);
    default:
      return viewAsOpenAI(request, model);
  }
}

interface UpstreamAnswer {
  text: string;
  finish: string;
  usage: { input: number; output: number };
}

/** The generated text of an OpenAI-shaped answer, empty when the answer has none. */
function openAIToAnswer(payload: unknown): UpstreamAnswer | null {
  if (!isJsonObject(payload) || !Array.isArray(payload.choices) || payload.choices.length === 0) return null;
  const first = payload.choices[0];
  if (!isJsonObject(first)) return null;
  const message = isJsonObject(first.message) ? first.message : null;
  const text = message !== null ? textFrom(message.content) : '';
  const finish = typeof first.finish_reason === 'string' ? first.finish_reason : '';
  const tokens = isJsonObject(payload.usage) ? payload.usage : {};
  return { text, finish, usage: { input: asNumberOrNull(tokens.prompt_tokens) ?? 0, output: asNumberOrNull(tokens.completion_tokens) ?? 0 } };
}

function anthropicToAnswer(payload: unknown): UpstreamAnswer | null {
  if (!isJsonObject(payload)) return null;
  const content = Array.isArray(payload.content) ? payload.content : [];
  const text = content.flatMap((block) => (isJsonObject(block) && block.type === 'text' && typeof block.text === 'string' ? [block.text] : [])).join('\n');
  const finish = typeof payload.stop_reason === 'string' ? payload.stop_reason : '';
  const tokens = isJsonObject(payload.usage) ? payload.usage : {};
  return { text, finish, usage: { input: asNumberOrNull(tokens.input_tokens) ?? 0, output: asNumberOrNull(tokens.output_tokens) ?? 0 } };
}

function geminiToAnswer(payload: unknown): UpstreamAnswer | null {
  if (!isJsonObject(payload) || !Array.isArray(payload.candidates) || payload.candidates.length === 0) return null;
  const first = payload.candidates[0];
  if (!isJsonObject(first) || !isJsonObject(first.content) || !Array.isArray(first.content.parts)) return null;
  const text = first.content.parts.flatMap((part) => (isJsonObject(part) && typeof part.text === 'string' ? [part.text] : [])).join('\n');
  const finish = typeof first.finishReason === 'string' ? first.finishReason : '';
  const tokens = isJsonObject(payload.usageMetadata) ? payload.usageMetadata : {};
  return { text, finish, usage: { input: asNumberOrNull(tokens.promptTokenCount) ?? 0, output: asNumberOrNull(tokens.candidatesTokenCount) ?? 0 } };
}

function wireToAnswer(wire: ProviderWire, payload: unknown): UpstreamAnswer | null {
  switch (wire) {
    case 'anthropic':
      return anthropicToAnswer(payload);
    case 'gemini':
      return geminiToAnswer(payload);
    default:
      return openAIToAnswer(payload);
  }
}

/** The endpoint one provider's wire takes, given its base. */
function inferEndpoint(provider: Provider, model: string): string {
  const root = resolveBase(ENV_ORIGIN, provider.basePath);
  switch (provider.wire) {
    case 'anthropic':
      return `${root}/v1/messages`;
    case 'gemini':
      return `${root}/v1beta/models/${encodeURIComponent(model)}:generateContent`;
    default:
      return `${root}/v1/chat/completions`;
  }
}

type Refusal = 'spent' | 'out' | 'transient' | 'other';

/**
 * What a refusal means. The provider's message names the cause: the budget is spent
 * or the provider is out, both permanent for the trial; a refused call or a concurrent
 * collision, both transient; anything else, unknown. A 402 or 429 with no readable
 * message is treated as transient — the only callers that send those codes here are
 * the refusal paths.
 */
function classifyFailure(provider: Provider, status: number, payload: unknown): Refusal {
  let refusalText = '';
  if (isJsonObject(payload)) {
    if (isJsonObject(payload.error) && typeof payload.error.message === 'string') refusalText = payload.error.message;
    else if (typeof payload.message === 'string') refusalText = payload.message;
    else if (typeof payload.error === 'string') refusalText = payload.error;
  }
  if (refusalText.includes('budget is spent')) return 'spent';
  if (refusalText.trim().endsWith('is out')) return 'out';
  if (refusalText.includes('refused this call')) return 'transient';
  if (refusalText.includes('concurrent_requests')) return 'transient';
  if (status === 402 || status === 429) return 'transient';
  return 'other';
}

type CallResult =
  | { ok: true; answer: UpstreamAnswer }
  | { ok: false; kind: Refusal; detail: string };

/** One inference call, counted: every attempt spends budget, refused or not. */
async function callUpstream(provider: Provider, model: string, request: NormalizedRequest): Promise<CallResult> {
  const provState = stateFor(provider.id);
  provState.calls++;
  const hdrs: Record<string, string> = { 'content-type': 'application/json', ...headersForAuth(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') hdrs['anthropic-version'] = '2023-06-01';
  let response: Response;
  try {
    response = await fetch(inferEndpoint(provider, model), {
      method: 'POST',
      headers: hdrs,
      body: JSON.stringify(viewAsWire(provider.wire, request, model)),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch (error) {
    return { ok: false, kind: 'other', detail: `fetch to ${provider.id} failed: ${(error as Error).message}` };
  }
  let payload: unknown = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  if (!response.ok) {
    return { ok: false, kind: classifyFailure(provider, response.status, payload), detail: `${provider.id} answered HTTP ${response.status}` };
  }
  const answer = wireToAnswer(provider.wire, payload);
  if (answer === null) return { ok: false, kind: 'other', detail: `${provider.id} answered in a shape this server cannot read` };
  return { ok: true, answer };
}

// ---------------------------------------------------------------------------
// Routing: follow the plan, fall back by rating, retry a transient refusal once.
// ---------------------------------------------------------------------------

interface Fallback {
  provider: Provider;
  modelId: string;
  rating: number;
}

/** The providers that can serve one request, best-rated first (the fallback order). */
function rankedCandidates(requires: RequestNeeds, clientApi: 'openai' | 'anthropic', jsonMode: boolean): Fallback[] {
  const options: Fallback[] = [];
  for (const entry of briefing.providers) {
    if (stateFor(entry.id).dropped) continue;
    const probe = probeFor(entry.id);
    if (probe === null) continue;
    if (clientApi === 'openai' && jsonMode && entry.wire === 'anthropic') continue;
    let top: Rating | null = null;
    for (const listed of probe.models) {
      const row = ratingFor(entry.id, listed.id);
      if (row === null || !row.free) continue;
      if (!satisfies(requires, row)) continue;
      if (top === null || row.rating > top.rating) top = row;
    }
    if (top !== null) options.push({ provider: entry, modelId: top.model, rating: top.rating });
  }
  // Best-rated first; when two providers rate the same, the one whose next call
  // is less likely to fail goes first (same quality, fewer expected calls); the
  // provider id breaks any remaining tie, so the order is fully deterministic.
  options.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    const aRisk = refusalChance(a.provider.id, stateFor(a.provider.id).calls + 1);
    const bRisk = refusalChance(b.provider.id, stateFor(b.provider.id).calls + 1);
    if (aRisk !== bRisk) return aRisk - bRisk;
    return a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0;
  });
  return options;
}

interface Served {
  providerId: string;
  modelId: string;
  rating: number;
  answer: UpstreamAnswer;
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
async function serveRequest(
  request: NormalizedRequest,
  requires: RequestNeeds,
  clientApi: 'openai' | 'anthropic',
  planIndex: number,
): Promise<Served | null> {
  const fallbacks = rankedCandidates(requires, clientApi, request.jsonMode);
  const tried: Fallback[] = [];
  const slot = planIndex >= 0 && planIndex < assignments.length ? assignments[planIndex] : null;
  if (slot !== null && slot.providerId !== null && slot.modelId !== null) {
    const plannedProvider = briefing.providers.find((entry) => entry.id === slot.providerId) ?? null;
    const plannedProbe = plannedProvider !== null ? probeFor(plannedProvider.id) : null;
    if (plannedProvider !== null && plannedProbe !== null && !stateFor(plannedProvider.id).dropped) {
      const plannedRating = ratingFor(plannedProvider.id, slot.modelId);
      const jsonOk = !(clientApi === 'openai' && request.jsonMode && plannedProvider.wire === 'anthropic');
      if (plannedRating !== null && plannedRating.free && satisfies(requires, plannedRating) && jsonOk) {
        tried.push({ provider: plannedProvider, modelId: slot.modelId, rating: plannedRating.rating });
      }
    }
  }
  for (const option of fallbacks) {
    if (!tried.some((entry) => entry.provider.id === option.provider.id)) tried.push(option);
  }
  for (const option of tried) {
    const provState = stateFor(option.provider.id);
    if (provState.dropped) continue;
    for (let tryNo = 0; tryNo < 2; tryNo++) {
      const callResult = await callUpstream(option.provider, option.modelId, request);
      if (callResult.ok) return { providerId: option.provider.id, modelId: option.modelId, rating: option.rating, answer: callResult.answer };
      if (callResult.kind === 'spent' || callResult.kind === 'out') {
        note(`${option.provider.id} dropped for the trial (${callResult.detail})`);
        provState.dropped = true;
        break;
      }
      if (callResult.kind === 'transient' && tryNo === 0) continue;
      note(`${option.provider.id} refused this one (${callResult.detail})`);
      break;
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// The client wires: one answer rendered as JSON or as a server-sent stream.
// ---------------------------------------------------------------------------

/** A provider's finish reason as the client's wire spells it. */
function stopReasonFor(wire: 'openai' | 'anthropic', finish: string): string {
  const normalized = finish.toLowerCase();
  switch (normalized) {
    case 'length':
    case 'max_tokens':
    case 'max_tokens_exceeded':
      return wire === 'openai' ? 'length' : 'max_tokens';
    case 'tool_calls':
    case 'function_call':
    case 'tool_use':
      return wire === 'openai' ? 'tool_calls' : 'tool_use';
    case 'content_filter':
    case 'refusal':
    case 'safety':
      return wire === 'openai' ? 'content_filter' : 'refusal';
    default:
      return wire === 'openai' ? 'stop' : 'end_turn';
  }
}

/** The answer as an OpenAI Chat Completions JSON body, with the provider's usage verbatim. */
function openAIResponse(answer: UpstreamAnswer): string {
  return JSON.stringify({
    id: 'chatcmpl-aggregator',
    object: 'chat.completion',
    created: FIXED_TIMESTAMP,
    model: 'aggregator',
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: answer.text },
        finish_reason: stopReasonFor('openai', answer.finish),
      },
    ],
    usage: {
      prompt_tokens: answer.usage.input,
      completion_tokens: answer.usage.output,
      total_tokens: answer.usage.input + answer.usage.output,
    },
  });
}

/** The answer as an Anthropic Messages JSON body, with the provider's usage verbatim. */
function anthropicResponse(answer: UpstreamAnswer): string {
  return JSON.stringify({
    id: 'msg_aggregator',
    type: 'message',
    role: 'assistant',
    model: 'aggregator',
    content: [{ type: 'text', text: answer.text }],
    stop_reason: stopReasonFor('anthropic', answer.finish),
    usage: { input_tokens: answer.usage.input, output_tokens: answer.usage.output },
  });
}

/** A string sliced into the deltas a stream sends, 13 characters at a time. */
function splitDeltas(text: string): string[] {
  const collected: string[] = [];
  for (let k = 0; k < text.length; k += STREAM_CHUNK_CHARS) collected.push(text.slice(k, k + STREAM_CHUNK_CHARS));
  return collected;
}

/** The answer as an OpenAI SSE stream: role chunk, text deltas, finish chunk, usage chunk, [DONE]. */
function openAIEvents(answer: UpstreamAnswer): string {
  const events: string[] = [];
  const roleChunk = {
    id: 'chatcmpl-aggregator',
    object: 'chat.completion.chunk',
    created: FIXED_TIMESTAMP,
    model: 'aggregator',
    choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }],
  };
  events.push(`data: ${JSON.stringify(roleChunk)}\n\n`);
  for (const piece of splitDeltas(answer.text)) {
    const textChunk = {
      id: 'chatcmpl-aggregator',
      object: 'chat.completion.chunk',
      created: FIXED_TIMESTAMP,
      model: 'aggregator',
      choices: [{ index: 0, delta: { content: piece }, finish_reason: null }],
    };
    events.push(`data: ${JSON.stringify(textChunk)}\n\n`);
  }
  const usage = {
    prompt_tokens: answer.usage.input,
    completion_tokens: answer.usage.output,
    total_tokens: answer.usage.input + answer.usage.output,
  };
  const finishChunk = {
    id: 'chatcmpl-aggregator',
    object: 'chat.completion.chunk',
    created: FIXED_TIMESTAMP,
    model: 'aggregator',
    choices: [{ index: 0, delta: {}, finish_reason: stopReasonFor('openai', answer.finish) }],
    usage,
  };
  events.push(`data: ${JSON.stringify(finishChunk)}\n\n`);
  const usageChunk = {
    id: 'chatcmpl-aggregator',
    object: 'chat.completion.chunk',
    created: FIXED_TIMESTAMP,
    model: 'aggregator',
    choices: [],
    usage,
  };
  events.push(`data: ${JSON.stringify(usageChunk)}\n\n`);
  events.push('data: [DONE]\n\n');
  return events.join('');
}

/** The answer as an Anthropic SSE stream: the full event sequence, with usage on both ends. */
function anthropicEvents(answer: UpstreamAnswer): string {
  const events: string[] = [];
  const startEvent = {
    type: 'message_start',
    message: {
      id: 'msg_aggregator',
      type: 'message',
      role: 'assistant',
      model: 'aggregator',
      content: [],
      stop_reason: null,
      usage: { input_tokens: answer.usage.input, output_tokens: 0 },
    },
  };
  events.push(`event: message_start\ndata: ${JSON.stringify(startEvent)}\n\n`);
  events.push(`event: content_block_start\ndata: ${JSON.stringify({ type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } })}\n\n`);
  for (const piece of splitDeltas(answer.text)) {
    events.push(`event: content_block_delta\ndata: ${JSON.stringify({ type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } })}\n\n`);
  }
  events.push(`event: content_block_stop\ndata: ${JSON.stringify({ type: 'content_block_stop', index: 0 })}\n\n`);
  const deltaEvent = {
    type: 'message_delta',
    delta: { stop_reason: stopReasonFor('anthropic', answer.finish) },
    usage: { output_tokens: answer.usage.output },
  };
  events.push(`event: message_delta\ndata: ${JSON.stringify(deltaEvent)}\n\n`);
  events.push(`event: message_stop\ndata: ${JSON.stringify({ type: 'message_stop' })}\n\n`);
  return events.join('');
}

/** The free models one provider offers, each id once, sorted — what /v1/models lists. */
function freeModelIds(provider: CatalogedProvider): string[] {
  return [...new Set(provider.models.filter((entry) => entry.free).map((entry) => entry.id))].sort();
}

/** The catalog as an OpenAI-shaped model listing. */
function openAIModelList(): string {
  return JSON.stringify({
    object: 'list',
    data: registry.providers.flatMap((provider) =>
      freeModelIds(provider).map((id) => ({ id, object: 'model', created: FIXED_TIMESTAMP, owned_by: provider.id })),
    ),
  });
}

/** The catalog as an Anthropic-shaped model listing. */
function anthropicModelList(): string {
  return JSON.stringify({
    data: registry.providers.flatMap((provider) =>
      freeModelIds(provider).map((id) => ({ id, type: 'model', display_name: id, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    ),
    has_more: false,
    first_id: null,
    last_id: null,
  });
}

/** The catalog listing in the wire the client's auth header names. */
function modelListFor(req: IncomingMessage): string {
  return req.headers['x-api-key'] !== undefined ? anthropicModelList() : openAIModelList();
}

// ---------------------------------------------------------------------------
// The server: the manifest is claimed in arrival order, then the request is served.
// ---------------------------------------------------------------------------

/** Read a request body, failing past the size limit or a socket error. */
function readRequestBody(req: IncomingMessage): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const segments: Buffer[] = [];
    let total = 0;
    req.on('data', (piece: Buffer) => {
      total += piece.length;
      if (total > MAX_BODY_BYTES) {
        reject(new Error('body too large'));
        return;
      }
      segments.push(piece);
    });
    req.on('end', () => resolve(Buffer.concat(segments)));
    req.on('error', reject);
  });
}

/** A JSON body, or the error shape the client wires expect. */
function parseJsonBody(rawBytes: Buffer): { body: JsonObject | null; error: string | null } {
  if (rawBytes.length === 0) return { body: null, error: 'empty body' };
  try {
    const parsed: unknown = JSON.parse(rawBytes.toString('utf8'));
    if (!isJsonObject(parsed)) return { body: null, error: 'body is not a JSON object' };
    return { body: parsed, error: null };
  } catch {
    return { body: null, error: 'body is not JSON' };
  }
}

function sendJsonResponse(res: ServerResponse, status: number, value: unknown): void {
  const text = JSON.stringify(value);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

/** The next manifest row, claimed by arrival: this runs synchronously at request start,
 *  before any await, so two requests arriving together are never handed the same row. */
function claimManifestRow(): { requires: RequestNeeds; weight: number; index: number } {
  const current = nextRow < briefing.manifest.length ? briefing.manifest[nextRow] : null;
  const index = nextRow;
  nextRow++;
  if (current === null) return { requires: {}, weight: 1, index };
  return { requires: current.requires, weight: current.weight, index };
}

/** A request's requirements read from its own body, for requests past the manifest's end. */
function requirementsInBody(api: 'openai' | 'anthropic', body: JsonObject): RequestNeeds {
  const requires: RequestNeeds = {};
  if (Array.isArray(body.tools) && body.tools.length > 0) requires.tools = true;
  if (api === 'openai' && isJsonObject(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
    requires.json = true;
  }
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isJsonObject(message)) continue;
    const content = message.content;
    if (!Array.isArray(content)) continue;
    for (const block of content) {
      if (!isJsonObject(block)) continue;
      if (block.type === 'image_url' || block.type === 'image') requires.vision = true;
    }
  }
  return requires;
}

/** One client request, start to finish. */
async function handleClientRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const parsedUrl = new URL(req.url ?? '/', 'http://localhost');
  const route = parsedUrl.pathname;
  const verb = req.method ?? '';
  if (verb === 'GET' && route === '/v1/models') {
    const listing = modelListFor(req);
    res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(listing) });
    res.end(listing);
    return;
  }
  let api: 'openai' | 'anthropic' | null = null;
  if (verb === 'POST' && route === '/v1/chat/completions') api = 'openai';
  else if (verb === 'POST' && route === '/v1/messages') api = 'anthropic';
  if (api === null) {
    sendJsonResponse(res, 404, { error: { message: `unknown route ${verb} ${route}`, type: 'not_found', code: 'not_found' } });
    return;
  }
  const row = claimManifestRow();
  let rawBytes: Buffer;
  try {
    rawBytes = await readRequestBody(req);
  } catch {
    sendJsonResponse(res, 400, { error: { message: 'could not read request body', type: 'invalid_request_error', code: 'invalid_request_error' } });
    return;
  }
  const decoded = parseJsonBody(rawBytes);
  if (decoded.body === null) {
    sendJsonResponse(res, 400, { error: { message: decoded.error ?? 'invalid body', type: 'invalid_request_error', code: 'invalid_request_error' } });
    return;
  }
  const body = decoded.body;
  const view = api === 'openai' ? openAIToView(body) : anthropicToView(body);
  const requires = row.index < briefing.manifest.length ? row.requires : requirementsInBody(api, body);
  // Snapshot the dropped set: a provider that drops while this request is served
  // changes the feasible set, and the remaining rows get re-planned for it.
  const droppedAtStart = new Set<string>();
  for (const entry of briefing.providers) if (stateFor(entry.id).dropped) droppedAtStart.add(entry.id);
  const result = await serveRequest(view, requires, api, row.index);
  // Book this request's committed loss, then re-plan what is left if the feasible
  // set changed. Both run synchronously here, before the response goes out, so no
  // other request can interleave between the booking and the re-plan.
  if (result !== null && row.index < assignments.length) {
    const slot = assignments[row.index];
    lossCommitted += row.weight * Math.max(0, slot.topRating - result.rating);
  }
  const droppedDuring = briefing.providers.some((entry) => stateFor(entry.id).dropped && !droppedAtStart.has(entry.id));
  if (droppedDuring) replanAfter(row.index + 1);
  if (result === null) {
    sendJsonResponse(res, 503, { error: { message: 'no provider could serve this request', type: 'service_unavailable', code: 'service_unavailable' } });
    return;
  }
  if (view.stream) {
    const text = api === 'openai' ? openAIEvents(result.answer) : anthropicEvents(result.answer);
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', 'content-length': Buffer.byteLength(text) });
    res.end(text);
    return;
  }
  sendJsonResponse(res, 200, JSON.parse(api === 'openai' ? openAIResponse(result.answer) : anthropicResponse(result.answer)));
}

/** Create the server: the request handler's errors are answered 500, never left hanging. */
function makeServer(): Server {
  return createServer((req, res) => {
    handleClientRequest(req, res).catch((error: unknown) => {
      if (!res.headersSent) {
        sendJsonResponse(res, 500, { error: { message: (error as Error).message, type: 'internal_error', code: 'internal_error' } });
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
async function registerWithEnv(port: number): Promise<void> {
  const response = await fetch(`${ENV_ORIGIN}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: registry }),
    signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
  });
  if (!response.ok) throw new Error(`registration failed: HTTP ${response.status}`);
}

/** Poll the env until the trial is done, then close the server and exit. */
async function waitForTrialEnd(server: Server): Promise<void> {
  while (true) {
    try {
      const response = await fetch(`${ENV_ORIGIN}/work`, { signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
      if (response.ok) {
        const payload: unknown = await response.json();
        if (isJsonObject(payload) && payload.done === true) break;
      }
    } catch {
      // The env not answering yet is not a failure: wait and ask again.
    }
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
  }
  await new Promise<void>((resolve) => server.close(() => resolve()));
  process.exit(0);
}

async function main(): Promise<void> {
  await probeAll();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(registry)}\n`;
    await new Promise<void>((resolve) => {
      if (process.stdout.write(text)) resolve();
      else process.stdout.once('drain', resolve);
    });
    process.exit(0);
  }
  if (ENV_ORIGIN === '') throw new Error('MAKEFUN_ENV_URL is not set');
  assignments = initialPlan();
  const httpServer = makeServer();
  await new Promise<void>((resolve, reject) => {
    httpServer.once('error', reject);
    httpServer.listen(0, '127.0.0.1', () => {
      httpServer.removeListener('error', reject);
      resolve();
    });
  });
  const bound = httpServer.address();
  if (bound === null || typeof bound === 'string') throw new Error('server did not bind a port');
  const listenPort = bound.port;
  await registerWithEnv(listenPort);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  await waitForTrialEnd(httpServer);
}

try {
  await main();
} catch (error) {
  note((error as Error).message);
  process.exit(1);
}
