const BASE = '';

// --- Types ---

export interface DataStats {
	total_markets: number;
	total_price_points: number;
}

export interface SyncStatus {
	running: boolean;
	fetched: number;
	skipped: number;
	errors: number;
	error?: string;
}

export interface EpisodeResultSummary {
	episode: number;
	total_reward: number;
	n_steps: number;
}

export interface RunMetrics {
	sharpe_ratio: number;
	win_rate: number;
	avg_pnl: number;
	max_drawdown: number;
	spread_capture: number;
	total_episodes: number;
	total_reward: number;
}

export interface TrainingRun {
	id: string;
	agent_type: string;
	agent_config: Record<string, any>;
	num_episodes: number;
	status: 'pending' | 'running' | 'completed' | 'cancelled' | 'failed';
	created_at: number;
	started_at: number | null;
	finished_at: number | null;
	current_episode: number;
	error: string | null;
	episode_results: EpisodeResultSummary[];
}

export interface AgentInfo {
	name: string;
	description: string;
	config_params: Record<string, any>;
}

// --- Data API ---

export async function fetchDataStats(): Promise<DataStats> {
	const res = await fetch(`${BASE}/api/data/stats`);
	return res.json();
}

export async function startSync(
	limit: number = 100
): Promise<{ status: string }> {
	const res = await fetch(`${BASE}/api/data/sync`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ limit })
	});
	return res.json();
}

export async function fetchSyncStatus(): Promise<SyncStatus> {
	const res = await fetch(`${BASE}/api/data/sync/status`);
	return res.json();
}

// --- Runs API ---

export async function createRun(params: {
	agent_type: string;
	num_episodes: number;
	agent_config?: Record<string, any>;
}): Promise<TrainingRun> {
	const res = await fetch(`${BASE}/api/runs`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(params)
	});
	return res.json();
}

export async function fetchRuns(): Promise<TrainingRun[]> {
	const res = await fetch(`${BASE}/api/runs`);
	const data = await res.json();
	return data.runs;
}

export async function fetchRun(runId: string): Promise<TrainingRun> {
	const res = await fetch(`${BASE}/api/runs/${runId}`);
	return res.json();
}

export async function fetchRunMetrics(runId: string): Promise<RunMetrics> {
	const res = await fetch(`${BASE}/api/runs/${runId}/metrics`);
	return res.json();
}

export async function deleteRun(runId: string): Promise<void> {
	await fetch(`${BASE}/api/runs/${runId}`, { method: 'DELETE' });
}

// --- RL Training API ---

export interface RLTrainStatus {
  running: boolean;
  timesteps: number;
  algorithm: string;
  save_path: string;
  progress: number;
  error: string | null;
}

export interface SavedModel {
  name: string;
  path: string;
  size_bytes: number;
}

export async function startRLTraining(params: {
  timesteps: number;
  algorithm: string;
  save_path?: string;
}): Promise<{ status: string }> {
  const res = await fetch(`${BASE}/api/train/rl`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params)
  });
  return res.json();
}

export async function fetchRLTrainStatus(): Promise<RLTrainStatus> {
  const res = await fetch(`${BASE}/api/train/rl/status`);
  return res.json();
}

export async function fetchModels(): Promise<SavedModel[]> {
  const res = await fetch(`${BASE}/api/models`);
  const data = await res.json();
  return data.models;
}

// --- Agents API ---

export async function fetchAgents(): Promise<Record<string, AgentInfo>> {
	const res = await fetch(`${BASE}/api/agents`);
	const data = await res.json();
	return data.agents;
}

// --- WebSocket ---

export function connectRunWebSocket(
	runId: string,
	onMessage: (data: any) => void
): WebSocket {
	const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
	const ws = new WebSocket(`${protocol}//${window.location.host}/ws/runs/${runId}`);
	ws.onmessage = (event) => {
		const data = JSON.parse(event.data);
		onMessage(data);
	};
	return ws;
}
