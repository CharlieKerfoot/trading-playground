<script lang="ts">
	import { page } from '$app/state';
	import {
		fetchRun,
		fetchRunMetrics,
		connectRunWebSocket,
		type TrainingRun,
		type RunMetrics
	} from '$lib/api';
	import MetricsCard from '$lib/components/MetricsCard.svelte';
	import RunProgress from '$lib/components/RunProgress.svelte';
	import PnlChart from '$lib/components/PnlChart.svelte';
	import { onMount, onDestroy } from 'svelte';

	let run: TrainingRun | null = $state(null);
	let metrics: RunMetrics | null = $state(null);
	let ws: WebSocket | null = $state(null);

	let chartData = $derived.by(() => {
		if (!run?.episode_results) return [];
		let cum = 0;
		return run.episode_results.map((e) => {
			cum += e.total_reward;
			return { episode: e.episode, total_reward: e.total_reward, cumulative: cum };
		});
	});

	onMount(async () => {
		const id = page.params.id!;
		run = await fetchRun(id);
		if (run && run.episode_results?.length > 0) {
			metrics = await fetchRunMetrics(id);
		}

		if (run?.status === 'running') {
			ws = connectRunWebSocket(id, (data) => {
				if (data.type === 'progress') {
					run = {
						...run!,
						current_episode: data.current_episode,
						episode_results: [
							...run!.episode_results,
							{
								episode: data.current_episode,
								total_reward: data.latest_reward,
								n_steps: 0
							}
						]
					};
					if (data.metrics) {
						metrics = data.metrics;
					}
				}
				if (data.type === 'state') {
					run = { ...run!, ...data };
				}
			});
		}
	});

	onDestroy(() => {
		if (ws) ws.close();
	});

	// Poll for completion
	let pollRef: ReturnType<typeof setInterval> | null = null;

	$effect(() => {
		if (run?.status === 'running' && !pollRef) {
			pollRef = setInterval(async () => {
				const id = page.params.id!;
				const updated = await fetchRun(id);
				run = updated;
				if (updated.status !== 'running') {
					if (pollRef) clearInterval(pollRef);
					pollRef = null;
					metrics = await fetchRunMetrics(id);
				}
			}, 2000);
		}
		return () => {
			if (pollRef) clearInterval(pollRef);
		};
	});
</script>

<div class="page">
	{#if !run}
		<p class="dim">Loading...</p>
	{:else}
		<div class="page-header">
			<div>
				<h2>Run {run.id}</h2>
				<span class="dim" style="font-size: 12px">
					{run.agent_type} agent &middot;
					{run.num_episodes} episodes
				</span>
			</div>
			<a href="/runs"><button>Back to Runs</button></a>
		</div>

		<div class="card">
			<RunProgress
				current={run.current_episode}
				total={run.num_episodes}
				status={run.status}
			/>
		</div>

		{#if metrics && Object.keys(metrics).length > 0}
			<div class="metrics-row">
				<div class="card">
					<MetricsCard label="Sharpe Ratio" value={metrics.sharpe_ratio} format="number" />
				</div>
				<div class="card">
					<MetricsCard label="Win Rate" value={metrics.win_rate} format="percent" />
				</div>
				<div class="card">
					<MetricsCard label="Avg P&L" value={metrics.avg_pnl} format="currency" colorize />
				</div>
				<div class="card">
					<MetricsCard label="Max Drawdown" value={metrics.max_drawdown} format="currency" colorize />
				</div>
				<div class="card">
					<MetricsCard label="Spread Capture" value={metrics.spread_capture} format="currency" />
				</div>
			</div>
		{/if}

		<div class="card">
			<h3>Cumulative P&L</h3>
			<PnlChart data={chartData} />
		</div>

		{#if run.episode_results.length > 0}
			<div class="card">
				<h3>Episode Results</h3>
				<div class="episodes-table-wrapper">
					<table class="episodes-table">
						<thead>
							<tr>
								<th>#</th>
								<th>Reward</th>
								<th>Steps</th>
							</tr>
						</thead>
						<tbody>
							{#each [...run.episode_results].reverse() as ep (ep.episode)}
								<tr>
									<td class="dim">{ep.episode}</td>
									<td>
										<span class={ep.total_reward > 0 ? 'green' : ep.total_reward < 0 ? 'red' : 'dim'}>
											{ep.total_reward >= 0 ? '+' : ''}{ep.total_reward.toFixed(4)}
										</span>
									</td>
									<td class="dim">{ep.n_steps}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			</div>
		{/if}

		{#if run.error}
			<div class="card error">
				<h3>Error</h3>
				<pre>{run.error}</pre>
			</div>
		{/if}
	{/if}
</div>

<style>
	.page {
		display: flex;
		flex-direction: column;
		gap: 20px;
	}

	.page-header {
		display: flex;
		justify-content: space-between;
		align-items: flex-start;
	}

	h2 {
		font-size: 16px;
		font-weight: 600;
	}

	.metrics-row {
		display: grid;
		grid-template-columns: repeat(5, 1fr);
		gap: 12px;
	}

	.episodes-table-wrapper {
		max-height: 400px;
		overflow-y: auto;
	}

	.episodes-table {
		width: 100%;
		border-collapse: collapse;
		font-size: 13px;
	}

	.episodes-table th {
		text-align: left;
		font-size: 10px;
		text-transform: uppercase;
		letter-spacing: 1px;
		color: var(--text-dim);
		padding: 8px 12px;
		border-bottom: 1px solid var(--border);
		position: sticky;
		top: 0;
		background: var(--bg-card);
	}

	.episodes-table td {
		padding: 6px 12px;
		border-bottom: 1px solid var(--border);
	}

	.error {
		border-color: var(--red);
	}

	.error pre {
		font-size: 12px;
		color: var(--red);
		white-space: pre-wrap;
	}
</style>
