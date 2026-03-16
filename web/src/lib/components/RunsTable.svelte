<script lang="ts">
	import type { TrainingRun } from '$lib/api';

	let { runs, onSelect, onDelete }: {
		runs: TrainingRun[];
		onSelect: (id: string) => void;
		onDelete: (id: string) => void;
	} = $props();

	function statusColor(status: string): string {
		switch (status) {
			case 'running': return 'var(--blue)';
			case 'completed': return 'var(--green)';
			case 'cancelled': return 'var(--yellow)';
			case 'failed': return 'var(--red)';
			default: return 'var(--text-dim)';
		}
	}

	function formatTime(ts: number | null): string {
		if (!ts) return '—';
		return new Date(ts * 1000).toLocaleString();
	}

	function formatDuration(run: TrainingRun): string {
		if (!run.started_at || !run.finished_at) return '—';
		const secs = run.finished_at - run.started_at;
		if (secs < 60) return `${secs.toFixed(1)}s`;
		return `${(secs / 60).toFixed(1)}m`;
	}
</script>

<div class="runs-table-wrapper">
	{#if runs.length === 0}
		<p class="dim" style="padding: 24px; text-align: center">No training runs yet</p>
	{:else}
		<table class="runs-table">
			<thead>
				<tr>
					<th>ID</th>
					<th>Agent</th>
					<th>Episodes</th>
					<th>Status</th>
					<th>Duration</th>
					<th>Avg P&L</th>
					<th></th>
				</tr>
			</thead>
			<tbody>
				{#each runs as run (run.id)}
					<tr class="clickable" onclick={() => onSelect(run.id)}>
						<td class="dim">{run.id}</td>
						<td>{run.agent_type}</td>
						<td>{run.current_episode}/{run.num_episodes}</td>
						<td>
							<span style="color: {statusColor(run.status)}">{run.status}</span>
						</td>
						<td class="dim">{formatDuration(run)}</td>
						<td>
							{#if run.episode_results.length > 0}
								{@const avg = run.episode_results.reduce((s, e) => s + e.total_reward, 0) / run.episode_results.length}
								<span class={avg > 0 ? 'green' : avg < 0 ? 'red' : 'dim'}>
									{avg.toFixed(4)}
								</span>
							{:else}
								<span class="dim">—</span>
							{/if}
						</td>
						<td>
							<button
								class="delete-btn"
								onclick={(e: MouseEvent) => { e.stopPropagation(); onDelete(run.id); }}
							>
								&times;
							</button>
						</td>
					</tr>
				{/each}
			</tbody>
		</table>
	{/if}
</div>

<style>
	.runs-table-wrapper {
		overflow-x: auto;
	}

	.runs-table {
		width: 100%;
		border-collapse: collapse;
		font-size: 13px;
	}

	th {
		text-align: left;
		font-size: 10px;
		text-transform: uppercase;
		letter-spacing: 1px;
		color: var(--text-dim);
		padding: 8px 12px;
		border-bottom: 1px solid var(--border);
	}

	td {
		padding: 10px 12px;
		border-bottom: 1px solid var(--border);
	}

	tr.clickable {
		cursor: pointer;
	}

	tr.clickable:hover td {
		background: var(--bg-hover);
	}

	.delete-btn {
		background: none;
		border: none;
		color: var(--text-dim);
		font-size: 18px;
		cursor: pointer;
		padding: 2px 6px;
	}

	.delete-btn:hover {
		color: var(--red);
	}
</style>
