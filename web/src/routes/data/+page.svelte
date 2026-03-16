<script lang="ts">
	import {
		fetchDataStats,
		fetchSyncStatus,
		startSync,
		type DataStats,
		type SyncStatus
	} from '$lib/api';
	import { onMount, onDestroy } from 'svelte';

	let stats: DataStats | null = $state(null);
	let syncStatus: SyncStatus = $state({ running: false, fetched: 0, skipped: 0, errors: 0 });
	let syncLimit = $state(100);
	let pollInterval: ReturnType<typeof setInterval> | null = null;

	onMount(async () => {
		stats = await fetchDataStats();
		syncStatus = await fetchSyncStatus();
		if (syncStatus.running) {
			startPolling();
		}
	});

	onDestroy(() => {
		if (pollInterval) clearInterval(pollInterval);
	});

	function startPolling() {
		pollInterval = setInterval(async () => {
			syncStatus = await fetchSyncStatus();
			if (!syncStatus.running) {
				if (pollInterval) clearInterval(pollInterval);
				pollInterval = null;
				stats = await fetchDataStats();
			}
		}, 1000);
	}

	async function handleSync() {
		await startSync(syncLimit);
		syncStatus = await fetchSyncStatus();
		startPolling();
	}
</script>

<div class="page">
	<h2>Data Management</h2>

	<div class="grid">
		<div class="card">
			<h3>Sync Markets</h3>
			<p class="dim" style="margin-bottom: 12px; font-size: 12px">
				Download resolved markets from Polymarket with their price histories.
			</p>

			<div class="sync-controls">
				<label>
					<span class="field-label">Limit</span>
					<input type="number" bind:value={syncLimit} min="10" max="500" step="10" />
				</label>
				<button
					class="primary"
					onclick={handleSync}
					disabled={syncStatus.running}
				>
					{syncStatus.running ? 'Syncing...' : 'Start Sync'}
				</button>
			</div>

			{#if syncStatus.running || syncStatus.fetched > 0}
				<div class="sync-progress">
					{#if syncStatus.running}
						<div class="progress-bar">
							<div
								class="progress-fill"
								style="width: {((syncStatus.fetched + syncStatus.skipped) / syncLimit) * 100}%"
							></div>
						</div>
					{/if}
					<div class="sync-stats dim">
						<span>Fetched: {syncStatus.fetched}</span>
						<span>Skipped: {syncStatus.skipped}</span>
						<span>Errors: {syncStatus.errors}</span>
					</div>
				</div>
			{/if}
		</div>

		{#if stats}
			<div class="card">
				<h3>Cache Stats</h3>
				<div class="stat-pair">
					<div>
						<span class="field-label">Markets</span>
						<span class="big-number">{stats.total_markets}</span>
					</div>
					<div>
						<span class="field-label">Price Points</span>
						<span class="big-number">{stats.total_price_points.toLocaleString()}</span>
					</div>
				</div>
				{#if stats.categories && Object.keys(stats.categories).length > 0}
					<div class="category-table">
						<h4>Categories</h4>
						<table>
							<thead>
								<tr>
									<th>Category</th>
									<th>Markets</th>
								</tr>
							</thead>
							<tbody>
								{#each Object.entries(stats.categories).sort((a, b) => b[1] - a[1]) as [cat, count]}
									<tr>
										<td>{cat}</td>
										<td>{count}</td>
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
				{/if}
			</div>
		{/if}
	</div>
</div>

<style>
	.page {
		display: flex;
		flex-direction: column;
		gap: 20px;
	}

	h2 {
		font-size: 16px;
		font-weight: 600;
	}

	.grid {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 16px;
	}

	.sync-controls {
		display: flex;
		gap: 12px;
		align-items: flex-end;
	}

	.sync-controls label {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}

	.field-label {
		font-size: 10px;
		text-transform: uppercase;
		letter-spacing: 1px;
		color: var(--text-dim);
	}

	.sync-controls input {
		width: 100px;
		font-family: var(--font);
		font-size: 13px;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius);
		padding: 8px 12px;
		color: var(--text);
	}

	.sync-progress {
		margin-top: 16px;
	}

	.progress-bar {
		width: 100%;
		height: 6px;
		background: var(--bg);
		border-radius: 3px;
		overflow: hidden;
		margin-bottom: 8px;
	}

	.progress-fill {
		height: 100%;
		background: var(--blue);
		border-radius: 3px;
		transition: width 0.3s;
	}

	.sync-stats {
		display: flex;
		gap: 16px;
		font-size: 12px;
	}

	.stat-pair {
		display: flex;
		gap: 32px;
		margin-top: 8px;
	}

	.big-number {
		font-size: 28px;
		font-weight: 700;
		color: var(--blue);
	}

	.category-table {
		margin-top: 16px;
		border-top: 1px solid var(--border);
		padding-top: 12px;
	}

	.category-table h4 {
		font-size: 12px;
		font-weight: 600;
		margin-bottom: 8px;
	}

	.category-table table {
		width: 100%;
		font-size: 12px;
		border-collapse: collapse;
	}

	.category-table th,
	.category-table td {
		text-align: left;
		padding: 4px 8px;
	}

	.category-table th {
		color: var(--text-dim);
		font-weight: 500;
		text-transform: uppercase;
		letter-spacing: 0.5px;
		font-size: 10px;
	}

	.category-table tr:nth-child(even) {
		background: var(--bg);
	}
</style>
