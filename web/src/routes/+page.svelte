<script lang="ts">
	import { fetchDataStats, fetchRuns, type DataStats, type TrainingRun } from '$lib/api';
	import MetricsCard from '$lib/components/MetricsCard.svelte';
	import RunsTable from '$lib/components/RunsTable.svelte';
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';

	let stats: DataStats | null = $state(null);
	let runs: TrainingRun[] = $state([]);

	onMount(async () => {
		[stats, runs] = await Promise.all([fetchDataStats(), fetchRuns()]);
	});

	function handleSelect(id: string) {
		goto(`/runs/${id}`);
	}

	async function handleDelete(id: string) {
		await fetch(`/api/runs/${id}`, { method: 'DELETE' });
		runs = await fetchRuns();
	}
</script>

<div class="page">
	<h2>Dashboard</h2>

	<div class="stats-row">
		<div class="card">
			<h3>Cached Markets</h3>
			<span class="big-number">{stats?.total_markets ?? '—'}</span>
		</div>
		<div class="card">
			<h3>Price Points</h3>
			<span class="big-number">{stats?.total_price_points?.toLocaleString() ?? '—'}</span>
		</div>
		<div class="card">
			<h3>Training Runs</h3>
			<span class="big-number">{runs.length}</span>
		</div>
	</div>

	{#if stats && stats.total_markets === 0}
		<div class="card empty-state">
			<p>No market data cached yet.</p>
			<a href="/data"><button class="primary">Sync Market Data</button></a>
		</div>
	{/if}

	<div class="card">
		<h3>Recent Runs</h3>
		<RunsTable runs={runs.slice(0, 10)} onSelect={handleSelect} onDelete={handleDelete} />
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

	.stats-row {
		display: grid;
		grid-template-columns: repeat(4, 1fr);
		gap: 16px;
	}

	.big-number {
		font-size: 28px;
		font-weight: 700;
		color: var(--blue);
	}

	.empty-state {
		text-align: center;
		padding: 32px;
	}

	.empty-state p {
		margin-bottom: 16px;
		color: var(--text-dim);
	}
</style>
