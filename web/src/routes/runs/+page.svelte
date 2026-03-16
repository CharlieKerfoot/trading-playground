<script lang="ts">
	import { fetchRuns, deleteRun, type TrainingRun } from '$lib/api';
	import RunsTable from '$lib/components/RunsTable.svelte';
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';

	let runs: TrainingRun[] = $state([]);

	onMount(async () => {
		runs = await fetchRuns();
	});

	function handleSelect(id: string) {
		goto(`/runs/${id}`);
	}

	async function handleDelete(id: string) {
		await deleteRun(id);
		runs = await fetchRuns();
	}
</script>

<div class="page">
	<div class="page-header">
		<h2>Training Runs</h2>
		<a href="/train"><button class="primary">New Run</button></a>
	</div>

	<div class="card">
		<RunsTable {runs} onSelect={handleSelect} onDelete={handleDelete} />
	</div>
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
		align-items: center;
	}

	h2 {
		font-size: 16px;
		font-weight: 600;
	}
</style>
