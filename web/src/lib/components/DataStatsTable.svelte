<script lang="ts">
	let { categories }: {
		categories: Record<string, number>;
	} = $props();

	let sorted = $derived(
		Object.entries(categories).sort((a, b) => b[1] - a[1])
	);

	let total = $derived(
		Object.values(categories).reduce((a, b) => a + b, 0)
	);
</script>

<div class="stats-table">
	<div class="stats-row header">
		<span>Category</span>
		<span>Markets</span>
	</div>
	{#each sorted as [cat, count] (cat)}
		<div class="stats-row">
			<span class="cat-name">{cat || 'uncategorized'}</span>
			<span class="cat-count">{count}</span>
		</div>
	{/each}
	<div class="stats-row total">
		<span>Total</span>
		<span>{total}</span>
	</div>
</div>

<style>
	.stats-table {
		font-size: 13px;
	}

	.stats-row {
		display: flex;
		justify-content: space-between;
		padding: 6px 0;
		border-bottom: 1px solid var(--border);
	}

	.stats-row.header {
		font-size: 10px;
		text-transform: uppercase;
		letter-spacing: 1px;
		color: var(--text-dim);
	}

	.stats-row.total {
		font-weight: 600;
		border-bottom: none;
		color: var(--blue);
	}

	.cat-count {
		color: var(--text-dim);
	}
</style>
