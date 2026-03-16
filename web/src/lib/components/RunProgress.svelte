<script lang="ts">
	let { current, total, status }: {
		current: number;
		total: number;
		status: string;
	} = $props();

	let pct = $derived(total > 0 ? (current / total) * 100 : 0);

	let statusColor = $derived.by(() => {
		switch (status) {
			case 'running': return 'var(--blue)';
			case 'completed': return 'var(--green)';
			case 'cancelled': return 'var(--yellow)';
			case 'failed': return 'var(--red)';
			default: return 'var(--text-dim)';
		}
	});
</script>

<div class="progress-container">
	<div class="progress-header">
		<span class="progress-label" style="color: {statusColor}">
			{status}
		</span>
		<span class="progress-count dim">
			{current} / {total}
		</span>
	</div>
	<div class="progress-bar">
		<div
			class="progress-fill"
			style="width: {pct}%; background: {statusColor}"
		></div>
	</div>
</div>

<style>
	.progress-container {
		width: 100%;
	}

	.progress-header {
		display: flex;
		justify-content: space-between;
		margin-bottom: 6px;
		font-size: 12px;
	}

	.progress-label {
		text-transform: uppercase;
		letter-spacing: 0.5px;
		font-weight: 600;
	}

	.progress-bar {
		width: 100%;
		height: 6px;
		background: var(--bg);
		border-radius: 3px;
		overflow: hidden;
	}

	.progress-fill {
		height: 100%;
		border-radius: 3px;
		transition: width 0.3s ease;
	}
</style>
