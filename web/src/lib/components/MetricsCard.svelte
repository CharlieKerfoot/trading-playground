<script lang="ts">
	let { label, value, format = 'number', colorize = false }: {
		label: string;
		value: number;
		format?: 'number' | 'percent' | 'currency';
		colorize?: boolean;
	} = $props();

	let formatted = $derived.by(() => {
		if (value === undefined || value === null) return '—';
		switch (format) {
			case 'percent':
				return (value * 100).toFixed(1) + '%';
			case 'currency':
				return (value >= 0 ? '+' : '') + value.toFixed(4);
			default:
				return value.toFixed(4);
		}
	});

	let colorClass = $derived(
		colorize ? (value > 0 ? 'green' : value < 0 ? 'red' : 'dim') : ''
	);
</script>

<div class="metric">
	<span class="metric-label">{label}</span>
	<span class="metric-value {colorClass}">{formatted}</span>
</div>

<style>
	.metric {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}

	.metric-label {
		font-size: 10px;
		text-transform: uppercase;
		letter-spacing: 1px;
		color: var(--text-dim);
	}

	.metric-value {
		font-size: 18px;
		font-weight: 600;
	}
</style>
