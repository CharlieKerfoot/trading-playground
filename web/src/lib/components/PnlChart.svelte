<script lang="ts">
	let { data }: {
		data: { episode: number; total_reward: number; cumulative: number }[];
	} = $props();

	const W = 600;
	const H = 200;
	const PAD = { top: 20, right: 20, bottom: 30, left: 60 };

	let plotW = W - PAD.left - PAD.right;
	let plotH = H - PAD.top - PAD.bottom;

	let yRange = $derived.by(() => {
		if (data.length === 0) return { min: -0.01, max: 0.01 };
		const vals = data.map((d) => d.cumulative);
		let min = Math.min(0, ...vals);
		let max = Math.max(0, ...vals);
		const pad = (max - min) * 0.1 || 0.01;
		return { min: min - pad, max: max + pad };
	});

	let points = $derived.by(() => {
		if (data.length === 0) return '';
		return data
			.map((d, i) => {
				const x = PAD.left + (i / Math.max(1, data.length - 1)) * plotW;
				const y = PAD.top + (1 - (d.cumulative - yRange.min) / (yRange.max - yRange.min)) * plotH;
				return `${x},${y}`;
			})
			.join(' ');
	});

	let zeroY = $derived(
		PAD.top + (1 - (0 - yRange.min) / (yRange.max - yRange.min)) * plotH
	);

	let finalValue = $derived(data.length > 0 ? data[data.length - 1].cumulative : 0);
	let lineColor = $derived(finalValue >= 0 ? 'var(--green)' : 'var(--red)');

	let yTicks = $derived.by(() => {
		const range = yRange.max - yRange.min;
		const step = range / 4;
		return Array.from({ length: 5 }, (_, i) => yRange.min + step * i);
	});
</script>

<div class="chart-container">
	{#if data.length === 0}
		<p class="dim" style="text-align: center; padding: 40px 0">Waiting for results...</p>
	{:else}
		<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet">
			<!-- Zero line -->
			<line
				x1={PAD.left}
				y1={zeroY}
				x2={W - PAD.right}
				y2={zeroY}
				stroke="var(--text-dim)"
				stroke-dasharray="4,4"
				opacity="0.3"
			/>

			<!-- Y-axis ticks -->
			{#each yTicks as tick}
				{@const y = PAD.top + (1 - (tick - yRange.min) / (yRange.max - yRange.min)) * plotH}
				<text x={PAD.left - 8} {y} text-anchor="end" fill="var(--text-dim)" font-size="9" dy="3">
					{tick.toFixed(3)}
				</text>
			{/each}

			<!-- X-axis label -->
			<text x={PAD.left + plotW / 2} y={H - 4} text-anchor="middle" fill="var(--text-dim)" font-size="9">
				Episode
			</text>

			<!-- Cumulative P&L line -->
			<polyline
				{points}
				fill="none"
				stroke={lineColor}
				stroke-width="2"
			/>

			<!-- Per-episode bars -->
			{#each data as d, i}
				{@const x = PAD.left + (i / Math.max(1, data.length - 1)) * plotW}
				{@const barH = Math.abs(d.total_reward) / (yRange.max - yRange.min) * plotH}
				<rect
					x={x - 2}
					y={d.total_reward >= 0 ? zeroY - barH : zeroY}
					width={4}
					height={Math.max(1, barH)}
					fill={d.total_reward >= 0 ? 'var(--green)' : 'var(--red)'}
					opacity="0.2"
				/>
			{/each}
		</svg>
	{/if}
</div>

<style>
	.chart-container {
		width: 100%;
	}

	svg {
		width: 100%;
		height: auto;
	}
</style>
