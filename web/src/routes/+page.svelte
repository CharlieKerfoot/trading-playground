<script lang="ts">
	import {
		createSession,
		sendAction,
		resetSession,
		runAgent,
		stopAgent,
		connectWebSocket,
		fetchMarkets,
		type Session,
		type MarketInfo
	} from '$lib/api';
	import PriceDisplay from '$lib/components/PriceDisplay.svelte';
	import PositionPanel from '$lib/components/PositionPanel.svelte';
	import TradePanel from '$lib/components/TradePanel.svelte';
	import SignalsPanel from '$lib/components/SignalsPanel.svelte';
	import HistoryTable from '$lib/components/HistoryTable.svelte';
	import PnlChart from '$lib/components/PnlChart.svelte';
	import { onMount } from 'svelte';

	let markets: MarketInfo[] = $state([]);
	let selectedMarket = $state('');
	let selectedAgent = $state<string | null>(null);
	let session: Session | null = $state(null);
	let ws: WebSocket | null = $state(null);
	let agentSpeed = $state(0.3);

	const uid = crypto.randomUUID();

	onMount(() => {
		fetchMarkets().then((m) => {
			markets = m;
			if (m.length > 0) selectedMarket = m[0].id;
		});
	});

	let resolvedMarkets = $derived(markets.filter((m) => m.type === 'polymarket' && m.resolved));
	let syntheticMarkets = $derived(markets.filter((m) => m.type !== 'polymarket'));

	async function startSession() {
		session = await createSession(selectedMarket, selectedAgent);
		if (ws) ws.close();
		ws = connectWebSocket(session.session_id, (data) => {
			if (data.type === 'step' || data.type === 'state' || data.type === 'done') {
				session = { ...session!, ...data };
			}
		});
	}

	async function handleTrade(action: string, size: number) {
		if (!session) return;
		session = await sendAction(session.session_id, action, size);
	}

	async function handleReset() {
		if (!session) return;
		session = await resetSession(session.session_id);
	}

	async function handleRunAgent() {
		if (!session) return;
		await runAgent(session.session_id, 100, agentSpeed);
	}

	async function handleStopAgent() {
		if (!session) return;
		await stopAgent(session.session_id);
		if (session) session.running = false;
	}

	let isManual = $derived(selectedAgent === null);

	let pnlWithCumulative = $derived.by(() => {
		let cum = 0;
		return (session?.history ?? []).map((h) => {
			cum += h.reward;
			return { step: h.step, reward: h.reward, cumulative: cum, price: h.yes_price };
		});
	});
</script>

<div class="app">
	<header>
		<h1>Trading Playground</h1>
		<div class="header-controls">
			{#if session}
				<span class="dim">Session: {session.session_id}</span>
				<span class="dim">|</span>
				<span class="dim">Step {session.step}</span>
			{/if}
		</div>
	</header>

	{#if !session}
		<div class="setup">
			<div class="card setup-card">
				<h3>New Session</h3>

				<div class="setup-row">
					<label for="{uid}-market">Market</label>
					<select id="{uid}-market" bind:value={selectedMarket}>
						{#if resolvedMarkets.length > 0}
							<optgroup label="Polymarket (Historical)">
								{#each resolvedMarkets as m (m.id)}
									<option value={m.id}>
										{m.name} - {(m.yes_price ?? 0) > 0.5 ? 'YES' : 'NO'}
									</option>
								{/each}
							</optgroup>
						{/if}
						<optgroup label="Synthetic">
							{#each syntheticMarkets as m (m.id)}
								<option value={m.id}>{m.name}</option>
							{/each}
						</optgroup>
					</select>
				</div>

				<div class="setup-row">
					<span class="field-label" id="{uid}-mode-label">Mode</span>
					<div class="mode-buttons" role="group" aria-labelledby="{uid}-mode-label">
						<button
							class:primary={selectedAgent === null}
							onclick={() => (selectedAgent = null)}
						>
							Manual
						</button>
						<button
							class:primary={selectedAgent === 'rule'}
							onclick={() => (selectedAgent = 'rule')}
						>
							Rule Agent
						</button>
						<button
							class:primary={selectedAgent === 'claude'}
							onclick={() => (selectedAgent = 'claude')}
						>
							Claude Agent
						</button>
						<button
							class:primary={selectedAgent === 'random'}
							onclick={() => (selectedAgent = 'random')}
						>
							Random Agent
						</button>
					</div>
				</div>

				{#if selectedAgent}
					<div class="setup-row">
						<label for="{uid}-speed">Agent Speed</label>
						<div class="speed-control">
							<input
								id="{uid}-speed"
								type="range"
								min="0.05"
								max="2"
								step="0.05"
								bind:value={agentSpeed}
							/>
							<span class="dim">{agentSpeed.toFixed(2)}s/step</span>
						</div>
					</div>
				{/if}

				<button class="primary start-btn" onclick={startSession}>
					{selectedAgent ? `Start ${selectedAgent} Agent` : 'Start Episode'}
				</button>
			</div>
		</div>
	{:else}
		<div class="dashboard">
			<div class="top-row">
				<PriceDisplay state={session.state} />
				<PositionPanel position={session.position} totalPnl={session.total_pnl} />
				<SignalsPanel signals={session.state.signals} timeElapsed={session.state.time_elapsed} timeRemaining={session.state.time_remaining} />
			</div>

			<div class="middle-row">
				<div class="chart-area card">
					<h3>P&L / Price</h3>
					<PnlChart data={pnlWithCumulative} />
				</div>
				<div class="trade-area">
					{#if isManual}
						<TradePanel
							onTrade={handleTrade}
							disabled={session.state.is_resolved || session.step >= 100}
						/>
					{:else}
						<div class="card agent-controls">
							<h3>Agent Controls</h3>
							<div class="agent-info">
								<span class="agent-badge">{session.agent_type}</span>
								<span class="dim">
									{session.running ? 'Running...' : 'Stopped'}
								</span>
							</div>
							<div class="agent-buttons">
								{#if !session.running}
									<button class="primary" onclick={handleRunAgent}>Run Agent</button>
								{:else}
									<button class="close-pos" onclick={handleStopAgent}>Stop</button>
								{/if}
							</div>
						</div>
					{/if}
					<div class="session-controls card">
						<h3>Session</h3>
						<div class="session-buttons">
							<button onclick={handleReset}>Reset Episode</button>
							<button onclick={() => { if (ws) ws.close(); session = null; }}>
								New Session
							</button>
						</div>
					</div>
				</div>
			</div>

			<div class="bottom-row">
				<HistoryTable history={session.history} />
			</div>
		</div>
	{/if}
</div>

<style>
	.app {
		max-width: 1400px;
		margin: 0 auto;
		padding: 16px;
	}

	header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 12px 0 24px;
		border-bottom: 1px solid var(--border);
		margin-bottom: 24px;
	}

	header h1 {
		font-size: 16px;
		font-weight: 600;
		letter-spacing: 0.5px;
	}

	.header-controls {
		display: flex;
		gap: 8px;
		font-size: 12px;
	}

	.setup {
		display: flex;
		justify-content: center;
		padding-top: 80px;
	}

	.setup-card {
		width: 500px;
	}

	.setup-row {
		margin-bottom: 16px;
	}

	.setup-row label,
	.setup-row .field-label {
		display: block;
		font-size: 11px;
		text-transform: uppercase;
		letter-spacing: 1px;
		color: var(--text-dim);
		margin-bottom: 6px;
	}

	.setup-row select {
		width: 100%;
	}

	.mode-buttons {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 8px;
	}

	.speed-control {
		display: flex;
		align-items: center;
		gap: 12px;
	}

	.speed-control input[type='range'] {
		flex: 1;
	}

	.start-btn {
		width: 100%;
		margin-top: 8px;
		padding: 12px;
		font-size: 14px;
	}

	.dashboard {
		display: flex;
		flex-direction: column;
		gap: 16px;
	}

	.top-row {
		display: grid;
		grid-template-columns: 1fr 1fr 1fr;
		gap: 16px;
	}

	.middle-row {
		display: grid;
		grid-template-columns: 1fr 320px;
		gap: 16px;
	}

	.chart-area {
		min-height: 250px;
	}

	.trade-area {
		display: flex;
		flex-direction: column;
		gap: 16px;
	}

	.agent-controls {
		flex: 1;
	}

	.agent-info {
		display: flex;
		align-items: center;
		gap: 8px;
		margin-bottom: 12px;
	}

	.agent-badge {
		background: var(--purple);
		color: #fff;
		padding: 4px 10px;
		border-radius: 4px;
		font-size: 12px;
		font-weight: 600;
	}

	.agent-buttons, .session-buttons {
		display: flex;
		gap: 8px;
	}

	.agent-buttons button, .session-buttons button {
		flex: 1;
	}
</style>
