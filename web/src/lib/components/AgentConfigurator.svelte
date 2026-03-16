<script lang="ts">
	import type { AgentInfo } from '$lib/api';

	let {
		agents,
		selectedAgent = $bindable('rule'),
		config = $bindable({})
	}: {
		agents: Record<string, AgentInfo>;
		selectedAgent: string;
		config: Record<string, any>;
	} = $props();

	let agentKeys = $derived(Object.keys(agents));
	let currentAgent = $derived(agents[selectedAgent]);

	function handleAgentChange(key: string) {
		selectedAgent = key;
		config = { ...(agents[key]?.config_params || {}) };
	}
</script>

<div class="agent-config">
	<div class="agent-selector">
		{#each agentKeys as key (key)}
			<button
				class:primary={selectedAgent === key}
				onclick={() => handleAgentChange(key)}
			>
				{agents[key].name}
			</button>
		{/each}
	</div>

	{#if currentAgent}
		<p class="agent-desc dim">{currentAgent.description}</p>
	{/if}
</div>

<style>
	.agent-config {
		display: flex;
		flex-direction: column;
		gap: 12px;
	}

	.agent-selector {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 8px;
	}

	.agent-desc {
		font-size: 12px;
	}
</style>
