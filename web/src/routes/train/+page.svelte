<script lang="ts">
  import {
    fetchAgents,
    fetchDataStats,
    fetchModels,
    fetchRLTrainStatus,
    createRun,
    startRLTraining,
    type AgentInfo,
    type DataStats,
    type SavedModel,
    type RLTrainStatus
  } from '$lib/api';
  import AgentConfigurator from '$lib/components/AgentConfigurator.svelte';
  import { onMount, onDestroy } from 'svelte';
  import { goto } from '$app/navigation';

  let agents: Record<string, AgentInfo> = $state({});
  let stats = $state<DataStats | null>(null);
  let models: SavedModel[] = $state([]);
  let selectedAgent = $state('rule');
  let agentConfig: Record<string, any> = $state({});
  let numEpisodes = $state(50);
  let launching = $state(false);

  // RL training state
  let rlTimesteps = $state(100000);
  let rlAlgorithm = $state('PPO');
  let rlTraining = $state(false);
  let rlStatus = $state<RLTrainStatus | null>(null);
  let rlPollRef: ReturnType<typeof setInterval> | null = null;

  onMount(async () => {
    const [a, s, m] = await Promise.all([
      fetchAgents(),
      fetchDataStats(),
      fetchModels()
    ]);
    agents = a;
    stats = s;
    models = m;

    // Check if RL training is already running
    rlStatus = await fetchRLTrainStatus();
    if (rlStatus?.running) {
      rlTraining = true;
      startRLPolling();
    }
  });

  onDestroy(() => {
    if (rlPollRef) clearInterval(rlPollRef);
  });

  async function handleLaunch() {
    launching = true;
    try {
      const run = await createRun({
        agent_type: selectedAgent,
        num_episodes: numEpisodes,
        agent_config: agentConfig,
      });
      goto(`/runs/${run.id}`);
    } catch (e) {
      launching = false;
    }
  }

  async function handleRLTrain() {
    rlTraining = true;
    await startRLTraining({
      timesteps: rlTimesteps,
      algorithm: rlAlgorithm,
    });
    startRLPolling();
  }

  function startRLPolling() {
    rlPollRef = setInterval(async () => {
      rlStatus = await fetchRLTrainStatus();
      if (!rlStatus?.running) {
        rlTraining = false;
        if (rlPollRef) clearInterval(rlPollRef);
        rlPollRef = null;
        models = await fetchModels();
      }
    }, 2000);
  }

  let hasData = $derived((stats?.total_markets ?? 0) > 0);
</script>

<div class="page">
  <h2>Launch Training Run</h2>

  {#if !hasData}
    <div class="card empty-state">
      <p>No market data available. Sync data first.</p>
      <a href="/data"><button class="primary">Go to Data</button></a>
    </div>
  {:else}
    <div class="grid">
      <div class="card">
        <h3>Agent</h3>
        {#if Object.keys(agents).length > 0}
          <AgentConfigurator {agents} bind:selectedAgent bind:config={agentConfig} />
        {/if}

        {#if selectedAgent === 'rl' && models.length > 0}
          <div class="config-field" style="margin-top: 12px">
            <label>
              <span class="field-label">Trained Model</span>
              <select bind:value={agentConfig.model_path}>
                <option value="">Random (no model)</option>
                {#each models as model (model.path)}
                  <option value={model.path}>{model.name}</option>
                {/each}
              </select>
            </label>
          </div>
        {/if}
      </div>

      <div class="card">
        <h3>Evaluate</h3>

        <div class="config-field">
          <label>
            <span class="field-label">Episodes</span>
            <input type="number" bind:value={numEpisodes} min="1" max="1000" step="10" />
          </label>
        </div>

        <button
          class="primary launch-btn"
          onclick={handleLaunch}
          disabled={launching}
        >
          {launching ? 'Launching...' : `Launch ${numEpisodes} Episodes`}
        </button>
      </div>
    </div>

    <div class="card">
      <h3>RL Training (Stable-Baselines3)</h3>
      <p class="dim" style="font-size: 12px; margin-bottom: 12px">
        Train a PPO or A2C policy on cached market data. Once trained, select the model above to evaluate.
      </p>

      <div class="rl-controls">
        <div class="config-field">
          <label>
            <span class="field-label">Timesteps</span>
            <input type="number" bind:value={rlTimesteps} min="1000" max="10000000" step="10000" />
          </label>
        </div>

        <div class="config-field">
          <label>
            <span class="field-label">Algorithm</span>
            <select bind:value={rlAlgorithm}>
              <option value="PPO">PPO</option>
              <option value="A2C">A2C</option>
            </select>
          </label>
        </div>

        <button
          class="primary"
          onclick={handleRLTrain}
          disabled={rlTraining}
          style="align-self: flex-end"
        >
          {rlTraining ? 'Training...' : 'Start RL Training'}
        </button>
      </div>

      {#if rlTraining || (rlStatus && !rlStatus.running && rlStatus.timesteps > 0)}
        <div class="rl-progress" style="margin-top: 12px">
          {#if rlTraining}
            <div class="progress-bar">
              <div class="progress-fill pulse"></div>
            </div>
            <p class="dim" style="font-size: 12px">
              Training {rlStatus?.algorithm ?? rlAlgorithm} for {(rlStatus?.timesteps ?? rlTimesteps).toLocaleString()} timesteps...
            </p>
          {:else if rlStatus?.error}
            <p class="red" style="font-size: 12px">Error: {rlStatus.error}</p>
          {:else}
            <p class="green" style="font-size: 12px">Training complete! Model saved to {rlStatus?.save_path}</p>
          {/if}
        </div>
      {/if}

      {#if models.length > 0}
        <div style="margin-top: 16px">
          <span class="field-label">Saved Models</span>
          <div class="model-list">
            {#each models as model (model.path)}
              <div class="model-item">
                <span>{model.name}</span>
                <span class="dim">{(model.size_bytes / 1024).toFixed(0)} KB</span>
              </div>
            {/each}
          </div>
        </div>
      {/if}
    </div>
  {/if}
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

  .config-field {
    margin-bottom: 16px;
  }

  .config-field label {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .field-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-dim);
  }

  .config-field input,
  .config-field select {
    width: 100%;
  }

  .launch-btn {
    width: 100%;
    padding: 14px;
    font-size: 14px;
    margin-top: 8px;
  }

  .empty-state {
    text-align: center;
    padding: 32px;
  }

  .empty-state p {
    margin-bottom: 16px;
    color: var(--text-dim);
  }

  .rl-controls {
    display: grid;
    grid-template-columns: 1fr 1fr auto;
    gap: 12px;
    align-items: flex-end;
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
    width: 100%;
  }

  .progress-fill.pulse {
    animation: pulse 1.5s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 0.3; }
    50% { opacity: 1; }
  }

  .model-list {
    margin-top: 8px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .model-item {
    display: flex;
    justify-content: space-between;
    padding: 6px 8px;
    border-radius: 4px;
    background: var(--bg);
    font-size: 12px;
  }
</style>
