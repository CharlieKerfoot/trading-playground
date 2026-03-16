<script lang="ts">
	import '../app.css';
	import { page } from '$app/state';
	let { children } = $props();

	const navItems = [
		{ href: '/', label: 'Dashboard' },
		{ href: '/data', label: 'Data' },
		{ href: '/train', label: 'Train' },
		{ href: '/runs', label: 'Runs' }
	];
</script>

<svelte:head>
	<title>Trading Playground</title>
</svelte:head>

<div class="app-shell">
	<nav>
		<a href="/" class="logo">Trading Playground</a>
		<div class="nav-links">
			{#each navItems as item (item.href)}
				<a
					href={item.href}
					class:active={item.href === '/' ? page.url.pathname === '/' : page.url.pathname.startsWith(item.href)}
				>
					{item.label}
				</a>
			{/each}
		</div>
	</nav>
	<main>
		{@render children()}
	</main>
</div>

<style>
	.app-shell {
		max-width: 1200px;
		margin: 0 auto;
		padding: 0 16px;
	}

	nav {
		display: flex;
		align-items: center;
		gap: 32px;
		padding: 16px 0;
		border-bottom: 1px solid var(--border);
		margin-bottom: 24px;
	}

	.logo {
		font-size: 14px;
		font-weight: 600;
		letter-spacing: 0.5px;
		color: var(--text);
		text-decoration: none;
	}

	.nav-links {
		display: flex;
		gap: 4px;
	}

	.nav-links a {
		font-size: 12px;
		color: var(--text-dim);
		text-decoration: none;
		padding: 6px 12px;
		border-radius: var(--radius);
		transition: all 0.15s;
	}

	.nav-links a:hover {
		color: var(--text);
		background: var(--bg-hover);
	}

	.nav-links a.active {
		color: var(--blue);
		background: var(--blue-dim);
	}

	main {
		padding-bottom: 48px;
	}
</style>
