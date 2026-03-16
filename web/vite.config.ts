import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

const apiPort = process.env.API_PORT || '8000';
const apiTarget = `http://localhost:${apiPort}`;

export default defineConfig({
	plugins: [sveltekit()],
	server: {
		proxy: {
			'/api': apiTarget,
			'/ws': {
				target: apiTarget,
				ws: true
			}
		}
	}
});
