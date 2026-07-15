import { KAMcamApp } from './app/KAMcamApp';

async function main() {
  const app = new KAMcamApp();
  await app.start();
}

main().catch((err) => {
  console.error('KAMcam failed to start:', err);
  const status = document.getElementById('status');
  if (status) status.textContent = `Error: ${err.message}`;
});
