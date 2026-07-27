/**
 * Example: JavaScript (Node 18+) client
 * Run: node examples/client_javascript.js
 * (Generate sample_request.json first)
 */

const fs = require('fs');
const path = require('path');

async function main() {
  const body = fs.readFileSync(path.join(__dirname, 'sample_request.json'), 'utf8');
  const res = await fetch('http://localhost:8000/forecast', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body,
  });
  const data = await res.json();
  console.log('status', data.status);
  console.log('first forecast values', data.forecast.predicted_cpu_percent.slice(0, 3));
}

main().catch(console.error);
