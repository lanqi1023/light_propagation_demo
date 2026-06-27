import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { readFileSync, statSync } from 'node:fs';
import path from 'node:path';

const root = import.meta.dirname;

const server = createServer((req, res) => {
  const filePath = req.url === '/' ? '/index.html' : req.url;
  const full = path.join(root, filePath);
  try {
    statSync(full);
    const data = readFileSync(full);
    const ext = path.extname(full);
    const types = { '.html': 'text/html', '.css': 'text/css', '.js': 'application/javascript' };
    res.writeHead(200, { 'Content-Type': types[ext] || 'application/octet-stream' });
    res.end(data);
  } catch {
    res.writeHead(404);
    res.end('Not Found');
  }
});

server.listen(0, async () => {
  const port = server.address().port;
  console.log(`Server on http://localhost:${port}`);

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });

  await page.goto(`http://localhost:${port}`, { waitUntil: 'networkidle' });
  console.log('Title:', await page.title());

  await page.waitForTimeout(2000);
  await page.screenshot({ path: 'screenshot.png', fullPage: true });
  console.log('screenshot.png saved');

  await browser.close();
  server.close();
});
