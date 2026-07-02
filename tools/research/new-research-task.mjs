#!/usr/bin/env node
// Stamp a research-note task .md for the loops orchestrator (pipelines/research-note.yaml
// in the loops repo). This generator is a road-to-master domain asset; see
// research-note-pipeline.md in this directory for the full workflow.
//
// Usage:
//   node tools/research/new-research-task.mjs \
//     --url https://example.org/paper.pdf \
//     --concept "Attention Is All You Need" \
//     --domain natural_language_processing \
//     --out <loops>/prompts/<batch-name>
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const DOMAINS = new Set([
  'natural_language_processing',
  'computer_vision',
  'recommender_system',
  'timeseries',
  'reinforcement_learning',
  'mlops',
  'utils',
  'vision_language',
]);

const args = parseArgs(process.argv.slice(2));

try {
  const paperUrl = required(args.url, '--url');
  const concept = required(args.concept, '--concept');
  const domain = required(args.domain, '--domain');
  const outDir = args.out || process.cwd();
  const codeRepo = args['code-repo'] || '';

  validateHttpsUrl(paperUrl, '--url');
  if (codeRepo) validateHttpsUrl(codeRepo, '--code-repo');
  if (!DOMAINS.has(domain)) {
    throw new Error(`invalid --domain "${domain}". Expected one of: ${[...DOMAINS].join(', ')}`);
  }

  // Resolve the template beside this script so the generator works from any cwd.
  const templatePath = path.join(path.dirname(fileURLToPath(import.meta.url)), 'task-template.md');
  const template = fs.readFileSync(templatePath, 'utf8');
  const slug = kebabCase(concept);
  const body = template
    .replaceAll('[[paper_url]]', paperUrl)
    .replaceAll('[[concept]]', concept)
    .replaceAll('[[domain]]', domain)
    .replaceAll('[[code_repo_url]]', codeRepo)
    .replaceAll('[[expected_title]]', args['expected-title'] || 'unknown')
    .replaceAll('[[expected_venue]]', args['expected-venue'] || 'unknown')
    .replaceAll('[[expected_year]]', args['expected-year'] || 'unknown');

  fs.mkdirSync(outDir, { recursive: true });
  const outPath = path.join(outDir, `${slug}.md`);
  fs.writeFileSync(outPath, body, { flag: 'wx' });
  console.log(outPath);
} catch (error) {
  console.error(error.message || error);
  process.exit(1);
}

function parseArgs(argv) {
  const parsed = {};
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (!arg.startsWith('--')) throw new Error(`unexpected argument "${arg}"`);
    const key = arg.slice(2);
    const value = argv[i + 1];
    if (!value || value.startsWith('--')) throw new Error(`missing value for ${arg}`);
    parsed[key] = value;
    i += 1;
  }
  return parsed;
}

function required(value, flag) {
  if (!value || !String(value).trim()) throw new Error(`missing required ${flag}`);
  return String(value).trim();
}

function validateHttpsUrl(value, flag) {
  if (value.startsWith('-')) throw new Error(`${flag} must not start with '-'`);
  let parsed;
  try {
    parsed = new URL(value);
  } catch {
    throw new Error(`${flag} must be an absolute https:// URL`);
  }
  if (parsed.protocol !== 'https:') {
    throw new Error(`${flag} must be https-only; reject ext::, file://, git://, ssh://, http://, and user@host: transports`);
  }
}

function kebabCase(value) {
  const slug = value
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .replace(/-{2,}/g, '-');
  if (!slug) throw new Error('--concept must contain at least one ASCII letter or digit for the task filename');
  return slug;
}
