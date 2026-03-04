#!/usr/bin/env npx tsx
/**
 * News Radar — Multi-Channel News Aggregation & Analysis Agent (Node.js, TypeScript)
 *
 * Uses A3S Code + WebMCP to:
 * 1. Fetch news from multiple channels (web search, RSS via MCP, site scraping)
 * 2. Deduplicate raw content
 * 3. LLM-powered extraction: structured news items with impact assessment
 * 4. Stream a daily briefing report
 *
 * Usage:
 *   npx tsx news_radar.ts
 *   npx tsx news_radar.ts --topics ai,dev
 */

import { Agent, Session, AgentResult, EventStream, AgentEvent, ToolResult } from '../index.js';
import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';

// ── Types ────────────────────────────────────────────────────────────

interface ChannelConfig {
  searchQueries: string[];
  rssFeeds: string[];
  sites: string[];
}

interface RawItem {
  channel: string;
  topic: string;
  content: string;
}

interface NewsItem {
  title: string;
  summary: string;
  source: string;
  topic: string;
  impact: 'high' | 'medium' | 'low';
  entities: string[];
}

// ── Channel Definitions ─────────────────────────────────────────────

const DEFAULT_CHANNELS: Record<string, ChannelConfig> = {
  tech: {
    searchQueries: [
      'latest technology news today',
      'AI artificial intelligence breakthroughs',
      'open source software releases',
    ],
    rssFeeds: [
      'https://news.ycombinator.com/rss',
      'https://www.reddit.com/r/technology/.rss',
    ],
    sites: ['https://news.ycombinator.com'],
  },
  ai: {
    searchQueries: [
      'AI research papers this week',
      'large language model news today',
    ],
    rssFeeds: ['https://arxiv.org/rss/cs.AI'],
    sites: ['https://huggingface.co/blog'],
  },
  dev: {
    searchQueries: [
      'software development news today',
      'Rust programming language updates',
    ],
    rssFeeds: ['https://dev.to/feed'],
    sites: ['https://github.com/trending'],
  },
};

// ============================================================================
// NewsRadar
// ============================================================================

class NewsRadar {
  private readonly agent: Agent;
  private readonly configPath: string;
  private session!: Session;

  constructor(agent: Agent, configPath: string) {
    this.agent = agent;
    this.configPath = configPath;
  }

  // --------------------------------------------------------------------------
  // Static helpers
  // --------------------------------------------------------------------------

  static findConfig(): string {
    const candidates: string[] = [
      path.join(__dirname, 'news-radar', 'agent.hcl'),
      path.join(process.env.HOME || '', '.a3s', 'config.hcl'),
    ];
    if (process.env.A3S_CONFIG) candidates.unshift(process.env.A3S_CONFIG);
    for (const p of candidates) {
      if (fs.existsSync(p)) return p;
    }
    throw new Error('Config not found. Set A3S_CONFIG or create ~/.a3s/config.hcl');
  }

  private static contentHash(text: string): string {
    return crypto
      .createHash('sha256')
      .update(text.trim().toLowerCase().slice(0, 500))
      .digest('hex')
      .slice(0, 12);
  }

  static nowStr(): string {
    return new Date().toISOString().replace('T', ' ').slice(0, 19) + ' UTC';
  }

  private static todayStr(): string {
    return new Date().toISOString().slice(0, 10);
  }

  // --------------------------------------------------------------------------
  // Phase 1: Multi-Channel Fetch
  // --------------------------------------------------------------------------

  async fetchViaSearch(queries: string[], topic: string): Promise<RawItem[]> {
    const results: RawItem[] = [];
    const tasks: Promise<void>[] = queries.map(async (query: string): Promise<void> => {
      console.log(`  🔍 search: ${query}`);
      try {
        const r: ToolResult = await this.session.tool('web_search', {
          query, limit: 10, timeout: 20, format: 'text',
        });
        if (r.exitCode === 0 && r.output) {
          results.push({ channel: 'search', topic, content: r.output });
        }
      } catch (e: unknown) {
        const err = e as { message?: string };
        console.log(`  ⚠ search failed: ${query} — ${err.message}`);
      }
    });
    await Promise.all(tasks);
    return results;
  }

  async fetchViaMcp(urls: string[], channelType: string, topic: string): Promise<RawItem[]> {
    const results: RawItem[] = [];
    const icon: string = channelType === 'rss' ? '📡' : '🌐';
    const tasks: Promise<void>[] = urls.map(async (url: string): Promise<void> => {
      console.log(`  ${icon} ${channelType}: ${url}`);
      try {
        const r: ToolResult = await this.session.tool('mcp__fetch__fetch', {
          url, max_length: 50000, raw: false,
        });
        if (r.exitCode === 0 && r.output) {
          results.push({ channel: channelType, topic, content: r.output });
        }
      } catch (e: unknown) {
        const err = e as { message?: string };
        console.log(`  ⚠ ${channelType} failed: ${url} — ${err.message}`);
      }
    });
    await Promise.all(tasks);
    return results;
  }

  async fetchAllChannels(channels: Record<string, ChannelConfig>): Promise<RawItem[]> {
    const all: RawItem[] = [];
    for (const [name, cfg] of Object.entries(channels)) {
      console.log(`\n📰 Channel: ${name}`);
      const [search, rss, scrape]: [RawItem[], RawItem[], RawItem[]] = await Promise.all([
        this.fetchViaSearch(cfg.searchQueries || [], name),
        this.fetchViaMcp(cfg.rssFeeds || [], 'rss', name),
        this.fetchViaMcp(cfg.sites || [], 'scrape', name),
      ]);
      all.push(...search, ...rss, ...scrape);
    }
    return all;
  }

  // --------------------------------------------------------------------------
  // Phase 2: Deduplicate
  // --------------------------------------------------------------------------

  deduplicate(items: RawItem[]): RawItem[] {
    const before: number = items.length;
    const seen: Set<string> = new Set();
    const unique: RawItem[] = items.filter((item: RawItem): boolean => {
      const h: string = NewsRadar.contentHash(item.content);
      if (seen.has(h)) return false;
      seen.add(h);
      return true;
    });
    console.log(`\n🔄 Dedup: ${before} → ${unique.length} unique items`);
    return unique;
  }

  // --------------------------------------------------------------------------
  // Phase 3: LLM Extraction
  // --------------------------------------------------------------------------

  async extractNewsItems(rawItems: RawItem[]): Promise<NewsItem[]> {
    console.log('\n🧠 Extracting news items via LLM...');
    const allItems: NewsItem[] = [];
    const chunkSize: number = 5;

    for (let i = 0; i < rawItems.length; i += chunkSize) {
      const chunk: RawItem[] = rawItems.slice(i, i + chunkSize);
      const combined: string = chunk
        .map((r: RawItem): string => `[${r.channel}:${r.topic}] ${r.content.slice(0, 3000)}`)
        .join('\n\n---\n\n');

      const prompt: string =
        'You are a news analyst. From the raw content below, extract structured news items.\n\n' +
        'For each distinct news story, output a JSON object with:\n' +
        '- "title": concise headline (< 80 chars)\n' +
        '- "summary": 2-3 sentence summary\n' +
        '- "source": original source name or URL\n' +
        '- "topic": the topic category\n' +
        '- "impact": "high" | "medium" | "low"\n' +
        '- "entities": list of key people, companies, or technologies mentioned\n\n' +
        'Return a JSON array of objects only. Skip ads and boilerplate.\n\n' +
        `Raw content:\n${combined}`;

      try {
        const result: AgentResult = await this.session.send(prompt);
        let text: string = result.text.trim();
        if (text.startsWith('```')) {
          text = text.replace(/^```\w*\n?/, '').replace(/\n?```$/, '');
        }
        const items: NewsItem[] = JSON.parse(text);
        if (Array.isArray(items)) {
          allItems.push(...items);
          console.log(`  ✓ Extracted ${items.length} items from batch ${Math.floor(i / chunkSize) + 1}`);
        }
      } catch (e: unknown) {
        const err = e as { message?: string };
        console.log(`  ⚠ Parse error in batch ${Math.floor(i / chunkSize) + 1}: ${err.message}`);
      }
    }

    console.log(`  📊 Total extracted: ${allItems.length} news items`);
    return allItems;
  }

  // --------------------------------------------------------------------------
  // Phase 4: Generate Report
  // --------------------------------------------------------------------------

  async generateReport(items: NewsItem[], outputDir: string): Promise<void> {
    const date: string = NewsRadar.todayStr();
    const timestamp: string = NewsRadar.nowStr();
    const sources: Set<string> = new Set(items.map((i: NewsItem): string => i.source || 'unknown'));
    const itemsJson: string = JSON.stringify(items, null, 2).slice(0, 60000);

    const prompt: string =
      `You are a senior news analyst producing a daily intelligence briefing.\n\n` +
      `Date: ${date}\nNews items (JSON): ${itemsJson}\n\n` +
      `Generate a structured markdown report with:\n\n` +
      `# 📡 News Radar — Daily Briefing (${date})\n\n` +
      `## 🔥 Top Stories\n(3-5 highest impact stories with detailed analysis)\n\n` +
      `## 📊 By Topic\n### Tech\n### AI & ML\n### Development\n` +
      `(Group stories by topic, 2-3 sentences each)\n\n` +
      `## 🏢 Key Entities\n(Table of most mentioned companies/people/technologies)\n\n` +
      `## 📈 Trend Analysis\n(What patterns emerge?)\n\n` +
      `## ⚡ Action Items\n(What should a tech professional pay attention to?)\n\n` +
      `---\n*Generated by News Radar Agent at ${timestamp}*\n` +
      `*Sources: ${sources.size} channels, ${items.length} stories analyzed*\n\n` +
      `Write in clear, professional language. Prioritize signal over noise.`;

    console.log('\n📝 Generating report...\n');
    console.log('='.repeat(60));

    const reportParts: string[] = [];
    const stream: EventStream = await this.session.stream(prompt);

    for await (const event of stream) {
      switch (event.type) {
        case 'text_delta':
          process.stdout.write(event.text!);
          reportParts.push(event.text!);
          break;
        case 'tool_start':
          console.log(`\n  🔧 ${event.toolName}...`);
          break;
        case 'end':
          console.log(`\n${'='.repeat(60)}`);
          console.log(`✓ Report generated (${event.totalTokens || '?'} tokens)`);
          break;
        case 'error':
          console.log(`\n❌ Error: ${event.error}`);
          break;
      }
    }

    // Save report
    fs.mkdirSync(outputDir, { recursive: true });
    const reportPath: string = path.join(outputDir, `news-radar-${date}.md`);
    fs.writeFileSync(reportPath, reportParts.join(''), 'utf-8');
    console.log(`\n💾 Saved to ${reportPath}`);
  }

  // --------------------------------------------------------------------------
  // Run All
  // --------------------------------------------------------------------------

  async runAll(channels: Record<string, ChannelConfig>): Promise<void> {
    this.session = this.agent.session('.', {
      permissive: true,
      autoCompact: true,
      autoCompactThreshold: 0.7,
      maxParseRetries: 3,
      toolTimeoutMs: 30000,
      circuitBreakerThreshold: 5,
    });

    // Phase 1: Fetch
    console.log('\n── Phase 1: Multi-Channel Fetch ──');
    const rawItems: RawItem[] = await this.fetchAllChannels(channels);
    console.log(`\n✓ Fetched ${rawItems.length} raw items`);

    if (rawItems.length === 0) {
      console.log('⚠ No items fetched. Check network and config.');
      return;
    }

    // Phase 2: Deduplicate
    console.log('\n── Phase 2: Deduplicate ──');
    const uniqueItems: RawItem[] = this.deduplicate(rawItems);

    // Phase 3: LLM extraction
    console.log('\n── Phase 3: LLM Analysis ──');
    const newsItems: NewsItem[] = await this.extractNewsItems(uniqueItems);

    if (newsItems.length === 0) {
      console.log('⚠ No news items extracted.');
      return;
    }

    // Phase 4: Generate report
    console.log('\n── Phase 4: Generate Report ──');
    await this.generateReport(newsItems, './reports');

    console.log(`\n${'━'.repeat(60)}`);
    console.log('✓ News Radar cycle complete');
    console.log('━'.repeat(60));
  }
}

// ============================================================================
// Main
// ============================================================================

async function main(): Promise<void> {
  const args: string[] = process.argv.slice(2);
  const topicsIdx: number = args.indexOf('--topics');
  const topicFilter: string[] | null = topicsIdx >= 0 ? args[topicsIdx + 1]?.split(',') : null;

  const channels: Record<string, ChannelConfig> = topicFilter
    ? Object.fromEntries(
        Object.entries(DEFAULT_CHANNELS).filter(([k]: [string, ChannelConfig]): boolean => topicFilter.includes(k))
      )
    : DEFAULT_CHANNELS;

  if (Object.keys(channels).length === 0) {
    console.log(`⚠ No matching topics. Available: ${Object.keys(DEFAULT_CHANNELS).join(', ')}`);
    process.exit(1);
  }

  const configPath: string = NewsRadar.findConfig();
  console.log('━'.repeat(60));
  console.log(`📡 News Radar — ${NewsRadar.nowStr()}`);
  console.log(`   Config:  ${configPath}`);
  console.log(`   Topics:  ${Object.keys(channels).join(', ')}`);
  console.log('━'.repeat(60));

  const agent: Agent = await Agent.create(configPath);
  const radar = new NewsRadar(agent, configPath);
  await radar.runAll(channels);
}

main().catch((e: unknown) => {
  const err = e as { message?: string };
  console.error('❌ Fatal:', err.message);
  process.exit(1);
});
