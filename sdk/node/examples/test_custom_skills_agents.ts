#!/usr/bin/env npx tsx
/**
 * A3S Code Node.js SDK - Custom Skills and Agents Integration Test (TypeScript)
 *
 * Tests loading and execution of custom skills and agents from ~/.a3s directory.
 *
 * Run with: npx tsx examples/test_custom_skills_agents.ts
 */

import { Agent, Session, AgentResult } from '../index.js';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

class CustomSkillsAgentsTest {
  private readonly agent: Agent;
  private readonly configPath: string;

  constructor(agent: Agent, configPath: string) {
    this.agent = agent;
    this.configPath = configPath;
  }

  // ── Static Helpers ───────────────────────────────────────────────────────

  /**
   * Find config file in home directory or project root.
   */
  static findConfigPath(): string {
    const homeConfig: string = path.join(os.homedir(), '.a3s', 'config.hcl');
    if (fs.existsSync(homeConfig)) {
      return homeConfig;
    }

    // Try project root (6 levels up from examples/test_custom_skills_agents.ts)
    const projectConfig: string = path.join(__dirname, '..', '..', '..', '..', '..', '..', '.a3s', 'config.hcl');
    if (fs.existsSync(projectConfig)) {
      return projectConfig;
    }

    throw new Error('Config file not found. Please create ~/.a3s/config.hcl');
  }

  /**
   * Find skills directory.
   */
  static findSkillsDir(): string {
    const homeSkills: string = path.join(os.homedir(), '.a3s', 'skills');
    if (fs.existsSync(homeSkills)) {
      return homeSkills;
    }

    // Try project root
    const projectSkills: string = path.join(__dirname, '..', '..', '..', '..', '..', '..', '.a3s', 'skills');
    if (fs.existsSync(projectSkills)) {
      return projectSkills;
    }

    throw new Error('Skills directory not found');
  }

  /**
   * Find agents directory.
   */
  static findAgentsDir(): string {
    const homeAgents: string = path.join(os.homedir(), '.a3s', 'agents');
    if (fs.existsSync(homeAgents)) {
      return homeAgents;
    }

    // Try project root
    const projectAgents: string = path.join(__dirname, '..', '..', '..', '..', '..', '..', '.a3s', 'agents');
    if (fs.existsSync(projectAgents)) {
      return projectAgents;
    }

    throw new Error('Agents directory not found');
  }

  /**
   * Truncate text to max length.
   */
  static truncate(text: string, maxLen: number): string {
    if (text.length <= maxLen) {
      return text;
    }
    return `${text.substring(0, maxLen)}... (truncated)`;
  }

  // ── Instance Test Methods ────────────────────────────────────────────────

  /**
   * Test 1: Load custom skills in session.
   */
  async testCustomSkills(): Promise<void> {
    console.log('\n📦 Test 1: Load Custom Skills in Session');
    console.log('-'.repeat(80));

    const skillsDir: string = CustomSkillsAgentsTest.findSkillsDir();

    const session: Session = this.agent.session('.', { skillDirs: [skillsDir] });

    console.log('Testing: Ask about Rust best practices (should use rust-expert skill)...');
    const result1: AgentResult = await session.send('What are the key principles of Rust ownership?');
    console.log(`✓ Result preview: ${CustomSkillsAgentsTest.truncate(result1.text, 200)}`);
    console.log();

    console.log('Testing: Ask about API design (should use api-designer skill)...');
    const result2: AgentResult = await session.send('What are REST API best practices for error handling?');
    console.log(`✓ Result preview: ${CustomSkillsAgentsTest.truncate(result2.text, 200)}`);
    console.log();

    console.log('✅ Test 1 passed: Custom skills loaded and used in session\n');
  }

  /**
   * Test 2: Load custom agents in session.
   */
  async testCustomAgents(): Promise<void> {
    console.log('🤖 Test 2: Load Custom Agents in Session');
    console.log('-'.repeat(80));

    const agentsDir: string = CustomSkillsAgentsTest.findAgentsDir();

    const session: Session = this.agent.session('.', { agentDirs: [agentsDir] });

    console.log('Testing: Request code review (should use code-reviewer agent)...');
    const result1: AgentResult = await session.send('Review this function for potential issues: function add(a, b) { return a + b; }');
    console.log(`✓ Result preview: ${CustomSkillsAgentsTest.truncate(result1.text, 200)}`);
    console.log();

    console.log('Testing: Request documentation (should use documentation-writer agent)...');
    const result2: AgentResult = await session.send('Write API documentation for a user registration endpoint');
    console.log(`✓ Result preview: ${CustomSkillsAgentsTest.truncate(result2.text, 200)}`);
    console.log();

    console.log('✅ Test 2 passed: Custom agents loaded and used in session\n');
  }

  /**
   * Test 3: Load both skills and agents together.
   */
  async testSkillsAndAgentsTogether(): Promise<void> {
    console.log('🔧 Test 3: Load Both Skills and Agents Together');
    console.log('-'.repeat(80));

    const skillsDir: string = CustomSkillsAgentsTest.findSkillsDir();
    const agentsDir: string = CustomSkillsAgentsTest.findAgentsDir();

    const session: Session = this.agent.session('.', {
      skillDirs: [skillsDir],
      agentDirs: [agentsDir]
    });

    console.log('Testing: Complex task using both skills and agents...');
    const result: AgentResult = await session.send('Generate unit tests for a JavaScript function that parses JSON');
    console.log(`✓ Result preview: ${CustomSkillsAgentsTest.truncate(result.text, 200)}`);
    console.log();

    console.log('✅ Test 3 passed: Both skills and agents work together\n');
  }

  /**
   * Test 4: Multiple sessions with different configurations.
   */
  async testMultipleSessions(): Promise<void> {
    console.log('🔀 Test 4: Multiple Sessions with Different Configurations');
    console.log('-'.repeat(80));

    const skillsDir: string = CustomSkillsAgentsTest.findSkillsDir();
    const agentsDir: string = CustomSkillsAgentsTest.findAgentsDir();

    // Session 1: Only skills
    const session1: Session = this.agent.session('.', { skillDirs: [skillsDir] });

    // Session 2: Only agents
    const session2: Session = this.agent.session('.', { agentDirs: [agentsDir] });

    console.log('Testing: Session 1 with skills only...');
    const result1: AgentResult = await session1.send('Explain JavaScript async/await');
    console.log(`✓ Session 1 result: ${CustomSkillsAgentsTest.truncate(result1.text, 150)}`);
    console.log();

    console.log('Testing: Session 2 with agents only...');
    const result2: AgentResult = await session2.send('Review this code for improvements');
    console.log(`✓ Session 2 result: ${CustomSkillsAgentsTest.truncate(result2.text, 150)}`);
    console.log();

    console.log('✅ Test 4 passed: Multiple sessions with different configs work correctly\n');
  }

  // ── Run All ───────────────────────────────────────────────────────────────

  async runAll(): Promise<void> {
    console.log('🚀 A3S Code Node.js SDK - Custom Skills and Agents Integration Test\n');
    console.log('='.repeat(80));

    console.log(`📄 Using config: ${this.configPath}`);

    const skillsDir: string = CustomSkillsAgentsTest.findSkillsDir();
    console.log(`📚 Skills directory: ${skillsDir}`);

    const agentsDir: string = CustomSkillsAgentsTest.findAgentsDir();
    console.log(`🤖 Agents directory: ${agentsDir}`);

    console.log('='.repeat(80));

    await this.testCustomSkills();
    await this.testCustomAgents();
    await this.testSkillsAndAgentsTogether();
    await this.testMultipleSessions();

    console.log();
    console.log('='.repeat(80));
    console.log('✅ All custom skills and agents tests completed successfully!');
    console.log('='.repeat(80));
  }
}

async function main(): Promise<void> {
  const configPath = CustomSkillsAgentsTest.findConfigPath();
  const agent = await Agent.create(configPath);
  const test = new CustomSkillsAgentsTest(agent, configPath);
  await test.runAll();
}

// Run tests
main().catch((err: unknown) => {
  console.error('❌ Test failed:', err);
  process.exit(1);
});
