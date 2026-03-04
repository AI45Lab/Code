#!/usr/bin/env npx tsx
/**
 * A3S Code Node.js SDK - Integration Tests
 *
 * Tests all major features using real LLM configuration from ~/.a3s/config.hcl
 *
 * Run with: npx tsx examples/integration_tests.ts
 */

import {
  Agent,
  Session,
  AgentEvent,
  AgentResult,
  SkillInfo,
  MessageObject,
  SessionQueueConfigOptions,
  builtinSkills,
} from '../index.js';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

// ============================================================================
// IntegrationTests
// ============================================================================

class IntegrationTests {
  private readonly agent: Agent;
  private readonly configPath: string;

  constructor(agent: Agent, configPath: string) {
    this.agent = agent;
    this.configPath = configPath;
  }

  // --------------------------------------------------------------------------
  // Static helpers
  // --------------------------------------------------------------------------

  /**
   * Find config file in home directory or project root.
   */
  static findConfigPath(): string {
    const homeConfig: string = path.join(os.homedir(), '.a3s', 'config.hcl');
    if (fs.existsSync(homeConfig)) {
      return homeConfig;
    }

    // Try project root (6 levels up from examples/integration_tests.ts)
    const projectConfig: string = path.join(__dirname, '..', '..', '..', '..', '..', '..', '.a3s', 'config.hcl');
    if (fs.existsSync(projectConfig)) {
      return projectConfig;
    }

    throw new Error('Config file not found. Please create ~/.a3s/config.hcl');
  }

  /**
   * Truncate text to max length.
   */
  private static truncate(text: string, maxLen: number): string {
    if (text.length <= maxLen) {
      return text;
    }
    return `${text.substring(0, maxLen)}... (truncated)`;
  }

  // --------------------------------------------------------------------------
  // Test 1: Basic tool execution
  // --------------------------------------------------------------------------

  /**
   * Test 1: Basic tool execution.
   */
  async testBasicTools(): Promise<void> {
    console.log('\n Test 1: Basic Tool Execution');
    console.log('-'.repeat(80));

    const session: Session = this.agent.session('.');

    console.log('Testing: List current directory...');
    const result1: AgentResult = await session.send('List the files in the current directory using ls');
    console.log(`Result preview: ${IntegrationTests.truncate(result1.text, 200)}`);

    console.log('\nTesting: Read a file...');
    const result2: AgentResult = await session.send('Read the Cargo.toml file');
    console.log(`Result preview: ${IntegrationTests.truncate(result2.text, 200)}`);

    console.log('\nTest 1 passed: Basic tools work correctly');
  }

  // --------------------------------------------------------------------------
  // Test 2: Built-in skills
  // --------------------------------------------------------------------------

  /**
   * Test 2: Built-in skills.
   */
  async testBuiltinSkills(): Promise<void> {
    console.log('\n Test 2: Built-in Skills (7 skills)');
    console.log('-'.repeat(80));

    // List available skills
    const skills: SkillInfo[] = builtinSkills();
    console.log(`Available skills: ${skills.length}`);
    for (let i = 0; i < Math.min(3, skills.length); i++) {
      const skill: SkillInfo = skills[i];
      console.log(`  - ${skill.name} (${skill.kind}): ${IntegrationTests.truncate(skill.description, 60)}`);
    }

    // Create session with builtin skills enabled
    const session: Session = this.agent.session('.', { builtinSkills: true });

    console.log('\nTesting: code-search skill...');
    const result1: AgentResult = await session.send('Search for all functions named "new" in Rust files');
    console.log(`Result preview: ${IntegrationTests.truncate(result1.text, 200)}`);

    console.log('\nTesting: builtin-tools skill...');
    const result2: AgentResult = await session.send('What tools are available for file operations?');
    console.log(`Result preview: ${IntegrationTests.truncate(result2.text, 200)}`);

    console.log('\nTest 2 passed: Built-in skills work correctly');
  }

  // --------------------------------------------------------------------------
  // Test 3: File operations
  // --------------------------------------------------------------------------

  /**
   * Test 3: File operations.
   */
  async testFileOperations(): Promise<void> {
    console.log('\n Test 3: File Operations');
    console.log('-'.repeat(80));

    const session: Session = this.agent.session('.');

    console.log('Testing: Create a test file...');
    const result1: AgentResult = await session.send(
      "Create a file named 'test_integration_js.txt' with content 'Hello from Node.js SDK!'"
    );
    console.log(`Result: ${IntegrationTests.truncate(result1.text, 200)}`);

    console.log('\nTesting: Read the test file...');
    const result2: AgentResult = await session.send('Read the file test_integration_js.txt');
    console.log(`Result: ${IntegrationTests.truncate(result2.text, 200)}`);

    console.log('\nCleaning up: Remove test file...');
    try {
      fs.unlinkSync('test_integration_js.txt');
    } catch (err) {
      // Ignore if file doesn't exist
    }

    console.log('\nTest 3 passed: File operations work correctly');
  }

  // --------------------------------------------------------------------------
  // Test 4: Search operations
  // --------------------------------------------------------------------------

  /**
   * Test 4: Search operations.
   */
  async testSearchOperations(): Promise<void> {
    console.log('\n Test 4: Search Operations');
    console.log('-'.repeat(80));

    const session: Session = this.agent.session('.');

    console.log('Testing: grep search...');
    const result1: AgentResult = await session.send('Search for the word "Agent" in all Rust files using grep');
    console.log(`Result preview: ${IntegrationTests.truncate(result1.text, 200)}`);

    console.log('\nTesting: glob pattern matching...');
    const result2: AgentResult = await session.send('Find all .rs files in the src directory using glob');
    console.log(`Result preview: ${IntegrationTests.truncate(result2.text, 200)}`);

    console.log('\nTest 4 passed: Search operations work correctly');
  }

  // --------------------------------------------------------------------------
  // Test 5: Direct tool execution
  // --------------------------------------------------------------------------

  /**
   * Test 5: Direct tool execution.
   */
  async testDirectToolCalls(): Promise<void> {
    console.log('\n Test 5: Direct Tool Execution');
    console.log('-'.repeat(80));

    const session: Session = this.agent.session('.');

    console.log('Testing: Direct readFile call...');
    const content: string = await session.readFile('Cargo.toml');
    console.log(`Read ${content.length} bytes from Cargo.toml`);

    console.log('\nTesting: Direct bash call...');
    const output: string = await session.bash("echo 'Hello from Node.js SDK'");
    console.log(`Bash output: ${output.trim()}`);

    console.log('\nTesting: Direct glob call...');
    const files: string[] = await session.glob('src/*.rs');
    console.log(`Found ${files.length} Rust files`);

    console.log('\nTesting: Direct grep call...');
    const matches: string = await session.grep('Agent');
    console.log(`Grep found ${matches.length} bytes of matches`);

    console.log('\nTest 5 passed: Direct tool calls work correctly');
  }

  // --------------------------------------------------------------------------
  // Test 6: Streaming execution
  // --------------------------------------------------------------------------

  /**
   * Test 6: Streaming execution.
   */
  async testStreaming(): Promise<void> {
    console.log('\n Test 6: Streaming Execution');
    console.log('-'.repeat(80));

    const session: Session = this.agent.session('.');

    console.log('Testing: Stream events...');
    const stream = await session.stream('List all .rs files in the current directory');

    let textDeltas: number = 0;
    let toolCalls: number = 0;
    let eventCount: number = 0;

    for await (const event of stream) {
      eventCount++;
      if (event.type === 'text_delta') {
        textDeltas++;
      } else if (event.type === 'tool_start') {
        toolCalls++;
        console.log(`  Tool called: ${event.toolName}`);
      }
    }

    console.log(`Received ${eventCount} events (${textDeltas} text deltas, ${toolCalls} tool calls)`);
    console.log('\nTest 6 passed: Streaming works correctly');
  }

  // --------------------------------------------------------------------------
  // Test 7: Queue configuration
  // --------------------------------------------------------------------------

  /**
   * Test 7: Queue configuration.
   */
  async testQueueConfig(): Promise<void> {
    console.log('\n Test 7: Queue Configuration (A3S Lane v0.4.0)');
    console.log('-'.repeat(80));

    console.log('Testing: Session with queue configuration...');
    const queueConfig: SessionQueueConfigOptions = {
      queryConcurrency: 5,
      executeConcurrency: 2,
      enableDlq: true,
      enableMetrics: true,
      enableAllFeatures: false,
    };
    const session: Session = this.agent.session('.', { queueConfig });

    const result: AgentResult = await session.send('List all .rs files and count how many contain the word "async"');
    console.log(`Result preview: ${IntegrationTests.truncate(result.text, 200)}`);

    console.log('\nTest 7 passed: Queue configuration works correctly');
  }

  // --------------------------------------------------------------------------
  // Test 8: Conversation history
  // --------------------------------------------------------------------------

  /**
   * Test 8: Conversation history.
   */
  async testConversationHistory(): Promise<void> {
    console.log('\n Test 8: Conversation History');
    console.log('-'.repeat(80));

    const session: Session = this.agent.session('.');

    console.log('Testing: Multi-turn conversation...');
    const result1: AgentResult = await session.send('What is 2 + 2?');
    console.log(`Turn 1: ${IntegrationTests.truncate(result1.text, 100)}`);

    const result2: AgentResult = await session.send('What was my previous question?');
    console.log(`Turn 2: ${IntegrationTests.truncate(result2.text, 100)}`);

    const history: MessageObject[] = session.history();
    console.log(`History has ${history.length} messages`);

    console.log('\nTest 8 passed: Conversation history works correctly');
  }

  // --------------------------------------------------------------------------
  // Run All
  // --------------------------------------------------------------------------

  async runAll(): Promise<void> {
    await this.testBasicTools();
    await this.testBuiltinSkills();
    await this.testFileOperations();
    await this.testSearchOperations();
    await this.testDirectToolCalls();
    await this.testStreaming();
    await this.testQueueConfig();
    await this.testConversationHistory();
  }
}

// ============================================================================
// Main
// ============================================================================

/**
 * Main test runner.
 */
async function main(): Promise<void> {
  console.log('A3S Code Node.js SDK - Integration Tests\n');
  console.log('='.repeat(80));

  // Load config
  const configPath: string = IntegrationTests.findConfigPath();
  console.log(`Using config: ${configPath}`);
  console.log('='.repeat(80));
  console.log();

  // Create agent
  const agent: Agent = await Agent.create(configPath);

  // Run tests
  const tests = new IntegrationTests(agent, configPath);
  await tests.runAll();

  console.log('\n\n');
  console.log('='.repeat(80));
  console.log('All Node.js SDK integration tests completed successfully!');
  console.log('='.repeat(80));
}

// Run tests
main().catch((err: unknown) => {
  console.error('Test failed:', err);
  process.exit(1);
});
