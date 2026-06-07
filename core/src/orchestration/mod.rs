//! Programmable, deterministic multi-agent orchestration.
//!
//! Today an agent fans work out only by *model-driven* delegation (the LLM
//! decides to call the `task` / `parallel_task` tool). This module adds a
//! *programmable* layer: a developer expresses fan-out / pipelines /
//! verification panels as code, so the orchestration is reproducible,
//! testable, budget-bounded, and (later) resumable — independent of what the
//! model chooses to do.
//!
//! ## The framework / host boundary
//!
//! Everything here is written against one seam, [`AgentExecutor`]: "run this
//! step, give me the result." The framework owns the **grammar** (which steps,
//! how they compose, the concurrency *hint*, the data contracts
//! [`AgentStepSpec`] / [`StepOutcome`]) and ships a default executor that runs
//! every step locally. The **placement** of those steps across a cluster —
//! transport, scheduling, where a step actually executes — belongs to the
//! host, which implements [`AgentExecutor`]. The combinators never
//! observe where a step ran, so the same orchestration scales from one process
//! to a cluster without change.
//!
//! The in-box implementation is [`crate::tools::TaskExecutor`].

mod checkpoint;
mod combinators;
mod executor;
mod workflow;
mod workflow_budget;

pub use checkpoint::{WorkflowCheckpoint, WorkflowStepRecord, WORKFLOW_CHECKPOINT_SCHEMA_VERSION};
pub use combinators::{
    execute_loop, execute_pipeline, execute_steps_parallel_resumable, LoopDecision, PipelineStage,
};
pub use executor::{execute_steps_parallel, AgentExecutor, AgentStepSpec, StepOutcome};
pub use workflow::{Workflow, WorkflowBuilder, WorkflowEvent};
pub use workflow_budget::{BudgetSnapshot, WorkflowBudget};
