//! Cron-driven schedules for filesystem-first agents (P1).
//!
//! A `schedules/<name>.md` file (parsed into a [`ScheduleSpec`] by the agent-dir
//! convention) fires a markdown prompt into the agent on a cron cadence. The
//! firing itself goes through a [`ScheduleSink`] — the serve daemon implements it
//! to route the prompt through `AgentSession::send`, so a scheduled turn is a
//! FULL harness turn (context, safety gate, verification), never a raw model call.

use std::str::FromStr;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use cron::Schedule;
use tokio_util::sync::CancellationToken;

use crate::config::ScheduleSpec;
use crate::error::{CodeError, Result};

/// A parsed, fireable schedule: a [`ScheduleSpec`] with its compiled cron.
pub struct ScheduledJob {
    pub spec: ScheduleSpec,
    schedule: Schedule,
}

impl ScheduledJob {
    /// Parse a spec's cron. Accepts standard 5-field cron
    /// (`min hour dom mon dow`) and the 6-field form (`sec min hour dom mon dow`).
    pub fn parse(spec: ScheduleSpec) -> Result<Self> {
        let schedule = Schedule::from_str(&normalize_cron(&spec.cron)).map_err(|e| {
            CodeError::Context(format!(
                "schedule '{}' has an invalid cron '{}': {e}",
                spec.name, spec.cron
            ))
        })?;
        Ok(Self { spec, schedule })
    }

    /// The next fire time strictly after `now` (None if it never fires again).
    pub fn next_fire_after(&self, now: DateTime<Utc>) -> Option<DateTime<Utc>> {
        self.schedule.after(&now).next()
    }
}

/// Normalize a cron expression to the 6-field form the `cron` crate expects:
/// standard 5-field cron gets a leading `0` seconds field.
fn normalize_cron(expr: &str) -> String {
    let expr = expr.trim();
    match expr.split_whitespace().count() {
        5 => format!("0 {expr}"),
        _ => expr.to_string(),
    }
}

/// What to do when a schedule fires. The serve daemon implements this to drive
/// the schedule's markdown prompt into `AgentSession::send` (a full harness turn).
#[async_trait::async_trait]
pub trait ScheduleSink: Send + Sync {
    async fn fire(&self, spec: &ScheduleSpec);
}

/// Drives a set of cron schedules: for each enabled job, sleep until its next
/// fire, invoke the sink, repeat — until cancelled.
pub struct Scheduler {
    jobs: Vec<ScheduledJob>,
}

impl Scheduler {
    /// Build from specs, skipping disabled ones. Errors on any invalid cron.
    pub fn new(specs: impl IntoIterator<Item = ScheduleSpec>) -> Result<Self> {
        let jobs = specs
            .into_iter()
            .filter(|s| s.enabled)
            .map(ScheduledJob::parse)
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { jobs })
    }

    /// Number of active (enabled, valid) jobs.
    pub fn job_count(&self) -> usize {
        self.jobs.len()
    }

    /// Run until `cancel` fires; one independent loop per job.
    pub async fn run(self, sink: Arc<dyn ScheduleSink>, cancel: CancellationToken) {
        let mut handles = Vec::new();
        for job in self.jobs {
            let sink = Arc::clone(&sink);
            let cancel = cancel.clone();
            handles.push(tokio::spawn(run_job(job, sink, cancel)));
        }
        for h in handles {
            let _ = h.await;
        }
    }
}

async fn run_job(job: ScheduledJob, sink: Arc<dyn ScheduleSink>, cancel: CancellationToken) {
    loop {
        let now = Utc::now();
        let Some(next) = job.next_fire_after(now) else {
            return; // never fires again
        };
        let wait = (next - now)
            .to_std()
            .unwrap_or(std::time::Duration::from_secs(0));
        tokio::select! {
            _ = tokio::time::sleep(wait) => sink.fire(&job.spec).await,
            _ = cancel.cancelled() => return,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn spec(name: &str, cron: &str, enabled: bool) -> ScheduleSpec {
        ScheduleSpec {
            name: name.to_string(),
            cron: cron.to_string(),
            prompt: "do the thing".to_string(),
            enabled,
        }
    }

    #[test]
    fn parses_5_field_cron_and_computes_next_fire() {
        let job = ScheduledJob::parse(spec("daily", "0 9 * * *", true)).unwrap();
        // Before 09:00 → fires at 09:00 the same day.
        let before = DateTime::parse_from_rfc3339("2026-01-01T08:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        assert_eq!(
            job.next_fire_after(before).unwrap().to_rfc3339(),
            "2026-01-01T09:00:00+00:00"
        );
        // After 09:00 → next is 09:00 the following day.
        let after = DateTime::parse_from_rfc3339("2026-01-01T10:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        assert_eq!(
            job.next_fire_after(after).unwrap().to_rfc3339(),
            "2026-01-02T09:00:00+00:00"
        );
    }

    #[test]
    fn invalid_cron_is_an_error() {
        assert!(ScheduledJob::parse(spec("bad", "not a cron", true)).is_err());
    }

    #[test]
    fn scheduler_skips_disabled_and_rejects_bad_cron() {
        let s = Scheduler::new([
            spec("a", "0 9 * * *", true),
            spec("b", "*/5 * * * *", false), // disabled → skipped
        ])
        .unwrap();
        assert_eq!(s.job_count(), 1);

        assert!(Scheduler::new([spec("bad", "xyz", true)]).is_err());
    }

    #[tokio::test]
    async fn fires_at_least_once_then_stops_on_cancel() {
        struct CountSink(Arc<AtomicUsize>);
        #[async_trait::async_trait]
        impl ScheduleSink for CountSink {
            async fn fire(&self, _spec: &ScheduleSpec) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let fires = Arc::new(AtomicUsize::new(0));
        // 6-field cron = every second; a short real wait yields >= 1 fire without
        // paused-clock plumbing (run_job mixes wall-clock next-fire with the timer).
        let s = Scheduler::new([spec("sec", "* * * * * *", true)]).unwrap();
        let cancel = CancellationToken::new();
        let sink = Arc::new(CountSink(Arc::clone(&fires)));
        let handle = tokio::spawn(s.run(sink, cancel.clone()));
        tokio::time::sleep(std::time::Duration::from_millis(2200)).await;
        cancel.cancel();
        let _ = handle.await;
        assert!(
            fires.load(Ordering::SeqCst) >= 1,
            "expected at least one fire in ~2.2s, got {}",
            fires.load(Ordering::SeqCst)
        );
    }
}
