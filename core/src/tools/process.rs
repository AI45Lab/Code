//! Process output reading utility

use super::MAX_OUTPUT_SIZE;
use crate::workspace::CommandOutputObserver;
use tokio::io::AsyncBufReadExt;
use tokio::io::BufReader;
use tokio::process::Child;

pub(crate) async fn read_process_output(
    child: &mut Child,
    timeout_secs: u64,
    observer: Option<&dyn CommandOutputObserver>,
) -> (String, bool) {
    let stdout = match child.stdout.take() {
        Some(s) => s,
        None => {
            return ("Internal error: child stdout not piped".to_string(), false);
        }
    };
    let stderr = match child.stderr.take() {
        Some(s) => s,
        None => {
            return ("Internal error: child stderr not piped".to_string(), false);
        }
    };

    let mut stdout_reader = BufReader::new(stdout).lines();
    let mut stderr_reader = BufReader::new(stderr).lines();

    let mut output = String::new();
    let mut total_size = 0usize;
    let mut stdout_done = false;
    let mut stderr_done = false;

    let timeout = tokio::time::Duration::from_secs(timeout_secs);
    let result = tokio::time::timeout(timeout, async {
        loop {
            if stdout_done && stderr_done {
                break;
            }
            tokio::select! {
                line = stdout_reader.next_line(), if !stdout_done => {
                    match line {
                        Ok(Some(line)) => {
                            if total_size < MAX_OUTPUT_SIZE {
                                output.push_str(&line);
                                output.push('\n');
                                total_size += line.len() + 1;
                            }
                            if let Some(obs) = observer {
                                let mut delta = line;
                                delta.push('\n');
                                obs.on_output_delta(&delta).await;
                            }
                        }
                        Ok(None) => stdout_done = true,
                        Err(_) => stdout_done = true,
                    }
                }
                line = stderr_reader.next_line(), if !stderr_done => {
                    match line {
                        Ok(Some(line)) => {
                            if total_size < MAX_OUTPUT_SIZE {
                                output.push_str(&line);
                                output.push('\n');
                                total_size += line.len() + 1;
                            }
                            if let Some(obs) = observer {
                                let mut delta = line;
                                delta.push('\n');
                                obs.on_output_delta(&delta).await;
                            }
                        }
                        Ok(None) => stderr_done = true,
                        Err(_) => stderr_done = true,
                    }
                }
            }
        }
    })
    .await;

    if result.is_err() {
        child.kill().await.ok();
        return (output, true);
    }

    (output, false)
}
