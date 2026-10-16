//! Governor state machine (PRD §9).
//!
//! Normal → Constrained → Paused → Draining → Failed. Warning pressure
//! evicts idle caches and stops speculative prefetch growth; critical
//! pressure stops new compute and I/O and pauses at the next safe boundary.
//! Resume requires a configurable stability interval (default 10 seconds) of
//! improved conditions to prevent oscillation. In-flight buffers are never
//! reclaimed to fake fast cancellation: draining waits for completion.
//!
//! Thermal throttling reduces work submission and speculation; it never
//! changes target precision or model semantics.

use std::time::{Duration, Instant};

use crate::pressure::{PressureLevel, ThermalLevel};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GovState {
    Normal,
    Constrained,
    Paused,
    Draining,
    Failed,
}

impl GovState {
    pub fn as_str(self) -> &'static str {
        match self {
            GovState::Normal => "normal",
            GovState::Constrained => "constrained",
            GovState::Paused => "paused",
            GovState::Draining => "draining",
            GovState::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GovernorSignal {
    Pressure(PressureLevel),
    Thermal(ThermalLevel),
    StorageLost(String),
    WorkerFailed(String),
    /// The condition that caused a pause has cleared (caller-observed).
    ConditionsImproved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GovernorAction {
    TrimCaches,
    StopPrefetchGrowth,
    PauseComputeAndIo,
    ResumeReads,
    ReduceSubmissionAndSpeculation,
    DrainThenRelease,
    FailRequest,
}

#[derive(Debug, Clone)]
pub struct GovernorConfig {
    /// Improved conditions must persist this long before resuming.
    pub stability_interval: Duration,
    /// Minimum dwell in a state before a further de-escalation transition
    /// (escalations are always immediate).
    pub min_dwell: Duration,
}

impl Default for GovernorConfig {
    fn default() -> Self {
        Self {
            stability_interval: Duration::from_secs(10),
            min_dwell: Duration::from_secs(2),
        }
    }
}

#[derive(Debug)]
pub struct Governor {
    pub config: GovernorConfig,
    state: GovState,
    entered_at: Instant,
    improved_since: Option<Instant>,
    last_pressure: PressureLevel,
    transitions: Vec<(Instant, GovState, String)>,
}

impl Governor {
    pub fn new(config: GovernorConfig) -> Self {
        Self {
            config,
            state: GovState::Normal,
            entered_at: Instant::now(),
            improved_since: None,
            last_pressure: PressureLevel::Normal,
            transitions: Vec::new(),
        }
    }

    pub fn state(&self) -> GovState {
        self.state
    }

    pub fn transitions(&self) -> &[(Instant, GovState, String)] {
        &self.transitions
    }

    /// Feed a signal; returns the actions to perform in order.
    pub fn on_signal(&mut self, signal: GovernorSignal, now: Instant) -> Vec<GovernorAction> {
        match signal {
            GovernorSignal::StorageLost(why) => {
                self.transition(GovState::Failed, now, format!("storage lost: {why}"));
                vec![
                    GovernorAction::DrainThenRelease,
                    GovernorAction::FailRequest,
                ]
            }
            GovernorSignal::WorkerFailed(why) => {
                self.transition(GovState::Failed, now, format!("worker failed: {why}"));
                vec![
                    GovernorAction::DrainThenRelease,
                    GovernorAction::FailRequest,
                ]
            }
            GovernorSignal::Pressure(level) => self.on_pressure(level, now),
            GovernorSignal::Thermal(level) => self.on_thermal(level, now),
            GovernorSignal::ConditionsImproved => self.try_resume(now),
        }
    }

    fn on_pressure(&mut self, level: PressureLevel, now: Instant) -> Vec<GovernorAction> {
        self.last_pressure = level.clone();
        match level {
            PressureLevel::Normal => self.try_resume(now),
            PressureLevel::Warning => {
                self.improved_since = None;
                let mut actions = Vec::new();
                if matches!(self.state, GovState::Normal) {
                    self.transition(GovState::Constrained, now, "warning pressure".into());
                }
                if self.state == GovState::Constrained {
                    actions.push(GovernorAction::TrimCaches);
                    actions.push(GovernorAction::StopPrefetchGrowth);
                } else if matches!(self.state, GovState::Paused) {
                    // Still constrained; stay paused.
                }
                actions
            }
            PressureLevel::Critical => {
                self.improved_since = None;
                if self.state != GovState::Paused {
                    self.transition(GovState::Paused, now, "critical pressure".into());
                    return vec![GovernorAction::PauseComputeAndIo];
                }
                Vec::new()
            }
            PressureLevel::Unknown(_) => {
                // Unknown readings neither pause nor resume; report nothing.
                Vec::new()
            }
        }
    }

    fn on_thermal(&mut self, level: ThermalLevel, now: Instant) -> Vec<GovernorAction> {
        match level {
            ThermalLevel::Serious => {
                if matches!(self.state, GovState::Normal) {
                    self.transition(GovState::Constrained, now, "thermal: serious".into());
                }
                vec![GovernorAction::ReduceSubmissionAndSpeculation]
            }
            ThermalLevel::Critical => {
                if self.state != GovState::Paused {
                    self.transition(GovState::Paused, now, "thermal: critical".into());
                    return vec![GovernorAction::PauseComputeAndIo];
                }
                Vec::new()
            }
            ThermalLevel::Nominal | ThermalLevel::Fair | ThermalLevel::Unknown(_) => Vec::new(),
        }
    }

    /// Resume only after the stability interval of improved conditions.
    fn try_resume(&mut self, now: Instant) -> Vec<GovernorAction> {
        match self.state {
            GovState::Paused => {
                // Improvement must persist.
                let since = *self.improved_since.get_or_insert(now);
                if now.duration_since(since) >= self.config.stability_interval {
                    self.transition(
                        GovState::Normal,
                        now,
                        format!(
                            "conditions improved for {:?}",
                            self.config.stability_interval
                        ),
                    );
                    self.improved_since = None;
                    return vec![GovernorAction::ResumeReads];
                }
                Vec::new()
            }
            GovState::Constrained => {
                let since = *self.improved_since.get_or_insert(now);
                if now.duration_since(since) >= self.config.stability_interval {
                    self.transition(GovState::Normal, now, "conditions normal".into());
                    self.improved_since = None;
                }
                Vec::new()
            }
            _ => Vec::new(),
        }
    }

    fn transition(&mut self, to: GovState, now: Instant, why: String) {
        if self.state == to {
            return;
        }
        // Escalations are immediate; de-escalations respect a minimum dwell.
        let escalating = to as u8 > self.state as u8;
        if !escalating && now.duration_since(self.entered_at) < self.config.min_dwell {
            return;
        }
        self.state = to;
        self.entered_at = now;
        self.transitions.push((now, to, why));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gov() -> Governor {
        Governor::new(GovernorConfig {
            stability_interval: Duration::from_secs(10),
            min_dwell: Duration::from_secs(2),
        })
    }

    #[test]
    fn warning_enters_constrained_and_trims() {
        let mut g = gov();
        let now = Instant::now();
        let actions = g.on_signal(GovernorSignal::Pressure(PressureLevel::Warning), now);
        assert_eq!(g.state(), GovState::Constrained);
        assert!(actions.contains(&GovernorAction::TrimCaches));
        assert!(actions.contains(&GovernorAction::StopPrefetchGrowth));
    }

    #[test]
    fn critical_pauses_immediately_even_below_dwell() {
        let mut g = gov();
        let t0 = Instant::now();
        g.on_signal(GovernorSignal::Pressure(PressureLevel::Warning), t0);
        let actions = g.on_signal(
            GovernorSignal::Pressure(PressureLevel::Critical),
            t0 + Duration::from_millis(100),
        );
        assert_eq!(g.state(), GovState::Paused);
        assert_eq!(actions, vec![GovernorAction::PauseComputeAndIo]);
    }

    #[test]
    fn resume_requires_the_full_stability_interval() {
        let mut g = gov();
        let t0 = Instant::now();
        g.on_signal(GovernorSignal::Pressure(PressureLevel::Critical), t0);
        assert_eq!(g.state(), GovState::Paused);

        // Improved but only for 5 seconds: stay paused.
        let actions = g.on_signal(
            GovernorSignal::ConditionsImproved,
            t0 + Duration::from_secs(1),
        );
        assert!(actions.is_empty());
        assert_eq!(g.state(), GovState::Paused);
        g.on_signal(
            GovernorSignal::Pressure(PressureLevel::Normal),
            t0 + Duration::from_secs(5),
        );
        assert_eq!(g.state(), GovState::Paused, "5s < 10s stability window");

        // At 11+ seconds of improvement: resume.
        let actions = g.on_signal(
            GovernorSignal::ConditionsImproved,
            t0 + Duration::from_secs(12),
        );
        assert_eq!(g.state(), GovState::Normal);
        assert_eq!(actions, vec![GovernorAction::ResumeReads]);
    }

    #[test]
    fn oscillation_is_prevented() {
        let mut g = gov();
        let t0 = Instant::now();
        // Rapid alternation: normal → critical → normal → critical ...
        g.on_signal(GovernorSignal::Pressure(PressureLevel::Critical), t0);
        g.on_signal(
            GovernorSignal::ConditionsImproved,
            t0 + Duration::from_secs(1),
        );
        g.on_signal(
            GovernorSignal::Pressure(PressureLevel::Normal),
            t0 + Duration::from_secs(2),
        );
        // Still paused because improvement has not persisted.
        assert_eq!(g.state(), GovState::Paused);
        // A new critical keeps it paused without duplicate actions.
        let actions = g.on_signal(
            GovernorSignal::Pressure(PressureLevel::Critical),
            t0 + Duration::from_secs(3),
        );
        assert!(actions.is_empty());
    }

    #[test]
    fn thermal_serious_reduces_work_without_changing_state_semantics() {
        let mut g = gov();
        let actions = g.on_signal(
            GovernorSignal::Thermal(ThermalLevel::Serious),
            Instant::now(),
        );
        assert_eq!(g.state(), GovState::Constrained);
        assert_eq!(
            actions,
            vec![GovernorAction::ReduceSubmissionAndSpeculation]
        );
    }

    #[test]
    fn worker_failure_drains_and_fails() {
        let mut g = gov();
        let actions = g.on_signal(
            GovernorSignal::WorkerFailed("crashed".into()),
            Instant::now(),
        );
        assert_eq!(g.state(), GovState::Failed);
        assert_eq!(
            actions,
            vec![
                GovernorAction::DrainThenRelease,
                GovernorAction::FailRequest
            ]
        );
    }

    #[test]
    fn unknown_pressure_is_inert() {
        let mut g = gov();
        let actions = g.on_signal(
            GovernorSignal::Pressure(PressureLevel::Unknown("n/a".into())),
            Instant::now(),
        );
        assert!(actions.is_empty());
        assert_eq!(g.state(), GovState::Normal);
    }

    #[test]
    fn transitions_are_recorded_for_diagnostics() {
        let mut g = gov();
        let t0 = Instant::now();
        g.on_signal(GovernorSignal::Pressure(PressureLevel::Warning), t0);
        assert_eq!(g.transitions().len(), 1);
        assert_eq!(g.transitions()[0].1, GovState::Constrained);
    }
}
