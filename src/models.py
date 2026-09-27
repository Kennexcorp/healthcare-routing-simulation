"""Pydantic data models shared by every component, kept in one place to avoid circular imports."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from src.config import DIAGNOSIS_CATEGORIES, STATES


class Patient(BaseModel):
    """A single community healthcare patient with clinical state, biometric
    history, and visit tracking."""

    id: int
    age: int = Field(ge=45, le=85)
    diagnosis: str
    coords: tuple[float, float]
    personal_baseline_mean: dict[str, float]
    current_clinical_state: str = "low"
    elapsed_time_in_state_hours: float = 0.0
    sojourn_time_hours: float = 0.0
    biometric_history: list[list[float]] = Field(default_factory=list)
    time_since_last_visit_mins: float = 0.0
    time_first_high_risk_mins: float | None = None
    visited: bool = False

    model_config = {"validate_assignment": True}

    @field_validator("current_clinical_state")
    @classmethod
    def state_must_be_valid(cls, value: str) -> str:
        """Reject any state outside the three-state CTMC."""
        if value not in STATES:
            raise ValueError(f"Invalid CTMC state: {value}")
        return value

    @field_validator("diagnosis")
    @classmethod
    def diagnosis_must_be_valid(cls, value: str) -> str:
        """Reject any diagnosis outside the five study categories."""
        if value not in DIAGNOSIS_CATEGORIES:
            raise ValueError(f"Invalid diagnosis category: {value}")
        return value


class Worker(BaseModel):
    """A community healthcare worker with current location and shift tracking."""

    id: int
    current_location: tuple[float, float]
    current_time_mins: float = Field(default=0.0, ge=0)
    visits_completed: list[int] = Field(default_factory=list)
    # Default of 480 keeps call sites that don't care about shift length working unchanged.
    shift_duration_mins: int = Field(default=480, gt=0)

    model_config = {"validate_assignment": True}

    @model_validator(mode="after")
    def current_time_cannot_exceed_the_shift(self) -> Worker:
        """Checks the clock against the shift length, since a static Field bound can't reference the runtime-configurable shift_duration_mins.

        Raises:
            ValueError: If the clock has passed the end of the shift.
        """
        if self.current_time_mins > self.shift_duration_mins:
            raise ValueError(
                f"current_time_mins ({self.current_time_mins}) exceeds "
                f"shift_duration_mins ({self.shift_duration_mins})"
            )
        return self


class TriggerEvent(BaseModel):
    """A single AND-gate trigger event with its condition outcomes and quality metrics.

    Outcome fields stay unset until the lead time elapses, since a trigger can't be judged false the moment it fires.
    """

    patient_id: int
    timestamp_mins: float = Field(ge=0)
    ml_risk_label: str
    high_risk_probability: float = Field(ge=0.0, le=1.0)
    biometric_deviation_detected: bool
    time_since_last_visit_mins: float = Field(ge=0)
    ctmc_ground_truth_state: str
    ctmc_state_at_lead_time: str | None = None
    is_false_trigger: bool | None = None
    rerouting_churn_proportion: float = Field(ge=0.0, le=1.0)
    additional_travel_mins: float = Field(default=0.0, ge=0)
    shift_duration_mins: int = Field(default=480, gt=0)

    model_config = {"validate_assignment": True}

    @model_validator(mode="after")
    def timestamp_cannot_exceed_the_shift(self) -> TriggerEvent:
        """Check the trigger time against the shift length it fired within.

        Raises:
            ValueError: If the timestamp falls after the end of the shift.
        """
        if self.timestamp_mins > self.shift_duration_mins:
            raise ValueError(
                f"timestamp_mins ({self.timestamp_mins}) exceeds "
                f"shift_duration_mins ({self.shift_duration_mins})"
            )
        return self

    @field_validator("ml_risk_label", "ctmc_ground_truth_state")
    @classmethod
    def must_be_valid_risk_label(cls, value: str) -> str:
        """Reject any risk label outside the three-state scheme."""
        if value not in STATES:
            raise ValueError(f"Invalid risk label: {value}")
        return value

    @field_validator("ctmc_state_at_lead_time")
    @classmethod
    def lead_time_state_must_be_valid(cls, value: str | None) -> str | None:
        """Reject any state outside the scheme, while allowing 'not yet known'."""
        if value is not None and value not in STATES:
            raise ValueError(f"Invalid risk label: {value}")
        return value

    @property
    def is_resolved(self) -> bool:
        """Whether the outcome of this trigger is known yet."""
        return self.is_false_trigger is not None

    def resolve(self, ctmc_state_at_lead_time: str) -> None:
        """Records the ground-truth state one lead time after firing, since judging at fire time would wrongly flag anticipated deterioration as false."""
        self.ctmc_state_at_lead_time = ctmc_state_at_lead_time
        self.is_false_trigger = ctmc_state_at_lead_time != "high"


class RoutingSolution(BaseModel):
    """Outcome of a single routing solve, carrying solver diagnostics alongside the routes."""

    worker_route_assignment: dict[int, list[int]]
    solver_status: str
    solve_time_secs: float = Field(ge=0)
    is_feasible: bool
    dropped_patient_ids: list[int] = Field(default_factory=list)


class SimulationResult(BaseModel):
    """Validated output record for a single simulation replication.

    Passing every result through this model means a bad metric fails here rather than corrupting the results file.
    """

    replication_id: int
    seed: int
    scenario: str
    n_patients: int = Field(ge=0)
    mean_response_time_high_risk: float = Field(ge=0)
    total_travel_distance_km: float = Field(ge=0)
    workforce_utilisation: float = Field(ge=0.0, le=1.0)
    rerouting_events_per_shift: int = Field(ge=0)
    false_trigger_rate: float = Field(ge=0.0, le=1.0)
    rerouting_churn_proportion: float = Field(ge=0.0, le=1.0)
    mean_additional_travel_per_false_trigger: float = Field(ge=0)
    unvisited_patients: int = Field(ge=0)
    # Distinct from unvisited_patients: a patient seen early who later deteriorates and isn't revisited counts here.
    unreached_high_risk_patients: int = Field(ge=0)
    solver_feasible: bool
    mean_reroute_time_secs: float = Field(ge=0)
    confidence_interval_half_width: float = Field(ge=0)

    @model_validator(mode="after")
    def patient_counts_cannot_exceed_the_cohort(self) -> SimulationResult:
        """Checks the patient counts against this result's own n_patients, since exceeding it indicates a bug in the simulation loop.

        Raises:
            ValueError: If either count exceeds the cohort size.
        """
        for field_name in ("unvisited_patients", "unreached_high_risk_patients"):
            count = getattr(self, field_name)
            if count > self.n_patients:
                raise ValueError(
                    f"{field_name} ({count}) exceeds n_patients ({self.n_patients})"
                )
        return self

    def to_csv_row(self) -> dict:
        """Serialise to a flat dictionary for CSV output."""
        return self.model_dump()
