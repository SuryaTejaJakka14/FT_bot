# src/modules/tracker_ui.py
"""
TrackerUI: Interactive terminal UI for the FT_Bot job tracker.

Built with Textual. Displays ranked jobs from JobStore in an
interactive DataTable. Keyboard shortcuts update application
status live and persist changes to the SQLite database.

Layout:
    ┌─────────────────────────────────────────────────────┐
    │  Header: FT_Bot | Role | stats summary              │
    ├─────────────────────────────────────────────────────┤
    │  DataTable: ranked jobs with score + status         │
    ├─────────────────────────────────────────────────────┤
    │  Footer: keyboard shortcut hints                    │
    └─────────────────────────────────────────────────────┘

Keyboard shortcuts:
    a  → Mark selected job as Applied
    i  → Mark as Interviewing
    o  → Mark as Offer Received
    r  → Mark as Rejected
    d  → Mark as Declined
    n  → Clear status back to Not Applied
    e  → Edit note for selected job
    u  → Open job URL in browser
    f  → Cycle status filter
    s  → Run a new job search
    q  → Quit

Usage:
    store  = JobStore("data/jobs.db")
    config = PipelineConfig(search_role="ML Engineer", ...)
    app    = TrackerApp(store=store, config=config)
    app.run()
"""

import webbrowser
import logging
from typing import List, Optional

from textual.app        import App, ComposeResult
from textual.binding    import Binding
from textual.widgets    import Header, Footer, DataTable, Static
from textual.containers import Vertical
from textual.screen     import Screen
from textual.reactive   import reactive

from src.modules.job_store       import JobStore, STATUS_LABELS
from src.modules.pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# FILTER CYCLE
# ------------------------------------------------------------------

FILTER_CYCLE = [
    (None,                              "All Jobs"),
    (["NOT_APPLIED"],                   "Not Applied"),
    (["APPLIED", "INTERVIEWING"],       "Active"),
    (["OFFER_RECEIVED"],                "Offers"),
    (["REJECTED", "DECLINED"],          "Closed"),
]

# Column definitions: (column_key, display_label, width)
COLUMNS = [
    ("rank",    "#",          4),
    ("title",   "Job Title", 28),
    ("company", "Company",   16),
    ("score",   "Score",      7),
    ("label",   "Match",     10),
    ("status",  "Status",    16),
    ("date",    "Posted",    11),
]


# ------------------------------------------------------------------
# NOTE SCREEN
# ------------------------------------------------------------------

class NoteScreen(Screen):
    """
    Full-screen modal for editing a job note.
    Pressing Enter saves, Escape cancels.
    """

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, job_id: str, current_note: str, **kwargs):
        super().__init__(**kwargs)
        self._job_id       = job_id
        self._current_note = current_note

    def compose(self) -> ComposeResult:
        from textual.widgets import Input
        yield Vertical(
            Static(
                f"[bold]Edit Note[/bold]\n"
                f"Job ID: {self._job_id}\n\n"
                f"Press Enter to save, Escape to cancel.",
                id="note-header",
            ),
            Input(
                value       = self._current_note,
                placeholder = "Type your note here...",
                id          = "note-input",
            ),
        )

    def on_input_submitted(self, event) -> None:
        """Called when user presses Enter in the input field."""
        self.dismiss(event.value)

    def action_cancel(self) -> None:
        """Called when user presses Escape."""
        self.dismiss(None)


# ------------------------------------------------------------------
# MAIN TRACKER APP
# ------------------------------------------------------------------

class TrackerApp(App):
    """
    FT_Bot Job Tracker — interactive terminal UI.

    Args:
        store:         JobStore instance (open SQLite connection)
        config:        PipelineConfig with search settings
        run_pipeline:  Optional callable — called when user presses 's'
                       Should return a RunResult. If None, 's' shows warning.
    """

    CSS = """
    Screen {
        background: $surface;
    }

    DataTable {
        height: 1fr;
        border: solid $primary;
    }

    #stats-bar {
        height: 3;
        background: $panel;
        padding: 0 2;
        border-bottom: solid $primary-darken-2;
    }

    NoteScreen {
        align: center middle;
    }

    NoteScreen Vertical {
        width: 60;
        height: auto;
        border: solid $accent;
        padding: 2 4;
        background: $panel;
    }

    NoteScreen Input {
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("a", "set_status('APPLIED')",        "Apply",       show=True),
        Binding("i", "set_status('INTERVIEWING')",   "Interview",   show=True),
        Binding("o", "set_status('OFFER_RECEIVED')", "Offer",       show=True),
        Binding("r", "set_status('REJECTED')",       "Reject",      show=True),
        Binding("d", "set_status('DECLINED')",       "Decline",     show=False),
        Binding("n", "not_applied",                  "Not Applied", show=False),
        Binding("e", "edit_note",                    "Note",        show=True),
        Binding("u", "open_url",                     "Open URL",    show=True),
        Binding("f", "cycle_filter",                 "Filter",      show=True),
        Binding("s", "new_search",                   "Search",      show=True),
        Binding("q", "quit",                         "Quit",        show=True),
    ]

    # Reactive — changing it auto-triggers watch__filter_index()
    _filter_index: reactive[int] = reactive(0)

    def __init__(
        self,
        store:       JobStore,
        config:      PipelineConfig,
        run_pipeline = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._store           = store
        self._config          = config
        self._run_pipeline    = run_pipeline
        self._jobs: List      = []
        self._last_stats_text = ""     # for test inspection

    # ------------------------------------------------------------------
    # LAYOUT
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("", id="stats-bar")
        yield DataTable(id="job-table", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        """Called once when the app starts — build the table."""
        self.title     = "FT_Bot Job Tracker"
        self.sub_title = self._config.search_role

        table = self.query_one(DataTable)
        for col_key, label, width in COLUMNS:
            table.add_column(label, key=col_key, width=width)

        self._refresh_table()

    # ------------------------------------------------------------------
    # REACTIVE: re-render when filter changes
    # ------------------------------------------------------------------

    def watch__filter_index(self, value: int) -> None:
        self._refresh_table()

    # ------------------------------------------------------------------
    # TABLE REFRESH
    # ------------------------------------------------------------------

    def _refresh_table(self) -> None:
        """Reload jobs from store and repopulate the DataTable."""
        filter_statuses, filter_name = FILTER_CYCLE[self._filter_index]

        self._jobs = self._store.get_all(
            status_filter = filter_statuses,
            min_score     = 0.0,
        )

        table = self.query_one(DataTable)
        table.clear()

        for i, job in enumerate(self._jobs, start=1):
            table.add_row(
                str(i),
                job["title"][:26],
                job["company"][:14],
                f"{job['match_score']:.2f}",
                self._score_bar(job["match_score"]),
                STATUS_LABELS.get(job["status"], job["status"]),
                job.get("date_posted", "")[:10],
                key=job["job_id"],
            )

        self._update_stats_bar(filter_name)

    def _update_stats_bar(self, filter_name: str) -> None:
        """Update the stats bar above the table."""
        stats = self._store.get_stats()

        # Plain text version stored for test inspection
        self._last_stats_text = (
            f"{self._config.search_role}  |  "
            f"Total: {stats['total']}  "
            f"Applied: {stats['applied']}  "
            f"Interviewing: {stats['interviewing']}  "
            f"Offers: {stats['offer_received']}  "
            f"Rejected: {stats['rejected']}  "
            f"  Filter: {filter_name}  "
            f"  Showing: {len(self._jobs)}"
        )

        # Rich markup version for the widget
        bar = self.query_one("#stats-bar", Static)
        bar.update(
            f"[bold]{self._config.search_role}[/bold]  |  "
            f"Total: {stats['total']}  "
            f"Applied: {stats['applied']}  "
            f"Interviewing: {stats['interviewing']}  "
            f"Offers: {stats['offer_received']}  "
            f"Rejected: {stats['rejected']}  "
            f"  [bold cyan]Filter: {filter_name}[/bold cyan]  "
            f"  [dim]Showing: {len(self._jobs)}[/dim]"
        )

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _score_bar(self, score: float) -> str:
        """Convert a 0-1 score to a 5-char block bar."""
        filled = round(score * 5)
        return "█" * filled + "░" * (5 - filled)

    def _get_selected_job(self) -> Optional[dict]:
        """Return the job dict for the currently highlighted row."""
        table = self.query_one(DataTable)
        if not self._jobs or table.cursor_row is None:
            return None
        try:
            return self._jobs[table.cursor_row]
        except IndexError:
            return None

    # ------------------------------------------------------------------
    # ACTIONS — status updates
    # ------------------------------------------------------------------

    def action_set_status(self, status: str) -> None:
        """Update status for the selected job."""
        job = self._get_selected_job()
        if not job:
            self.notify("No job selected", severity="warning")
            return

        ok = self._store.update_status(job["job_id"], status)
        if ok:
            self.notify(
                f"{STATUS_LABELS.get(status, status)} — {job['title'][:30]}",
                timeout=2,
            )
            self._refresh_table()
        else:
            self.notify("Update failed", severity="error")

    def action_not_applied(self) -> None:
        """Reset status to NOT_APPLIED."""
        self.action_set_status("NOT_APPLIED")

    # ------------------------------------------------------------------
    # ACTIONS — note editing
    # ------------------------------------------------------------------

    def action_edit_note(self) -> None:
        """Open note editor for selected job."""
        job = self._get_selected_job()
        if not job:
            self.notify("No job selected", severity="warning")
            return

        def handle_note(result: Optional[str]) -> None:
            if result is not None:
                self._store.add_note(job["job_id"], result)
                self.notify("Note saved", timeout=2)

        self.push_screen(
            NoteScreen(job["job_id"], job.get("notes", "")),
            handle_note,
        )

    # ------------------------------------------------------------------
    # ACTIONS — open URL
    # ------------------------------------------------------------------

    def action_open_url(self) -> None:
        """Open the selected job's URL in the default browser."""
        job = self._get_selected_job()
        if not job:
            self.notify("No job selected", severity="warning")
            return
        url = job.get("url", "")
        if url:
            webbrowser.open(url)
            self.notify(f"Opened: {url[:50]}", timeout=2)
        else:
            self.notify("No URL available", severity="warning")

    # ------------------------------------------------------------------
    # ACTIONS — filter cycle
    # ------------------------------------------------------------------

    def action_cycle_filter(self) -> None:
        """Cycle through status filters."""
        self._filter_index = (self._filter_index + 1) % len(FILTER_CYCLE)
        _, name = FILTER_CYCLE[self._filter_index]
        self.notify(f"Filter: {name}", timeout=1)

    # ------------------------------------------------------------------
    # ACTIONS — new search
    # ------------------------------------------------------------------

    def action_new_search(self) -> None:
        """Trigger a new pipeline run if run_pipeline was provided."""
        if self._run_pipeline is None:
            self.notify(
                "No pipeline connected — start via run.py",
                severity="warning",
            )
            return

        self.notify("🔍 Running new search...", timeout=3)
        try:
            result = self._run_pipeline()
            self._refresh_table()
            self.notify(
                f"✓ {result.jobs_new} new jobs found | "
                f"Top: {result.top_match}",
                timeout=4,
            )
        except Exception as e:
            self.notify(f"Search failed: {e}", severity="error")
