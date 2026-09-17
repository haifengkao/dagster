import logging
import time

import pytest
from dagster import DagsterEvent, DagsterEventType, EventLogEntry
from dagster._core.instance import DagsterInstance
from dagster._core.test_utils import create_run_for_test
from dagster._daemon.auto_run_reexecution.event_log_consumer import (
    EventLogConsumerDaemon,
    get_new_cursor,
)

TEST_EVENT_LOG_FETCH_LIMIT = 10


class MockEventLogConsumerDaemon(EventLogConsumerDaemon):
    """Override the actual handlers so that we can just test which run records they receive."""

    def __init__(self):
        super().__init__(event_log_fetch_limit=TEST_EVENT_LOG_FETCH_LIMIT)
        self.run_records = []

    @property
    def handle_updated_runs_fns(self):
        def stash_run_records(_ctx, run_records, _logger):
            self.run_records = run_records
            yield

        return [stash_run_records]


def _create_success_event(instance, run):
    dagster_event = DagsterEvent(
        event_type_value=DagsterEventType.RUN_SUCCESS.value,
        job_name="foo",
        message="yay success",
    )
    event_record = EventLogEntry(
        user_message="",
        level=logging.INFO,
        job_name="foo",
        run_id=run.run_id,
        error_info=None,
        timestamp=time.time(),
        dagster_event=dagster_event,
    )

    instance.handle_new_event(event_record)


def test_daemon(instance: DagsterInstance, empty_workspace_context):
    daemon = MockEventLogConsumerDaemon()

    list(daemon.run_iteration(empty_workspace_context))
    assert daemon.run_records == []

    run = create_run_for_test(instance, "test_job")
    instance.report_run_failed(run)

    list(daemon.run_iteration(empty_workspace_context))
    assert [record.dagster_run.run_id for record in daemon.run_records] == [run.run_id]

    # not called again for same event
    daemon.run_records = []  # reset this since it will keep the value from the last call
    list(daemon.run_iteration(empty_workspace_context))
    assert daemon.run_records == []


def test_events_exceed_limit(instance: DagsterInstance, empty_workspace_context):
    daemon = MockEventLogConsumerDaemon()
    list(daemon.run_iteration(empty_workspace_context))

    for _ in range(TEST_EVENT_LOG_FETCH_LIMIT + 1):
        run = create_run_for_test(instance, "test_job")
        instance.report_run_failed(run)

    list(daemon.run_iteration(empty_workspace_context))
    assert len(daemon.run_records) == TEST_EVENT_LOG_FETCH_LIMIT

    list(daemon.run_iteration(empty_workspace_context))
    assert len(daemon.run_records) == 1


def test_success_and_failure_events(instance: DagsterInstance, empty_workspace_context):
    daemon = MockEventLogConsumerDaemon()
    list(daemon.run_iteration(empty_workspace_context))

    for _ in range(TEST_EVENT_LOG_FETCH_LIMIT + 1):
        run = create_run_for_test(instance, "foo")
        instance.report_run_failed(run)

        run = create_run_for_test(instance, "foo")
        _create_success_event(instance, run)

    list(daemon.run_iteration(empty_workspace_context))
    assert len(daemon.run_records) == TEST_EVENT_LOG_FETCH_LIMIT * 2

    list(daemon.run_iteration(empty_workspace_context))
    assert len(daemon.run_records) == 2


FAILURE_KEY = "EVENT_LOG_CONSUMER_CURSOR-PIPELINE_FAILURE"
SUCCESS_KEY = "EVENT_LOG_CONSUMER_CURSOR-PIPELINE_SUCCESS"


def test_cursors(instance: DagsterInstance, empty_workspace_context, caplog):
    assert instance.run_storage.get_cursor_values({FAILURE_KEY, SUCCESS_KEY}) == {}

    daemon = MockEventLogConsumerDaemon()
    with caplog.at_level(logging.INFO):
        list(daemon.run_iteration(empty_workspace_context))

    assert len(caplog.records) == 2
    assert all(record.levelno == logging.INFO for record in caplog.records)
    assert all(
        "at 0; existing events will not be replayed" in record.message for record in caplog.records
    )

    assert instance.run_storage.get_cursor_values({FAILURE_KEY, SUCCESS_KEY}) == {
        FAILURE_KEY: str(0),
        SUCCESS_KEY: str(0),
    }
    caplog.clear()
    daemon = MockEventLogConsumerDaemon()
    with caplog.at_level(logging.INFO):
        list(daemon.run_iteration(empty_workspace_context))
    assert caplog.records == [], "A restarted daemon reuses its persisted cursors"

    run1 = create_run_for_test(instance, "foo")
    run2 = create_run_for_test(instance, "foo")

    instance.report_run_failed(run1)
    instance.report_run_failed(run2)

    list(daemon.run_iteration(empty_workspace_context))
    assert len(daemon.run_records) == 2

    cursors = instance.run_storage.get_cursor_values({FAILURE_KEY, SUCCESS_KEY})

    list(daemon.run_iteration(empty_workspace_context))
    assert instance.run_storage.get_cursor_values({FAILURE_KEY, SUCCESS_KEY}) == cursors

    for _ in range(5):
        instance.report_engine_event("foo", run1)
        instance.report_engine_event("foo", run2)

    list(daemon.run_iteration(empty_workspace_context))
    # Cursors are per-type, so engine events (which are neither RUN_FAILURE nor RUN_SUCCESS)
    # do not advance the cursor. The cursor stays at the last seen event of the relevant type.
    assert instance.run_storage.get_cursor_values({FAILURE_KEY, SUCCESS_KEY}) == cursors

    run3 = create_run_for_test(instance, "foo")
    run4 = create_run_for_test(instance, "foo")

    instance.report_run_failed(run3)
    instance.report_run_failed(run4)

    list(daemon.run_iteration(empty_workspace_context))
    assert len(daemon.run_records) == 2


@pytest.mark.parametrize("event_type", [DagsterEventType.RUN_FAILURE, DagsterEventType.RUN_SUCCESS])
def test_cursor_init(instance: DagsterInstance, empty_workspace_context, caplog, event_type):
    instance.run_storage.wipe()
    daemon = MockEventLogConsumerDaemon()

    run1 = create_run_for_test(instance, "foo")
    run2 = create_run_for_test(instance, "foo")

    for run in [run1, run2]:
        if event_type == DagsterEventType.RUN_FAILURE:
            instance.report_run_failed(run)
        else:
            _create_success_event(instance, run)
    latest_event_id = instance.event_log_storage.get_maximum_record_id()

    with caplog.at_level(logging.INFO):
        list(daemon.run_iteration(empty_workspace_context))
    assert len(daemon.run_records) == 0, "Cursors init to latest event"
    assert len(caplog.records) == 2
    assert all(record.levelno == logging.INFO for record in caplog.records)
    assert all(
        f"at {latest_event_id}; existing events will not be replayed" in record.message
        for record in caplog.records
    )
    assert instance.daemon_cursor_storage.get_cursor_values({FAILURE_KEY, SUCCESS_KEY}) == {
        FAILURE_KEY: str(latest_event_id),
        SUCCESS_KEY: str(latest_event_id),
    }
    caplog.clear()
    daemon = MockEventLogConsumerDaemon()
    with caplog.at_level(logging.INFO):
        list(daemon.run_iteration(empty_workspace_context))
    assert caplog.records == []
    assert daemon.run_records == []

    run3 = create_run_for_test(instance, "foo")
    instance.report_run_failed(run3)

    list(daemon.run_iteration(empty_workspace_context))
    assert [record.dagster_run.run_id for record in daemon.run_records] == [run3.run_id]
    daemon.run_records = []
    list(daemon.run_iteration(empty_workspace_context))
    assert daemon.run_records == []


@pytest.mark.parametrize("persisted_key", [FAILURE_KEY, SUCCESS_KEY])
def test_partial_cursor_initialization_warns(
    instance: DagsterInstance, empty_workspace_context, caplog, persisted_key
):
    instance.daemon_cursor_storage.set_cursor_values({persisted_key: "0"})
    daemon = MockEventLogConsumerDaemon()

    list(daemon.run_iteration(empty_workspace_context))

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert "ignoring older events" in caplog.records[0].message


@pytest.mark.parametrize("invalid_key", [FAILURE_KEY, SUCCESS_KEY])
def test_invalid_cursor_still_raises(
    instance: DagsterInstance, empty_workspace_context, caplog, invalid_key
):
    cursors = {FAILURE_KEY: "0", SUCCESS_KEY: "0", invalid_key: "invalid"}
    instance.daemon_cursor_storage.set_cursor_values(cursors)
    daemon = MockEventLogConsumerDaemon()

    with pytest.raises(ValueError, match="invalid"):
        list(daemon.run_iteration(empty_workspace_context))

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.ERROR
    assert "Invalid cursor" in caplog.records[0].message
    assert instance.daemon_cursor_storage.get_cursor_values({FAILURE_KEY, SUCCESS_KEY}) == cursors


def test_get_new_cursor():
    # new events: advances to max
    assert get_new_cursor(0, [3, 4, 5, 6, 7, 8, 9, 10]) == 10

    # no new events: stays at persisted cursor
    assert get_new_cursor(0, []) == 0
    assert get_new_cursor(10, []) == 10
