"""
Tests for the patient store and its storage backends.

Covers the three defects this refactor fixed: a shared queue across sessions,
non-atomic writes, and a silent reset-to-empty that destroyed data on any read
error.
"""

import json

import pytest

from memory.backends import DictBackend, FileBackend, empty_store
from memory.patient_store import PatientStore, load_seed_cards, session_store
from models import Patient, PatientCard, TriageStatus, VitalSigns

NORMAL = dict(
    heart_rate=78, systolic_bp=122, diastolic_bp=78,
    respiratory_rate=16, spo2=98.0, temperature_c=36.8,
)


def make_patient(name="Test Patient", age=40):
    return Patient(
        patient_id="PENDING", name=name, age=age, weight_kg=70.0,
        chief_complaint="ankle pain", vitals=VitalSigns(**NORMAL),
    )


@pytest.fixture
def store(tmp_path):
    return PatientStore(FileBackend(tmp_path / "store.json"))


# ─── The shallow-copy bug ─────────────────────────────────────────────────────

class TestEmptyStoreIsolation:
    """
    Regression: a module-level `_EMPTY_STORE` was handed out via `.copy()`,
    a shallow copy, so every "fresh" store shared one `cards` dict.
    """

    def test_each_empty_store_is_independent(self):
        a, b = empty_store(), empty_store()
        a["cards"]["PT-0001"] = {"x": 1}
        assert b["cards"] == {}

    def test_clear_then_write_does_not_leak_into_the_next_store(self, tmp_path):
        s1 = PatientStore(FileBackend(tmp_path / "a.json"))
        s1.clear_all()
        s1.check_in(make_patient("Alice"))

        s2 = PatientStore(FileBackend(tmp_path / "b.json"))
        assert s2.get_all_cards() == []


# ─── Atomic writes ────────────────────────────────────────────────────────────

class TestAtomicWrite:

    def test_write_then_read_round_trips(self, tmp_path):
        backend = FileBackend(tmp_path / "store.json")
        backend.write({"id_counter": 3, "cards": {"PT-0003": {"a": 1}}})
        assert backend.read()["id_counter"] == 3

    def test_failed_write_leaves_the_previous_file_intact(self, tmp_path, monkeypatch):
        """A crash mid-write must not truncate an existing queue."""
        path = tmp_path / "store.json"
        backend = FileBackend(path)
        backend.write({"id_counter": 1, "cards": {"PT-0001": {"keep": True}}})

        def boom(*a, **k):
            raise OSError("disk full")

        monkeypatch.setattr("memory.backends.os.replace", boom)
        with pytest.raises(OSError):
            backend.write({"id_counter": 2, "cards": {}})

        # Original content survives.
        assert backend.read()["cards"] == {"PT-0001": {"keep": True}}

    def test_failed_write_leaves_no_temp_files_behind(self, tmp_path, monkeypatch):
        path = tmp_path / "store.json"
        backend = FileBackend(path)
        monkeypatch.setattr(
            "memory.backends.os.replace",
            lambda *a, **k: (_ for _ in ()).throw(OSError("nope")),
        )
        with pytest.raises(OSError):
            backend.write(empty_store())
        assert list(tmp_path.glob(".store-*.tmp")) == []


# ─── Corrupt data is preserved, not discarded ─────────────────────────────────

class TestCorruptStore:
    """
    Regression: any read error returned an empty store, turning one bad byte
    into silent, total loss of the queue.
    """

    def test_corrupt_file_is_quarantined_not_deleted(self, tmp_path):
        path = tmp_path / "store.json"
        path.write_text("{ this is not json", encoding="utf-8")

        backend = FileBackend(path)
        assert backend.read() == empty_store()

        quarantined = list(tmp_path.glob("store.corrupt-*.json"))
        assert len(quarantined) == 1
        assert "not json" in quarantined[0].read_text()

    def test_wrong_structure_is_quarantined(self, tmp_path):
        path = tmp_path / "store.json"
        path.write_text('["unexpected", "list"]', encoding="utf-8")

        assert FileBackend(path).read() == empty_store()
        assert len(list(tmp_path.glob("store.corrupt-*.json"))) == 1

    def test_absent_file_is_not_an_error(self, tmp_path):
        assert FileBackend(tmp_path / "missing.json").read() == empty_store()


# ─── Session isolation ────────────────────────────────────────────────────────

class TestSessionIsolation:
    """
    Regression: a module-level file-backed singleton meant every visitor to a
    deployment shared one queue and could clear it for everyone.
    """

    def test_two_sessions_do_not_see_each_others_patients(self):
        alice_session, bob_session = {}, {}
        alice = session_store(alice_session)
        bob = session_store(bob_session)

        alice.check_in(make_patient("Alice"))

        assert len(alice.get_all_cards()) == 1
        assert bob.get_all_cards() == []

    def test_one_session_clearing_does_not_empty_another(self):
        a_session, b_session = {}, {}
        a, b = session_store(a_session), session_store(b_session)
        a.check_in(make_patient("Alice"))
        b.check_in(make_patient("Bob"))

        a.clear_all()

        assert a.get_all_cards() == []
        assert len(b.get_all_cards()) == 1

    def test_session_store_writes_nothing_to_disk(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        session_store({}).check_in(make_patient("Alice"))
        assert list(tmp_path.iterdir()) == []


# ─── Read caching ─────────────────────────────────────────────────────────────

class TestReadCaching:
    """
    Regression: the store hit its backend on every method call, so one render
    performed six full reads and five complete deserialisations.
    """

    def test_repeated_reads_hit_the_backend_once(self, tmp_path):
        class CountingBackend(FileBackend):
            reads = 0

            def read(self):
                CountingBackend.reads += 1
                return super().read()

        store = PatientStore(CountingBackend(tmp_path / "store.json"))
        for _ in range(6):
            store.get_all_cards()
            store.queue_stats()

        assert CountingBackend.reads == 1

    def test_write_refreshes_the_cached_view(self, store):
        assert store.get_all_cards() == []
        store.check_in(make_patient("Alice"))
        assert len(store.get_all_cards()) == 1

    def test_refresh_forces_a_reread(self, tmp_path):
        path = tmp_path / "store.json"
        store = PatientStore(FileBackend(path))
        store.check_in(make_patient("Alice"))

        # Change the file underneath the store.
        path.write_text(json.dumps(empty_store()), encoding="utf-8")
        assert len(store.get_all_cards()) == 1  # still cached

        store.refresh()
        assert store.get_all_cards() == []


# ─── Check-in behaviour ───────────────────────────────────────────────────────

class TestCheckIn:

    def test_assigns_sequential_ids(self, store):
        a = store.check_in(make_patient("Alice"))
        b = store.check_in(make_patient("Bob"))
        assert (a.patient_id, b.patient_id) == ("PT-0001", "PT-0002")

    def test_first_visit_is_not_returning(self, store):
        assert store.check_in(make_patient("Alice")).is_returning is False

    def test_second_visit_is_returning(self, store):
        store.check_in(make_patient("Alice"))
        assert store.check_in(make_patient("Alice")).is_returning is True

    def test_patient_does_not_match_their_own_stub(self, store):
        """is_returning must be computed before the stub is persisted."""
        assert store.check_in(make_patient("Alice")).is_returning is False

    def test_returning_match_ignores_case_and_padding(self, store):
        store.check_in(make_patient("Alice Smith"))
        assert store.check_in(make_patient("  alice smith  ")).is_returning is True

    def test_check_in_reserves_the_id_with_a_pending_stub(self, store):
        p = store.check_in(make_patient("Alice"))
        card = store.get_card(p.patient_id)
        assert card is not None
        assert card.patient.status is TriageStatus.PENDING


# ─── Seeding ──────────────────────────────────────────────────────────────────

class TestSeeding:

    def _card(self, pid, name):
        p = make_patient(name)
        p.patient_id = pid
        return PatientCard(patient=p)

    def test_seed_replaces_the_queue(self, store):
        store.check_in(make_patient("Pre-existing"))
        store.seed([self._card("PT-0001", "Alice")])
        assert [c.patient.name for c in store.get_all_cards()] == ["Alice"]

    def test_ids_continue_past_the_seeded_range(self, store):
        store.seed([self._card("PT-0001", "A"), self._card("PT-0007", "B")])
        assert store.check_in(make_patient("New")).patient_id == "PT-0008"

    def test_unparseable_seed_id_does_not_raise(self, store):
        store.seed([self._card("WEIRD", "A")])
        assert len(store.get_all_cards()) == 1

    def test_missing_seed_file_gives_an_empty_list(self, tmp_path):
        assert load_seed_cards(tmp_path / "absent.json") == []

    def test_corrupt_seed_file_gives_an_empty_list(self, tmp_path):
        bad = tmp_path / "seed.json"
        bad.write_text("{not json", encoding="utf-8")
        assert load_seed_cards(bad) == []


# ─── Queue and stats ──────────────────────────────────────────────────────────

class TestQueue:

    def test_resolved_patients_leave_the_queue(self, store):
        p = store.check_in(make_patient("Alice"))
        card = store.get_card(p.patient_id)
        card.patient.status = TriageStatus.RESOLVED
        store.save_card(card)

        assert store.get_queue() == []
        assert len(store.get_all_cards()) == 1

    def test_search_by_id_is_case_insensitive(self, store):
        p = store.check_in(make_patient("Alice"))
        assert store.search_by_id(p.patient_id.lower()) is not None

    def test_search_by_name_is_a_substring_match(self, store):
        store.check_in(make_patient("Alice Smith"))
        assert len(store.search_by_name("smith")) == 1

    def test_stats_count_pending_patients(self, store):
        store.check_in(make_patient("Alice"))
        stats = store.queue_stats()
        assert stats["total"] == 1 and stats["pending"] == 1
