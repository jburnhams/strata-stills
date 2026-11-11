"""The report flags on `report_flag` (see flags_store.py for the contract it shares with FileStore, and for the switch: `database_url` AND 'flags' in `db_reads`).
A row per (report, picture): rating, reasons, note, status, `first_at` (first flagged) and `at` (last change; both local clock times, as the file has them). Taking a flag off sets `removed_at` and keeps the row; flagging again clears it. Reading: the files' flags with the database's laid over them; a picture the database marked removed is left out whatever the file says. Every write also goes to the files (dual-write)."""

from datetime import UTC, datetime

from sqlalchemy import select

from stratastills.flags_store import FileStore


def _t(s):
    try:
        return datetime.fromisoformat(s) if s else None
    except ValueError:
        return None


def _iso(d):
    return d.isoformat(timespec="seconds") if d else None


class DbStore:
    kind = "database"

    def __init__(self, url):
        from omarchy_db import connect

        self.engine = connect(url)
        self.files = FileStore()

    def _session(self):
        from omarchy_db import session

        return session(self.engine)

    def read(self, report):
        """The files' flags with the database's laid over them."""
        return self._overlay(self.files.read(report), report)

    def read_db(self, report):
        """What the database alone holds (parity checks)."""
        return self._overlay({}, report)

    def _overlay(self, out, report):
        from .models import ReportFlag

        with self._session() as s:
            for r in s.scalars(select(ReportFlag).where(ReportFlag.report == report)):
                if r.removed_at is not None:
                    out.pop(r.pic, None)
                    continue
                out[r.pic] = dict(
                    flag=True,
                    rating=r.rating,
                    reasons=r.reasons or [],
                    note=r.note,
                    status=r.status,
                    at=_iso(r.at),
                    first=_iso(r.first_at),
                )
        return out

    def write(self, report, pic, entry):
        from .models import ReportFlag

        with self._session() as s:
            r = s.get(ReportFlag, (report, pic))
            if entry:
                vals = dict(
                    rating=entry.get("rating") or "",
                    reasons=list(entry.get("reasons") or []),
                    note=entry.get("note") or "",
                    status=entry.get("status") or "open",
                    first_at=_t(entry.get("first")),
                    at=_t(entry.get("at")),
                    removed_at=None,
                )
                if r is None:
                    s.add(ReportFlag(report=report, pic=pic, **vals))  # pyright: ignore[reportArgumentType]
                else:
                    for k, v in vals.items():
                        setattr(r, k, v)
            elif r is not None and r.removed_at is None:
                r.removed_at = datetime.now(UTC).replace(tzinfo=None)
            s.commit()
        return self.files.write(report, pic, entry)  # dual-write: the file stays a fresh backup
