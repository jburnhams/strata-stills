"""The records of schema `photos`: everything about the pictures except the pictures (decisions of Jonathan, 2026-11-07: all non-binary data in the database; a photo is identified by the SHA-256 of its original file, the raw; votes are never deleted, undo marks them; embeddings are pgvector).

Why it is shaped like this:
- `Photo` is one exposure, keyed by the SHA-256 of its original. Where the file was seen (root, relative path, size, mtime) is `File`, so a moved or renamed shoot keeps its votes, flags and analysis. `legacy_item_id` and `arena_pic.name` keep the old path/name based keys so the JSON files of today map across.
- New facts need no migration: `PhotoMeta.exif_raw/xmp_raw` hold everything the readers found, `Analysis` holds any computed result (scene, person count, noise estimate, Lightroom-derived features ...) with the producer, its version and what it was computed from, `Detection` holds boxes (faces, persons, nsfw parts), `Embedding` holds vectors. A result computed again from changed input is a new row; the old one stays.
- Votes, flags and their changes are append-only history: `ArenaVote.undone_at` marks an undo, flag changes are `ArenaFlagEvent` rows and `ArenaFlag` is the current state of them.
- Cache keys are stored as the strings the code makes today (`Render.cache_key`); this schema never changes how they are computed.
Portable types only (the unit tests run on SQLite): free-form blobs are `sa.JSON`, vectors are `types.Vector`. Times are UTC."""

from datetime import datetime

import sqlalchemy as sa
from omarchy_db import make_base
from sqlalchemy import ForeignKey, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from .types import Vector

Base = make_base()


def _now():
    return mapped_column(server_default=func.current_timestamp(), init=False)


# ---- identity of a picture and where it lives -----------------------------------------------------------------------------------------------------------


class Photo(Base):
    """One exposure, identified by the SHA-256 of its original file (the raw; the camera JPEG for a camera with no raw)."""

    __tablename__ = "photo"
    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    sha256: Mapped[str] = mapped_column(sa.String(64), unique=True)
    stem: Mapped[str | None] = mapped_column(
        default=None
    )  # the shot's stem as ingest.py says it (DSC00010, DJI_20250402190211_0046_D)
    kind: Mapped[str] = mapped_column(sa.String(16), server_default="still", default="still")  # still | pano | video
    parent_id: Mapped[int | None] = mapped_column(
        ForeignKey("photo.id"), default=None, index=True
    )  # a pano's shot it is named after
    legacy_item_id: Mapped[str | None] = mapped_column(
        default=None, index=True
    )  # '<sha1(abspath)[:8]>~<stem>' of projects.py
    created_at: Mapped[datetime] = _now()


class File(Base):
    """A file seen on disk: path relative to a source root (as projects store sources), so it means the same on every machine. A file that changed is a new row; the old one stays (facts about it stay true of its bytes)."""

    __tablename__ = "file"
    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    root: Mapped[str] = mapped_column(sa.String(64))  # label of the source root (drone, raw ...)
    relpath: Mapped[str]
    size: Mapped[int] = mapped_column(sa.BigInteger)
    mtime_ns: Mapped[int] = mapped_column(sa.BigInteger)
    photo_id: Mapped[int | None] = mapped_column(ForeignKey("photo.id"), default=None, index=True)
    role: Mapped[str | None] = mapped_column(
        sa.String(16), default=None
    )  # raw | jpeg | derived | video | sidecar | final (a Lightroom export)
    tag: Mapped[str | None] = mapped_column(default=None)  # the editor's suffix of a derived file (Pano, Enhanced-NR)
    sha256: Mapped[str | None] = mapped_column(sa.String(64), default=None, index=True)
    current: Mapped[bool] = mapped_column(server_default=sa.true(), default=True)
    first_seen: Mapped[datetime] = _now()
    last_seen: Mapped[datetime | None] = mapped_column(default=None)
    __table_args__ = (UniqueConstraint("root", "relpath", "size", "mtime_ns"),)


class PhotoMeta(Base):
    """What the file says about the picture: the fields queries filter on as columns, everything else in the raw JSON."""

    __tablename__ = "photo_meta"
    photo_id: Mapped[int] = mapped_column(ForeignKey("photo.id"), primary_key=True)
    taken_at: Mapped[datetime | None] = mapped_column(default=None, index=True)
    width: Mapped[int | None] = mapped_column(default=None)
    height: Mapped[int | None] = mapped_column(default=None)
    make: Mapped[str | None] = mapped_column(default=None)
    model: Mapped[str | None] = mapped_column(default=None, index=True)
    lens: Mapped[str | None] = mapped_column(default=None)
    iso: Mapped[int | None] = mapped_column(default=None)
    exposure_s: Mapped[float | None] = mapped_column(default=None)
    f_number: Mapped[float | None] = mapped_column(default=None)
    focal_mm: Mapped[float | None] = mapped_column(default=None)
    orientation: Mapped[int | None] = mapped_column(default=None)
    gps_lat: Mapped[float | None] = mapped_column(default=None)
    gps_lon: Mapped[float | None] = mapped_column(default=None)
    gps_alt: Mapped[float | None] = mapped_column(default=None)
    dji: Mapped[dict | None] = mapped_column(
        sa.JSON(none_as_null=True), default=None
    )  # meta.dji_fields (gimbal, flight attitude, relative altitude ...)
    is_nadir: Mapped[bool | None] = mapped_column(default=None)
    exif_raw: Mapped[dict | None] = mapped_column(sa.JSON(none_as_null=True), default=None)
    xmp_raw: Mapped[dict | None] = mapped_column(sa.JSON(none_as_null=True), default=None)
    read_at: Mapped[datetime] = _now()


# ---- results computed about a picture (grows without migrations) ---------------------------------------------------------------------------------


class Analysis(Base):
    """Any computed result: `kind` (nsfw, scene, person_count, noise, detail, lr_settings, look_features, faces ...), who made it (`producer`, `producer_version`), from what (`input_key`: the file's sha256 or size+mtime plus the settings), the headline number in `score` and the rest in `data`."""

    __tablename__ = "analysis"
    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    photo_id: Mapped[int] = mapped_column(ForeignKey("photo.id"), index=True)
    kind: Mapped[str] = mapped_column(sa.String(64), index=True)
    producer: Mapped[str] = mapped_column(sa.String(128))
    producer_version: Mapped[str] = mapped_column(sa.String(64), server_default="", default="")
    input_key: Mapped[str] = mapped_column(sa.String(128), server_default="", default="")
    file_id: Mapped[int | None] = mapped_column(ForeignKey("file.id"), default=None)
    score: Mapped[float | None] = mapped_column(default=None)
    label: Mapped[str | None] = mapped_column(default=None)
    data: Mapped[dict | None] = mapped_column(sa.JSON(none_as_null=True), default=None)
    created_at: Mapped[datetime] = _now()
    __table_args__ = (UniqueConstraint("photo_id", "kind", "producer", "producer_version", "input_key"),)


class Detection(Base):
    """A box found in a picture (face, person, object, exposed body part); x, y, w, h are fractions of the picture."""

    __tablename__ = "detection"
    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    analysis_id: Mapped[int] = mapped_column(ForeignKey("analysis.id"), index=True)
    label: Mapped[str] = mapped_column(sa.String(64), index=True)
    x: Mapped[float]
    y: Mapped[float]
    w: Mapped[float]
    h: Mapped[float]
    score: Mapped[float | None] = mapped_column(default=None)
    data: Mapped[dict | None] = mapped_column(
        sa.JSON(none_as_null=True), default=None
    )  # e.g. the face's median Lab in the camera preview and in the final
    person_id: Mapped[int | None] = mapped_column(default=None)  # reserved for people identities


class Embedding(Base):
    """A vector for a picture (or one detection) from a model: pgvector on Postgres."""

    __tablename__ = "embedding"
    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    photo_id: Mapped[int] = mapped_column(ForeignKey("photo.id"), index=True)
    model: Mapped[str] = mapped_column(sa.String(128), index=True)  # clip-vit-b32, clip-pca32, scrfd-512 ...
    vector: Mapped[list] = mapped_column(Vector())
    detection_id: Mapped[int | None] = mapped_column(ForeignKey("detection.id"), default=None)
    created_at: Mapped[datetime] = _now()
    __table_args__ = (UniqueConstraint("photo_id", "model", "detection_id"),)


class NsfwResult(Base):
    """nsfw.py's cache: the highest score of the exposed classes of a file for one model (its own table because every scan reads it)."""

    __tablename__ = "nsfw_result"
    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    file_id: Mapped[int] = mapped_column(ForeignKey("file.id"), index=True)
    model: Mapped[str] = mapped_column(sa.String(64))
    score: Mapped[float]
    label: Mapped[str] = mapped_column(server_default="", default="")
    checked_at: Mapped[datetime] = _now()
    __table_args__ = (UniqueConstraint("file_id", "model"),)


class NsfwDecision(Base):
    """The person's call on a flagged picture (the allow list)."""

    __tablename__ = "nsfw_decision"
    photo_id: Mapped[int] = mapped_column(ForeignKey("photo.id"), primary_key=True)
    allowed: Mapped[bool]
    decided_at: Mapped[datetime] = _now()
    file_id: Mapped[int | None] = mapped_column(
        ForeignKey("file.id"), default=None
    )  # the file version (size, time stamp) the decision was made on; one decision per photo, the latest one


class LightroomEdit(Base):
    """A finished Lightroom export paired with its raw: the settings the export carries (crs: XMP) and its crop, the data the look is learned from."""

    __tablename__ = "lightroom_edit"
    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    photo_id: Mapped[int] = mapped_column(ForeignKey("photo.id"), index=True)
    final_file_id: Mapped[int | None] = mapped_column(ForeignKey("file.id"), default=None)
    settings: Mapped[dict | None] = mapped_column(sa.JSON(none_as_null=True), default=None)
    crop: Mapped[dict | None] = mapped_column(sa.JSON(none_as_null=True), default=None)
    error: Mapped[str | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = _now()


# ---- pipeline products, projects ----------------------------------------------------------------------------------------------------------------------


class Render(Base):
    """A picture the app made (a generated variant, a preview render ...); the picture stays a file, this row is its sidecar JSON."""

    __tablename__ = "render"
    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    photo_id: Mapped[int] = mapped_column(ForeignKey("photo.id"), index=True)
    cache_key: Mapped[str] = mapped_column(sa.String(64), index=True)  # the string the code makes today, unchanged
    kind: Mapped[str] = mapped_column(sa.String(16))  # generated | preview | detail | arena
    variant: Mapped[str | None] = mapped_column(default=None)
    preset: Mapped[str | None] = mapped_column(default=None)
    stages: Mapped[list | None] = mapped_column(sa.JSON(none_as_null=True), default=None)
    summary: Mapped[str | None] = mapped_column(default=None)
    out_w: Mapped[int | None] = mapped_column(default=None)
    out_h: Mapped[int | None] = mapped_column(default=None)
    bytes: Mapped[int | None] = mapped_column(sa.BigInteger, default=None)
    quality: Mapped[int | None] = mapped_column(default=None)
    path: Mapped[str | None] = mapped_column(default=None)
    info: Mapped[dict | None] = mapped_column(
        sa.JSON(none_as_null=True), default=None
    )  # the rest of the sidecar JSON as written
    created_at: Mapped[datetime] = _now()
    __table_args__ = (UniqueConstraint("kind", "cache_key"),)


class Project(Base):
    __tablename__ = "project"
    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    slug: Mapped[str] = mapped_column(unique=True)  # the folder name
    name: Mapped[str]
    created: Mapped[datetime | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = _now()


class ProjectSource(Base):
    __tablename__ = "project_source"
    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    project_id: Mapped[int] = mapped_column(ForeignKey("project.id"), index=True)
    root: Mapped[str] = mapped_column(sa.String(64))
    relpath: Mapped[str]
    added: Mapped[datetime | None] = mapped_column(default=None)
    original_path: Mapped[str | None] = mapped_column(
        default=None
    )  # the path as the project stored it (a Windows UNC path, say); root + relpath is where it is on this machine
    stems: Mapped[list | None] = mapped_column(
        sa.JSON(none_as_null=True), default=None
    )  # the pictures the source holds (their stems), None = every picture in the folder (projects.include)
    __table_args__ = (UniqueConstraint("project_id", "root", "relpath"),)


class ProjectRule(Base):
    """An `auto` rule of project.json: which preset runs on which source (source '*' = all)."""

    __tablename__ = "project_rule"
    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    project_id: Mapped[int] = mapped_column(ForeignKey("project.id"), index=True)
    source: Mapped[str]
    pipeline: Mapped[str]
    variant: Mapped[str | None] = mapped_column(default=None)


class ItemChoice(Base):
    """What the person chose for an item of a project (items.json): the primary variant."""

    __tablename__ = "item_choice"
    project_id: Mapped[int] = mapped_column(ForeignKey("project.id"), primary_key=True)
    photo_id: Mapped[int] = mapped_column(ForeignKey("photo.id"), primary_key=True)
    primary_variant: Mapped[str]
    chosen_at: Mapped[datetime] = _now()


class PipelinePreset(Base):
    """A saved pipeline preset (`~/.stratastills/pipelines/*.json`)."""

    __tablename__ = "pipeline_preset"
    name: Mapped[str] = mapped_column(primary_key=True)
    label: Mapped[str]
    stages: Mapped[list] = mapped_column(sa.JSON)
    updated_at: Mapped[datetime] = _now()


# ---- the arena: votes, flags --------------------------------------------------------------------------------------------------------------------------


class ArenaPic(Base):
    """A picture of the arena's pool, named as the pool names it ('2023-01-11_DSC00010'); `photo_id` is filled once the name is matched to its raw."""

    __tablename__ = "arena_pic"
    name: Mapped[str] = mapped_column(primary_key=True)
    photo_id: Mapped[int | None] = mapped_column(ForeignKey("photo.id"), default=None, index=True)


class ArenaVariant(Base):
    __tablename__ = "arena_variant"
    name: Mapped[str] = mapped_column(primary_key=True)
    retired: Mapped[bool] = mapped_column(server_default=sa.false(), default=False)


class ArenaEntry(Base):
    """The pool: the picture a variant of a pic is shown as. A rebuild adds a row and sets `superseded_at` on the old one, so a flag can say which version it was about."""

    __tablename__ = "arena_entry"
    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    pic: Mapped[str] = mapped_column(ForeignKey("arena_pic.name"), index=True)
    variant: Mapped[str] = mapped_column(ForeignKey("arena_variant.name"))
    file: Mapped[str]
    hash: Mapped[str]
    key: Mapped[str | None] = mapped_column(default=None)
    built_at: Mapped[datetime | None] = mapped_column(default=None)
    superseded_at: Mapped[datetime | None] = mapped_column(default=None)
    revision: Mapped[dict | None] = mapped_column(
        sa.JSON(none_as_null=True), default=None
    )  # {same: [earlier hashes whose votes still count], changed: unix time of the last real change, change: share of pixels changed} (arena_revision)


class ArenaVote(Base):
    """A vote: `a` and `b` are the variants as shown left and right, `winner` is a | b | tie. Undo sets `undone_at`; ratings count the rows where it is null. `legacy_key` makes the import of votes.jsonl repeatable; `raw` is the line as it was."""

    __tablename__ = "arena_vote"
    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    pic: Mapped[str] = mapped_column(ForeignKey("arena_pic.name"), index=True)
    a: Mapped[str]
    b: Mapped[str]
    winner: Mapped[str] = mapped_column(sa.String(8))
    voted_at: Mapped[float] = mapped_column(sa.Float, index=True)  # unix seconds, as the JSON lines have it
    ms: Mapped[int | None] = mapped_column(default=None)
    voter: Mapped[str | None] = mapped_column(default=None)
    undone_at: Mapped[datetime | None] = mapped_column(default=None)
    legacy_key: Mapped[str | None] = mapped_column(sa.String(40), unique=True, default=None)
    raw: Mapped[dict | None] = mapped_column(sa.JSON(none_as_null=True), default=None)


class ArenaFlag(Base):
    """A flag (something is wrong with one version of a picture): its current state. status: open | investigating | fixed | confirmed | wontfix."""

    __tablename__ = "arena_flag"
    id: Mapped[str] = mapped_column(sa.String(32), primary_key=True)  # the id the flag has in alerts.jsonl
    pic: Mapped[str] = mapped_column(ForeignKey("arena_pic.name"), index=True)
    variant: Mapped[str]
    status: Mapped[str] = mapped_column(sa.String(16), server_default="open", default="open", index=True)
    note: Mapped[str] = mapped_column(server_default="", default="")
    other: Mapped[str | None] = mapped_column(default=None)
    side: Mapped[str | None] = mapped_column(default=None)
    token: Mapped[str | None] = mapped_column(default=None)
    votes_on_pic: Mapped[int | None] = mapped_column(default=None)
    votes_on_variant: Mapped[int | None] = mapped_column(default=None)
    ms: Mapped[int | None] = mapped_column(default=None)
    flagged_at: Mapped[float | None] = mapped_column(sa.Float, default=None)
    entry_id: Mapped[int | None] = mapped_column(
        ForeignKey("arena_entry.id"), default=None
    )  # the pool entry that was on screen


class ArenaFlagEvent(Base):
    """Every change of a flag, in order (created, note, status): the history `ArenaFlag` is the latest of."""

    __tablename__ = "arena_flag_event"
    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    flag_id: Mapped[str] = mapped_column(ForeignKey("arena_flag.id"), index=True)
    at: Mapped[float] = mapped_column(sa.Float)
    status: Mapped[str | None] = mapped_column(default=None)
    note: Mapped[str | None] = mapped_column(default=None)
    legacy_key: Mapped[str | None] = mapped_column(sa.String(40), unique=True, default=None)
    raw: Mapped[dict | None] = mapped_column(sa.JSON(none_as_null=True), default=None)


class ReportFlag(Base):
    """A flag on a picture of a visual report (`neutral`, `auto`): rating worst..perfect, reasons, a note and a status (open | fixable | deferred | fixed). Not used to train anything."""

    __tablename__ = "report_flag"
    report: Mapped[str] = mapped_column(sa.String(32), primary_key=True)
    pic: Mapped[str] = mapped_column(primary_key=True)
    rating: Mapped[str] = mapped_column(sa.String(16), server_default="", default="")
    reasons: Mapped[list | None] = mapped_column(sa.JSON(none_as_null=True), default=None)
    note: Mapped[str] = mapped_column(server_default="", default="")
    status: Mapped[str] = mapped_column(sa.String(16), server_default="open", default="open")
    first_at: Mapped[datetime | None] = mapped_column(default=None)
    at: Mapped[datetime | None] = mapped_column(default=None)
    removed_at: Mapped[datetime | None] = mapped_column(default=None)  # a flag taken off is marked, not deleted
