"""User settings: one JSON file, `~/.stratastills/settings.json` (the folder can be moved with the STRATASTILLS_HOME environment variable), declared once as strata-python's
`Settings` (`SETTINGS`): it reads the file and the environment, validates, writes atomically under a lock, and gives the server its GET/PUT /api/settings and the CLI its `set`.

  projects_dir   where the projects live (each in its own folder, see projects.py). Default: `projects/` in this repository (gitignored) until the person picks a place.
  source_roots   list of folders (mounts) that hold the photos on THIS machine, e.g. `\\\\Server\\Drone` and `\\\\Server\\Raw` on Windows, `~/drone` and `~/raw` on Linux. A project stores a source as a path relative to a root (projects.py), so the same project opens on every machine.
  extra_roots    more roots the database's file paths may be under (the Lightroom exports in `personal`), not browsed as sources.
  report_dir, thumbs_dir, look_dir, db_reads, nsfw_filter, nsfw_threshold: see their functions below.

Each has an environment variable that overrides the file (handy for tests and one-off runs): STRATASTILLS_PROJECTS, STRATASTILLS_SOURCES (roots separated by os.pathsep),
STRATASTILLS_EXTRA_ROOTS, STRATASTILLS_REPORT, STRATASTILLS_THUMBS, STRATASTILLS_LOOK, STRATASTILLS_DB_READS (comma separated), STRATASTILLS_NSFW, STRATASTILLS_NSFW_THRESHOLD.
A value the file or the environment holds that does not pass its check is ignored (the default applies, and the log says so once).

`database_url` is read from the same file but is not one of the declared settings, on purpose: GET /api/settings sends every declared value, and the URL can hold a password."""

import os

from strata_python.settings import ListSetting, Setting, Settings, absolute_path, boolean, number_between

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_PROJECTS_DIR = os.path.join(REPO, "projects")
NSFW_THRESHOLD = 0.45  # the nudity filter's default (nsfw.DEFAULT_THRESHOLD); here so that reading the settings never imports the detector


def _root(value):
    """A source root as this machine names it: `~` expanded, the separators and a trailing one tidied (`\\\\Server\\Drone` stays a UNC path)."""
    v = value.strip()
    if not v:
        raise ValueError("must not be blank")
    return os.path.normpath(os.path.expanduser(v))


def _home():
    return os.environ.get("STRATASTILLS_HOME") or os.path.join(os.path.expanduser("~"), ".stratastills")


SETTINGS = Settings(
    "stratastills",
    [
        Setting(
            "projects_dir", default=lambda: DEFAULT_PROJECTS_DIR, normalise=absolute_path, env="STRATASTILLS_PROJECTS"
        ),
        ListSetting("source_roots", item=_root, env="STRATASTILLS_SOURCES"),
        ListSetting("extra_roots", item=_root),
        Setting(
            "report_dir",
            default=lambda: os.path.join(REPO, "out", "report"),
            normalise=absolute_path,
            env="STRATASTILLS_REPORT",
        ),
        Setting(
            "thumbs_dir",
            default=lambda: os.path.join(_home(), "thumbs"),
            normalise=absolute_path,
            env="STRATASTILLS_THUMBS",
        ),
        Setting(
            "look_dir",
            default=lambda: os.path.join(os.path.expanduser("~"), "models", "strata-stills", "look"),
            normalise=absolute_path,
            env="STRATASTILLS_LOOK",
        ),
        ListSetting("db_reads", sep=","),
        Setting("nsfw_filter", default="true", normalise=boolean, env="STRATASTILLS_NSFW"),
        Setting("nsfw_threshold", default=str(NSFW_THRESHOLD), normalise=number_between(0, 1)),
    ],
    prefix="STRATASTILLS",
)


def home():
    """The folder of the settings file and the app's own state (the pipelines, the nudity filter's cache): STRATASTILLS_HOME, else `~/.stratastills`."""
    return SETTINGS.home()


def load():
    """The settings file as it is (keys the app does not declare included: `database_url`)."""
    return SETTINGS.load()


def save(changes):
    """Merge `changes` into the settings file; returns what it holds. An unknown key is a KeyError and a value its check refuses a ValueError naming the key, both before
    anything is written; None removes the setting (back to its default). A list setting takes a list or one string joined by its separator."""
    return SETTINGS.save(changes)


def projects_dir():
    """The folder that holds the projects (created when asked for)."""
    return SETTINGS.folder("projects_dir")


def source_roots():
    """The folders that hold the photos on this machine, in search order (`~` expanded, trailing separators dropped)."""
    return SETTINGS.get_list("source_roots")


def extra_roots():
    """The roots the database's paths may be under besides the source roots (`extra_roots`)."""
    return SETTINGS.get_list("extra_roots")


def report_flags_dir():
    """Where the flags set on the report's pictures are kept (one JSON file per report): beside the report folder, so that rebuilding a report never loses them (`out/flags` by default)."""
    return (
        os.path.join(os.path.dirname(report_dir()), "flags")
        if not os.environ.get("STRATASTILLS_REPORT_FLAGS")
        else os.path.abspath(os.path.expanduser(os.environ["STRATASTILLS_REPORT_FLAGS"]))
    )


def database_url():
    """The shared database (schema `photos`) the arena keeps its votes and flags in: STRATASTILLS_DATABASE_URL or `database_url` of the settings file. Unset = the arena uses its files (arena_store.py); a bare `photos` means strata_python.db's default connection for that app (the shared database named by STRATA_DB_DATABASE, schema photos; STRATA_DB_LIVE=1 is needed outside tests)."""
    return (
        os.environ.get("STRATASTILLS_DATABASE_URL") or load().get("database_url") or None
    )  # not a declared setting: see the module notes


def db_reads():
    """The app's read paths that use the database once `database_url` is set, besides the arena (which uses it as soon as the URL is set): a set of names ('nsfw', ...) from STRATASTILLS_DB_READS (comma separated) or `db_reads` of the settings file. Empty by default, so merging a read path (or a nightly restart) never switches it by itself: each is switched on deliberately, one at a time."""
    return set(SETTINGS.get_list("db_reads"))


def nsfw_filter():
    """Whether the nudity filter runs (STRATASTILLS_NSFW, `nsfw_filter`; on by default)."""
    return SETTINGS.get_bool("nsfw_filter")


def nsfw_threshold():
    """The detector score from which a picture is held back (STRATASTILLS_NSFW_THRESHOLD, `nsfw_threshold`, from 0 to 1)."""
    return SETTINGS.get_float("nsfw_threshold")


def arena_dir():
    """Where the rating arena keeps its pictures, pool and votes (`out/arena` by default; STRATASTILLS_ARENA moves it)."""
    return os.path.abspath(
        os.path.expanduser(os.environ.get("STRATASTILLS_ARENA") or os.path.join(REPO, "out", "arena"))
    )


def report_dir():
    """Where scripts/build_report.py writes the visual report (`out/report` of the repository), which the server shows at /report/: the STRATASTILLS_REPORT environment variable or `report_dir` of the settings file move it."""
    return SETTINGS.get("report_dir")


def thumbs_dir():
    """Where the Browser keeps the thumbnails it makes (browser.py): STRATASTILLS_THUMBS, `thumbs_dir` of the settings file, else `thumbs/` in the settings folder."""
    return SETTINGS.get("thumbs_dir")


def look_dir():
    """Where the frozen look models are (stratastills.look, made by scripts/export_look_bundle.py): STRATASTILLS_LOOK, `look_dir` of the settings file, else ~/models/strata-stills/look. Private: trained on the person's photos."""
    return SETTINGS.get("look_dir")
