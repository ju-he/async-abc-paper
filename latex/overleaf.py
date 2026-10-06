#!/usr/bin/env python3
"""Two-way sync between the paper directories and their Overleaf projects.

Overleaf exposes every project as a git repository (Overleaf premium feature,
URL ``https://git.overleaf.com/<project-id>``).  This tool keeps one clone per
target in ``latex/.overleaf/<target>/`` (gitignored) and mirrors the paper
sources between the local paper directory and that clone with rsync.  The
clone is the only place that talks to Overleaf; the main repository never
sees Overleaf's history, so neither side's git history is rewritten.

Targets
-------
  tmlr       latex/tmlr                 (the manuscript being worked on;
                                         figures/ and references.bib are symlinks
                                         into the Springer directory and are
                                         synced through the links)
  springer   latex/sn-article-template  (the sn-jnl version, frozen 2026-10-06)

Commands
--------
  TARGET init URL        clone the Overleaf project into latex/.overleaf/TARGET
  TARGET pull            Overleaf -> local paper directory (then review `git diff`)
  TARGET push [-m MSG]   local paper directory -> Overleaf (one commit per push)
  TARGET status          what differs, and whether Overleaf has unpulled commits

Safety rules
------------
* `pull` refuses when the paper directory (or a directory it symlinks into)
  has uncommitted changes, so that anything it overwrites is one
  `git checkout` away.  `--force` overrides.
* `pull` does not delete local files that are absent on Overleaf unless
  `--delete` is given; it lists them instead.
* `push` refuses when Overleaf has commits that were never pulled, because
  the push would silently revert them.  Pull first, or `--force`.
* Build artifacts, the KIT-scheme figures (thesis only), drafting notes and
  unused template material are never synced (see the per-target excludes).

Credentials: Overleaf's git bridge authenticates with user ``git`` and an
Overleaf *git authentication token* (Account settings -> Git integration) as
the password.  `init` enables ``credential.helper store`` for the clone when
no helper is configured globally, so the token is asked once and then kept in
~/.git-credentials (plain text).
"""
from __future__ import annotations

import argparse
import re
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
CLONES_ROOT = HERE / ".overleaf"
LAST_SYNCED_KEY = "overleaf.lastSynced"
BRANCH_KEY = "overleaf.branch"  # the Overleaf project's single branch, detected at init

# rsync filter rules, relative to the paper directory.  Applied in both
# directions, so a file listed here is invisible to the sync entirely.
COMMON_EXCLUDES = [
    ".git/",
    # LaTeX build artifacts: Overleaf compiles on its own.
    "*.aux",
    "*.bbl",
    "*.blg",
    "*.log",
    "*.out",
    "*.fls",
    "*.fdb_latexmk",
    "*.synctex.gz",
    "*.toc",
    "/*.pdf",  # the compiled paper (and user-manual.pdf); figures/*.pdf stay in
    # Figure pipeline side products (gitignored locally as well).
    "/figures/kit/",  # KIT colour scheme feeds the thesis, not the paper
    "/figures/*.png",
    "/figures/*_meta.json",
]


@dataclass(frozen=True)
class Target:
    name: str
    paper_dir: Path
    excludes: list[str]
    after_pull_hint: str

    @property
    def clone_dir(self) -> Path:
        return CLONES_ROOT / self.name

    @property
    def rel_paper_dir(self) -> Path:
        return self.paper_dir.relative_to(REPO_ROOT)


TARGETS = {
    "springer": Target(
        name="springer",
        paper_dir=HERE / "sn-article-template",
        excludes=COMMON_EXCLUDES
        + [
            # Drafting support and unused Springer template material.
            "/notes/",
            "/tables/",
            "/bst/",
            "/empty.eps",
            "/fig.eps",
        ],
        after_pull_hint=(
            "Note: the Springer version is frozen (since 2026-10-06); the TMLR version\n"
            "is the manuscript being worked on."
        ),
    ),
    "tmlr": Target(
        name="tmlr",
        paper_dir=HERE / "tmlr",
        excludes=COMMON_EXCLUDES
        + [
            "/README.md",  # local workflow notes, not part of the manuscript
        ],
        after_pull_hint="",
    ),
}


class SyncError(RuntimeError):
    """A precondition failed; the message tells the user what to do."""


def run(cmd: list[str], *, cwd: Path | None = None, capture: bool = False) -> str:
    """Run a command, raising on failure.  Returns stdout when captured."""
    result = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    if result.returncode != 0:
        detail = (result.stderr or "").strip() if capture else ""
        where = f" (in {cwd})" if cwd else ""
        raise SyncError(
            f"command failed{where}: {' '.join(cmd)}" + (f"\n{detail}" if detail else "")
        )
    return result.stdout if capture else ""


def git(*args: str, cwd: Path, capture: bool = True) -> str:
    return run(["git", *args], cwd=cwd, capture=capture).strip()


def require_clone(target: Target) -> None:
    if not (target.clone_dir / ".git").is_dir():
        raise SyncError(
            f"no Overleaf clone at {target.clone_dir}.\n"
            f"Run:  latex/overleaf.py {target.name} init https://git.overleaf.com/<project-id>"
        )


def require_paper_dir(target: Target) -> None:
    if not target.paper_dir.is_dir():
        raise SyncError(f"paper directory does not exist: {target.paper_dir}")


def fetch_and_fast_forward(target: Target) -> str:
    """Bring the clone to the Overleaf branch tip; return that commit's sha."""
    clone = target.clone_dir
    dirty = git("status", "--porcelain", cwd=clone)
    if dirty:
        raise SyncError(
            f"the Overleaf clone at {clone} has uncommitted changes, which the sync "
            f"tool never leaves behind.  Inspect it (git -C {clone.relative_to(REPO_ROOT)} status) "
            "and clean it up before syncing."
        )
    branch = clone_branch(target)
    git("fetch", "origin", branch, cwd=clone)
    remote = git("rev-parse", f"origin/{branch}", cwd=clone)
    local = git("rev-parse", "HEAD", cwd=clone)
    if local != remote:
        # ff-only fails if a previous push left an unpushed commit behind.
        git("merge", "--ff-only", f"origin/{branch}", cwd=clone)
    return remote


def clone_config(target: Target, key: str) -> str | None:
    result = subprocess.run(
        ["git", "config", "--get", key], cwd=target.clone_dir, text=True, capture_output=True, check=False
    )
    if result.returncode == 1:  # key not set
        return None
    if result.returncode != 0:
        raise SyncError(f"git config failed: {result.stderr.strip()}")
    return result.stdout.strip()


def last_synced(target: Target) -> str | None:
    return clone_config(target, LAST_SYNCED_KEY)


def clone_branch(target: Target) -> str:
    branch = clone_config(target, BRANCH_KEY)
    if branch is None:
        raise SyncError(
            f"{target.clone_dir} has no {BRANCH_KEY} set; it predates branch detection.\n"
            f"Remove it and run `{target.name} init` again."
        )
    return branch


def set_last_synced(target: Target, sha: str) -> None:
    git("config", LAST_SYNCED_KEY, sha, cwd=target.clone_dir)


def unpulled_commits(target: Target, remote: str) -> list[str]:
    """Overleaf commits made since the last pull/push, oldest first."""
    synced = last_synced(target)
    if synced is None:
        return []
    out = git(
        "log", "--reverse", "--format=%h %ad %s", "--date=short", f"{synced}..{remote}", cwd=target.clone_dir
    )
    return [line for line in out.splitlines() if line]


def rsync(
    target: Target, src: Path, dst: Path, *, delete: bool, dry_run: bool, extra_excludes: list[str] = ()
) -> list[str]:
    """Mirror src into dst.  Returns rsync's itemized change lines.

    --copy-links sends symlinks as their referents (Overleaf cannot hold
    symlinks); --keep-dirlinks lets a pull write through a local symlinked
    directory (tmlr/figures -> ../sn-article-template/figures) instead of
    replacing the link with a real directory.
    """
    cmd = ["rsync", "--recursive", "--copy-links", "--keep-dirlinks", "--checksum", "--itemize-changes"]
    if delete:
        cmd.append("--delete-after")
    if dry_run:
        cmd.append("--dry-run")
    for pattern in [*target.excludes, *extra_excludes]:
        cmd += ["--exclude", pattern]
    cmd += [f"{src}/", f"{dst}/"]
    out = run(cmd, capture=True)
    # rsync -i prints one line per changed path; skip the summary/noise lines.
    return [line for line in out.splitlines() if line and not line.startswith(("sending", "sent ", "total "))]


def describe_changes(lines: list[str]) -> str:
    if not lines:
        return "  (no changes)"
    rows = []
    for line in lines:
        flags, _, path = line.partition(" ")
        path = path.strip()
        if flags.startswith("*deleting"):
            rows.append(f"  delete  {path}")
        elif flags[0] in "<>" and flags[1] == "f":
            rows.append(f"  {'new    ' if '+' in flags else 'update '} {path}")
        elif flags[0] == "c" and flags[1] == "d":
            rows.append(f"  mkdir   {path}")
        else:
            rows.append(f"  {flags:11s} {path}")
    return "\n".join(rows)


def top_level_symlinks(target: Target) -> dict[Path, Path]:
    """Top-level symlinks in the paper dir, mapped to their resolved referents."""
    return {entry: entry.resolve() for entry in sorted(target.paper_dir.iterdir()) if entry.is_symlink()}


def guarded_paths(target: Target) -> list[Path]:
    """The paper dir plus everything its top-level symlinks point to."""
    return [target.paper_dir, *top_level_symlinks(target).values()]


def file_symlinks(target: Target) -> dict[Path, Path]:
    """Top-level symlinks to regular files, mapped to their referents."""
    return {path: ref for path, ref in top_level_symlinks(target).items() if ref.is_file()}


def pull(target: Target, *, delete: bool, dry_run: bool) -> list[str]:
    """Overleaf clone -> paper dir, writing through top-level file symlinks.

    --keep-dirlinks lets rsync write through symlinked directories, but a
    symlinked *file* (tmlr/references.bib -> ../sn-article-template/
    sn-bibliography.bib) would be replaced by a regular file and reported as
    changed on every run.  Such files are excluded from rsync and handled
    here: differing content is written onto the referent, and with --delete a
    file missing on Overleaf removes the link (the referent belongs to the
    other paper directory and stays).  Returns rsync-style itemized lines.
    """
    links = file_symlinks(target)
    lines = rsync(
        target,
        target.clone_dir,
        target.paper_dir,
        delete=delete,
        dry_run=dry_run,
        extra_excludes=[f"/{path.name}" for path in links],
    )
    for path, referent in links.items():
        remote = target.clone_dir / path.name
        if remote.is_file():
            new_content = remote.read_bytes()
            if new_content != referent.read_bytes():
                lines.append(f">f.st...... {path.name}")
                if not dry_run:
                    referent.write_bytes(new_content)
        elif delete:
            lines.append(f"*deleting   {path.name}")
            if not dry_run:
                path.unlink()
    return lines


def paper_dir_git_status(target: Target) -> str | None:
    """`git status --porcelain` over the guarded paths, or None if not in a repo."""
    probe = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=target.paper_dir, text=True, capture_output=True, check=False
    )
    if probe.returncode != 0:
        return None
    return git("status", "--porcelain", "--", *map(str, guarded_paths(target)), cwd=target.paper_dir)


def repo_description(target: Target) -> str:
    """Short sha of the paper dir's repo plus a dirty marker, for commit messages."""
    status = paper_dir_git_status(target)
    if status is None:
        return f"{target.paper_dir.name} (not under git)"
    sha = git("rev-parse", "--short", "HEAD", cwd=target.paper_dir)
    return sha + ("+dirty" if status else "")


# ----------------------------------------------------------------- commands


def git_url(url: str) -> str:
    """Accept the project's browser URL and translate it to the git bridge URL."""
    match = re.fullmatch(r"https://(?:www\.)?overleaf\.com/project/([0-9a-f]{24})/?", url)
    if match:
        return f"https://git.overleaf.com/{match.group(1)}"
    return url


def cmd_init(target: Target, args: argparse.Namespace) -> None:
    clone = target.clone_dir
    args.url = git_url(args.url)
    if clone.exists():
        raise SyncError(f"{clone} already exists; remove it to re-initialise.")
    clone.parent.mkdir(exist_ok=True)
    print(f"Cloning {args.url} into {clone} ...")
    run(["git", "clone", args.url, str(clone)])
    # Overleaf serves exactly one branch; take whatever the clone checked out.
    branch = git("symbolic-ref", "--short", "HEAD", cwd=clone)
    git("config", BRANCH_KEY, branch, cwd=clone)
    print(f"Overleaf branch: {branch}")
    helper = subprocess.run(
        ["git", "config", "--get", "credential.helper"], text=True, capture_output=True, check=False
    )
    if helper.returncode == 1:
        git("config", "credential.helper", "store", cwd=clone)
        print("Enabled credential.helper=store for the clone (token kept in ~/.git-credentials).")
    set_last_synced(target, git("rev-parse", "HEAD", cwd=clone))
    print(
        "Done.  The Overleaf project's current content counts as already seen, so\n"
        f"  {target.name} push   replaces whatever Overleaf holds with {target.rel_paper_dir};\n"
        f"  {target.name} pull   brings Overleaf's content into that directory.\n"
        f"Run `{target.name} status` first to see what each direction would change."
    )


def cmd_status(target: Target, args: argparse.Namespace) -> None:
    require_clone(target)
    require_paper_dir(target)
    remote = fetch_and_fast_forward(target)
    synced = last_synced(target)
    pending = unpulled_commits(target, remote)
    print(f"Target         : {target.name}  ({target.rel_paper_dir})")
    print(f"Overleaf clone : {target.clone_dir}")
    print(f"Overleaf HEAD  : {remote[:10]}" + ("" if synced else "  (never synced)"))
    if pending:
        print(f"Unpulled Overleaf commits ({len(pending)}):")
        for line in pending:
            print(f"  {line}")
    else:
        print("Unpulled Overleaf commits: none")
    print("\npush would change on Overleaf (local -> Overleaf, mirror):")
    print(describe_changes(rsync(target, target.paper_dir, target.clone_dir, delete=True, dry_run=True)))
    print("\npull would change locally (Overleaf -> local, no deletions):")
    print(describe_changes(pull(target, delete=False, dry_run=True)))


def cmd_pull(target: Target, args: argparse.Namespace) -> None:
    require_clone(target)
    require_paper_dir(target)
    dirty = paper_dir_git_status(target)
    if dirty is None:
        print(f"warning: {target.paper_dir} is not under git; a pull cannot be undone with git checkout.")
    elif dirty and not args.force:
        raise SyncError(
            "the paper directory (or a directory it symlinks into) has uncommitted changes:\n"
            + "\n".join(f"  {line}" for line in dirty.splitlines())
            + "\nCommit or stash them so a pull cannot destroy work, or use --force."
        )
    remote = fetch_and_fast_forward(target)
    pending = unpulled_commits(target, remote)
    if pending:
        print(f"Pulling {len(pending)} Overleaf commit(s):")
        for line in pending:
            print(f"  {line}")
    changes = pull(target, delete=args.delete, dry_run=False)
    print("Changed locally:")
    print(describe_changes(changes))
    if not args.delete:
        extra = [line for line in pull(target, delete=True, dry_run=True) if line.startswith("*deleting")]
        if extra:
            print("Present locally but absent on Overleaf (kept; pass --delete to remove):")
            for line in extra:
                print(f"  {line.partition(' ')[2].strip()}")
    set_last_synced(target, remote)
    if changes:
        print(f"\nReview with  git diff -- {target.rel_paper_dir}  and commit.")
        if target.after_pull_hint:
            print(target.after_pull_hint)


def cmd_push(target: Target, args: argparse.Namespace) -> None:
    require_clone(target)
    require_paper_dir(target)
    remote = fetch_and_fast_forward(target)
    pending = unpulled_commits(target, remote)
    if pending and not args.force:
        raise SyncError(
            f"Overleaf has {len(pending)} commit(s) that were never pulled:\n"
            + "\n".join(f"  {line}" for line in pending)
            + f"\nPushing now would revert them.  Run `{target.name} pull` first, or --force to overwrite."
        )
    clone = target.clone_dir
    changes = rsync(target, target.paper_dir, clone, delete=True, dry_run=False)
    git("add", "--all", cwd=clone)
    staged = git("status", "--porcelain", cwd=clone)
    if not staged:
        set_last_synced(target, remote)
        print("Overleaf is already up to date; nothing to push.")
        return
    print("Changes pushed to Overleaf:")
    print(describe_changes(changes))
    message = args.message or f"Sync from async-abc-paper {repo_description(target)}"
    git("commit", "--quiet", "--message", message, cwd=clone)
    git("push", "origin", clone_branch(target), cwd=clone, capture=False)
    set_last_synced(target, git("rev-parse", "HEAD", cwd=clone))
    print(f"Pushed: {message}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="latex/overleaf.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("target", choices=sorted(TARGETS), help="which paper version to sync")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="clone the Overleaf project")
    p_init.add_argument("url", help="https://git.overleaf.com/<id> or the project page https://www.overleaf.com/project/<id>")
    p_init.set_defaults(func=cmd_init)

    p_status = sub.add_parser("status", help="show pending differences in both directions")
    p_status.set_defaults(func=cmd_status)

    p_pull = sub.add_parser("pull", help="Overleaf -> local paper directory")
    p_pull.add_argument("--delete", action="store_true", help="also delete local files absent on Overleaf")
    p_pull.add_argument("--force", action="store_true", help="pull even if the paper directory has uncommitted changes")
    p_pull.set_defaults(func=cmd_pull)

    p_push = sub.add_parser("push", help="local paper directory -> Overleaf")
    p_push.add_argument("-m", "--message", help="commit message on Overleaf")
    p_push.add_argument("--force", action="store_true", help="push even if Overleaf has unpulled commits (overwrites them)")
    p_push.set_defaults(func=cmd_push)

    args = parser.parse_args(argv)
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    sys.stdout.reconfigure(line_buffering=True)  # keep stdout/stderr ordered when piped
    try:
        args.func(TARGETS[args.target], args)
    except SyncError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
