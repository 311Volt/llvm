# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import hashlib
import json
import os
from pathlib import Path
import shutil

from utils.logger import log
from utils.utils import run
from options import options


class GitProject:
    def __init__(
        self,
        url: str,
        ref: str,
        directory: Path,
        name: str,
        use_installdir: bool = True,
        no_suffix_src: bool = False,
        shallow_clone: bool = True,
        src_dir_override: Path | None = None,
    ) -> None:
        self._url = url
        self._ref = ref
        self._directory = directory
        self._name = name
        self._use_installdir = use_installdir
        self._no_suffix_src = no_suffix_src
        self._shallow_clone = shallow_clone
        # When set, build from this already-present source tree instead of
        # cloning/fetching. Used for local development against a checkout the
        # user manages themselves - no git operations are performed on it.
        self._src_dir_override = (
            Path(src_dir_override) if src_dir_override is not None else None
        )
        self._rebuild_needed = self._setup_repo()

    @property
    def name(self):
        return self._name

    @property
    def src_dir(self) -> Path:
        if self._src_dir_override is not None:
            return self._src_dir_override
        suffix = "" if self._no_suffix_src else "-src"
        return self._directory / f"{self._name}{suffix}"

    @property
    def build_dir(self) -> Path:
        return self._directory / f"{self._name}-build"

    @property
    def install_dir(self) -> Path:
        return self._directory / f"{self._name}-install"

    @property
    def _build_complete_marker(self) -> Path:
        # marker lives in whichever dir needs_rebuild() inspects
        base = self.install_dir if self._use_installdir else self.build_dir
        return base / ".llvm_bench_build_complete.json"

    def _git_worktree_fingerprint(self) -> str | None:
        """Fingerprint the source tree's git working state, or None if src_dir
        is not a git repository.

        Combines the HEAD commit, the porcelain status (which files are
        modified/staged/untracked), and the diff of tracked changes so that any
        committed or uncommitted change to tracked files invalidates a prior
        build.
        """
        if not Path(self.src_dir, ".git").exists():
            return None
        try:
            parts = []
            for cmd in (
                "git rev-parse HEAD",
                "git status --porcelain",
                "git diff HEAD",
            ):
                parts.append(run(cmd, cwd=self.src_dir).stdout.decode(errors="replace"))
            return hashlib.sha256("\0".join(parts).encode()).hexdigest()
        except Exception as e:
            log.debug(f"Could not fingerprint git worktree at {self.src_dir}: {e}")
            return None

    def _read_build_marker(self) -> dict:
        """Return the parsed build-completion marker, or {} if absent/invalid."""
        if not self._build_complete_marker.exists():
            return {}
        try:
            return json.loads(self._build_complete_marker.read_text())
        except (json.JSONDecodeError, OSError) as e:
            log.debug(f"Could not read build marker {self._build_complete_marker}: {e}")
            return {}

    def _mark_build_complete(self) -> None:
        """Record that the build/install finished successfully.

        Written only after the relevant command returns without raising, so a
        failed (or merely configured) directory is never treated as built. The
        marker stores the ref, url, and (for a git source tree) the working-tree
        fingerprint, so a later run can skip rebuilding an unchanged tree.
        """
        base = self.install_dir if self._use_installdir else self.build_dir
        base.mkdir(parents=True, exist_ok=True)
        metadata = {
            "ref": self._ref,
            "url": self._url,
            "src_dir": str(self.src_dir),
        }
        if self._src_dir_override is not None:
            metadata["worktree_fingerprint"] = self._git_worktree_fingerprint()
        self._build_complete_marker.write_text(json.dumps(metadata, indent=2))

    def needs_rebuild(self) -> bool:
        if options.offline:
            log.debug("Rebuild is disabled due to --offline option.")
            return False
        if self._rebuild_needed:
            log.debug(
                f"Rebuild needed because new sources were detected for project {self._name}."
            )
            return True

        dir_to_check = self.install_dir if self._use_installdir else self.build_dir

        if not self._build_complete_marker.exists():
            log.debug(
                f"{dir_to_check} has no build-completion marker, rebuild needed."
            )
            return True
        log.debug(f"{dir_to_check} build previously completed, no rebuild needed.")
        return False

    def configure(
        self,
        extra_args: list | None = None,
        add_sycl: bool = False,
    ) -> None:
        """Configures the project."""

        is_gdb_mode = os.environ.get("LLVM_BENCHMARKS_USE_GDB", "") == "1"
        build_type = "RelWithDebInfo" if is_gdb_mode else "Release"

        cmd = [
            "cmake",
            f"-S {self.src_dir}",
            f"-B {self.build_dir}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
        ]
        if self._use_installdir:
            cmd.append(f"-DCMAKE_INSTALL_PREFIX={self.install_dir}")
        if extra_args:
            cmd.extend(extra_args)

        run(cmd, add_sycl=add_sycl)

    def build(
        self,
        target: str = "",
        add_sycl: bool = False,
        ld_library: list = [],
        timeout: int | None = None,
    ) -> None:
        """Builds the project."""
        target_arg = f"--target {target}" if target else ""
        run(
            f"cmake --build {self.build_dir} {target_arg} -j {options.build_jobs}",
            add_sycl=add_sycl,
            ld_library=ld_library,
            timeout=timeout,
        )
        # When the build dir is the final artifact dir, the build succeeding
        # (run() didn't raise) means the project is ready to use.
        if not self._use_installdir:
            self._mark_build_complete()

    def install(self) -> None:
        """Installs the project."""
        run(f"cmake --install {self.build_dir}")
        # The install dir is only complete once cmake --install succeeds.
        if self._use_installdir:
            self._mark_build_complete()

    def _can_shallow_clone_ref(self, ref: str) -> bool:
        """Check if we can do a shallow clone with this ref using git ls-remote."""
        try:
            result = run(f"git ls-remote --heads --tags {self._url} {ref}")
            output = result.stdout.decode().strip()

            if output:
                # Found the ref as a branch or tag
                log.debug(
                    f"Ref '{ref}' found as branch/tag via ls-remote, can shallow clone"
                )
                return True
            else:
                # Not found as branch/tag, likely a SHA commit or a special ref
                log.debug(
                    f"Ref '{ref}' not found as branch/tag via ls-remote, likely SHA commit or a special ref"
                )
                return False
        except Exception as e:
            log.debug(
                f"Could not check ref '{ref}' via ls-remote: {e}, assuming SHA commit"
            )
            return False

    def _git_clone(self) -> None:
        """Clone the git repository."""
        try:
            log.debug(f"Cloning {self._url} into {self.src_dir} at ref {self._ref}")
            git_clone_cmd = (
                f"git clone --recursive --depth 1 {self._url} {self.src_dir}"
            )
            if self._shallow_clone:
                if self._can_shallow_clone_ref(self._ref):
                    # Shallow clone for branches and tags only
                    git_clone_cmd = f"git clone --recursive --depth 1 --branch {self._ref} {self._url} {self.src_dir}"
                else:
                    log.debug(
                        f"Cannot shallow clone ref '{self._ref}', clone default branch"
                    )

            run(git_clone_cmd)
            run(f"git fetch {self._url} {self._ref}", cwd=self.src_dir)
            run(f"git checkout FETCH_HEAD", cwd=self.src_dir)
            log.debug(f"Cloned {self._url} into {self.src_dir} at ref {self._ref}")
        except Exception as e:
            log.error(f"Failed to clone repository {self._url}: {e}")
            raise

    def _git_fetch(self) -> None:
        """Fetch the ref from the remote repository."""
        try:
            log.debug(f"Fetching ref '{self._ref}' for {self._url} in {self.src_dir}")
            run("git reset --hard", cwd=self.src_dir)
            run(f"git fetch {self._url} {self._ref}", cwd=self.src_dir)
            run(f"git checkout FETCH_HEAD", cwd=self.src_dir)
            log.debug(f"Fetched changes for {self._url} in {self.src_dir}")
        except Exception as e:
            log.error(f"Failed to fetch updates for repository {self._url}: {e}")
            raise

    def _setup_repo(self) -> bool:
        """Clone a git repository into a specified directory at a specific ref.
        Returns:
            bool: True if the repository was cloned or updated, False if it was already up-to-date.
        """
        if self._src_dir_override is not None:
            if not self.src_dir.exists():
                raise Exception(
                    f"Specified source directory {self.src_dir} does not exist."
                )
            # No git operations are performed on a user-managed tree. If it's a
            # git repo and its working-tree state matches the last successful
            # build, skip rebuilding; otherwise (non-git, or changed) rebuild.
            fingerprint = self._git_worktree_fingerprint()
            if fingerprint is None:
                log.debug(
                    f"Source tree at {self.src_dir} is not a git repository; "
                    "rebuilding to be safe."
                )
                return True
            if self._read_build_marker().get("worktree_fingerprint") == fingerprint:
                log.debug(
                    f"Source tree at {self.src_dir} unchanged since last build; "
                    "no rebuild needed."
                )
                return False
            log.debug(
                f"Source tree at {self.src_dir} changed since last build; "
                "rebuild needed."
            )
            return True
        if os.environ.get("LLVM_BENCHMARKS_UNIT_TESTING") == "1":
            log.debug(
                f"Skipping git operations during unit testing of {self._name} (LLVM_BENCHMARKS_UNIT_TESTING=1)."
            )
            return False
        if options.offline:
            log.debug(
                f"Skipping git operations for {self._name} due to --offline option."
            )
            return False
        if not self.src_dir.exists():
            self._git_clone()
            return True
        elif Path(self.src_dir, ".git").exists():
            log.debug(
                f"Repository {self._url} already exists at {self.src_dir}, checking for updates."
            )
            current_commit = (
                run("git rev-parse HEAD^{commit}", cwd=self.src_dir)
                .stdout.decode()
                .strip()
            )
            try:
                target_commit = (
                    run(f"git rev-parse {self._ref}^{{commit}}", cwd=self.src_dir)
                    .stdout.decode()
                    .strip()
                )
                if current_commit != target_commit:
                    log.debug(
                        f"Current commit {current_commit} does not match target {target_commit}, checking out {self._ref}."
                    )
                    run("git reset --hard", cwd=self.src_dir)
                    run(f"git checkout {self._ref}", cwd=self.src_dir)
                    return True
            except Exception:
                log.error(
                    f"Failed to resolve target commit {self._ref}. Fetching updates."
                )
                if self._shallow_clone:
                    log.debug(f"Cloning a clean shallow copy.")
                    shutil.rmtree(self.src_dir)
                    self._git_clone()
                    return True
                else:
                    self._git_fetch()
                    return True
            else:
                log.debug(
                    f"Current commit {current_commit} matches target {target_commit}, no update needed."
                )
                return False
        else:
            raise Exception(
                f"The directory {self.src_dir} exists but is not a git repository."
            )
