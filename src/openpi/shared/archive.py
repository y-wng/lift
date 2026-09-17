"""Extraction rules for recordings received from external sources."""

from pathlib import Path
import tarfile


def extract_recording(archive: tarfile.TarFile, destination: str | Path) -> None:
    """Extract data files only, rejecting traversal, links, and special files."""
    if not hasattr(tarfile, "data_filter"):
        raise RuntimeError("Safe archive extraction requires Python 3.11.8 or newer.")

    def recording_filter(member: tarfile.TarInfo, target: str) -> tarfile.TarInfo:
        if Path(member.name).is_absolute():
            raise tarfile.FilterError(f"Recording archive paths must be relative: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise tarfile.FilterError(f"Recording archives cannot contain links or special files: {member.name}")
        return tarfile.data_filter(member, target)

    archive.extractall(path=destination, filter=recording_filter)
