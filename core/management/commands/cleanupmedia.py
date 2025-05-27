import os
import time
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    """
    Command to clean up old media files in MEDIA_ROOT.
    Deletes files older than a specified number of hours.
    """

    help = "Deletes files in MEDIA_ROOT older than the given number of hours (default: 1 hour)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--hours",
            type=float,
            default=1.0,
            help="Delete files older than this many hours (default: 1.0)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="List files that would be deleted, but do not delete.",
        )

    def handle(self, *args, **options):
        media_root = Path(settings.MEDIA_ROOT)
        if not media_root.exists():
            self.stderr.write(f"MEDIA_ROOT '{media_root}' does not exist.")
            return

        cutoff = time.time() - options["hours"] * 3600
        deleted = 0
        checked = 0
        for file in media_root.glob("**/*"):
            if file.is_file():
                checked += 1
                if file.stat().st_mtime < cutoff:
                    if options["dry_run"]:
                        self.stdout.write(f"[DRY RUN] Would delete: {file}")
                    else:
                        try:
                            file.unlink()
                            self.stdout.write(f"Deleted: {file}")
                            deleted += 1
                        except Exception as e:
                            self.stderr.write(f"Failed to delete {file}: {e}")
        if options["dry_run"]:
            self.stdout.write(
                f"Checked {checked} files. Would delete files older than {options['hours']} hour(s)."
            )
        else:
            self.stdout.write(
                f"Checked {checked} files. Deleted {deleted} files older than {options['hours']} hour(s)."
            )
