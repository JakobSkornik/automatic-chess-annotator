"""Upload any completed comparison responses that are not yet in the Drive backup folder."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / '.env')
from app import drive_backup


if __name__ == '__main__':
    if not drive_backup.configured():
        raise SystemExit('Set ' + ', '.join(drive_backup.ENV) + ' first (see scripts/setup_drive_backup.py).')
    print(f'Uploaded {drive_backup.backup_pending()} response(s) to Google Drive.')
