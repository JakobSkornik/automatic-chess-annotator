# Frozen survey deployment data

`study-20261024.json.gz` is the original Luna-only pilot: ten selected games and
their saved English commentary. The expanded comparison grid is a separate
study; see `docs/survey-grid.md`. Its bundle is added only after generation of
every requested cell completes. The backend imports existing bundles at startup
when that study is absent from its response database. Redeployment preserves
existing responses and refuses conflicting changes to the frozen material.

The bundle includes capability hashes, never the raw participant/researcher
tokens. It includes no participant responses. The private link receipt and
raw generation logs remain outside the repository. Do not edit a published
bundle; prepare and publish a new study if the material needs changing.

Source games: the first 12,000 records of the January 2025 standard-games
archive at https://database.lichess.org/ (CC0). Selection and generation
provenance are included in the bundle. See `docs/survey.md` in this repository
for the study design, export command, and persistent-storage requirements.
