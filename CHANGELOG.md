# Changelog

All notable changes to this project will be documented in this file.

## [V1.2] - 2026-02-18

### Added
- **Duplicate Tag Validation**: Added real-time check when adding exclude tags. If a tag already exists, it is highlighted in yellow to alert the user.
- **Grid Layout for Settings**: Refactored the settings area below the exclude list to use a responsive grid layout, saving vertical space and improving aesthetics.

### Optimized
- **Startup Performance**: significantly reduced application startup time by skipping non-essential path validation checks during initialization (`_validate_paths` is now optional during startup).

### Fixed
- Addressed potential UI clutter in settings panel.
