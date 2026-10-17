
## Plan

Include-list (whitelist) naast ignore (gebruikersvraag): .codesearchinclude (repo-root) + ~/.codesearch/.codesearchinclude (globaal), gitignore-stijl incl. !-negatie. Semantiek: ignore wint altijd; include-list niet-leeg = alleen matches worden geindexeerd (pad zelf of ancestor); afwezig/leeg = historisch gedrag. Walker (file/mod.rs build_include_matcher + is_included + gate vóór binary-check, skipped_ignored-teller nu gebruikt) en watcher (watch/mod.rs zelfde matcher, gate in is_watchable).

1. [x] Stage 1: matcher + gates + 4 tests (walker whitelist/negatie/absent, watcher gate)
2. [x] Stage 2: CHANGELOG + highlights + full validation
