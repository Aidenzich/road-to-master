tree:
	git ls-tree -r --name-only HEAD | tree --fromfile

# --- research note-quality gate (see tools/research/README.md) ---
# Run all deterministic validators against one note; exits non-zero on any violation.
#   make research-check NOTE=<domain>/<Topic>
research-check:
	@if [ -z "$(NOTE)" ]; then \
		echo "usage: make research-check NOTE=<domain>/<Topic>"; exit 2; \
	fi
	python3 tools/research/check_note.py --note "$(NOTE)"

# Red/green proof: the good fixture PASSes all gates and each broken fixture FAILs exactly
# its intended gate(s). Exits non-zero if any fixture deviates from its expected verdict.
research-check-fixtures:
	python3 tools/research/check_note.py --all-fixtures

.PHONY: tree research-check research-check-fixtures
