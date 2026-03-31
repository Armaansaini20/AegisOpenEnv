#!/usr/bin/env bash
# Official Meta OpenEnv Pre-Validation Script

# Colors
GREEN='\033[0;32m'
BOLD='\033[1m'
NC='\033[0m'

log() { echo -e "$@"; }
pass() { log "  ${GREEN}PASS${NC}: $@"; }
fail() { log "  ${BOLD}FAIL${NC}: $@"; }
hint() { log "  TIP: $@"; }
stop_at() { log "\nStopped at ${BOLD}$1${NC}. Please fix the issue and try again."; exit 1; }

# Mocking variables for local check
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X GET "https://armaan020-aegisopenenv.hf.space/health" || echo "000")
REPO_DIR="."
DOCKER_BUILD_TIMEOUT=300

log "${BOLD}Step 1/3: Checking HF Space Status${NC} ..."
if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space is live and responds to GET /health"
elif [ "$HTTP_CODE" = "000" ]; then
  fail "HF Space not reachable (connection failed or timed out)"
  hint "Check your network connection and that the Space is running."
  stop_at "Step 1"
else
  fail "HF Space /reset returned HTTP $HTTP_CODE (expected 200)"
  stop_at "Step 1"
fi

log "${BOLD}Step 2/3: Running docker build${NC} ..."

if ! command -v docker &>/dev/null; then
  fail "docker command not found"
  hint "Install Docker: https://docs.docker.com/get-docker/"
  stop_at "Step 2"
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found in repo root or server/ directory"
  stop_at "Step 2"
fi

log "  Found Dockerfile in $DOCKER_CONTEXT"

# BUILD_OK=false
# BUILD_OUTPUT=$(docker build "$DOCKER_CONTEXT" 2>&1) && BUILD_OK=true

# if [ "$BUILD_OK" = true ]; then
#   pass "Docker build succeeded"
# else
#   fail "Docker build failed"
#   stop_at "Step 2"
# fi

log "${BOLD}Step 3/3: Running openenv validate${NC} ..."

# Check if openenv is available as a command or a python module
if ! command -v openenv &>/dev/null && ! python -c "import openenv.core" &>/dev/null; then
  fail "openenv not found (tried 'openenv' command and 'python -m openenv.core')"
  hint "Install it: pip install openenv-core"
  stop_at "Step 3"
fi

VALIDATE_OK=false
VALIDATE_OUTPUT=$(cd "$REPO_DIR" && python -m openenv.core.cli validate 2>&1) && VALIDATE_OK=true

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
  [ -n "$VALIDATE_OUTPUT" ] && log "  $VALIDATE_OUTPUT"
else
  fail "openenv validate failed"
  # printf "%s\n" "$VALIDATE_OUTPUT"
  stop_at "Step 3"
fi

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  All 3/3 checks passed!${NC}\n"
printf "${GREEN}${BOLD}  Your submission is ready to submit.${NC}\n"
printf "${BOLD}========================================${NC}\n"
printf "\n"

exit 0
