# Doseedo dev / deploy shortcuts. Run `make help` for a listing.
#
# This Makefile lives in the Do/ monorepo (doseedo-next + modal_stemphonic.py +
# CRA source). The Fly auth-service lives in the doseedo-desktop repo, so the
# fly-* targets cd into that sibling checkout.

SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c

AUTH_SERVICE := ../doseedo-desktop/auth-service
MODAL_APP    := doseedo-stemphonic
FLY_APP      := doseedo-api

# Colors for help output
BLUE  := \033[34m
GREEN := \033[32m
RESET := \033[0m

.PHONY: help
help: ## Show this help
	@printf "\n"
	@printf "  $(BLUE)Doseedo dev / deploy commands$(RESET)\n"
	@printf "\n"
	@printf "  $(GREEN)Dev loop (fast iteration):$(RESET)\n"
	@grep -E '^[a-zA-Z_-]+:.*?## dev' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## dev "}; {printf "    %-20s %s\n", $$1, $$2}'
	@printf "\n  $(GREEN)Deploy (prod):$(RESET)\n"
	@grep -E '^[a-zA-Z_-]+:.*?## deploy' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## deploy "}; {printf "    %-20s %s\n", $$1, $$2}'
	@printf "\n  $(GREEN)Observability:$(RESET)\n"
	@grep -E '^[a-zA-Z_-]+:.*?## obs' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## obs "}; {printf "    %-20s %s\n", $$1, $$2}'
	@printf "\n  $(GREEN)Checks / tests:$(RESET)\n"
	@grep -E '^[a-zA-Z_-]+:.*?## test' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## test "}; {printf "    %-20s %s\n", $$1, $$2}'
	@printf "\n"

# ─── Dev loop ──────────────────────────────────────────────────────────

.PHONY: modal-dev
modal-dev: ## dev Hot-reload Modal (edits to stemphonic_server.py sync instantly; temp URL)
	modal serve modal_stemphonic.py

.PHONY: fly-dev
fly-dev: ## dev Tail Fly logs while editing — fastest way to watch auth-service behavior
	cd $(AUTH_SERVICE) && fly logs --app $(FLY_APP)

.PHONY: next-dev
next-dev: ## dev Local Next.js at localhost:3000 (Vercel rewrites don't work; use `vercel dev` for that)
	cd doseedo-next && npm run dev

.PHONY: vercel-dev
vercel-dev: ## dev Local Next.js with Vercel rewrites applied (proxies to prod backends)
	cd doseedo-next && vercel dev

# ─── Deploy ────────────────────────────────────────────────────────────

.PHONY: check-deps
check-deps: ## test Fast local check: do pinned pip packages actually contain the modules we import?
	@bash scripts/check-deps.sh

.PHONY: modal-deploy
modal-deploy: check-deps ## deploy Modal stemphonic app (10-15 min). Auto-runs check-deps first.
	modal deploy modal/modal_stemphonic.py

.PHONY: modal-deploy-chatbot
modal-deploy-chatbot: ## deploy Modal chatbot app (vLLM + Moondream vision, single L4).
	modal deploy modal/modal_chatbot.py

.PHONY: modal-deploy-staging
modal-deploy-staging: check-deps ## deploy Modal stemphonic STAGING app (separate URL, same image).
	modal deploy modal/modal_stemphonic_staging.py

.PHONY: modal-deploy-all
modal-deploy-all: check-deps ## deploy Both Modal apps sequentially (stemphonic + chatbot).
	$(MAKE) modal-deploy
	$(MAKE) modal-deploy-chatbot

.PHONY: fly-deploy
fly-deploy: ## deploy Fly auth-service (2-4 min rolling restart).
	cd $(AUTH_SERVICE) && fly deploy --app $(FLY_APP) --remote-only --now

.PHONY: vercel-deploy
vercel-deploy: ## deploy Vercel manually (normally auto-deploys on `git push`).
	cd doseedo-next && vercel deploy --prod --yes

.PHONY: push
push: ## deploy git push main → triggers Vercel auto-deploy.
	git push origin main

.PHONY: deploy-all
deploy-all: check-deps ## deploy Everything: Modal + Fly + push (Vercel follows push). Sequential; ~20 min total.
	$(MAKE) modal-deploy
	$(MAKE) fly-deploy
	$(MAKE) push

# ─── Observability ─────────────────────────────────────────────────────

.PHONY: modal-logs
modal-logs: ## obs Tail Modal logs (wsgi app).
	modal app logs $(MODAL_APP)

.PHONY: fly-logs
fly-logs: ## obs Tail Fly logs.
	cd $(AUTH_SERVICE) && fly logs --app $(FLY_APP)

.PHONY: fly-status
fly-status: ## obs Fly machine status + health.
	cd $(AUTH_SERVICE) && fly status --app $(FLY_APP)

.PHONY: modal-history
modal-history: ## obs Recent Modal deployments.
	modal app history $(MODAL_APP)

.PHONY: vercel-last
vercel-last: ## obs Last 3 Vercel production deploys.
	@cd doseedo-next && vercel list --scope doseedo --prod | head -6

# ─── Tests ─────────────────────────────────────────────────────────────

.PHONY: smoke
smoke: ## test Full end-to-end smoke: health + auth + live Modal stem-separation POST.
	@python3 scripts/e2e_smoke.py

.PHONY: verify-frontend
verify-frontend: ## test Verify sem4Decoder deploy: R2 model routes + bundle markers + live ONNX pipeline. No browser.
	@python3 scripts/verify_frontend_deploy.py

.PHONY: smoke-preview
smoke-preview: ## test Like `smoke` but against a Vercel preview URL (set BASE env var).
	@BASE=$${BASE:-$$(cd doseedo-next && vercel list --scope doseedo --prod 2>/dev/null | grep '● Ready' | head -1 | awk '{print $$3}')} python3 scripts/e2e_smoke.py

.PHONY: types-next
types-next: ## test TypeScript type-check on doseedo-next (fast).
	cd doseedo-next && npx tsc --noEmit

.PHONY: build-next
build-next: ## test Full Next.js production build (catches everything tsc misses).
	cd doseedo-next && npm run build

.PHONY: types-backend
types-backend: ## test Python compile check on auth-service.
	cd $(AUTH_SERVICE) && python3 -m py_compile app/*.py app/routers/*.py && echo "✓ auth-service compiles"
