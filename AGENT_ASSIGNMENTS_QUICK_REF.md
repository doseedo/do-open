# Multi-Agent Research - Quick Reference Sheet

## How to Use This Prompt

1. **Copy** the entire content of `AGENT_RESEARCH_MASTER_PROMPT.md`
2. **Send** the same prompt to 10 different AI agents (Claude, ChatGPT, etc.)
3. **Specify** which agent number they are: "You are Agent #3"
4. **Collect** their research deliverables
5. **Synthesize** the results into a unified enhancement roadmap

## Agent Roster & Assignments

| Agent # | Role | Primary Focus | Expected Output |
|---------|------|---------------|-----------------|
| **#1** | Genre Expansion | EDM, metal, hip-hop, classical, R&B, country | 30+ genre specs, pattern databases |
| **#2** | ML Integration | Transformers, VAEs, diffusion models | Hybrid ML+rule architecture |
| **#3** | Control Systems | Constraints, parameters, emotional controls | 50+ control dimensions, API |
| **#4** | Rhythm & Groove | Polyrhythms, humanization, cultural patterns | Groove extraction, 100+ rhythm patterns |
| **#5** | Harmonic Theory | Counterpoint, spectral, atonal, quartal | Voice leading optimizer, 20+ techniques |
| **#6** | Melodic Intelligence | Contour, motif development, narrative arc | Melodic tension modeling, transformations |
| **#7** | Testing & QA | Test coverage 5% → 90%, CI/CD | 2,000+ test plan, automation pipeline |
| **#8** | Performance | Vectorization, caching, parallelization | 10x speedup roadmap, profiling report |
| **#9** | Format Support | MusicXML, ABC, LilyPond, DAW integration | Export/import pipelines, VST architecture |
| **#10** | Real-Time Performance | Live MIDI I/O, interactive modes, low latency | Real-time architecture, <10ms latency |

## Deliverable Checklist (Per Agent)

Each agent should provide:

- [ ] **Executive Summary** (1 paragraph)
- [ ] **Current State Analysis** (what's missing vs. competitors)
- [ ] **State-of-the-Art Survey** (5-10 key findings from 2024-2025)
- [ ] **Top 5 Proposals** (What, Why, How, Complexity, API Example)
- [ ] **Prioritized Roadmap** (Phase 1: Quick Wins, Phase 2: Core, Phase 3: Advanced)
- [ ] **References** (papers, repos, tools, datasets)

## Expected Timeline

- **Agent Research Phase:** 1-2 days (all agents work in parallel)
- **Synthesis Phase:** 1 day (combine findings into master plan)
- **Implementation Phase:** 3-6 months (based on priorities)

## Coordination Notes

### Dependencies to Watch:
- **Agent #7 (Testing)** blocks all other work → highest priority
- **Agent #2 (ML)** + **Agent #10 (Real-time)** may have architecture conflicts (resolve early)
- **Agent #5 (Harmony)** + **Agent #6 (Melody)** should coordinate on music theory
- **Agent #8 (Performance)** needs to review all other proposals (can't optimize later)

### Synthesis Strategy:
1. Collect all 10 deliverables
2. Extract **quick wins** (low complexity, high impact) from each
3. Build **Phase 1 roadmap** (1-2 weeks of work)
4. Identify **architectural decisions** that affect multiple domains (resolve conflicts)
5. Create **unified API** across all enhancements (consistent design)
6. Estimate **total effort** (person-months)
7. Prioritize based on: (a) Testing first, (b) User-facing features, (c) Performance last

## Success Metrics

After agent research + implementation, the library should achieve:

| Metric | Current | Target | Agent Responsible |
|--------|---------|--------|-------------------|
| **Test Coverage** | <5% | 90%+ | Agent #7 |
| **Genres Supported** | ~10 | 50+ | Agent #1 |
| **Control Parameters** | ~20 | 100+ | Agent #3 |
| **Generation Speed** | 1x | 10x | Agent #8 |
| **ML Capabilities** | None | 3+ models | Agent #2 |
| **Format Support** | MIDI only | +3 formats | Agent #9 |
| **Real-time Capable** | No | Yes (<10ms) | Agent #10 |
| **Rhythm Patterns** | ~50 | 200+ | Agent #4 |
| **Harmonic Techniques** | ~30 | 80+ | Agent #5 |
| **Melodic Intelligence** | Basic | Advanced | Agent #6 |

## Next Steps

1. ✅ **Done:** Master prompt created (`AGENT_RESEARCH_MASTER_PROMPT.md`)
2. **TODO:** Deploy prompt to 10 agents
3. **TODO:** Collect deliverables in `/research_reports/agent_01.md` ... `agent_10.md`
4. **TODO:** Synthesize findings into `ENHANCEMENT_MASTER_PLAN.md`
5. **TODO:** Implement Phase 1 (quick wins)

---

**File Location:** `AGENT_RESEARCH_MASTER_PROMPT.md` (502 lines)
**Committed & Pushed:** ✅ `claude/expand-music-genres-01MCCFchdpgpDRc6CV6neTmm`

Ready to revolutionize the library! 🎵🚀
