# Web Audio Effects Library - Agent Prompts Index

This document provides an overview of all 10 agent assignments for creating a comprehensive JavaScript Web Audio API implementation of Ableton Live Suite stock effects.

## Project Overview

**Goal**: Create 20+ professional-grade audio effects plugins using modern Web Audio API
**Architecture**: Modular, routable, with efficient parameter automation
**Team**: 10 specialized agents, each handling a different category of effects

## Quick Reference

| Agent | Category | Plugins | File |
|-------|----------|---------|------|
| 1 | Dynamics | Compressor, Gate, Limiter, Glue Compressor | [AGENT_01_DYNAMICS.md](./agent-prompts/AGENT_01_DYNAMICS.md) |
| 2 | EQ & Filters | EQ Eight, EQ Three, Auto Filter | [AGENT_02_EQ_FILTERS.md](./agent-prompts/AGENT_02_EQ_FILTERS.md) |
| 3 | Delays | Simple Delay, Ping Pong Delay, Filter Delay | [AGENT_03_DELAYS.md](./agent-prompts/AGENT_03_DELAYS.md) |
| 4 | Modulation | Chorus, Flanger, Phaser, Tremolo | [AGENT_04_MODULATION.md](./agent-prompts/AGENT_04_MODULATION.md) |
| 5 | Reverb | Reverb, Hybrid Reverb, Echo | [AGENT_05_REVERB.md](./agent-prompts/AGENT_05_REVERB.md) |
| 6 | Distortion | Overdrive, Saturator, Distortion, Redux | [AGENT_06_DISTORTION.md](./agent-prompts/AGENT_06_DISTORTION.md) |
| 7 | Creative | Beat Repeat, Grain Delay, Erosion, Vinyl | [AGENT_07_CREATIVE.md](./agent-prompts/AGENT_07_CREATIVE.md) |
| 8 | Spectral | Spectral Time, Spectral Resonator, Frequency Shifter, Vocoder | [AGENT_08_SPECTRAL.md](./agent-prompts/AGENT_08_SPECTRAL.md) |
| 9 | Utility | Utility, Spectrum Analyzer, Tuner, Channel EQ | [AGENT_09_UTILITY.md](./agent-prompts/AGENT_09_UTILITY.md) |
| 10 | Integration | Router, Preset Manager, Automation, Base Class | [AGENT_10_INTEGRATION.md](./agent-prompts/AGENT_10_INTEGRATION.md) |

**Total Plugins**: 31 unique effects/tools

## How to Use These Prompts

### For AI Agents
1. Read the [MASTER_PROMPT_WEB_AUDIO_EFFECTS.md](./MASTER_PROMPT_WEB_AUDIO_EFFECTS.md) first for global context
2. Open your assigned agent prompt (e.g., AGENT_01_DYNAMICS.md)
3. Complete the research phase before coding
4. Follow the architecture patterns provided
5. Test thoroughly against the checklist
6. Create examples demonstrating your plugins
7. Document everything clearly

### For Project Coordinators
1. Assign one agent prompt to each developer/AI
2. Ensure Agent 10 (Integration) starts early to define interfaces
3. Review deliverables against success criteria
4. Coordinate integration testing
5. Ensure consistent naming and coding standards

## Project Structure

```
web-audio-effects/
├── core/                  # Agent 10: Base classes and infrastructure
│   ├── BasePlugin.js
│   ├── Router.js
│   ├── PresetManager.js
│   ├── ParamAutomation.js
│   └── PerformanceMonitor.js
├── dynamics/              # Agent 1
│   ├── Compressor.js
│   ├── Gate.js
│   ├── Limiter.js
│   └── GlueCompressor.js
├── eq/                    # Agent 2
│   ├── EQEight.js
│   └── EQThree.js
├── filters/               # Agent 2
│   └── AutoFilter.js
├── delay/                 # Agent 3
│   ├── SimpleDelay.js
│   ├── PingPongDelay.js
│   └── FilterDelay.js
├── modulation/            # Agent 4
│   ├── Chorus.js
│   ├── Flanger.js
│   ├── Phaser.js
│   └── Tremolo.js
├── reverb/                # Agent 5
│   ├── Reverb.js
│   ├── HybridReverb.js
│   ├── Echo.js
│   └── impulse-responses/
├── distortion/            # Agent 6
│   ├── Overdrive.js
│   ├── Saturator.js
│   ├── Distortion.js
│   └── Redux.js
├── creative/              # Agent 7
│   ├── BeatRepeat.js
│   ├── GrainDelay.js
│   ├── Erosion.js
│   └── VinylDistortion.js
├── spectral/              # Agent 8
│   ├── SpectralTime.js
│   ├── SpectralResonator.js
│   ├── FrequencyShifter.js
│   ├── Vocoder.js
│   └── worklets/
├── utility/               # Agent 9
│   ├── Utility.js
│   ├── SpectrumAnalyzer.js
│   ├── Tuner.js
│   └── ChannelEQ.js
├── examples/              # All agents contribute examples
│   ├── dynamics-chain-example.html
│   ├── eq-filter-example.html
│   ├── delay-rhythms-example.html
│   ├── modulation-showcase-example.html
│   ├── spatial-effects-example.html
│   ├── distortion-shootout-example.html
│   ├── creative-sound-design-example.html
│   ├── spectral-processing-example.html
│   ├── utility-tools-example.html
│   ├── master-routing-example.html
│   └── full-mixing-console-example.html
└── README.md
```

## Development Phases

### Phase 1: Research (Week 1)
- All agents study their assigned effect types
- Research DSP fundamentals
- Review Web Audio API capabilities
- Study reference implementations (Tone.js, etc.)
- Understand Ableton's plugin behavior

### Phase 2: Core Development (Week 2-3)
- Agent 10 creates base infrastructure first
- Agents 1-9 implement their plugins
- Regular integration testing
- Code reviews and refinements

### Phase 3: Integration & Testing (Week 3-4)
- Cross-agent integration testing
- Performance optimization
- Browser compatibility testing
- Documentation completion
- Example creation

### Phase 4: Polish (Week 4)
- Bug fixes
- Performance tuning
- Documentation review
- Final integration tests
- Release preparation

## Key Technical Requirements

### All Agents Must:
- ✅ Extend BasePlugin class (provided by Agent 10)
- ✅ Use ES6 modules for exports
- ✅ Implement standard interface (input, output, connect, disconnect)
- ✅ Support parameter automation
- ✅ Provide preset save/load
- ✅ Handle resource cleanup (dispose method)
- ✅ Create working examples
- ✅ Write comprehensive documentation

### Performance Targets:
- **Latency**: < 10ms total chain latency
- **CPU**: < 5% per plugin on average hardware
- **Memory**: No leaks, efficient cleanup
- **Real-time**: No dropouts during parameter changes

### Browser Support:
- Chrome/Edge (primary target)
- Firefox (secondary)
- Safari (if time permits)

## Critical Dependencies

### Agent 10 Must Complete First:
- BasePlugin class
- Router architecture
- Parameter interface standards
- This allows other agents to build correctly

### Recommended Order:
1. **Agent 10**: Complete core infrastructure
2. **Agents 1-9**: Parallel development
3. **All**: Integration and testing
4. **All**: Documentation and examples

## Communication Guidelines

### Naming Conventions:
- **Classes**: PascalCase (e.g., `Compressor`)
- **Files**: Match class name (e.g., `Compressor.js`)
- **Parameters**: camelCase (e.g., `attackTime`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_DELAY_TIME`)

### Git Commits:
- Format: `[Category] Plugin Name - Description`
- Example: `[Dynamics] Compressor - Add sidechain support`

### Code Review Criteria:
- Follows BasePlugin interface
- Parameters validated
- Memory properly cleaned up
- Examples demonstrate features
- JSDoc comments complete
- Tested across browsers

## Success Metrics

### Individual Agent Success:
- ✅ All assigned plugins implemented
- ✅ Parameters respond in real-time
- ✅ Sound quality matches Ableton
- ✅ Code is clean and documented
- ✅ Examples work correctly
- ✅ Performance targets met

### Project Success:
- ✅ All 31 plugins functional
- ✅ Arbitrary routing works
- ✅ Presets save/load reliably
- ✅ Automation system functional
- ✅ Performance acceptable
- ✅ Documentation complete
- ✅ Examples comprehensive

## Resources

### Essential Reading:
1. [Web Audio API Specification](https://www.w3.org/TR/webaudio/)
2. [Ableton Live Manual](https://www.ableton.com/en/manual/live-audio-effect-reference/)
3. [Tone.js Documentation](https://tonejs.github.io/)
4. "Designing Audio Effect Plugins in C++" by Will Pirkle (concepts applicable to Web Audio)

### Reference Implementations:
- [Tone.js](https://github.com/Tonejs/Tone.js) - Well-architected Web Audio library
- [Tuna.js](https://github.com/Theodeus/tuna) - Audio effects library
- [Web Audio Modules](https://www.webaudiomodules.org/) - Plugin standard

### Tools:
- Chrome DevTools (Performance tab)
- Audacity (for analyzing output)
- Oscilloscope visualizations
- Spectrum analyzers

## Timeline

### Week 1 (Research):
- Read documentation
- Study reference implementations
- Understand DSP fundamentals
- Plan architecture

### Week 2 (Development):
- Agent 10: Complete core infrastructure
- Agents 1-9: Begin plugin implementation
- Daily integration checks

### Week 3 (Refinement):
- Complete all plugins
- Integration testing
- Performance optimization
- Bug fixes

### Week 4 (Polish):
- Final testing
- Documentation
- Examples
- Release preparation

## Questions?

Each agent prompt contains:
- Detailed plugin specifications
- Research topics and resources
- Architecture patterns and code examples
- Testing checklists
- Success criteria

Refer to the master prompt and your specific agent prompt for complete details.

## Let's Build Something Amazing! 🎵🎛️🎚️

This is an ambitious project that will result in a comprehensive, professional-grade Web Audio effects library. Each agent plays a crucial role in bringing this vision to life.

Remember:
- **Research thoroughly** before coding
- **Test extensively** at each step
- **Document clearly** for others to use
- **Optimize performance** for real-time audio
- **Communicate** with other agents for integration

Good luck to all agents! Let's create the most complete Web Audio effects library available! 🚀
