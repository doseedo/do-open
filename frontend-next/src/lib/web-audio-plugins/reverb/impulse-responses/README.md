# Impulse Response Files

This directory contains impulse response (IR) files for use with the HybridReverb plugin.

## What are Impulse Responses?

An impulse response is a recording that captures the acoustic characteristics of a physical space or hardware reverb unit. When used with convolution reverb, IRs provide realistic spatial effects that sound like actual rooms, halls, or vintage gear.

## Supported Formats

- **WAV files** (16-bit or 24-bit, mono or stereo)
- Sample rates: 44.1kHz, 48kHz, 96kHz (automatically converted)
- Duration: Typically 1-6 seconds

## Where to Get Impulse Responses

### Free IR Libraries

1. **OpenAIR** (http://www.openairlib.net/)
   - Academic library with concert halls, churches, studios
   - High-quality recordings from real spaces

2. **EchoThief** (http://www.echothief.com/)
   - Free IRs from vintage hardware reverbs
   - Lexicon, EMT, AMS units

3. **Voxengo Impulse Modeler**
   - Generate synthetic IRs
   - Great for learning and experimentation

4. **freeverb3** (https://github.com/freeverb3/freeverb3)
   - Open-source IR collection
   - Various spaces and effects

### Creating Your Own IRs

You can record impulse responses from real spaces:

1. **Equipment needed:**
   - Portable recorder or audio interface
   - Speaker or starter pistol (for impulse)
   - Quiet environment

2. **Recording process:**
   - Record a loud, short impulse (balloon pop, clap, etc.)
   - Capture the room's decay
   - Normalize and trim in audio editor
   - Export as WAV file

## Included Presets (Examples)

This directory should contain at least these IRs:

### Small Room (`small-room.wav`)
- Room size: 3x4 meters
- Decay time: ~0.3s
- Use for: Vocals, acoustic instruments
- Character: Intimate, clear

### Large Hall (`large-hall.wav`)
- Room size: 20x30 meters
- Decay time: ~2.5s
- Use for: Orchestral, ambient
- Character: Spacious, grand

### Plate Reverb (`plate.wav`)
- Emulates vintage plate reverb
- Decay time: ~2.0s
- Use for: Drums, vocals
- Character: Bright, dense

## Using IRs with HybridReverb

```javascript
// Load impulse response
const reverb = new HybridReverb(audioContext);
await reverb.loadImpulseResponse('reverb/impulse-responses/large-hall.wav');

// Or from file input
const fileInput = document.getElementById('ir-file');
fileInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  await reverb.loadImpulseResponseFile(file);
});

// Adjust IR parameters
reverb.setIRLength(80);  // Use 80% of IR (trim tail)
reverb.normalizeIR();     // Prevent clipping
```

## Best Practices

1. **Normalize IRs**: Always normalize to prevent clipping
2. **Trim silence**: Remove leading/trailing silence
3. **Match sample rates**: Use IRs matching your project's sample rate
4. **Fade tails**: Apply short fade-out to prevent clicks
5. **Test levels**: Start with low IR level and adjust up

## IR File Naming Convention

Use descriptive names:
- `{space-type}_{size}_{characteristic}.wav`
- Examples:
  - `hall_large_bright.wav`
  - `room_small_wood.wav`
  - `plate_vintage_emt140.wav`
  - `spring_guitar_amp.wav`

## Technical Specifications

### Optimal IR Characteristics

- **Length**: 1-4 seconds (longer for large spaces)
- **Sample rate**: Match your project (44.1/48kHz)
- **Bit depth**: 24-bit preferred
- **Channels**: Stereo (for spatial information)
- **Dynamic range**: High SNR, minimal noise floor

### IR Processing Tips

1. **Deconvolution**: If using swept sine recording method
2. **EQ**: Flatten response if needed
3. **Normalization**: Peak normalize to -3dB
4. **Fade in/out**: 1ms fade-in, 10-50ms fade-out
5. **Trim**: Remove silence before impulse

## Performance Considerations

- Longer IRs require more CPU
- Stereo IRs use 2x processing vs mono
- Consider trimming IRs for real-time use
- Use algorithmic tail for long decay times

## License Notes

When distributing your project:
- Check IR licenses (many are free for personal use only)
- Commercial projects may require licensed IRs
- Give attribution where required
- Consider recording your own IRs for full ownership

## Resources

- **Impulse Response Primer**: https://www.native-instruments.com/en/reaktor-community/reaktor-user-library/impulse-responses/
- **OpenAIR Library**: http://www.openairlib.net/
- **Web Audio Convolver API**: https://developer.mozilla.org/en-US/docs/Web/API/ConvolverNode

---

**Note**: This directory currently contains placeholder documentation. Add your own IR files here in WAV format. For testing, you can generate simple IRs or download from the free libraries listed above.
