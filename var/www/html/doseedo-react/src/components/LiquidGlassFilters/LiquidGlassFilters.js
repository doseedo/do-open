import React from 'react';

/**
 * LiquidGlassFilters Component
 * SVG filters for liquid glass distortion effects
 * Based on Apple's Liquid Glass design language
 */
function LiquidGlassFilters() {
  return (
    <svg
      style={{
        position: 'absolute',
        width: 0,
        height: 0,
        overflow: 'hidden',
        pointerEvents: 'none'
      }}
      aria-hidden="true"
    >
      <defs>
        {/* Main liquid glass distortion filter */}
        <filter id="liquid-glass-distortion" x="-50%" y="-50%" width="200%" height="200%">
          {/* Generate fractal noise for organic distortion */}
          <feTurbulence
            type="fractalNoise"
            baseFrequency="0.01 0.008"
            numOctaves="3"
            seed="42"
            result="noise"
          />

          {/* Blur the noise for smoother distortion */}
          <feGaussianBlur in="noise" stdDeviation="2.5" result="blurredNoise" />

          {/* Create displacement map for the distortion effect */}
          <feDisplacementMap
            in="SourceGraphic"
            in2="blurredNoise"
            scale="25"
            xChannelSelector="R"
            yChannelSelector="G"
            result="displaced"
          />

          {/* Add specular lighting for glass highlights */}
          <feSpecularLighting
            in="blurredNoise"
            surfaceScale="3"
            specularConstant="1.2"
            specularExponent="15"
            lightingColor="#ffffff"
            result="specular"
          >
            <fePointLight x="150" y="100" z="200" />
          </feSpecularLighting>

          {/* Blend the specular highlights with the displaced image */}
          <feComposite
            in="specular"
            in2="displaced"
            operator="in"
            result="composite"
          />

          {/* Merge everything together */}
          <feMerge>
            <feMergeNode in="displaced" />
            <feMergeNode in="composite" />
          </feMerge>
        </filter>

        {/* Subtle liquid glass filter for panels */}
        <filter id="liquid-glass-subtle" x="-20%" y="-20%" width="140%" height="140%">
          <feTurbulence
            type="fractalNoise"
            baseFrequency="0.012 0.01"
            numOctaves="2"
            seed="78"
            result="noise"
          />

          <feGaussianBlur in="noise" stdDeviation="1.5" result="blurredNoise" />

          <feDisplacementMap
            in="SourceGraphic"
            in2="blurredNoise"
            scale="12"
            xChannelSelector="R"
            yChannelSelector="G"
            result="displaced"
          />

          <feSpecularLighting
            in="blurredNoise"
            surfaceScale="2"
            specularConstant="0.8"
            specularExponent="12"
            lightingColor="#ffffff"
            result="specular"
          >
            <fePointLight x="100" y="80" z="150" />
          </feSpecularLighting>

          <feComposite
            in="specular"
            in2="displaced"
            operator="in"
            result="composite"
          />

          <feMerge>
            <feMergeNode in="displaced" />
            <feMergeNode in="composite" />
          </feMerge>
        </filter>

        {/* Strong liquid glass filter for interactive elements */}
        <filter id="liquid-glass-strong" x="-50%" y="-50%" width="200%" height="200%">
          <feTurbulence
            type="fractalNoise"
            baseFrequency="0.008 0.006"
            numOctaves="4"
            seed="123"
            result="noise"
          />

          <feGaussianBlur in="noise" stdDeviation="3" result="blurredNoise" />

          <feDisplacementMap
            in="SourceGraphic"
            in2="blurredNoise"
            scale="35"
            xChannelSelector="R"
            yChannelSelector="G"
            result="displaced"
          />

          <feSpecularLighting
            in="blurredNoise"
            surfaceScale="4"
            specularConstant="1.5"
            specularExponent="18"
            lightingColor="#ffffff"
            result="specular"
          >
            <fePointLight x="200" y="120" z="250" />
          </feSpecularLighting>

          <feComposite
            in="specular"
            in2="displaced"
            operator="in"
            result="composite"
          />

          <feMerge>
            <feMergeNode in="displaced" />
            <feMergeNode in="composite" />
          </feMerge>
        </filter>

        {/* Animated liquid glass filter (for hover states) */}
        <filter id="liquid-glass-animated" x="-50%" y="-50%" width="200%" height="200%">
          <feTurbulence
            type="fractalNoise"
            baseFrequency="0.009 0.007"
            numOctaves="3"
            seed="256"
            result="noise"
          >
            {/* Animate the turbulence */}
            <animate
              attributeName="baseFrequency"
              from="0.009 0.007"
              to="0.011 0.009"
              dur="3s"
              repeatCount="indefinite"
            />
          </feTurbulence>

          <feGaussianBlur in="noise" stdDeviation="2.5" result="blurredNoise" />

          <feDisplacementMap
            in="SourceGraphic"
            in2="blurredNoise"
            scale="28"
            xChannelSelector="R"
            yChannelSelector="G"
            result="displaced"
          />

          <feSpecularLighting
            in="blurredNoise"
            surfaceScale="3.5"
            specularConstant="1.3"
            specularExponent="16"
            lightingColor="#ffffff"
            result="specular"
          >
            <fePointLight x="175" y="110" z="220">
              {/* Animate the light position */}
              <animate
                attributeName="x"
                from="175"
                to="185"
                dur="2s"
                repeatCount="indefinite"
                direction="alternate"
              />
            </fePointLight>
          </feSpecularLighting>

          <feComposite
            in="specular"
            in2="displaced"
            operator="in"
            result="composite"
          />

          <feMerge>
            <feMergeNode in="displaced" />
            <feMergeNode in="composite" />
          </feMerge>
        </filter>
      </defs>
    </svg>
  );
}

export default LiquidGlassFilters;
