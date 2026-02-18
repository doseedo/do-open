import { forwardRef } from "react";

/**
 * GlassCard Component - Simplified for performance
 * Previously contained heavy SVG chromatic aberration filters
 * Now just renders a simple glass-styled container
 */

const GlassContainer = forwardRef(
    (
        {
            children,
            className = "",
            style,
            cornerRadius = 10,
            padding = "0px 0px",
            shadowMode = false,
            onClick,
        },
        ref,
    ) => {
        return (
            <div style={{ display: "flex" }}>
                <div
                    className={className}
                    ref={ref}
                    style={{
                        ...style,
                        borderRadius: `${cornerRadius}px`
                    }}
                    onClick={onClick}
                >
                    <div
                        className="glass"
                        style={{
                            borderRadius: `${cornerRadius}px`,
                            position: "relative",
                            display: "inline-flex",
                            alignItems: "center",
                            gap: "24px",
                            padding,
                            overflow: "hidden",
                            background: "rgba(255, 255, 255, 0.08)",
                            border: "1px solid rgba(255, 255, 255, 0.15)",
                            boxShadow: shadowMode
                                ? "0px 8px 32px rgba(0, 0, 0, 0.4)"
                                : "0px 4px 16px rgba(0, 0, 0, 0.2)",
                        }}
                    >
                        <div
                            style={{
                                position: "relative",
                                zIndex: 1,
                                color: "white",
                            }}
                        >
                            {children}
                        </div>
                    </div>
                </div>
            </div>
        );
    },
);

GlassContainer.displayName = "GlassContainer";

export default function GlassCard({
    children,
    cornerRadius = 10,
    className = "",
    padding = "0px 0px",
    shadowMode = false,
    style = {},
    onClick,
    // These props are kept for API compatibility but no longer used
    displacementScale,
    blurAmount,
    mouseOffset,
    mouseContainer,
}) {
    return (
        <div style={{ position: "relative" }}>
            <GlassContainer
                className={className}
                style={style}
                cornerRadius={cornerRadius}
                padding={padding}
                shadowMode={shadowMode}
                onClick={onClick}
            >
                {children}
            </GlassContainer>
        </div>
    );
}
