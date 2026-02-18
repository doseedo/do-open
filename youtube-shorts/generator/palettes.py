# Clean/minimal color palette

BACKGROUND = (245, 245, 245)       # light gray
PEG_COLOR = (60, 60, 60)           # dark gray
WALL_COLOR = (100, 100, 100)       # medium gray
BIN_BORDER = (180, 180, 180)       # light border

# Ball/marble colors — solid, distinct
BALL_COLORS = [
    (231, 76, 60),    # red
    (52, 152, 219),   # blue
    (46, 204, 113),   # green
    (243, 156, 18),   # orange
    (155, 89, 182),   # purple
    (26, 188, 156),   # teal
    (241, 196, 15),   # yellow
    (230, 126, 34),   # dark orange
    (52, 73, 94),     # navy
    (231, 76, 120),   # pink
]

FINISH_LINE = (46, 204, 113)       # green
OBSTACLE_COLOR = (80, 80, 80)      # dark gray
TEXT_COLOR = (40, 40, 40)          # near-black

# --- Named palette sets for variety ---

PALETTES = {
    "classic": {
        "bg": (245, 245, 245),
        "wall": (100, 100, 100),
        "peg": (60, 60, 60),
        "obstacle": (80, 80, 80),
        "text": (40, 40, 40),
        "balls": BALL_COLORS,
    },
    "neon": {
        "bg": (15, 15, 30),
        "wall": (40, 40, 60),
        "peg": (60, 60, 80),
        "obstacle": (50, 50, 70),
        "text": (220, 220, 255),
        "balls": [
            (255, 0, 102),    # hot pink
            (0, 255, 204),    # cyan
            (255, 255, 0),    # yellow
            (102, 0, 255),    # violet
            (0, 204, 255),    # sky blue
            (255, 102, 0),    # orange
            (0, 255, 102),    # green
            (255, 0, 255),    # magenta
            (102, 255, 0),    # lime
            (255, 153, 204),  # pink
        ],
    },
    "sunset": {
        "bg": (255, 240, 230),
        "wall": (140, 80, 60),
        "peg": (120, 60, 40),
        "obstacle": (130, 70, 50),
        "text": (80, 30, 10),
        "balls": [
            (255, 87, 51),    # flame
            (255, 150, 50),   # orange
            (255, 195, 0),    # gold
            (200, 50, 80),    # crimson
            (255, 120, 80),   # coral
            (180, 40, 100),   # berry
            (255, 170, 120),  # peach
            (220, 80, 40),    # rust
            (255, 210, 100),  # amber
            (160, 30, 60),    # wine
        ],
    },
    "ocean": {
        "bg": (230, 242, 255),
        "wall": (50, 90, 130),
        "peg": (40, 70, 110),
        "obstacle": (45, 80, 120),
        "text": (20, 40, 80),
        "balls": [
            (0, 119, 190),    # ocean blue
            (0, 180, 216),    # cerulean
            (72, 202, 228),   # light blue
            (144, 224, 239),  # sky
            (0, 150, 136),    # teal
            (38, 166, 154),   # sea green
            (0, 96, 100),     # deep teal
            (100, 181, 246),  # periwinkle
            (0, 200, 180),    # aqua
            (30, 130, 180),   # steel blue
        ],
    },
    "forest": {
        "bg": (235, 245, 235),
        "wall": (70, 100, 60),
        "peg": (50, 80, 40),
        "obstacle": (60, 90, 50),
        "text": (30, 50, 20),
        "balls": [
            (76, 175, 80),    # green
            (139, 195, 74),   # light green
            (205, 220, 57),   # lime
            (255, 193, 7),    # amber
            (121, 85, 72),    # brown
            (56, 142, 60),    # dark green
            (174, 213, 129),  # sage
            (230, 180, 50),   # gold
            (100, 160, 90),   # moss
            (180, 140, 80),   # tan
        ],
    },
    "candy": {
        "bg": (255, 245, 250),
        "wall": (200, 150, 180),
        "peg": (180, 130, 160),
        "obstacle": (190, 140, 170),
        "text": (120, 60, 90),
        "balls": [
            (255, 105, 180),  # hot pink
            (147, 112, 219),  # medium purple
            (255, 182, 193),  # light pink
            (138, 43, 226),   # blue violet
            (255, 140, 200),  # pink
            (186, 85, 211),   # orchid
            (255, 160, 122),  # light salmon
            (219, 112, 147),  # pale violet red
            (255, 130, 171),  # rose
            (199, 21, 133),   # medium violet red
        ],
    },
}


def get_palette(name="classic"):
    """Get a palette by name. Returns classic if name not found."""
    return PALETTES.get(name, PALETTES["classic"])
