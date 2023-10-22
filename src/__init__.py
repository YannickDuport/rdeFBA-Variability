from pathlib import Path
from pyrrrateFBA.util.runge_kutta import RungeKuttaPars

PACKAGE_PATH = Path(__file__).parent
PROJECT_PATH = PACKAGE_PATH.parent
RESULT_PATH = PROJECT_PATH / 'results'
FIGURE_PATH = RESULT_PATH / 'figures'

discretization_schemes = {
    'default': None,
    'trapezoidal': RungeKuttaPars(family='LobattoIIIA', s=2),
    'implicit_euler': RungeKuttaPars(family='RadauIIA', s=1),
    'radau3': RungeKuttaPars(family='RadauIIA', s=2),
    'radau5': RungeKuttaPars(family='RadauIIA', s=3),
    'implicit_midpoint': RungeKuttaPars(family='Gauss', s=1),
    'explicit_euler': RungeKuttaPars(family='Explicit1', s=1),
    'heun': RungeKuttaPars(family='Explicit1', s=2),
    'kutta_simpson': RungeKuttaPars(family='Explicit1', s=3),
    'rk4': RungeKuttaPars(family='Explicit1', s=4),
}