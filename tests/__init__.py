import warnings

import matplotlib.pyplot as plt

# switch matplotlib backend to non-Gui, preventing plots being displayed
plt.switch_backend("Agg")
# suppress UserWarning that the current backend cannot show plots
warnings.filterwarnings("ignore", "Matplotlib is currently using agg")
