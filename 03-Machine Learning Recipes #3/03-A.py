#https://www.youtube.com/watch?v=N9fDIAflCMY&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal&index=3

# Install 
#- matplotlib using this command "pip install -U matplotlib"

import numpy as np
import matplotlib.pyplot as plt

# we define a population of 1000 dogs 50% of levrieri and 50% labrador
grayhounds = 500 #levreri
labs = 500       #labrador

# we define for eatch dog the height
gray_height = 28 + 1 * np.random.randn(grayhounds)
lab_height = 24 + 4 * np.random.randn(labs)

# we plot the chart
plt.hist([gray_height, lab_height], stacked=True, color=['r', 'b'])
plt.show()
