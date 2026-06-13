import numpy as np
import pandas as pd
import sys
# sys.path.append("../")
from algorithms.core.cpc import CPC
from algorithms.utils.graph_utils import visualize_graph_color

# Initialize and run CPC algorithm
n_rows = 100
n_cols = 10

# Generate random binary data (0 or 1)
data = np.random.randint(0, 2, size=(n_rows, n_cols))

# Create column names: X0, X1, ..., X9
columns = [f'X{i}' for i in range(n_cols)]

# Create DataFrame
df_binary = pd.DataFrame(data, columns=columns)

I = [{}, {'X0'}]
tester = 'chisq'

alpha = 0.05

D, _ = CPC(df_binary, tester, I, alpha=alpha, data_names=columns)
cpc_adj = D.graph

# Visualize the resulting graph
visualize_graph_color(D, name= 'cpc_output_p{}'.format(alpha)) 