from __future__ import annotations

import pandas as pd
from dash import Dash, dash_table


df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/solar.csv")
app = Dash(__name__)
app.layout = dash_table.DataTable(df.to_dict("records"), [{"name": i, "id": i} for i in df.columns])

if __name__ == "__main__":
    app.run_server(debug=True, host ="0.0.0.0")