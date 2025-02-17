import numpy as np
import pandas as pd

def ensure_interventions(df: pd.DataFrame, intervention_list: list) -> pd.DataFrame:
    """
    Ensure that each intervention in `intervention_list` has its own column in df.
    If an intervention column does not exist, create it with zeros (meaning 'not applied').

    Args:
      df: Input DataFrame.
      intervention_list: List of strings naming each intervention.

    Returns:
      DataFrame with columns for each intervention.
    """
    for intervention in intervention_list:
        if intervention not in df.columns:
            df[intervention] = 0  # default: no intervention
    return df

def simulate_interventions(df: pd.DataFrame, intervention_list: list, probability=0.1) -> pd.DataFrame:
    """
    Optionally simulate random interventions. For demonstration or testing.
    We assign '1' with some probability to each intervention.

    Args:
      df: Input DataFrame.
      intervention_list: List of intervention names.
      probability: Probability of assigning an intervention at each row.

    Returns:
      DataFrame with random interventions assigned.
    """
    np.random.seed(42)
    for intervention in intervention_list:
        if df[intervention].sum() == 0:  # only simulate if column is all zeros
            df[intervention] = np.random.choice(
                [0, 1],
                size=len(df),
                p=[1-probability, probability]
            )
    return df
