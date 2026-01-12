import pandas as pd

def demo_paddlers():
    return pd.DataFrame({
        "name":["Ana","Ben","Carlos","Diana","Eve","Gina","Hiro","Frank","Ivan"],
        "role":["pacer","pacer","pacer",
                "regular","regular","regular","regular",
                "steerer","steerer"]
    })
