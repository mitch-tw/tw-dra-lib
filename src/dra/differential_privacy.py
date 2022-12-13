import pandas as pd

database = pd.DataFrame(
    [
        ('Sara Gray', 8, False, False, False),
        ('Joseph Collins', 18, False, True, True),
        ('Vincent Porter', 24, True, False, True),
        ('Tiffany Brown', 30, True, True, True),
        ('Brenda Small', 36, True, False, False),
        ('Dr. Tina Ayala', 66, True, False, False),
        ('Rodney Gonzalez', 84, True, True, False),
    ],
    columns=['name', 'age', 'married', 'smoker', 'employed'],
)
