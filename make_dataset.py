import pandas as pd
import numpy as np

# Set seed so data is consistent
np.random.seed(42)
n_students = 300

# 1. Generate realistic CGPA between 5.5 and 9.8
cgpa = np.round(np.random.uniform(5.5, 9.8, n_students), 2)

# 2. Generate skills score between 1 and 10
skills = np.random.randint(2, 11, n_students)

# 3. Generate number of internships between 0 and 4
internships = np.random.randint(0, 5, n_students)

# 4. Logical placement formula (High CGPA + Skills + Internships = Placed)
placement_score = (cgpa * 0.45) + (skills * 0.35) + (internships * 0.6)
# If score is above 6.5, student is placed (1), otherwise not placed (0)
placed = (placement_score + np.random.normal(0, 0.5, n_students) > 6.5).astype(int)

# 5. Save to placement.csv
df = pd.DataFrame({
    "cgpa": cgpa,
    "skills": skills,
    "internships": internships,
    "placed": placed
})

df.to_csv("placement.csv", index=False)
print("placement.csv successfully created with 300 student records!") 