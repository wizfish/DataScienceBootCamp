import pandas as pd

# ----------------------------
# Part 1: Create Custom Dataset
# ----------------------------
data = {
    "Name": ["Amina", "Biniam", "Catherine", "Daniel", "Eden",
             "Fikadu", "Grace", "Hassan", "Ivy", "John",
             "Kalkidan", "Liya", "Meles", "Nadia", "Omar"],
    "Age": [20, 21, 19, 22, 20, 23, 21, 24, 19, 22, 20, 21, 23, 19, 24],
    "Department": ["CSE", "ECE", "CSE", "ME", "CE",
                   "CSE", "ECE", "ME", "CE", "CSE",
                   "ECE", "CE", "ME", "CSE", "ECE"],
    "Score": [88, 79, 91, 85, 76, 90, 82, 74, 89, 95, 80, 87, 73, 92, 78],
    "City": ["Addis", "Dire Dawa", "Adama", "Mekelle", "Bahir Dar",
             "Addis", "Adama", "Mekelle", "Addis", "Dire Dawa",
             "Bahir Dar", "Adama", "Mekelle", "Addis", "Bahir Dar"]
}

df_custom = pd.DataFrame(data, index=[
    "S01", "S02", "S03", "S04", "S05",
    "S06", "S07", "S08", "S09", "S10",
    "S11", "S12", "S13", "S14", "S15"
])

print("Custom Dataset:")
print(df_custom)

# ----------------------------
# Part 2: Titanic Dataset
# ----------------------------
titanic = pd.read_csv("train.csv")

# Exploration
print("\nHead:")
print(titanic.head())

print("\nInfo:")
print(titanic.info())

print("\nDescribe:")
print(titanic.describe())

# Cleaning
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
titanic["Embarked"] = titanic["Embarked"].fillna(titanic["Embarked"].mode()[0])
titanic = titanic.drop(columns=["Cabin"])
titanic = titanic.drop_duplicates()

# Analysis
print("\nSurvival rate by gender:")
print(titanic.groupby("Sex")["Survived"].mean())

print("\nSurvival rate by class:")
print(titanic.groupby("Pclass")["Survived"].mean())

print("\nAverage age per class:")
print(titanic.groupby("Pclass")["Age"].mean())

# Age groups
titanic["AgeGroup"] = pd.cut(
    titanic["Age"],
    bins=[0, 12, 18, 30, 50, 80],
    labels=["Child", "Teen", "Young Adult", "Adult", "Senior"]
)

print("\nSurvival rate by age group:")
print(titanic.groupby("AgeGroup")["Survived"].mean())

# Filtering
print("\nFemale passengers who survived:")
print(titanic[(titanic["Sex"] == "female") & (titanic["Survived"] == 1)])

print("\nChildren who survived:")
print(titanic[(titanic["Age"] <= 12) & (titanic["Survived"] == 1)])

print("\n1st class passengers who survived:")
print(titanic[(titanic["Pclass"] == 1) & (titanic["Survived"] == 1)])

print("\nSurvival rate by sex and class:")
print(titanic.groupby(["Sex", "Pclass"])["Survived"].mean())